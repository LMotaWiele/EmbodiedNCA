"""Per-step measurements. Dead patterns (mass < DEATH_MASS_EPS) record NaN, not fallback averages."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# Mass below this is treated as "pattern gone" for derived metrics.
DEATH_MASS_EPS = 1e-6
DEFAULT_ACTIVATION_THRESHOLD = 0.05


def activation_map(S: torch.Tensor) -> torch.Tensor:
    """Per-cell L2 magnitude over channels. S: [B, C, H, W] -> [B, H, W]."""
    return S.norm(dim=1)


def self_mass(S: torch.Tensor, threshold: float = DEFAULT_ACTIVATION_THRESHOLD) -> float:
    """
    Sum of per-cell magnitude restricted to cells above `threshold`.
    Proxy for "how much pattern still exists."
    """
    mag = activation_map(S)  # [B, H, W]
    mask = mag > threshold
    return float((mag * mask.float()).sum().item())


def is_alive(mass: float, eps: float = DEATH_MASS_EPS) -> bool:
    return mass > eps


def coherence(
    S: torch.Tensor,
    S0: torch.Tensor,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    use_union_support: bool = True,
    mass: Optional[float] = None,
) -> float:
    """Cosine(S, S0) on active support. Dead patterns return NaN, not a residual cosine on noise."""
    if mass is None:
        mass = self_mass(S, threshold=threshold)
    if not is_alive(mass):
        return float("nan")

    mag = activation_map(S)
    mag0 = activation_map(S0)
    if use_union_support:
        support = (mag > threshold) | (mag0 > threshold)
    else:
        support = mag0 > threshold

    if support.sum() == 0:
        return float("nan")

    support_c = support.unsqueeze(1).float()
    a = (S * support_c).reshape(-1)
    b = (S0 * support_c).reshape(-1)
    denom = a.norm() * b.norm()
    if denom.item() < 1e-12:
        return float("nan")
    return float((a @ b / denom).item())


def total_resource(R: torch.Tensor) -> float:
    return float(R.sum().item())


def pattern_centroid(
    S: torch.Tensor,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    mass: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Center of mass of the pattern, weighted by per-cell magnitude.
    Returns (nan, nan) if the pattern is dead — never a grid-center fallback.
    """
    if mass is None:
        mass = self_mass(S, threshold=threshold)
    if not is_alive(mass):
        return float("nan"), float("nan")

    mag = activation_map(S)[0]  # [H, W]
    weights = mag * (mag > threshold).float()
    total = weights.sum()
    if total.item() < 1e-12:
        return float("nan"), float("nan")

    H, W = mag.shape
    ys = torch.arange(H, device=mag.device, dtype=mag.dtype)
    xs = torch.arange(W, device=mag.device, dtype=mag.dtype)
    cy = (weights.sum(dim=1) * ys).sum() / total
    cx = (weights.sum(dim=0) * xs).sum() / total
    return float(cy.item()), float(cx.item())


def sample_R_at(R: torch.Tensor, y: float, x: float) -> float:
    """Bilinear sample of R at continuous (y, x). R: [B, 1, H, W]."""
    if math.isnan(y) or math.isnan(x):
        return float("nan")
    _, _, H, W = R.shape
    y0 = int(max(0, min(H - 1, int(y))))
    x0 = int(max(0, min(W - 1, int(x))))
    y1 = min(H - 1, y0 + 1)
    x1 = min(W - 1, x0 + 1)
    wy = y - y0
    wx = x - x0
    r = R[0, 0]
    v00 = r[y0, x0]
    v01 = r[y0, x1]
    v10 = r[y1, x0]
    v11 = r[y1, x1]
    v0 = v00 * (1 - wx) + v01 * wx
    v1 = v10 * (1 - wx) + v11 * wx
    return float((v0 * (1 - wy) + v1 * wy).item())


def local_resource_at_pattern_centroid(
    R: torch.Tensor,
    S: torch.Tensor,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    mass: Optional[float] = None,
) -> float:
    cy, cx = pattern_centroid(S, threshold=threshold, mass=mass)
    return sample_R_at(R, cy, cx)


def distance_to_nearest_well(
    S: torch.Tensor,
    wells: Sequence[Tuple[int, int]],
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    mass: Optional[float] = None,
) -> float:
    cy, cx = pattern_centroid(S, threshold=threshold, mass=mass)
    if math.isnan(cy) or math.isnan(cx) or not wells:
        return float("nan")
    best = min(((cy - wy) ** 2 + (cx - wx) ** 2) ** 0.5 for wy, wx in wells)
    return float(best)


def compute_all(
    S: torch.Tensor,
    S0: torch.Tensor,
    R: torch.Tensor,
    wells: Sequence[Tuple[int, int]],
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    mean_update_magnitude: float = 0.0,
    total_consumption: float = 0.0,
    t: int = 0,
) -> Dict[str, float]:
    """
    One row of metrics. Pattern-relative fields are NaN once mass is dead.
    `total_resource` and update/consumption accounting remain defined always
    (they are world-level, not pattern-relative).
    """
    mass = self_mass(S, threshold=threshold)
    alive = 1.0 if is_alive(mass) else 0.0
    return {
        "t": t,
        "self_mass": mass,
        "alive": alive,
        "coherence": coherence(S, S0, threshold=threshold, mass=mass),
        "total_resource": total_resource(R),
        "local_resource_at_pattern_centroid": local_resource_at_pattern_centroid(
            R, S, threshold=threshold, mass=mass
        ),
        "distance_to_nearest_well": distance_to_nearest_well(
            S, wells, threshold=threshold, mass=mass
        ),
        "mean_update_magnitude": mean_update_magnitude,
        "total_consumption": total_consumption,
    }


def time_of_death(
    history: List[Dict[str, float]], eps: float = DEATH_MASS_EPS
) -> Optional[int]:
    """First timestep where self_mass drops to/below eps, or None if survived."""
    for row in history:
        if row.get("self_mass", 0.0) <= eps:
            return int(row["t"])
    return None


def last_alive_value(
    history: List[Dict[str, float]], key: str, eps: float = DEATH_MASS_EPS
) -> float:
    """Last finite value of `key` while the pattern was alive; else NaN."""
    last = float("nan")
    for row in history:
        if row.get("self_mass", 0.0) <= eps:
            break
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            last = float(v)
    return last


def save_history_csv(history: List[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("")
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            # Write NaN as empty-friendly string; csv will serialize float nan as "nan"
            writer.writerow(row)


def history_column(history: List[Dict[str, float]], key: str) -> List[float]:
    return [float(row[key]) for row in history]


def finite_mean(values: Sequence[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))
