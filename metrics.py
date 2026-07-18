"""
metrics.py
==========
Per-step measurements for the resource-gated embedded-pattern NCA.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def activation_map(S: torch.Tensor) -> torch.Tensor:
    """Per-cell L2 magnitude over channels. S: [B, C, H, W] -> [B, H, W]."""
    return S.norm(dim=1)


def self_mass(S: torch.Tensor, threshold: float = 0.05) -> float:
    """
    Sum of per-cell magnitude restricted to cells above `threshold`.
    Proxy for "how much pattern still exists."
    """
    mag = activation_map(S)  # [B, H, W]
    mask = mag > threshold
    return float((mag * mask.float()).sum().item())


def coherence(
    S: torch.Tensor,
    S0: torch.Tensor,
    threshold: float = 0.05,
    use_union_support: bool = True,
) -> float:
    """
    Cosine similarity between current S and the t=0 pattern, restricted to
    the active support (cells where either state exceeds the threshold).

    Returns a scalar in roughly [-1, 1]; 1.0 means still recognizably itself.
    """
    # Flatten over channels + spatial (batch assumed 1 for v0)
    mag = activation_map(S)
    mag0 = activation_map(S0)
    if use_union_support:
        support = (mag > threshold) | (mag0 > threshold)
    else:
        support = mag0 > threshold

    if support.sum() == 0:
        return 0.0

    # Broadcast support over channels: [B, 1, H, W]
    support_c = support.unsqueeze(1).float()
    a = (S * support_c).reshape(-1)
    b = (S0 * support_c).reshape(-1)
    denom = a.norm() * b.norm()
    if denom.item() < 1e-12:
        return 0.0
    return float((a @ b / denom).item())


def total_resource(R: torch.Tensor) -> float:
    return float(R.sum().item())


def pattern_centroid(
    S: torch.Tensor,
    threshold: float = 0.05,
) -> Tuple[float, float]:
    """
    Center of mass of the pattern, weighted by per-cell magnitude.
    Returns (y, x) in continuous grid coordinates.
    Falls back to geometric grid center if mass is empty.
    """
    mag = activation_map(S)[0]  # [H, W]
    weights = mag * (mag > threshold).float()
    total = weights.sum()
    H, W = mag.shape
    if total.item() < 1e-12:
        return (H - 1) / 2.0, (W - 1) / 2.0

    ys = torch.arange(H, device=mag.device, dtype=mag.dtype)
    xs = torch.arange(W, device=mag.device, dtype=mag.dtype)
    cy = (weights.sum(dim=1) * ys).sum() / total
    cx = (weights.sum(dim=0) * xs).sum() / total
    return float(cy.item()), float(cx.item())


def sample_R_at(R: torch.Tensor, y: float, x: float) -> float:
    """Bilinear sample of R at continuous (y, x). R: [B, 1, H, W]."""
    _, _, H, W = R.shape
    # Clamp to valid range for grid_sample-style coords
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
    threshold: float = 0.05,
) -> float:
    cy, cx = pattern_centroid(S, threshold=threshold)
    return sample_R_at(R, cy, cx)


def distance_to_nearest_well(
    S: torch.Tensor,
    wells: Sequence[Tuple[int, int]],
    threshold: float = 0.05,
) -> float:
    cy, cx = pattern_centroid(S, threshold=threshold)
    if not wells:
        return float("nan")
    best = min(((cy - wy) ** 2 + (cx - wx) ** 2) ** 0.5 for wy, wx in wells)
    return float(best)


def compute_all(
    S: torch.Tensor,
    S0: torch.Tensor,
    R: torch.Tensor,
    wells: Sequence[Tuple[int, int]],
    threshold: float = 0.05,
    mean_update_magnitude: float = 0.0,
    t: int = 0,
) -> Dict[str, float]:
    return {
        "t": t,
        "self_mass": self_mass(S, threshold=threshold),
        "coherence": coherence(S, S0, threshold=threshold),
        "total_resource": total_resource(R),
        "local_resource_at_pattern_centroid": local_resource_at_pattern_centroid(
            R, S, threshold=threshold
        ),
        "distance_to_nearest_well": distance_to_nearest_well(
            S, wells, threshold=threshold
        ),
        "mean_update_magnitude": mean_update_magnitude,
    }


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
            writer.writerow(row)


def history_column(history: List[Dict[str, float]], key: str) -> List[float]:
    return [float(row[key]) for row in history]
