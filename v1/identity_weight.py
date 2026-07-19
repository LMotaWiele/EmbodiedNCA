"""
identity_weight.py
==================
Derive a per-cell identity_weight map from a trained (or ablated) SelfModel.

identity_weight[y, x] = max(0, baseline_coh − probed_coh)

Read-only: does not step the v0 simulator or write into S/R.

Operational ranking (v1 findings)
---------------------------------
Two different untrustworthy regimes (do not conflate):

1. **No signal** — plain background far from pattern/wells: weight ≈ 0
   (bg_p95 ≈ 0). Benign if you only look at support.
2. **False signal** — inter-well *equidistant boundary* (locus of cells with
   |d_well_A − d_well_B| small). Weight is *high* and confidently wrong; can
   exceed the pattern's own peak at typical operating distances. Hardcoded
   coordinates are layout-specific; compute from well positions at runtime.

Ranking must use `validity_mask` / `argmax_identity_weight` — never raw
full-grid argmax. Mass-only masks exclude (1) but not the danger of (2) on
full-grid maps; pass `wells=` so the equidistant locus is excluded too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

_V1_ROOT = Path(__file__).resolve().parent
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from rollout_dataset import make_perturbation_map  # noqa: E402
from self_model import SelfModel  # noqa: E402

# Match v0 activation threshold for "which cells are candidate probe sites"
DEFAULT_MASS_THRESHOLD = 0.05
# Cells with |d_A - d_B| < this (in grid units) are on the equidistant boundary
# between a pair of wells. ~0.75 catches the discrete grid locus for the
# default two-well layout (x+y=32 between (8,8) and (24,24)).
DEFAULT_EQUIDISTANT_EPS = 0.75


def mass_map(S: torch.Tensor) -> torch.Tensor:
    """Per-cell ‖S‖. Accepts [1,C,H,W] or [C,H,W]; returns [H,W]."""
    if S.dim() == 4:
        return S.norm(dim=1)[0]
    if S.dim() == 3:
        return S.norm(dim=0)
    raise ValueError(f"Expected S with 3 or 4 dims, got {S.shape}")


def equidistant_boundary_mask(
    H: int,
    W: int,
    wells: Sequence[Tuple[int, int]],
    eps: float = DEFAULT_EQUIDISTANT_EPS,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Boolean [H, W]: True where the cell lies on the equidistant boundary
    between *any pair* of wells — i.e. min over pairs of |d_i - d_j| < eps.

    For two wells at (8,8) and (24,24) this is the perpendicular bisector
    locus (on the discrete grid: cells with x+y ≈ 32), including (24,8),
    (8,24), and (16,16) — one phenomenon, not three special cases.

    Computed from well positions at runtime so it survives well relocation.
    """
    if len(wells) < 2:
        return torch.zeros(H, W, dtype=torch.bool, device=device)

    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    dists = []
    for wy, wx in wells:
        dists.append(torch.sqrt((yy - float(wy)) ** 2 + (xx - float(wx)) ** 2))

    on_boundary = torch.zeros(H, W, dtype=torch.bool, device=device)
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            on_boundary |= (dists[i] - dists[j]).abs() < eps
    return on_boundary


def support_mask(
    S: torch.Tensor,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
) -> torch.Tensor:
    """True where ‖S‖ > mass_threshold (pattern support — has signal to rank)."""
    return mass_map(S) > mass_threshold


def validity_mask(
    S: torch.Tensor,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
    wells: Optional[Sequence[Tuple[int, int]]] = None,
    equidistant_eps: float = DEFAULT_EQUIDISTANT_EPS,
    H: Optional[int] = None,
    W: Optional[int] = None,
) -> torch.Tensor:
    """
    Boolean [H, W] mask of cells where ranking/argmax is allowed.

    Conditions (all required):
      1. support: ‖S‖ > mass_threshold  — excludes *no-signal* background
      2. if wells given: not on equidistant boundary between any well pair —
         excludes *false-signal* inter-well locus (even if mass somehow nonzero)

    Mass-only masks are enough for "rank cells on the live pattern." Full-grid
    or diagnostic maps must pass wells so the equidistant locus cannot win.
    """
    m = support_mask(S, mass_threshold=mass_threshold)
    if wells is not None and len(wells) >= 2:
        if H is None or W is None:
            H, W = m.shape[-2], m.shape[-1]
        eq = equidistant_boundary_mask(
            H, W, wells, eps=equidistant_eps, device=m.device
        )
        m = m & ~eq
    return m


def apply_validity_mask(
    weight_map: torch.Tensor,
    mask: torch.Tensor,
    fill: float = float("nan"),
) -> torch.Tensor:
    """Return a copy of weight_map with invalid cells set to `fill` (default NaN)."""
    out = weight_map.clone().to(dtype=torch.float32)
    out = out.masked_fill(~mask.to(device=out.device), fill)
    return out


def support_restricted_weight(
    weight_map: torch.Tensor,
    S: torch.Tensor,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
    wells: Optional[Sequence[Tuple[int, int]]] = None,
    equidistant_eps: float = DEFAULT_EQUIDISTANT_EPS,
    fill: float = float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (masked_weights, validity_mask). Preferred form to ship with identity_weight.

    Pass `wells` so equidistant-boundary false-signal cells are excluded.
    """
    mask = validity_mask(
        S,
        mass_threshold=mass_threshold,
        wells=wells,
        equidistant_eps=equidistant_eps,
    ).to(device=weight_map.device)
    return apply_validity_mask(weight_map, mask, fill=fill), mask.cpu()


def argmax_identity_weight(
    weight_map: torch.Tensor,
    S: torch.Tensor,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
    wells: Optional[Sequence[Tuple[int, int]]] = None,
    equidistant_eps: float = DEFAULT_EQUIDISTANT_EPS,
) -> Dict:
    """
    Argmax restricted by validity_mask (support ± equidistant exclusion).

    Never compares against no-signal background or false-signal equidistant
    boundary cells.
    """
    w = weight_map.detach().float()
    if w.dim() != 2:
        raise ValueError(f"weight_map must be [H,W], got {w.shape}")
    H, W = w.shape
    mask = validity_mask(
        S,
        mass_threshold=mass_threshold,
        wells=wells,
        equidistant_eps=equidistant_eps,
        H=H,
        W=W,
    ).to(device=w.device)
    support = support_mask(S, mass_threshold=mass_threshold).to(device=w.device)
    eq = (
        equidistant_boundary_mask(H, W, wells, eps=equidistant_eps, device=w.device)
        if wells is not None and len(wells) >= 2
        else torch.zeros(H, W, dtype=torch.bool, device=w.device)
    )

    gflat = int(torch.argmax(w).item())
    gy, gx = divmod(gflat, W)
    gval = float(w[gy, gx].item())
    g_on_support = bool(support[gy, gx].item())
    g_on_equidistant = bool(eq[gy, gx].item())

    if not mask.any():
        return {
            "y": -1,
            "x": -1,
            "value": float("nan"),
            "on_support": False,
            "global_y": gy,
            "global_x": gx,
            "global_value": gval,
            "global_on_support": g_on_support,
            "global_on_equidistant_boundary": g_on_equidistant,
            "global_argmax_is_background": not g_on_support,
            "global_argmax_is_false_signal": g_on_equidistant and not g_on_support,
            "safe": False,
        }

    w_safe = w.clone().masked_fill(~mask, float("-inf"))
    sflat = int(torch.argmax(w_safe).item())
    sy, sx = divmod(sflat, W)
    sval = float(w[sy, sx].item())
    return {
        "y": sy,
        "x": sx,
        "value": sval,
        "on_support": True,
        "global_y": gy,
        "global_x": gx,
        "global_value": gval,
        "global_on_support": g_on_support,
        "global_on_equidistant_boundary": g_on_equidistant,
        "global_argmax_is_background": not g_on_support,
        "global_argmax_is_false_signal": g_on_equidistant and not g_on_support,
        "safe": True,
    }


@torch.no_grad()
def probe_identity_at_locations(
    self_model: SelfModel,
    S_t: torch.Tensor,
    R_t: torch.Tensor,
    locations: Sequence[Tuple[int, int]],
    probe_magnitude: float = 1.0,
    probe_sigma: float = 1.5,
    device: Optional[str] = None,
) -> Dict[Tuple[int, int], float]:
    """
    identity_weight at an explicit list of (y, x) sites only.

    Used for targeted diagnostics (e.g. multi-well saddles) without a dense map.
    """
    self_model.eval()
    if S_t.dim() == 3:
        S_t = S_t.unsqueeze(0)
    if R_t.dim() == 3:
        R_t = R_t.unsqueeze(0)
    if R_t.dim() == 2:
        R_t = R_t.view(1, 1, *R_t.shape)

    device = device or next(self_model.parameters()).device
    S_t = S_t.to(device)
    R_t = R_t.to(device)
    _, _, H, W = S_t.shape

    null = torch.zeros(1, 1, H, W, device=device)
    base_coh, _, _ = self_model(S_t, R_t, null)
    base = float(base_coh.item())

    out: Dict[Tuple[int, int], float] = {}
    if not locations:
        return out

    maps = [
        make_perturbation_map(H, W, y, x, probe_magnitude, device=device, sigma=probe_sigma)
        for y, x in locations
    ]
    P = torch.cat(maps, dim=0)
    B = len(locations)
    pred_coh, _, _ = self_model(
        S_t.expand(B, -1, -1, -1).contiguous(),
        R_t.expand(B, -1, -1, -1).contiguous(),
        P,
    )
    damage = (base - pred_coh).clamp(min=0.0)
    for i, (y, x) in enumerate(locations):
        out[(int(y), int(x))] = float(damage[i].item())
    return out


@torch.no_grad()
def compute_identity_weight(
    self_model: SelfModel,
    S_t: torch.Tensor,
    R_t: torch.Tensor,
    probe_magnitude: float = 1.0,
    probe_sigma: float = 1.5,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
    candidate_locations: Optional[Sequence[Tuple[int, int]]] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Compute H×W identity_weight map for a single live state.

    Args:
        self_model: trained or untrained SelfModel
        S_t: [1, C, H, W] or [C, H, W]
        R_t: [1, 1, H, W] or [1, H, W]
        probe_magnitude: amplitude of the Gaussian probe encoding
        probe_sigma: spatial width of the probe encoding
        mass_threshold: only probe cells with ‖S‖ > threshold (unless candidates given)
        candidate_locations: optional explicit (y, x) list; default = active mass cells
            plus a sparse background sample so empty cells are also scored
        device: inference device
        batch_size: probe batching for speed

    Returns:
        weight_map: [H, W] float tensor, clamped ≥ 0
    """
    self_model.eval()
    if S_t.dim() == 3:
        S_t = S_t.unsqueeze(0)
    if R_t.dim() == 3:
        R_t = R_t.unsqueeze(0)
    if R_t.dim() == 2:
        R_t = R_t.view(1, 1, *R_t.shape)

    device = device or next(self_model.parameters()).device
    S_t = S_t.to(device)
    R_t = R_t.to(device)
    _, _, H, W = S_t.shape

    # Baseline: null perturbation
    null = torch.zeros(1, 1, H, W, device=device)
    base_coh, _, _ = self_model(S_t, R_t, null)
    base = float(base_coh.item())

    if candidate_locations is None:
        mag = S_t.norm(dim=1)[0]  # [H, W]
        active = (mag > mass_threshold).nonzero(as_tuple=False)
        locs = [(int(y), int(x)) for y, x in active.tolist()]
        # Sparse background grid so empty cells get a score too (structure check)
        step = max(2, min(H, W) // 8)
        for y in range(0, H, step):
            for x in range(0, W, step):
                locs.append((y, x))
        # de-dupe
        locs = list(dict.fromkeys(locs))
    else:
        locs = list(candidate_locations)

    weight_map = torch.zeros(H, W, device=device)
    if not locs:
        return weight_map.cpu()

    # Batch probes
    for i0 in range(0, len(locs), batch_size):
        chunk = locs[i0 : i0 + batch_size]
        B = len(chunk)
        S_b = S_t.expand(B, -1, -1, -1).contiguous()
        R_b = R_t.expand(B, -1, -1, -1).contiguous()
        maps = []
        for y, x in chunk:
            maps.append(
                make_perturbation_map(
                    H, W, y, x, probe_magnitude, device=device, sigma=probe_sigma
                )
            )
        P_b = torch.cat(maps, dim=0)
        pred_coh, _, _ = self_model(S_b, R_b, P_b)
        damage = (base - pred_coh).clamp(min=0.0)
        for j, (y, x) in enumerate(chunk):
            # max if location appears multiple times
            weight_map[y, x] = torch.maximum(weight_map[y, x], damage[j])

    return weight_map.cpu()


@torch.no_grad()
def dense_identity_weight(
    self_model: SelfModel,
    S_t: torch.Tensor,
    R_t: torch.Tensor,
    probe_magnitude: float = 1.0,
    probe_sigma: float = 1.5,
    stride: int = 1,
    device: Optional[str] = None,
    batch_size: int = 64,
    active_only: bool = False,
    mass_threshold: float = DEFAULT_MASS_THRESHOLD,
) -> torch.Tensor:
    """
    Probe on a spatial grid for visualization.

    active_only=True restricts probes to cells with ‖S‖ > mass_threshold
    (matches training distribution and the v1 spec's candidate set). Background
    stays 0 rather than OOD empty-cell probes.
    """
    if S_t.dim() == 3:
        S_t = S_t.unsqueeze(0)
    H, W = S_t.shape[-2], S_t.shape[-1]
    if active_only:
        mag = S_t.norm(dim=1)[0]
        locs = [
            (int(y), int(x))
            for y, x in (mag > mass_threshold).nonzero(as_tuple=False).tolist()
        ]
        # still include a coarse empty sample for contrast plots (optional small set)
        step = max(4, min(H, W) // 4)
        for y in range(0, H, step):
            for x in range(0, W, step):
                if float(mag[y, x]) <= mass_threshold:
                    locs.append((y, x))
        locs = list(dict.fromkeys(locs))
    else:
        locs = [(y, x) for y in range(0, H, stride) for x in range(0, W, stride)]
    return compute_identity_weight(
        self_model,
        S_t,
        R_t,
        probe_magnitude=probe_magnitude,
        probe_sigma=probe_sigma,
        candidate_locations=locs,
        device=device,
        batch_size=batch_size,
    )
