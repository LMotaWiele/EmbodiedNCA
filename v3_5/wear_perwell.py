"""
wear_perwell.py
===============
Per-well wear for v3.5: two independent wear scalars (one per well).

Ambient accrual is shared (both wells gain equally every step). Boost-driven
accrual hits only the well *nearest* the boosted location — so *where* you
boost finally has a wear consequence, not just *whether*.

Integration with v0 is via a modified spatial generation **mask**, not a
scalar gen_rate override:

    effective_multiplier[i] = max(gen_floor_frac, 1 - wear_decay_coeff * wear[i])
    effective_generation_mask = sum_i generation_mask_i * effective_multiplier[i]
    step_resource(R, generation_mask=effective_generation_mask,
                  gen_rate=gen_rate_base, ...)   # v0 unmodified signature

Does not modify grid_world.py, SimpleNCA, or v1/v2 checkpoints.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

V35_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V35_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V3_ROOT = REPO_ROOT / "v3"
for p in (str(V0_ROOT), str(V3_ROOT), str(V35_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_world import (  # noqa: E402
    DEFAULT_GEN_RATE,
    DEFAULT_WELL_SIGMA,
    DEFAULT_WELLS,
    ResourceGrid,
    diffuse,
    gated_nca_step,
    step_resource,
)

# Reuse Option A direct charge from v3 (sibling, unmodified import)
from boost_cost import apply_direct_boost_cost  # noqa: E402


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------


@dataclass
class PerWellWearConfig:
    """Per-well wear parameters (two wells; same rates for both)."""

    wear_ambient_rate: float = 0.001  # v3 OP
    wear_boost_rate: float = 0.004  # v3 OP
    wear_decay_coeff: float = 0.20  # v3 OP
    gen_floor_frac: float = 0.40  # v3 OP
    gen_rate_base: float = DEFAULT_GEN_RATE
    well_sigma: float = DEFAULT_WELL_SIGMA
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS
    wear0: Tuple[float, ...] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.gen_floor_frac <= 0.0 or self.gen_floor_frac > 1.0:
            raise ValueError(
                f"gen_floor_frac must be in (0, 1] (got {self.gen_floor_frac})"
            )
        if self.wear_ambient_rate < 0 or self.wear_boost_rate < 0:
            raise ValueError("wear rates must be non-negative")
        if self.wear_decay_coeff < 0:
            raise ValueError("wear_decay_coeff must be non-negative")
        n = len(self.wells)
        if len(self.wear0) != n:
            # broadcast scalar-like single value
            if len(self.wear0) == 1:
                object.__setattr__(self, "wear0", tuple(self.wear0[0] for _ in range(n)))
            else:
                raise ValueError(
                    f"wear0 length {len(self.wear0)} != n_wells {n}"
                )


def nearest_well_index(
    y: float,
    x: float,
    wells: Sequence[Tuple[int, int]],
) -> int:
    """
    Index of the well nearest to (y, x). Ties broken by lower index.

    Reuses the same Euclidean distance used by v1 equidistant-boundary /
    nearest-well geometry — do not reimplement a different metric.
    """
    best_i = 0
    best_d2 = float("inf")
    for i, (wy, wx) in enumerate(wells):
        d2 = (float(y) - float(wy)) ** 2 + (float(x) - float(wx)) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def nearest_well_distance(
    y: float,
    x: float,
    wells: Sequence[Tuple[int, int]],
) -> float:
    """Euclidean distance to the nearest well (v1 convention)."""
    return min(math.hypot(float(y) - wy, float(x) - wx) for wy, wx in wells)


def make_per_well_masks(
    H: int,
    W: int,
    wells: Sequence[Tuple[int, int]],
    device: str | torch.device = "cpu",
    sigma: float = DEFAULT_WELL_SIGMA,
) -> List[torch.Tensor]:
    """
    One soft generation field [1,1,H,W] per well (peak 1.0 at the well cell).

    Same Gaussian footprint as v0 make_generation_mask, but uncombined so each
    well's multiplier can scale its own contribution independently.
    """
    masks: List[torch.Tensor] = []
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    for y, x in wells:
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"Well ({y}, {x}) outside grid {H}x{W}")
        mask = torch.zeros(1, 1, H, W, device=device)
        if sigma <= 0:
            mask[0, 0, y, x] = 1.0
        else:
            dist2 = (yy - float(y)) ** 2 + (xx - float(x)) ** 2
            blob = torch.exp(-dist2 / (2.0 * sigma ** 2))
            mask[0, 0] = blob
        masks.append(mask)
    return masks


def effective_multiplier(
    wear: float,
    wear_decay_coeff: float,
    gen_floor_frac: float,
) -> float:
    """max(gen_floor_frac, 1 - wear_decay_coeff * wear)."""
    return float(max(gen_floor_frac, 1.0 - wear_decay_coeff * wear))


def build_effective_generation_mask(
    per_well_masks: Sequence[torch.Tensor],
    multipliers: Sequence[float],
) -> torch.Tensor:
    """
    sum_i generation_mask_i * effective_multiplier[i]

    Keeps gen_rate_base a scalar; all degradation folds into the spatial mask.
    """
    if len(per_well_masks) != len(multipliers):
        raise ValueError("per_well_masks and multipliers length mismatch")
    out = torch.zeros_like(per_well_masks[0])
    for m, mult in zip(per_well_masks, multipliers):
        out = out + m * float(mult)
    return out


class PerWellWearState:
    """Monotonic wear vector [wear_A, wear_B, ...] + effective generation mask."""

    def __init__(
        self,
        cfg: Optional[PerWellWearConfig] = None,
        H: int = 32,
        W: int = 32,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg or PerWellWearConfig()
        n = len(self.cfg.wells)
        self.wear = [float(w) for w in self.cfg.wear0]
        if len(self.wear) != n:
            self.wear = [0.0] * n
        self.H = H
        self.W = W
        self.device = device
        self.per_well_masks = make_per_well_masks(
            H, W, self.cfg.wells, device=device, sigma=self.cfg.well_sigma
        )
        self._cached_mask: Optional[torch.Tensor] = None
        self._cache_key: Optional[Tuple[float, ...]] = None

    @property
    def n_wells(self) -> int:
        return len(self.cfg.wells)

    def multipliers(self) -> List[float]:
        return [
            effective_multiplier(w, self.cfg.wear_decay_coeff, self.cfg.gen_floor_frac)
            for w in self.wear
        ]

    def effective_generation_mask(self) -> torch.Tensor:
        key = tuple(self.wear)
        if self._cached_mask is not None and self._cache_key == key:
            return self._cached_mask
        mults = self.multipliers()
        mask = build_effective_generation_mask(self.per_well_masks, mults)
        self._cached_mask = mask
        self._cache_key = key
        return mask

    def wear_at_floor(self) -> float:
        """Wear level at which a single well's multiplier first hits the floor."""
        if self.cfg.wear_decay_coeff <= 0:
            return float("inf")
        return (1.0 - self.cfg.gen_floor_frac) / self.cfg.wear_decay_coeff

    def floor_gen_contribution(self) -> float:
        """gen_rate_base * gen_floor_frac (scalar contribution scale at floor)."""
        return float(self.cfg.gen_rate_base * self.cfg.gen_floor_frac)

    def ambient_steps_to_floor(self, well_idx: int = 0) -> float:
        if self.cfg.wear_ambient_rate <= 0:
            return float("inf")
        remaining = max(0.0, self.wear_at_floor() - self.wear[well_idx])
        return remaining / self.cfg.wear_ambient_rate

    def is_floored(self, well_idx: int, eps: float = 1e-9) -> bool:
        mult = self.multipliers()[well_idx]
        return mult <= self.cfg.gen_floor_frac + eps

    def all_floored(self, eps: float = 1e-9) -> bool:
        return all(self.is_floored(i, eps=eps) for i in range(self.n_wells))

    def step(
        self,
        boosted_this_step: bool,
        boost_location: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Accrue ambient on all wells; if boosted, also accrue boost rate on the
        nearest well to boost_location. Returns the new effective generation mask.
        """
        amb = self.cfg.wear_ambient_rate
        for i in range(self.n_wells):
            self.wear[i] = self.wear[i] + amb
        if boosted_this_step and boost_location is not None:
            ni = nearest_well_index(
                boost_location[0], boost_location[1], self.cfg.wells
            )
            self.wear[ni] = self.wear[ni] + self.cfg.wear_boost_rate
        self._cached_mask = None
        self._cache_key = None
        return self.effective_generation_mask()

    def snapshot(self) -> dict:
        mults = self.multipliers()
        return {
            "wear": list(self.wear),
            "multipliers": mults,
            "wear_at_floor": self.wear_at_floor(),
            "gen_rate_base": self.cfg.gen_rate_base,
            "gen_floor_frac": self.cfg.gen_floor_frac,
            "wells": list(self.cfg.wells),
        }

    def clone(self) -> "PerWellWearState":
        other = PerWellWearState(
            cfg=self.cfg, H=self.H, W=self.W, device=self.device
        )
        other.wear = list(self.wear)
        other._cached_mask = None
        other._cache_key = None
        return other


@torch.no_grad()
def step_with_perwell_wear(
    world: ResourceGrid,
    wear_state: PerWellWearState,
    boost_map: Optional[torch.Tensor] = None,
    gate_mode: str = "gated",
    direct_boost_cost: float = 0.0,
    boost_location: Optional[Tuple[int, int]] = None,
    charge_direct_cost: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], torch.Tensor]:
    """
    One timestep under per-well wear + optional direct boost cost.

    Structure mirrors v3 step_with_wear / ResourceGrid.step:
      0. diffuse S
      1. gated NCA (optional additive boost)
      2. consume on pre-gate ‖update‖
      2b. optional direct R charge at boost_location
      3. wear accrual → effective generation mask → step_resource
         with gen_rate = gen_rate_base (scalar, unchanged)

    Returns (S, update_magnitude_map, wear_list, effective_generation_mask).
    """
    cfg = world.config
    boosted_this_step = boost_map is not None and gate_mode == "gated"

    # 0. diffuse S
    world.S = diffuse(world.S, cfg.diffusion_rate)

    # 1. gated NCA step
    if boosted_this_step:
        perception_out = world.nca.perception(world.S)
        update = world.nca.update_net(perception_out)
        update_magnitude_map = update.norm(dim=1, keepdim=True)
        gate_from_R = (world.R / cfg.R_max).clamp(0.0, 1.0)
        b = boost_map.to(device=world.S.device, dtype=world.S.dtype)
        if b.dim() == 3:
            b = b.unsqueeze(0)
        gate = (gate_from_R + b).clamp(0.0, 1.0)
        world.S = torch.clamp(world.S + gate * update, -10.0, 10.0)
    else:
        world.S, update_magnitude_map = gated_nca_step(
            world.nca, world.S, world.R, R_max=cfg.R_max, gate_mode=gate_mode
        )

    # 2. consume resource on pre-gate ‖update‖ (v0 accounting)
    consumption = cfg.consumption_rate * update_magnitude_map
    world.R = (world.R - consumption).clamp(min=0.0)

    # 2b. direct boost cost (Option A)
    if charge_direct_cost and direct_boost_cost > 0 and boost_location is not None:
        apply_direct_boost_cost(
            world.R, boost_location[0], boost_location[1], direct_boost_cost
        )

    # 3. wear → effective mask → step_resource (v0 unmodified call)
    #    boost_location for nearest-well attribution: use provided location, or
    #    if missing while boosted, fall back to argmax of boost_map.
    loc = boost_location
    if boosted_this_step and loc is None and boost_map is not None:
        bm = boost_map.detach()
        if bm.dim() == 4:
            bm = bm[0, 0]
        elif bm.dim() == 3:
            bm = bm[0]
        flat = bm.reshape(-1)
        idx = int(flat.argmax().item())
        loc = (idx // bm.shape[-1], idx % bm.shape[-1])

    eff_mask = wear_state.step(
        boosted_this_step=boosted_this_step,
        boost_location=loc if boosted_this_step else None,
    )
    # Keep mask on same device as R
    if eff_mask.device != world.R.device:
        eff_mask = eff_mask.to(world.R.device)

    world.R = step_resource(
        world.R,
        eff_mask,
        decay_rate=cfg.decay_rate,
        gen_rate=wear_state.cfg.gen_rate_base,  # scalar base; degradation in mask
        R_max=cfg.R_max,
        resource_diffusion=cfg.resource_diffusion,
    )
    world.t += 1
    return world.S, update_magnitude_map, list(wear_state.wear), eff_mask


def closed_form_ambient_perwell(
    cfg: PerWellWearConfig,
    horizon: int,
    well_idx: int = 0,
) -> Tuple[List[float], List[float]]:
    """Ambient-only wear[well_idx] and multiplier trajectory (no boost)."""
    wears: List[float] = []
    mults: List[float] = []
    w = float(cfg.wear0[well_idx] if well_idx < len(cfg.wear0) else 0.0)
    for _ in range(horizon + 1):
        wears.append(w)
        mults.append(
            effective_multiplier(w, cfg.wear_decay_coeff, cfg.gen_floor_frac)
        )
        w = w + cfg.wear_ambient_rate
    return wears, mults


def mask_value_at_well(
    mask: torch.Tensor,
    well: Tuple[int, int],
) -> float:
    """Scalar generation-mask value at a well cell."""
    y, x = well
    if mask.dim() == 4:
        return float(mask[0, 0, y, x].item())
    if mask.dim() == 3:
        return float(mask[0, y, x].item())
    if mask.dim() == 2:
        return float(mask[y, x].item())
    raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")
