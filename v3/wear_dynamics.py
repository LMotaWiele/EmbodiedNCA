"""
wear_dynamics.py
================
Monotonic wear accumulator and effective gen_rate wrapper for v3.

Wear never decreases. It accrues ambiently every step and faster while a
deliberate boost is active. Effective generation rate degrades toward a
nonzero floor:

    wear += wear_ambient_rate + (wear_boost_rate if boosted_this_step else 0)
    effective_gen_rate = gen_rate_base * max(gen_floor_frac,
                                             1.0 - wear_decay_coeff * wear)

The sole integration point with v0 is passing `effective_gen_rate` into the
existing, unmodified `step_resource(..., gen_rate=effective_gen_rate)`.
Does not modify grid_world.py, SimpleNCA, or v1/v2 models.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V2_ROOT = REPO_ROOT / "v2"
for p in (str(V0_ROOT), str(V2_ROOT), str(V3_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_world import (  # noqa: E402
    DEFAULT_GEN_RATE,
    ResourceGrid,
    diffuse,
    gated_nca_step,
    step_resource,
)


@dataclass
class WearConfig:
    """Global wear parameters (one scalar degrading both wells uniformly)."""

    wear_ambient_rate: float = 0.002
    wear_boost_rate: float = 0.008
    wear_decay_coeff: float = 0.10
    gen_floor_frac: float = 0.40  # must be > 0; wells degrade to a reduced floor
    gen_rate_base: float = DEFAULT_GEN_RATE
    wear0: float = 0.0

    def __post_init__(self) -> None:
        if self.gen_floor_frac <= 0.0:
            raise ValueError(
                f"gen_floor_frac must be > 0 (got {self.gen_floor_frac}); "
                "a zero floor guarantees eventual death for every policy and "
                "makes Check 2 uninformative."
            )
        if self.gen_floor_frac > 1.0:
            raise ValueError(f"gen_floor_frac must be ≤ 1 (got {self.gen_floor_frac})")
        if self.wear_ambient_rate < 0 or self.wear_boost_rate < 0:
            raise ValueError("wear rates must be non-negative")
        if self.wear_decay_coeff < 0:
            raise ValueError("wear_decay_coeff must be non-negative")


class WearState:
    """Monotonic wear scalar + effective gen_rate."""

    def __init__(self, cfg: Optional[WearConfig] = None):
        self.cfg = cfg or WearConfig()
        self.wear = float(self.cfg.wear0)

    def effective_gen_rate(self) -> float:
        """gen_rate_base * max(gen_floor_frac, 1 - wear_decay_coeff * wear)."""
        raw = 1.0 - self.cfg.wear_decay_coeff * self.wear
        scale = max(self.cfg.gen_floor_frac, raw)
        return float(self.cfg.gen_rate_base * scale)

    def floor_gen_rate(self) -> float:
        return float(self.cfg.gen_rate_base * self.cfg.gen_floor_frac)

    def wear_at_floor(self) -> float:
        """Wear level at which effective_gen_rate first hits the floor."""
        # 1 - coeff * wear = floor  =>  wear = (1 - floor) / coeff
        if self.cfg.wear_decay_coeff <= 0:
            return float("inf")
        return (1.0 - self.cfg.gen_floor_frac) / self.cfg.wear_decay_coeff

    def ambient_steps_to_floor(self) -> float:
        """Steps of ambient-only accrual to reach the floor (closed form)."""
        if self.cfg.wear_ambient_rate <= 0:
            return float("inf")
        remaining = max(0.0, self.wear_at_floor() - self.wear)
        return remaining / self.cfg.wear_ambient_rate

    def step(self, boosted_this_step: bool) -> float:
        """
        Accrue wear for one timestep. Returns the *new* effective_gen_rate
        after the update (use for the resource step on this same timestep).
        """
        self.wear = self.wear + self.cfg.wear_ambient_rate
        if boosted_this_step:
            self.wear = self.wear + self.cfg.wear_boost_rate
        # Wear never decreases — no clamp-to-zero, no repair.
        return self.effective_gen_rate()

    def snapshot(self) -> dict:
        return {
            "wear": self.wear,
            "effective_gen_rate": self.effective_gen_rate(),
            "floor_gen_rate": self.floor_gen_rate(),
            "wear_at_floor": self.wear_at_floor(),
            "gen_rate_base": self.cfg.gen_rate_base,
            "gen_floor_frac": self.cfg.gen_floor_frac,
        }


def effective_gen_rate_formula(
    wear: float,
    gen_rate_base: float,
    wear_decay_coeff: float,
    gen_floor_frac: float,
) -> float:
    """Pure function form of the effective-gen_rate mapping (for sweeps/tests)."""
    return float(
        gen_rate_base * max(gen_floor_frac, 1.0 - wear_decay_coeff * wear)
    )


@torch.no_grad()
def step_with_wear(
    world: ResourceGrid,
    wear_state: WearState,
    boost_map: Optional[torch.Tensor] = None,
    gate_mode: str = "gated",
    direct_boost_cost: float = 0.0,
    boost_location: Optional[Tuple[int, int]] = None,
    charge_direct_cost: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    One timestep identical in structure to ResourceGrid.step / step_with_boost,
    with two v3 additions:

      1. wear accrues; effective_gen_rate is passed into step_resource
      2. optional direct R charge at boost_location (Option A) when
         charge_direct_cost is True (only when a boost is actually applied)

    Does not modify grid_world.py. Returns (S, update_magnitude_map, effective_gen_rate).
    """
    from boost_cost import apply_direct_boost_cost  # local import; sibling module

    cfg = world.config
    boosted_this_step = boost_map is not None and gate_mode == "gated"

    # 0. diffuse S
    world.S = diffuse(world.S, cfg.diffusion_rate)

    # 1. gated NCA step (optional additive boost on gate)
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

    # 2. consume resource on pre-gate ‖update‖ (v0 accounting, unchanged)
    consumption = cfg.consumption_rate * update_magnitude_map
    world.R = (world.R - consumption).clamp(min=0.0)

    # 2b. direct boost cost (Option A) — additive, only when charged
    if charge_direct_cost and direct_boost_cost > 0 and boost_location is not None:
        apply_direct_boost_cost(
            world.R, boost_location[0], boost_location[1], direct_boost_cost
        )

    # 3. wear accrual → effective gen_rate → step_resource (v0 unmodified call)
    egr = wear_state.step(boosted_this_step=boosted_this_step)
    world.R = step_resource(
        world.R,
        world.generation_mask,
        decay_rate=cfg.decay_rate,
        gen_rate=egr,  # <-- the entire v0 integration point
        R_max=cfg.R_max,
        resource_diffusion=cfg.resource_diffusion,
    )
    world.t += 1
    return world.S, update_magnitude_map, egr


def closed_form_ambient_trajectory(
    cfg: WearConfig,
    horizon: int,
) -> Tuple[list, list]:
    """
    Closed-form ambient-only wear and effective_gen_rate over `horizon` steps
    (no boost). Useful for Check 1 sanity without running the full world.
    """
    wears = []
    egrs = []
    w = float(cfg.wear0)
    for _ in range(horizon + 1):
        wears.append(w)
        egrs.append(
            effective_gen_rate_formula(
                w, cfg.gen_rate_base, cfg.wear_decay_coeff, cfg.gen_floor_frac
            )
        )
        w = w + cfg.wear_ambient_rate
    return wears, egrs
