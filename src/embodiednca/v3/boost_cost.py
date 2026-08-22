"""Direct R charge at the boost cell, in addition to v0 pre-gate consumption. Skip/never-boost is free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BoostCostConfig:
    """Direct R charge parameters (Option A)."""

    direct_boost_cost: float = 0.15  # R units charged at the boost cell
    # Charge once when the boost is first applied (armed), not every active step.
    charge_on_apply_only: bool = True

    def __post_init__(self) -> None:
        if self.direct_boost_cost < 0:
            raise ValueError(
                f"direct_boost_cost must be ≥ 0 (got {self.direct_boost_cost})"
            )


def apply_direct_boost_cost(
    R: torch.Tensor,
    y: int,
    x: int,
    cost: float,
) -> torch.Tensor:
    """
    Charge `cost` from R at grid cell (y, x), clamped at 0.

    R shapes accepted: [1,1,H,W], [1,H,W], [H,W].
    Returns the same tensor (in-place mutation of the storage used by the
    world so ResourceGrid.R stays consistent).
    """
    if cost <= 0:
        return R
    if R.dim() == 4:
        # [B,1,H,W] or [B,C,H,W]
        H, W = R.shape[-2], R.shape[-1]
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"boost location ({y},{x}) outside {H}x{W}")
        R[..., y, x] = (R[..., y, x] - cost).clamp(min=0.0)
    elif R.dim() == 3:
        H, W = R.shape[-2], R.shape[-1]
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"boost location ({y},{x}) outside {H}x{W}")
        R[..., y, x] = (R[..., y, x] - cost).clamp(min=0.0)
    elif R.dim() == 2:
        H, W = R.shape
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"boost location ({y},{x}) outside {H}x{W}")
        R[y, x] = max(0.0, float(R[y, x].item()) - cost)
    else:
        raise ValueError(f"Unexpected R shape {tuple(R.shape)}")
    return R


def read_R_at(R: torch.Tensor, y: int, x: int) -> float:
    """Scalar R value at (y, x) for logging / Check 3."""
    if R.dim() == 4:
        return float(R[0, 0, y, x].item())
    if R.dim() == 3:
        return float(R[0, y, x].item()) if R.shape[0] == 1 else float(R[y, x].item())
    if R.dim() == 2:
        return float(R[y, x].item())
    raise ValueError(f"Unexpected R shape {tuple(R.shape)}")


def should_charge_direct_cost(
    boosted_this_decision: bool,
    cost: float,
) -> bool:
    """
    True only when a boost is actually applied and cost > 0.
    Never true on budget-skipped / no-boost / never_boost intervals.
    """
    return bool(boosted_this_decision) and cost > 0.0
