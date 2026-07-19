"""
grid_world.py
=============
Resource-gated embedded-pattern NCA dynamics (v0).

Wraps SimpleNCA without modifying it. Resource field R is independent world
physics; pattern state S is updated only through a resource-gated residual.

World-physics additions beyond the pure residual NCA (documented in README):

1. Mild spatial diffusion on S — unopposed erosion when the gate is low.
   SimpleNCA is a pure residual; with gate=0 the state would otherwise freeze.
2. Mild spatial diffusion on R + a narrow soft well footprint — so fixed wells
   create a smooth, distance-dependent resource field a multi-cell pattern can
   actually sit in. A pure single-pixel delta cannot sustain multi-cell gating.

There is no `if R <= 0: reset()` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from nca_module import SimpleNCA


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_H = 32
DEFAULT_W = 32
DEFAULT_CHANNELS = 32
DEFAULT_HIDDEN = 64
DEFAULT_R_MAX = 1.0
DEFAULT_DECAY_RATE = 0.02
# Spec defaults were gen=0.05 / cons=0.01; those leave the multi-cell pattern
# under-supplied once R is a smooth field. Operating point below was found by
# the Check-1 sweep so a pattern parked on a well is near break-even.
DEFAULT_GEN_RATE = 0.20
DEFAULT_CONSUMPTION_RATE = 1.0
DEFAULT_DIFFUSION_RATE = 0.04
DEFAULT_RESOURCE_DIFFUSION = 0.40
DEFAULT_WELL_SIGMA = 2.5
DEFAULT_PATTERN_RADIUS = 2.5

# Two wells, well-separated so distance-from-nearest-well is unambiguous.
# Coordinates are (y, x).
DEFAULT_WELLS: Tuple[Tuple[int, int], ...] = (
    (8, 8),
    (24, 24),
)


@dataclass
class WorldConfig:
    H: int = DEFAULT_H
    W: int = DEFAULT_W
    channels: int = DEFAULT_CHANNELS
    hidden_dim: int = DEFAULT_HIDDEN
    R_max: float = DEFAULT_R_MAX
    decay_rate: float = DEFAULT_DECAY_RATE
    gen_rate: float = DEFAULT_GEN_RATE
    consumption_rate: float = DEFAULT_CONSUMPTION_RATE
    diffusion_rate: float = DEFAULT_DIFFUSION_RATE
    resource_diffusion: float = DEFAULT_RESOURCE_DIFFUSION
    well_sigma: float = DEFAULT_WELL_SIGMA
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS
    device: str = "cpu"


def make_generation_mask(
    H: int,
    W: int,
    wells: Sequence[Tuple[int, int]],
    device: str | torch.device = "cpu",
    sigma: float = DEFAULT_WELL_SIGMA,
) -> torch.Tensor:
    """Soft generation field [1,1,H,W], peaking at 1.0 on each well cell."""
    mask = torch.zeros(1, 1, H, W, device=device)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    for y, x in wells:
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"Well ({y}, {x}) outside grid {H}x{W}")
        if sigma <= 0:
            mask[0, 0, y, x] = 1.0
        else:
            dist2 = (yy - y) ** 2 + (xx - x) ** 2
            blob = torch.exp(-dist2 / (2.0 * sigma ** 2))
            mask[0, 0] = torch.maximum(mask[0, 0], blob)
    return mask


def diffuse(field: torch.Tensor, rate: float) -> torch.Tensor:
    """field <- field + rate * (local_mean - field)."""
    if rate <= 0:
        return field
    local_mean = F.avg_pool2d(field, kernel_size=3, stride=1, padding=1)
    return field + rate * (local_mean - field)


def step_resource(
    R: torch.Tensor,
    generation_mask: torch.Tensor,
    decay_rate: float = DEFAULT_DECAY_RATE,
    gen_rate: float = DEFAULT_GEN_RATE,
    R_max: float = DEFAULT_R_MAX,
    resource_diffusion: float = DEFAULT_RESOURCE_DIFFUSION,
) -> torch.Tensor:
    """Decay, generate at wells, then diffuse."""
    R_next = R * (1.0 - decay_rate) + generation_mask * gen_rate
    R_next = diffuse(R_next, resource_diffusion)
    return R_next.clamp(0.0, R_max)


def gated_nca_step(
    nca: SimpleNCA,
    S: torch.Tensor,
    R: torch.Tensor,
    R_max: float = DEFAULT_R_MAX,
    gate_mode: str = "gated",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap SimpleNCA perception -> update_net -> residual, with optional gate.

    gate_mode:
      - "gated":   S += (R/R_max) * update
      - "ungated": S += update
      - "zero":    S unchanged by NCA  (update is still *computed*)

    Always returns the raw pre-gate `update_magnitude_map = ‖update‖₂` per cell.
    Consumption (see ResourceGrid.step) is charged on this raw magnitude —
    independent of gate — so a starved pattern that *attempts* repair still
    pays for the attempt even when gate=0 nullifies the effect on S. That is a
    deliberate match to the v0 spec formula
    `consumption = consumption_rate * ||update||`, not gate * ||update||.
    """
    perception_out = nca.perception(S)
    update = nca.update_net(perception_out)
    # Pre-gate magnitude: used for resource accounting regardless of gate_mode.
    update_magnitude_map = update.norm(dim=1, keepdim=True)

    if gate_mode == "gated":
        gate = (R / R_max).clamp(0.0, 1.0)
        S_next = S + gate * update
    elif gate_mode == "ungated":
        S_next = S + update
    elif gate_mode == "zero":
        # Gate forced to 0 for S; update still computed above for consumption.
        S_next = S
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode!r}")

    S_next = torch.clamp(S_next, -10.0, 10.0)
    return S_next, update_magnitude_map


def make_seed_pattern(
    channels: int,
    H: int,
    W: int,
    center: Tuple[int, int],
    radius: float = DEFAULT_PATTERN_RADIUS,
    amplitude: float = 1.0,
    device: str | torch.device = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Localized Gaussian blob of random channel activity, shape [1, C, H, W]."""
    cy, cx = center
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    envelope = torch.exp(-dist2 / (2.0 * radius ** 2))
    envelope = envelope * (dist2 <= (radius * 3.0) ** 2).float()
    noise = torch.randn(1, channels, H, W, device=device, generator=generator)
    return noise * envelope.view(1, 1, H, W) * amplitude


def place_pattern(
    base: torch.Tensor,
    center: Tuple[int, int],
    base_center: Tuple[int, int] = (16, 16),
) -> torch.Tensor:
    """Translate a base pattern so its content sits at `center` (circular roll)."""
    dy = center[0] - base_center[0]
    dx = center[1] - base_center[1]
    return torch.roll(base, shifts=(dy, dx), dims=(-2, -1))


def settle_resource_field(
    generation_mask: torch.Tensor,
    decay_rate: float,
    gen_rate: float,
    R_max: float,
    resource_diffusion: float,
    steps: int = 300,
) -> torch.Tensor:
    R = torch.zeros_like(generation_mask)
    for _ in range(steps):
        R = step_resource(
            R,
            generation_mask,
            decay_rate=decay_rate,
            gen_rate=gen_rate,
            R_max=R_max,
            resource_diffusion=resource_diffusion,
        )
    return R


def init_resource_field(
    generation_mask: torch.Tensor,
    decay_rate: float,
    gen_rate: float,
    R_max: float,
    resource_diffusion: float,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    R = settle_resource_field(
        generation_mask,
        decay_rate,
        gen_rate,
        R_max,
        resource_diffusion,
        steps=300,
    )
    return R.to(device).clamp(0.0, R_max)


class ResourceGrid:
    """Co-located pattern state S and resource field R on an H x W grid."""

    def __init__(
        self,
        nca: SimpleNCA,
        config: Optional[WorldConfig] = None,
        S0: Optional[torch.Tensor] = None,
        R0: Optional[torch.Tensor] = None,
        pattern_center: Optional[Tuple[int, int]] = None,
        seed: int = 0,
    ):
        self.config = config or WorldConfig()
        self.nca = nca
        self.device = self.config.device
        self.wells = list(self.config.wells)
        self.generation_mask = make_generation_mask(
            self.config.H,
            self.config.W,
            self.wells,
            device=self.device,
            sigma=self.config.well_sigma,
        )

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        if S0 is not None:
            self.S = S0.to(self.device)
        else:
            cy, cx = pattern_center or (self.config.H // 2, self.config.W // 2)
            self.S = make_seed_pattern(
                self.config.channels,
                self.config.H,
                self.config.W,
                center=(cy, cx),
                device="cpu",
                generator=g,
            ).to(self.device)

        if R0 is not None:
            self.R = R0.to(self.device)
        else:
            self.R = init_resource_field(
                self.generation_mask,
                self.config.decay_rate,
                self.config.gen_rate,
                self.config.R_max,
                self.config.resource_diffusion,
                device=self.device,
            )

        self.S0 = self.S.detach().clone()
        self.t = 0

    def step(self, gate_mode: str = "gated") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep:
          0. diffuse S
          1. gated_nca_step  (returns pre-gate update magnitude)
          2. consume resource BEFORE gen/decay:
                consumption = consumption_rate * ‖update‖   (NOT gate * ‖update‖)
             Attempted self-maintenance costs resource even when the gate
             nullifies the effect on S (including gate_mode="zero").
          3. step_resource (decay + well generation + R diffusion)

        Returns (S, update_magnitude_map). Caller may also read the last
        consumption via the returned magnitude * consumption_rate.
        """
        cfg = self.config
        self.S = diffuse(self.S, cfg.diffusion_rate)
        self.S, update_magnitude_map = gated_nca_step(
            self.nca, self.S, self.R, R_max=cfg.R_max, gate_mode=gate_mode
        )
        # Spec-faithful accounting: cost tracks the attempted update, not the
        # applied (gated) residual. Failure is not free.
        consumption = cfg.consumption_rate * update_magnitude_map
        self.R = (self.R - consumption).clamp(min=0.0)
        self.R = step_resource(
            self.R,
            self.generation_mask,
            decay_rate=cfg.decay_rate,
            gen_rate=cfg.gen_rate,
            R_max=cfg.R_max,
            resource_diffusion=cfg.resource_diffusion,
        )
        self.t += 1
        return self.S, update_magnitude_map



def train_self_maintenance(
    nca: SimpleNCA,
    base_pattern: torch.Tensor,
    diffusion_rate: float = DEFAULT_DIFFUSION_RATE,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> list[float]:
    """
    Curriculum + translation-augmented + spatially-varying partial-gate
    training so SimpleNCA learns a local self-maintenance rule that:
      - works at any grid position (random rolls of one base pattern)
      - holds under smooth spatial gate fields in ~[0.5, 1.0]
      - stays stable for long horizons (curriculum up to 128 steps)

    Runtime has no external controller — this only sets the local update weights.
    """
    nca = nca.to(device)
    base_pattern = base_pattern.to(device).detach()
    opt = torch.optim.Adam(nca.parameters(), lr=1e-3)
    losses: list[float] = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    _, _, H, W = base_pattern.shape

    schedule: List[int] = (
        [8] * 60
        + [16] * 100
        + [32] * 150
        + [64] * 250
        + [128] * 350
    )

    nca.train()
    for it, steps in enumerate(schedule):
        B = 8
        targets = []
        for _ in range(B):
            dy = int(torch.randint(-12, 13, (1,), generator=g).item())
            dx = int(torch.randint(-12, 13, (1,), generator=g).item())
            targets.append(torch.roll(base_pattern, shifts=(dy, dx), dims=(-2, -1)))
        target = torch.cat(targets, dim=0)

        S = target + 0.04 * torch.randn_like(target)
        S = torch.clamp(S, -10.0, 10.0)

        # Mixture: full gate, or smooth spatial field in [0.5, 1]
        if it % 3 == 0:
            gate = torch.ones(B, 1, 1, 1, device=device)
        else:
            noise = torch.randn(B, 1, max(1, H // 4), max(1, W // 4), device=device)
            gate = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
            gate = torch.sigmoid(gate) * 0.5 + 0.5

        for _ in range(steps):
            S = diffuse(S, diffusion_rate)
            update = nca.update_net(nca.perception(S))
            S = torch.clamp(S + gate * update, -10.0, 10.0)

        loss = F.mse_loss(S, target) + 5e-4 * (update ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

        if verbose and (it + 1) % 100 == 0:
            print(
                f"  self-maintenance train iter {it + 1}/{len(schedule)}  "
                f"steps={steps}  loss={loss.item():.6f}"
            )

    nca.eval()
    return losses
