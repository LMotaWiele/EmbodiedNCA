"""v1-style rollouts plus boost probes so the self-model sees the intervention used at action time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from embodiednca.paths import results_dir

import torch
import torch.nn.functional as F

from embodiednca.v0.grid_world import (
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    diffuse,
    gated_nca_step,
    place_pattern,
    step_resource,
)
from embodiednca.v0.metrics import (
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from embodiednca.v0.nca_module import SimpleNCA
from embodiednca.v1.rollout_dataset import (
    BASE_CENTER,
    DEFAULT_CKPT,
    apply_perturbation,
    load_v0_world,
    make_perturbation_map,
    nearest_well_distance,
    sample_center_at_distance,
)

# Perturbation type codes (stored as int64)
PERT_NULL = 0
PERT_NOISE = 1
PERT_ZERO = 2
PERT_BOOST = 3
PERT_TYPE_NAMES = {
    PERT_NULL: "null",
    PERT_NOISE: "noise",
    PERT_ZERO: "zero",
    PERT_BOOST: "boost",
}


@dataclass
class BoostDatasetConfig:
    n_examples: int = 14_000
    burn_in: int = 20
    horizon: int = 100
    dist_min: float = 0.0
    dist_max: float = 12.0
    # Fractions among non-reserved mass; null is separate.
    null_frac: float = 0.10
    # Among non-null: equal weight noise/zero/boost unless overridden
    boost_frac_of_non_null: float = 0.40  # boost share of non-null examples
    patch_sizes: Tuple[int, ...] = (3, 5)
    noise_std: float = 1.5
    # Boost sampling (gate additive, clamped so total gate ≤ 1)
    boost_mag_min: float = 0.20
    boost_mag_max: float = 0.60
    boost_duration_min: int = 15
    boost_duration_max: int = 25  # default center ≈ 20, matches decision interval K
    boost_sigma: float = 1.5
    seed: int = 0
    device: str = "cpu"
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS


def make_boost_map(
    H: int,
    W: int,
    y: int,
    x: int,
    magnitude: float,
    device: str | torch.device,
    sigma: float = 1.5,
    k: Optional[int] = None,
) -> torch.Tensor:
    """
    Spatial boost field [1, 1, H, W] added to gate_from_R (then clamped to 1).

    Soft Gaussian bump peaking at `magnitude` at (y, x). Optional hard k×k
    crop is not applied — soft falloff matches the attention patch used at
    runtime and the pert_map encoding.
    """
    if magnitude <= 0:
        return torch.zeros(1, 1, H, W, device=device)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    dist2 = (yy - float(y)) ** 2 + (xx - float(x)) ** 2
    bump = torch.exp(-dist2 / (2.0 * sigma ** 2))
    # Soft spatial support: drop far tails so boost is local
    if k is not None and k > 0:
        half = k // 2
        mask = ((yy - y).abs() <= half) & ((xx - x).abs() <= half)
        bump = bump * mask.float()
    return (magnitude * bump).view(1, 1, H, W)


@torch.no_grad()
def step_with_boost(
    world: ResourceGrid,
    boost_map: Optional[torch.Tensor] = None,
    gate_mode: str = "gated",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One timestep identical to ResourceGrid.step, except the gate applied to S
    may be elevated by an additive boost_map:

        gate_effective = min(1.0, gate_from_R + boost_map)

    Resource physics (consumption on pre-gate ‖update‖, gen/decay/diffusion of
    R) are unchanged. Does not modify grid_world.py — this is a v2 layer.
    """
    cfg = world.config
    world.S = diffuse(world.S, cfg.diffusion_rate)

    if gate_mode == "gated" and boost_map is not None:
        # Manual gated step with additive boost (mirrors gated_nca_step)
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

    consumption = cfg.consumption_rate * update_magnitude_map
    world.R = (world.R - consumption).clamp(min=0.0)
    world.R = step_resource(
        world.R,
        world.generation_mask,
        decay_rate=cfg.decay_rate,
        gen_rate=cfg.gen_rate,
        R_max=cfg.R_max,
        resource_diffusion=cfg.resource_diffusion,
    )
    world.t += 1
    return world.S, update_magnitude_map


def _sample_mode(dcfg: BoostDatasetConfig, rng: torch.Generator) -> int:
    """Sample pert type code."""
    u = float(torch.rand(1, generator=rng).item())
    if u < dcfg.null_frac:
        return PERT_NULL
    # non-null mass
    v = float(torch.rand(1, generator=rng).item())
    b = dcfg.boost_frac_of_non_null
    # Remaining (1-b) split equally noise / zero
    if v < b:
        return PERT_BOOST
    if v < b + 0.5 * (1.0 - b):
        return PERT_NOISE
    return PERT_ZERO


@torch.no_grad()
def generate_one_example(
    nca: SimpleNCA,
    base_pattern: torch.Tensor,
    cfg: WorldConfig,
    dcfg: BoostDatasetConfig,
    rng: torch.Generator,
    example_seed: int,
) -> Dict[str, torch.Tensor]:
    """One (state, intervention, future outcome) tuple — damage or boost."""
    H, W = cfg.H, cfg.W
    device = cfg.device

    dist = float(
        dcfg.dist_min
        + (dcfg.dist_max - dcfg.dist_min) * torch.rand(1, generator=rng).item()
    )
    center = sample_center_at_distance(dcfg.wells, H, W, dist, rng)
    S0 = place_pattern(base_pattern, center, base_center=BASE_CENTER)

    world = ResourceGrid(
        nca,
        config=cfg,
        S0=S0,
        pattern_center=center,
        seed=example_seed,
    )
    for _ in range(dcfg.burn_in):
        world.step(gate_mode="gated")

    S_t = world.S.detach().clone()
    R_t = world.R.detach().clone()
    S_ref = S_t.clone()

    mode = _sample_mode(dcfg, rng)

    mag_map = S_t.norm(dim=1)[0]
    active = (mag_map > DEFAULT_ACTIVATION_THRESHOLD).nonzero(as_tuple=False)
    if active.numel() > 0 and mode != PERT_NULL:
        idx = int(torch.randint(0, active.shape[0], (1,), generator=rng).item())
        py, px = int(active[idx, 0].item()), int(active[idx, 1].item())
    else:
        py = int(torch.randint(0, H, (1,), generator=rng).item())
        px = int(torch.randint(0, W, (1,), generator=rng).item())

    k = int(
        dcfg.patch_sizes[
            int(torch.randint(0, len(dcfg.patch_sizes), (1,), generator=rng).item())
        ]
    )

    boost_duration = 0
    boost_mag = 0.0
    boost_field: Optional[torch.Tensor] = None

    if mode == PERT_NULL:
        mag = 0.0
        # no change to world.S
    elif mode in (PERT_NOISE, PERT_ZERO):
        mode_str = "noise" if mode == PERT_NOISE else "zero"
        S_pert, mag = apply_perturbation(
            S_t, py, px, mode_str, k, dcfg.noise_std, rng
        )
        world.S = S_pert
    elif mode == PERT_BOOST:
        boost_mag = float(
            dcfg.boost_mag_min
            + (dcfg.boost_mag_max - dcfg.boost_mag_min)
            * torch.rand(1, generator=rng).item()
        )
        boost_duration = int(
            torch.randint(
                dcfg.boost_duration_min,
                dcfg.boost_duration_max + 1,
                (1,),
                generator=rng,
            ).item()
        )
        mag = boost_mag
        boost_field = make_boost_map(
            H, W, py, px, boost_mag, device=device, sigma=dcfg.boost_sigma, k=k
        )
        # S_t stays unperturbed; intervention is gate boost during rollout
    else:
        raise ValueError(f"Unknown mode code {mode}")

    died_t: Optional[int] = None
    for h in range(1, dcfg.horizon + 1):
        if mode == PERT_BOOST and h <= boost_duration and boost_field is not None:
            step_with_boost(world, boost_map=boost_field, gate_mode="gated")
        else:
            world.step(gate_mode="gated")
        if died_t is None and self_mass(world.S) <= DEATH_MASS_EPS:
            died_t = h

    coh = coherence(world.S, S_ref)
    local_r = local_resource_at_pattern_centroid(world.R, world.S)
    died = 1.0 if died_t is not None else 0.0
    coh_t = 0.0 if (isinstance(coh, float) and math.isnan(coh)) else float(coh)
    lr_t = 0.0 if (isinstance(local_r, float) and math.isnan(local_r)) else float(local_r)

    # Encoding convention (same SelfModel input shape as v1 — no arch change):
    #   damage (noise/zero/null): non-negative Gaussian bump, mag ≥ 0  (v1)
    #   boost: *negative* Gaussian bump, mag_signed = −boost_magnitude
    # Without a sign difference the two interventions are input-identical and
    # the model cannot learn distinct boost vs damage outcome maps.
    enc_mag = -mag if mode == PERT_BOOST else mag
    pert_map = make_perturbation_map(H, W, py, px, abs(enc_mag), device=device)
    if enc_mag < 0:
        pert_map = -pert_map

    return {
        "S": S_t[0].cpu().float(),
        "R": R_t[0].cpu().float(),
        "pert_map": pert_map[0].cpu().float(),
        "pert_y": torch.tensor(py, dtype=torch.int64),
        "pert_x": torch.tensor(px, dtype=torch.int64),
        # Signed magnitude: negative ⇔ boost (see above)
        "pert_mag": torch.tensor(enc_mag, dtype=torch.float32),
        "pert_type": torch.tensor(mode, dtype=torch.int64),
        "boost_duration": torch.tensor(boost_duration, dtype=torch.int64),
        "coherence_future": torch.tensor(coh_t, dtype=torch.float32),
        "local_R_future": torch.tensor(lr_t, dtype=torch.float32),
        "survived": torch.tensor(1.0 - died, dtype=torch.float32),
        "died_within_horizon": torch.tensor(died, dtype=torch.float32),
        "time_of_death": torch.tensor(
            -1 if died_t is None else died_t, dtype=torch.int64
        ),
        "distance_to_well": torch.tensor(
            nearest_well_distance(center[0], center[1], dcfg.wells),
            dtype=torch.float32,
        ),
        "center_y": torch.tensor(center[0], dtype=torch.int64),
        "center_x": torch.tensor(center[1], dtype=torch.int64),
        "coherence_was_nan": torch.tensor(
            1.0 if (isinstance(coh, float) and math.isnan(coh)) else 0.0
        ),
        "local_R_was_nan": torch.tensor(
            1.0 if (isinstance(local_r, float) and math.isnan(local_r)) else 0.0
        ),
    }


@torch.no_grad()
def generate_dataset(
    dcfg: Optional[BoostDatasetConfig] = None,
    ckpt_path: Path | str = DEFAULT_CKPT,
    out_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate and optionally save a stacked damage+boost dataset dict."""
    dcfg = dcfg or BoostDatasetConfig()
    nca, base_pattern, cfg = load_v0_world(ckpt_path, device=dcfg.device)
    cfg = WorldConfig(
        H=cfg.H,
        W=cfg.W,
        channels=cfg.channels,
        hidden_dim=cfg.hidden_dim,
        gen_rate=cfg.gen_rate,
        consumption_rate=cfg.consumption_rate,
        decay_rate=cfg.decay_rate,
        diffusion_rate=cfg.diffusion_rate,
        resource_diffusion=cfg.resource_diffusion,
        well_sigma=cfg.well_sigma,
        wells=dcfg.wells,
        device=dcfg.device,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(dcfg.seed)

    rows: List[Dict[str, torch.Tensor]] = []
    log_every = max(1, dcfg.n_examples // 20)
    type_counts = {k: 0 for k in PERT_TYPE_NAMES}
    for i in range(dcfg.n_examples):
        ex = generate_one_example(
            nca, base_pattern, cfg, dcfg, rng, example_seed=dcfg.seed + 20_000 + i
        )
        rows.append(ex)
        type_counts[int(ex["pert_type"].item())] = (
            type_counts.get(int(ex["pert_type"].item()), 0) + 1
        )
        if verbose and ((i + 1) % log_every == 0 or i == 0):
            print(f"  dataset {i + 1}/{dcfg.n_examples}")

    keys = rows[0].keys()
    data = {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}
    data["meta"] = {
        "n_examples": dcfg.n_examples,
        "burn_in": dcfg.burn_in,
        "horizon": dcfg.horizon,
        "seed": dcfg.seed,
        "null_frac": dcfg.null_frac,
        "boost_frac_of_non_null": dcfg.boost_frac_of_non_null,
        "boost_mag_min": dcfg.boost_mag_min,
        "boost_mag_max": dcfg.boost_mag_max,
        "boost_duration_min": dcfg.boost_duration_min,
        "boost_duration_max": dcfg.boost_duration_max,
        "dist_min": dcfg.dist_min,
        "dist_max": dcfg.dist_max,
        "ckpt": str(ckpt_path),
        "type_counts": {
            PERT_TYPE_NAMES[k]: v for k, v in type_counts.items()
        },
        "pert_type_names": PERT_TYPE_NAMES,
    }
    if verbose:
        print("  type counts:", data["meta"]["type_counts"])
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        if verbose:
            print(f"  wrote {out_path} ({dcfg.n_examples} examples)")
    return data


def train_val_split(
    data: Dict[str, torch.Tensor],
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n = data["S"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def take(idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in data.items() if isinstance(v, torch.Tensor)}

    return take(train_idx), take(val_idx)


def filter_by_pert_type(
    data: Dict[str, torch.Tensor],
    pert_type: int,
) -> Dict[str, torch.Tensor]:
    """Subset a stacked dataset dict to a single pert_type code."""
    m = data["pert_type"] == pert_type
    return {k: v[m] for k, v in data.items() if isinstance(v, torch.Tensor)}


class RolloutTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self.n = data["S"].shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "S": self.data["S"][i],
            "R": self.data["R"][i],
            "pert_map": self.data["pert_map"][i],
            "coherence_future": self.data["coherence_future"][i],
            "local_R_future": self.data["local_R_future"][i],
            "survived": self.data["survived"][i],
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate v2 damage+boost rollout dataset")
    p.add_argument("--n", type=int, default=14_000)
    p.add_argument("--out", type=str, default=str(results_dir("v2") / "boost_rollout_dataset.pt"))
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--burn_in", type=int, default=20)
    args = p.parse_args()

    dcfg = BoostDatasetConfig(
        n_examples=args.n,
        horizon=args.horizon,
        burn_in=args.burn_in,
        seed=args.seed,
        device=args.device,
    )
    generate_dataset(dcfg, out_path=Path(args.out), verbose=True)
