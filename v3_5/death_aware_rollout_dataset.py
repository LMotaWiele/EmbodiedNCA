"""
death_aware_rollout_dataset.py
==============================
Rollout dataset under full v3.5 physics (per-well wear + direct boost cost),
with horizons long enough to contain real deaths and explicit null/skip
perturbations so the self-model can supervise "what if nothing happens."

Generation strategy (efficiency):
  - Run long episodes (horizon ≥ 2500–3000) under evolving per-well wear.
  - Sample states along the trajectory (including late times / far distances
    where death is common).
  - At each sample, fork a short prediction-horizon rollout under a random
    intervention (null / noise / zero / boost) and record future labels.

Does not retrain under constant-gen_rate physics (that would reproduce v2's
blind spot). Does not modify v0/v1/v2.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

V35_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V35_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V1_ROOT = REPO_ROOT / "v1"
V2_ROOT = REPO_ROOT / "v2"
V3_ROOT = REPO_ROOT / "v3"
for p in (str(V0_ROOT), str(V1_ROOT), str(V2_ROOT), str(V3_ROOT), str(V35_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boost_cost import BoostCostConfig  # noqa: E402
from boost_rollout_dataset import (  # noqa: E402
    make_boost_map,
    make_perturbation_map,
)
from grid_world import (  # noqa: E402
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    place_pattern,
)
from metrics import (  # noqa: E402
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from nca_module import SimpleNCA  # noqa: E402
from rollout_dataset import (  # noqa: E402
    BASE_CENTER,
    DEFAULT_CKPT,
    apply_perturbation,
    load_v0_world,
    sample_center_at_distance,
)
from wear_perwell import (  # noqa: E402
    PerWellWearConfig,
    PerWellWearState,
    nearest_well_distance,
    nearest_well_index,
    step_with_perwell_wear,
)

# Perturbation type codes
PERT_NULL = 0  # skip / nothing happens
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
class DeathAwareDatasetConfig:
    n_examples: int = 8_000
    # Full episode length under wear (must contain real deaths)
    episode_horizon: int = 2800
    burn_in: int = 30
    # Prediction window after intervention (coherence / local_R / survival)
    pred_horizon: int = 120
    # Sample every this many steps along an episode
    sample_interval: int = 80
    # Distance sampling: overweight far range where death is common
    dist_min: float = 0.0
    dist_max: float = 13.0
    far_dist_min: float = 6.0
    far_frac: float = 0.65  # fraction of episodes with dist in [far_min, dist_max]
    # Intervention mix — null/skip is first-class (not residual)
    null_frac: float = 0.20
    boost_frac_of_non_null: float = 0.40
    patch_sizes: Tuple[int, ...] = (3, 5)
    noise_std: float = 1.5
    boost_mag_min: float = 0.20
    boost_mag_max: float = 0.60
    boost_duration_min: int = 15
    boost_duration_max: int = 25
    boost_sigma: float = 1.5
    # Background boosts during host episode (to create asymmetric wear)
    host_boost_prob: float = 0.35  # per decision interval
    host_K: int = 20
    seed: int = 0
    device: str = "cpu"
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS
    # Min death rate target; oversample far distances if below
    min_death_rate: float = 0.10
    max_oversample_rounds: int = 3
    wear_cfg: Optional[PerWellWearConfig] = None
    cost_cfg: Optional[BoostCostConfig] = None


def _sample_mode(dcfg: DeathAwareDatasetConfig, rng: torch.Generator) -> int:
    u = float(torch.rand(1, generator=rng).item())
    if u < dcfg.null_frac:
        return PERT_NULL
    v = float(torch.rand(1, generator=rng).item())
    b = dcfg.boost_frac_of_non_null
    if v < b:
        return PERT_BOOST
    if v < b + 0.5 * (1.0 - b):
        return PERT_NOISE
    return PERT_ZERO


def _sample_distance(dcfg: DeathAwareDatasetConfig, rng: torch.Generator) -> float:
    u = float(torch.rand(1, generator=rng).item())
    if u < dcfg.far_frac:
        lo, hi = dcfg.far_dist_min, dcfg.dist_max
    else:
        lo, hi = dcfg.dist_min, dcfg.far_dist_min
    return float(lo + (hi - lo) * torch.rand(1, generator=rng).item())


@torch.no_grad()
def _fork_outcome(
    nca: SimpleNCA,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    S_t: torch.Tensor,
    R_t: torch.Tensor,
    wear_snapshot: List[float],
    dcfg: DeathAwareDatasetConfig,
    wear_cfg: PerWellWearConfig,
    cost_cfg: BoostCostConfig,
    rng: torch.Generator,
    example_seed: int,
) -> Dict[str, torch.Tensor]:
    """
    From a host-episode snapshot, apply a random intervention and roll
    pred_horizon steps under full v3.5 physics. Returns one training example.
    """
    H, W = world_cfg.H, world_cfg.W
    device = world_cfg.device

    world = ResourceGrid(
        nca,
        config=world_cfg,
        S0=S_t.clone(),
        pattern_center=None,
        seed=example_seed,
    )
    world.S = S_t.clone()
    world.R = R_t.clone()
    world.t = 0

    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    wear.wear = list(wear_snapshot)

    S_ref = world.S.detach().clone()
    mode = _sample_mode(dcfg, rng)

    mag_map = world.S.norm(dim=1)[0]
    active = (mag_map > DEFAULT_ACTIVATION_THRESHOLD).nonzero(as_tuple=False)
    if active.numel() > 0 and mode != PERT_NULL:
        idx = int(torch.randint(0, active.shape[0], (1,), generator=rng).item())
        py, px = int(active[idx, 0].item()), int(active[idx, 1].item())
    else:
        # null: still pick a location for encoding (zeros); random for damage
        if active.numel() > 0:
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
    boost_loc: Optional[Tuple[int, int]] = None
    charged = False

    if mode == PERT_NULL:
        mag = 0.0
    elif mode in (PERT_NOISE, PERT_ZERO):
        mode_str = "noise" if mode == PERT_NOISE else "zero"
        S_pert, mag = apply_perturbation(
            world.S, py, px, mode_str, k, dcfg.noise_std, rng
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
        boost_loc = (py, px)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Record pre-intervention wear for model inputs
    wear_at_t = list(wear.wear)
    mults_at_t = wear.multipliers()

    died_t: Optional[int] = None
    # Already dead at snapshot?
    if self_mass(world.S) <= DEATH_MASS_EPS:
        died_t = 0

    for h in range(1, dcfg.pred_horizon + 1):
        if died_t is not None:
            # still advance wear/R for consistency? stop stepping S once dead
            # but keep rolling for label simplicity — mass stays ~0
            pass
        boost_map = None
        charge = False
        loc = None
        if mode == PERT_BOOST and h <= boost_duration and boost_field is not None:
            boost_map = boost_field
            loc = boost_loc
            if not charged and cost_cfg.direct_boost_cost > 0:
                charge = True
                charged = True
        step_with_perwell_wear(
            world,
            wear,
            boost_map=boost_map,
            gate_mode="gated",
            direct_boost_cost=cost_cfg.direct_boost_cost,
            boost_location=loc,
            charge_direct_cost=charge,
        )
        if died_t is None and self_mass(world.S) <= DEATH_MASS_EPS:
            died_t = h

    coh = coherence(world.S, S_ref)
    local_r = local_resource_at_pattern_centroid(world.R, world.S)
    died = 1.0 if died_t is not None else 0.0
    coh_t = 0.0 if (isinstance(coh, float) and math.isnan(coh)) else float(coh)
    lr_t = 0.0 if (isinstance(local_r, float) and math.isnan(local_r)) else float(local_r)

    # Encoding: damage non-neg; boost negative; null zeros
    enc_mag = -mag if mode == PERT_BOOST else mag
    if mode == PERT_NULL:
        pert_map = torch.zeros(1, 1, H, W, device=device)
        enc_mag = 0.0
    else:
        pert_map = make_perturbation_map(H, W, py, px, abs(enc_mag), device=device)
        if enc_mag < 0:
            pert_map = -pert_map

    # Distance of snapshot pattern centroid (approx via mass centroid or center)
    mass = self_mass(S_t)
    if mass > DEATH_MASS_EPS:
        mag_s = S_t.norm(dim=1)[0]
        mask = mag_s > DEFAULT_ACTIVATION_THRESHOLD
        if mask.any():
            ys, xs = torch.where(mask)
            w = mag_s[mask]
            cy = float((ys.float() * w).sum() / w.sum())
            cx = float((xs.float() * w).sum() / w.sum())
        else:
            cy, cx = H / 2.0, W / 2.0
    else:
        cy, cx = H / 2.0, W / 2.0

    return {
        "S": S_t[0].cpu().float(),
        "R": R_t[0].cpu().float(),
        "pert_map": pert_map[0].cpu().float(),
        "wear": torch.tensor(wear_at_t, dtype=torch.float32),
        "multipliers": torch.tensor(mults_at_t, dtype=torch.float32),
        "pert_y": torch.tensor(py, dtype=torch.int64),
        "pert_x": torch.tensor(px, dtype=torch.int64),
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
            nearest_well_distance(cy, cx, dcfg.wells), dtype=torch.float32
        ),
        "centroid_y": torch.tensor(cy, dtype=torch.float32),
        "centroid_x": torch.tensor(cx, dtype=torch.float32),
        "coherence_was_nan": torch.tensor(
            1.0 if (isinstance(coh, float) and math.isnan(coh)) else 0.0
        ),
        "local_R_was_nan": torch.tensor(
            1.0 if (isinstance(local_r, float) and math.isnan(local_r)) else 0.0
        ),
    }


@torch.no_grad()
def _run_host_episode_samples(
    nca: SimpleNCA,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    dcfg: DeathAwareDatasetConfig,
    wear_cfg: PerWellWearConfig,
    cost_cfg: BoostCostConfig,
    rng: torch.Generator,
    episode_seed: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    One long host episode under v3.5 physics; sample states and fork outcomes.
    """
    H, W = world_cfg.H, world_cfg.W
    device = world_cfg.device

    dist = _sample_distance(dcfg, rng)
    center = sample_center_at_distance(dcfg.wells, H, W, dist, rng)
    S0 = place_pattern(base_pattern, center, base_center=BASE_CENTER)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=episode_seed
    )
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)

    samples: List[Dict[str, torch.Tensor]] = []
    # Host-episode random boosting to create wear asymmetry / early floor
    boost_remaining = 0
    boost_field: Optional[torch.Tensor] = None
    boost_loc: Optional[Tuple[int, int]] = None
    pending_charge = False

    for t in range(1, dcfg.episode_horizon + 1):
        # Decision interval: maybe arm a random host boost
        if t % dcfg.host_K == 0 or t == 1:
            if (
                float(torch.rand(1, generator=rng).item()) < dcfg.host_boost_prob
                and self_mass(world.S) > DEATH_MASS_EPS
            ):
                mag_map = world.S.norm(dim=1)[0]
                active = (mag_map > DEFAULT_ACTIVATION_THRESHOLD).nonzero(
                    as_tuple=False
                )
                if active.numel() > 0:
                    idx = int(
                        torch.randint(0, active.shape[0], (1,), generator=rng).item()
                    )
                    by, bx = int(active[idx, 0].item()), int(active[idx, 1].item())
                    bmag = float(
                        dcfg.boost_mag_min
                        + (dcfg.boost_mag_max - dcfg.boost_mag_min)
                        * torch.rand(1, generator=rng).item()
                    )
                    boost_field = make_boost_map(
                        H, W, by, bx, bmag, device=device, sigma=dcfg.boost_sigma
                    )
                    boost_remaining = dcfg.boost_duration_max
                    boost_loc = (by, bx)
                    pending_charge = True

        boost_map = None
        charge = False
        loc = None
        if boost_remaining > 0 and boost_field is not None:
            boost_map = boost_field
            loc = boost_loc
            boost_remaining -= 1
            if pending_charge and cost_cfg.direct_boost_cost > 0:
                charge = True
                pending_charge = False
            if boost_remaining <= 0:
                boost_field = None

        step_with_perwell_wear(
            world,
            wear,
            boost_map=boost_map,
            gate_mode="gated",
            direct_boost_cost=cost_cfg.direct_boost_cost,
            boost_location=loc,
            charge_direct_cost=charge,
        )

        # Sample after burn-in
        if t < dcfg.burn_in:
            continue
        if (t - dcfg.burn_in) % dcfg.sample_interval != 0:
            continue
        if self_mass(world.S) <= DEATH_MASS_EPS:
            # Still sample a dead-state example occasionally (survived=0, trivial)
            # but limit — mostly skip to keep labels informative
            if float(torch.rand(1, generator=rng).item()) > 0.15:
                continue

        ex = _fork_outcome(
            nca=nca,
            base_pattern=base_pattern,
            world_cfg=world_cfg,
            S_t=world.S.detach().clone(),
            R_t=world.R.detach().clone(),
            wear_snapshot=list(wear.wear),
            dcfg=dcfg,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            rng=rng,
            example_seed=episode_seed + t * 17,
        )
        # tag host distance for diagnostics
        ex["host_distance"] = torch.tensor(dist, dtype=torch.float32)
        ex["host_t"] = torch.tensor(t, dtype=torch.int64)
        samples.append(ex)

        if len(samples) >= dcfg.n_examples:
            break

    return samples


@torch.no_grad()
def generate_dataset(
    dcfg: Optional[DeathAwareDatasetConfig] = None,
    ckpt_path: Path | str = DEFAULT_CKPT,
    out_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate death-aware dataset under full v3.5 physics."""
    dcfg = dcfg or DeathAwareDatasetConfig()
    wear_cfg = dcfg.wear_cfg or PerWellWearConfig(wells=dcfg.wells)
    cost_cfg = dcfg.cost_cfg or BoostCostConfig(direct_boost_cost=0.15)

    nca, base_pattern, cfg = load_v0_world(ckpt_path, device=dcfg.device)
    world_cfg = WorldConfig(
        H=cfg.H,
        W=cfg.W,
        channels=cfg.channels,
        hidden_dim=cfg.hidden_dim,
        gen_rate=wear_cfg.gen_rate_base,
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
    episode = 0
    max_episodes = max(50, dcfg.n_examples // 5 + 100)

    if verbose:
        print(
            f"  generating death-aware dataset: target n={dcfg.n_examples}  "
            f"episode_H={dcfg.episode_horizon}  pred_H={dcfg.pred_horizon}  "
            f"far_frac={dcfg.far_frac}"
        )

    while len(rows) < dcfg.n_examples and episode < max_episodes:
        episode += 1
        samples = _run_host_episode_samples(
            nca=nca,
            base_pattern=base_pattern,
            world_cfg=world_cfg,
            dcfg=dcfg,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            rng=rng,
            episode_seed=dcfg.seed + 50_000 + episode * 997,
        )
        for s in samples:
            rows.append(s)
            if len(rows) >= dcfg.n_examples:
                break
        if verbose and (episode % 10 == 0 or episode == 1):
            n_dead = sum(1 for r in rows if float(r["died_within_horizon"].item()) > 0.5)
            print(
                f"  episode {episode}: collected {len(rows)}/{dcfg.n_examples}  "
                f"death_rate={n_dead / max(1, len(rows)):.3f}"
            )

    rows = rows[: dcfg.n_examples]

    # Class-balance report + oversample far if death still rare
    def death_rate(rs: List[Dict]) -> float:
        if not rs:
            return 0.0
        return sum(1 for r in rs if float(r["died_within_horizon"].item()) > 0.5) / len(
            rs
        )

    dr = death_rate(rows)
    if verbose:
        print(f"  pre-oversample death_rate={dr:.4f}  n={len(rows)}")

    oversample_round = 0
    # Temporarily force farther placements / more host boosts if needed
    while dr < dcfg.min_death_rate and oversample_round < dcfg.max_oversample_rounds:
        oversample_round += 1
        if verbose:
            print(
                f"  death rate {dr:.3f} < {dcfg.min_death_rate}; "
                f"oversample round {oversample_round} (far-distance heavy)"
            )
        # Create a far-only config
        far_cfg = DeathAwareDatasetConfig(
            n_examples=dcfg.n_examples,
            episode_horizon=dcfg.episode_horizon,
            burn_in=dcfg.burn_in,
            pred_horizon=dcfg.pred_horizon,
            sample_interval=max(40, dcfg.sample_interval // 2),
            dist_min=dcfg.far_dist_min,
            dist_max=dcfg.dist_max,
            far_dist_min=dcfg.far_dist_min,
            far_frac=1.0,
            null_frac=dcfg.null_frac,
            boost_frac_of_non_null=dcfg.boost_frac_of_non_null,
            host_boost_prob=min(0.7, dcfg.host_boost_prob + 0.15),
            seed=dcfg.seed + 10_000 * oversample_round,
            device=dcfg.device,
            wells=dcfg.wells,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
        )
        need = max(500, int(0.3 * dcfg.n_examples))
        added = 0
        for ep in range(need):
            samples = _run_host_episode_samples(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                dcfg=far_cfg,
                wear_cfg=wear_cfg,
                cost_cfg=cost_cfg,
                rng=rng,
                episode_seed=dcfg.seed + 90_000 + oversample_round * 10_000 + ep,
            )
            # Prefer dead outcomes for balance
            dead = [s for s in samples if float(s["died_within_horizon"].item()) > 0.5]
            live = [s for s in samples if float(s["died_within_horizon"].item()) <= 0.5]
            # Keep all dead + a fraction of live
            for s in dead:
                rows.append(s)
                added += 1
            for s in live[::3]:
                rows.append(s)
                added += 1
            if added >= need:
                break
        # Shuffle and cap (keep mix)
        g = torch.Generator().manual_seed(dcfg.seed + oversample_round)
        perm = torch.randperm(len(rows), generator=g).tolist()
        rows = [rows[i] for i in perm]
        # If we overshot, prefer keeping dead examples
        dead_rows = [r for r in rows if float(r["died_within_horizon"].item()) > 0.5]
        live_rows = [r for r in rows if float(r["died_within_horizon"].item()) <= 0.5]
        target_dead = max(
            int(dcfg.min_death_rate * dcfg.n_examples),
            min(len(dead_rows), int(0.4 * dcfg.n_examples)),
        )
        target_live = dcfg.n_examples - min(target_dead, len(dead_rows))
        rows = dead_rows[:target_dead] + live_rows[:target_live]
        # If still short, fill with remaining
        if len(rows) < dcfg.n_examples:
            remaining = dead_rows[target_dead:] + live_rows[target_live:]
            rows = rows + remaining[: dcfg.n_examples - len(rows)]
        rows = rows[: dcfg.n_examples]
        dr = death_rate(rows)
        if verbose:
            print(f"  after oversample: n={len(rows)} death_rate={dr:.4f}")

    if not rows:
        raise RuntimeError("death-aware dataset generation produced 0 examples")

    keys = [k for k in rows[0].keys()]
    data: Dict[str, torch.Tensor] = {
        k: torch.stack([r[k] for r in rows], dim=0) for k in keys
    }

    type_counts = {name: 0 for name in PERT_TYPE_NAMES.values()}
    for t in data["pert_type"].tolist():
        type_counts[PERT_TYPE_NAMES[int(t)]] += 1

    n_dead = int(data["died_within_horizon"].sum().item())
    n_surv = int(data["survived"].sum().item())
    data["meta"] = {
        "n_examples": len(rows),
        "episode_horizon": dcfg.episode_horizon,
        "pred_horizon": dcfg.pred_horizon,
        "burn_in": dcfg.burn_in,
        "seed": dcfg.seed,
        "null_frac": dcfg.null_frac,
        "boost_frac_of_non_null": dcfg.boost_frac_of_non_null,
        "dist_min": dcfg.dist_min,
        "dist_max": dcfg.dist_max,
        "far_frac": dcfg.far_frac,
        "far_dist_min": dcfg.far_dist_min,
        "ckpt": str(ckpt_path),
        "type_counts": type_counts,
        "pert_type_names": PERT_TYPE_NAMES,
        "n_died": n_dead,
        "n_survived": n_surv,
        "death_rate": float(n_dead / len(rows)),
        "survival_rate": float(n_surv / len(rows)),
        "wear_cfg": {
            "wear_ambient_rate": wear_cfg.wear_ambient_rate,
            "wear_boost_rate": wear_cfg.wear_boost_rate,
            "wear_decay_coeff": wear_cfg.wear_decay_coeff,
            "gen_floor_frac": wear_cfg.gen_floor_frac,
            "gen_rate_base": wear_cfg.gen_rate_base,
        },
        "direct_boost_cost": cost_cfg.direct_boost_cost,
        "oversample_rounds": oversample_round,
        "n_host_episodes": episode,
    }
    if verbose:
        print("  type counts:", type_counts)
        print(
            f"  FINAL class balance: died={n_dead}  survived={n_surv}  "
            f"death_rate={data['meta']['death_rate']:.4f}"
        )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        if verbose:
            print(f"  wrote {out_path} ({len(rows)} examples)")
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
    m = data["pert_type"] == pert_type
    return {k: v[m] for k, v in data.items() if isinstance(v, torch.Tensor)}


class DeathAwareTensorDataset(torch.utils.data.Dataset):
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
            "wear": self.data["wear"][i],
            "multipliers": self.data["multipliers"][i],
            "coherence_future": self.data["coherence_future"][i],
            "local_R_future": self.data["local_R_future"][i],
            "survived": self.data["survived"][i],
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate v3.5 death-aware rollout dataset")
    p.add_argument("--n", type=int, default=8_000)
    p.add_argument("--out", type=str, default="results/death_aware_dataset.pt")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode_horizon", type=int, default=2800)
    p.add_argument("--pred_horizon", type=int, default=120)
    args = p.parse_args()

    dcfg = DeathAwareDatasetConfig(
        n_examples=args.n,
        episode_horizon=args.episode_horizon,
        pred_horizon=args.pred_horizon,
        seed=args.seed,
        device=args.device,
    )
    generate_dataset(dcfg, out_path=Path(args.out), verbose=True)
