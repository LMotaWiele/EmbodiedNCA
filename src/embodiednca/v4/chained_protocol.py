"""N challenge segments as one run: S/R/wear/A persist; death is terminal; pattern is relocated between segments."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from embodiednca.v2.action_selection import AttentionConfig
from embodiednca.v2.action_selection import mass_weighted_centroid
from embodiednca.v3_5.action_selection_v3_5 import (
    SKIP_SECTOR,
    WearAwareAttentionController,
)
from embodiednca.v3.boost_cost import BoostCostConfig
from embodiednca.v2.boost_rollout_dataset import make_boost_map, make_perturbation_map
from embodiednca.v0.grid_world import ResourceGrid, WorldConfig, place_pattern
from embodiednca.v0.metrics import (
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from embodiednca.v3_5.self_model_v3_5 import SelfModelV35
from embodiednca.v3_5.wear_perwell import (
    PerWellWearConfig,
    PerWellWearState,
    nearest_well_index,
    step_with_perwell_wear,
)


SegmentSpec = Tuple[int, int]  # (distance, duration)


def segment_center(
    distance: int,
    well: Tuple[int, int] = (8, 8),
    world_cfg: Optional[WorldConfig] = None,
    placement_angle: Optional[float] = None,
) -> Tuple[int, int]:
    """Place center at `distance` from `well` (default: east; optional angle)."""
    H = world_cfg.H if world_cfg is not None else 32
    W = world_cfg.W if world_cfg is not None else 32
    if placement_angle is not None:
        cy = int(round(well[0] + distance * math.sin(placement_angle)))
        cx = int(round(well[1] + distance * math.cos(placement_angle)))
    else:
        cy, cx = well[0], well[1] + distance
    cy = max(2, min(H - 3, cy))
    cx = max(2, min(W - 3, cx))
    return cy, cx


def relocate_pattern_state(
    S: torch.Tensor,
    new_center: Tuple[int, int],
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Tuple[torch.Tensor, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Circular-roll S so its mass-weighted centroid moves to `new_center`.

    Returns (S_relocated, old_centroid_int, new_center).
    If no mass, returns S unchanged and None for old_centroid.
    """
    cent = mass_weighted_centroid(S, threshold=mass_threshold)
    if cent is None:
        return S, None, new_center
    old_y, old_x = int(round(cent[0])), int(round(cent[1]))
    dy = new_center[0] - old_y
    dx = new_center[1] - old_x
    if dy == 0 and dx == 0:
        return S, (old_y, old_x), new_center
    S_new = torch.roll(S, shifts=(dy, dx), dims=(-2, -1))
    return S_new, (old_y, old_x), new_center


def _is_dead(S: torch.Tensor) -> bool:
    return self_mass(S) <= DEATH_MASS_EPS


@torch.no_grad()
def evaluate_model_at_state(
    model: SelfModelV35,
    S: torch.Tensor,
    R: torch.Tensor,
    wear_state: PerWellWearState,
    chosen_sector: Optional[int],
    boosted: bool,
    boost_loc: Optional[Tuple[int, int]],
    magnitude: float,
    device: str,
    sigma: float = 1.5,
    pred_horizon: int = 200,
    boost_duration: int = 20,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    nca=None,
    world_cfg: Optional[WorldConfig] = None,
) -> Dict[str, float]:
    """
    Fork a short prediction-horizon rollout from the current state under the
    chosen action encoding, and compare self-model heads to realized labels.

    Mirrors death-aware training: S_ref = state at probe time; apply boost for
    boost_duration (or null); roll pred_horizon under per-well wear.
    """
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    assert nca is not None and world_cfg is not None

    mults = torch.tensor(
        wear_state.multipliers(), dtype=torch.float32, device=device
    )
    H, W = S.shape[-2], S.shape[-1]

    if boosted and boost_loc is not None:
        pert = -make_perturbation_map(
            H, W, boost_loc[0], boost_loc[1], abs(magnitude), device=device, sigma=sigma
        )
        boost_field = make_boost_map(
            H, W, boost_loc[0], boost_loc[1], abs(magnitude), device=device, sigma=sigma
        )
    else:
        pert = torch.zeros(1, 1, H, W, device=device)
        boost_field = None

    model.eval()
    S_in = S if S.dim() == 4 else S.unsqueeze(0)
    R_in = R
    if R_in.dim() == 2:
        R_in = R_in.view(1, 1, *R_in.shape)
    elif R_in.dim() == 3:
        R_in = R_in.unsqueeze(0)
    p_coh, p_lr, p_surv = model(S_in, R_in, pert, mults.unsqueeze(0))

    # Fork world
    world = ResourceGrid(nca, config=world_cfg, S0=S.detach().clone(), seed=0)
    world.S = S.detach().clone()
    world.R = R.detach().clone()
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    wear.wear = list(wear_state.wear)
    S_ref = world.S.detach().clone()

    died_h: Optional[int] = None
    if _is_dead(world.S):
        died_h = 0
    charged = False
    for h in range(1, pred_horizon + 1):
        bmap = None
        charge = False
        loc = None
        if boost_field is not None and h <= boost_duration:
            bmap = boost_field
            loc = boost_loc
            if not charged and cost_cfg.direct_boost_cost > 0:
                charge = True
                charged = True
        step_with_perwell_wear(
            world,
            wear,
            boost_map=bmap,
            gate_mode="gated",
            direct_boost_cost=cost_cfg.direct_boost_cost,
            boost_location=loc,
            charge_direct_cost=charge,
        )
        if died_h is None and _is_dead(world.S):
            died_h = h

    coh = coherence(world.S, S_ref)
    lr = local_resource_at_pattern_centroid(world.R, world.S)
    true_coh = (
        0.0 if (isinstance(coh, float) and math.isnan(coh)) else float(coh)
    )
    true_lr = (
        0.0 if (isinstance(lr, float) and math.isnan(lr)) else float(lr)
    )
    true_surv = 0.0 if died_h is not None else 1.0

    pred_coh = float(p_coh.item())
    pred_lr = float(p_lr.item())
    pred_surv = float(p_surv.item())
    return {
        "pred_coherence": pred_coh,
        "true_coherence": true_coh,
        "pred_local_R": pred_lr,
        "true_local_R": true_lr,
        "pred_survival": pred_surv,
        "true_survival": true_surv,
        "abs_err_coherence": abs(pred_coh - true_coh),
        "abs_err_local_R": abs(pred_lr - true_lr),
        "chosen_sector": float(chosen_sector if chosen_sector is not None else SKIP_SECTOR),
        "boosted": 1.0 if boosted else 0.0,
    }


@torch.no_grad()
def run_wear_schedule_preamble(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    schedule: Sequence[Tuple[str, int]],
    *,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    seed: int = 0,
    device: str = "cpu",
    base_center: Tuple[int, int] = (16, 16),
    well: Tuple[int, int] = (8, 8),
    distance: int = 2,
    boost_magnitude: float = 0.40,
    boost_sigma: float = 1.5,
    placement_angle: Optional[float] = None,
) -> Dict:
    """
    Controlled preamble that accrues a prescribed (ambient / boost_well0) schedule.

    `schedule` is a list of ("ambient"|"boost0", n_steps) phases. Ambient accrues
    wear on both wells; boost0 also accrues boost_rate on well 0 only (boost map
    fixed at well 0). Final wear depends only on total ambient steps T and total
    boost0 steps N0 — *not* on phase order — so two schedules with the same
    (T, N0) end at the same wear vector via different intermediate paths.

    `placement_angle` (radians) places the pattern at `distance` from `well`
    along that bearing (D3-style). Seed alone does not diversify a deterministic
    pipeline; angle must vary for genuine multi-run sample size.

    Returns world/wear snapshots plus the living S/R at the end.
    """
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    H, W = world_cfg.H, world_cfg.W
    center = segment_center(
        distance, well=well, world_cfg=world_cfg, placement_angle=placement_angle
    )
    S0 = place_pattern(base_pattern, center, base_center=base_center)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed
    )
    world.config.gen_rate = wear_cfg.gen_rate_base
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)

    wy, wx = wear_cfg.wells[0]
    bmap = make_boost_map(
        H, W, wy, wx, boost_magnitude, device=device, sigma=boost_sigma
    )

    total_steps = 0
    n_ambient = 0
    n_boost0 = 0
    died = False
    for phase, n in schedule:
        phase = str(phase).lower()
        for _ in range(int(n)):
            if phase in ("boost0", "boost", "boost_well0"):
                step_with_perwell_wear(
                    world,
                    wear,
                    boost_map=bmap,
                    gate_mode="gated",
                    direct_boost_cost=cost_cfg.direct_boost_cost,
                    boost_location=(wy, wx),
                    charge_direct_cost=True,
                )
                n_boost0 += 1
            else:
                step_with_perwell_wear(
                    world,
                    wear,
                    boost_map=None,
                    gate_mode="gated",
                    direct_boost_cost=0.0,
                    boost_location=None,
                    charge_direct_cost=False,
                )
                n_ambient += 1
            total_steps += 1
            if _is_dead(world.S):
                died = True
                break
        if died:
            break

    return {
        "S": world.S.detach().clone(),
        "R": world.R.detach().clone(),
        "wear": list(wear.wear),
        "multipliers": wear.multipliers(),
        "total_steps": total_steps,
        "n_ambient": n_ambient,
        "n_boost0": n_boost0,
        "died_in_preamble": died,
        "center": center,
        "distance": distance,
        "placement_angle": placement_angle,
        "schedule": [(str(p), int(n)) for p, n in schedule],
        "world": world,
        "wear_state": wear,
    }


@torch.no_grad()
def run_probe_from_wear(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: Optional[SelfModelV35],
    policy: str,
    inherited_wear: Sequence[float],
    *,
    probe_distance: int = 4,
    probe_horizon: int = 1500,
    seed: int = 0,
    device: str = "cpu",
    base_center: Tuple[int, int] = (16, 16),
    well: Tuple[int, int] = (8, 8),
    attn_cfg: Optional[AttentionConfig] = None,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    placement_angle: Optional[float] = None,
    # If True: fresh S0/R0/A0, only wear carried (isolates wear as the matched state).
    # If False: not used here — residual-state probes use run_probe_from_live_state.
    reset_SRA: bool = True,
) -> Dict:
    """
    Control / apparatus probe: place at probe_distance with inherited wear and
    *fresh* S/R/A. With a deterministic simulator + frozen model, front vs back
    paths that match wear bit-for-bit and use the same placement cannot diverge
    — this is a leakage/sanity control, not an empirical test of path dependence.
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    assert reset_SRA, "use run_probe_from_live_state for residual S/R/A probes"

    # Single-segment chained run with initial_wear via direct construction
    H, W = world_cfg.H, world_cfg.W
    center = segment_center(
        probe_distance,
        well=well,
        world_cfg=world_cfg,
        placement_angle=placement_angle,
    )
    S0 = place_pattern(base_pattern, center, base_center=base_center)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed + 991
    )
    world.config.gen_rate = wear_cfg.gen_rate_base
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    wear.wear = [float(w) for w in inherited_wear]

    ctrl = WearAwareAttentionController(attn_cfg)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1009)
    S_ref = world.S.detach().clone()
    history: List[Dict[str, float]] = []

    def record(t: int) -> None:
        mass = self_mass(world.S)
        coh = coherence(world.S, S_ref, mass=mass)
        lr = local_resource_at_pattern_centroid(world.R, world.S)
        mults = wear.multipliers()
        history.append(
            {
                "t": float(t),
                "self_mass": mass,
                "coherence": (
                    float("nan")
                    if (isinstance(coh, float) and math.isnan(coh))
                    else float(coh)
                ),
                "local_R": (
                    float("nan")
                    if (isinstance(lr, float) and math.isnan(lr))
                    else float(lr)
                ),
                "A": ctrl.A,
                "alive": 1.0 if mass > DEATH_MASS_EPS else 0.0,
                "wear_0": wear.wear[0],
                "wear_1": wear.wear[1],
                "mult_0": mults[0],
                "mult_1": mults[1],
                "mean_mult": float(sum(mults) / len(mults)),
            }
        )

    record(0)
    ctrl.maybe_decide(0, world.S, world.R, model, policy, device, rng, wear)
    died = False
    tod: Optional[int] = None
    for t in range(1, probe_horizon + 1):
        ctrl.regen_step()
        boost_map = None
        if ctrl.active_boost_map is not None and ctrl.boost_remaining > 0:
            boost_map = ctrl.active_boost_map
            ctrl.total_boost_steps += 1
        charge = False
        loc = None
        if (
            ctrl.pending_direct_charge
            and cost_cfg.charge_on_apply_only
            and boost_map is not None
            and cost_cfg.direct_boost_cost > 0
        ):
            charge = True
            loc = ctrl.last_applied_boost_location
            ctrl.pending_direct_charge = False
            ctrl.total_direct_charges += 1
        elif boost_map is not None:
            loc = ctrl.last_applied_boost_location
        step_with_perwell_wear(
            world,
            wear,
            boost_map=boost_map,
            gate_mode="gated",
            direct_boost_cost=cost_cfg.direct_boost_cost,
            boost_location=loc,
            charge_direct_cost=charge,
        )
        ctrl.tick_boost()
        if _is_dead(world.S):
            died = True
            tod = t
            record(t)
            break
        if t % attn_cfg.K == 0:
            ctrl.maybe_decide(t, world.S, world.R, model, policy, device, rng, wear)
        record(t)

    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    final = history[-1]
    return {
        "history": history,
        "policy": policy,
        "probe_distance": probe_distance,
        "probe_horizon": probe_horizon,
        "seed": seed,
        "start_wear": list(inherited_wear),
        "end_wear": list(wear.wear),
        "start_multipliers": [
            float(max(wear_cfg.gen_floor_frac, 1.0 - wear_cfg.wear_decay_coeff * w))
            for w in inherited_wear
        ],
        "mean_coherence": (
            float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
        ),
        "final_coherence": final["coherence"],
        "final_self_mass": final["self_mass"],
        "time_of_death": tod,
        "survived": not died,
        "steps_run": int(final["t"]),
        "n_boosts": ctrl.total_boosts,
        "mode": "wear_only_control_reset_SRA",
        "placement_angle": placement_angle,
        "center": center,
    }


@torch.no_grad()
def run_probe_from_live_state(
    nca,
    world_cfg: WorldConfig,
    model: Optional[SelfModelV35],
    policy: str,
    S: torch.Tensor,
    R: torch.Tensor,
    inherited_wear: Sequence[float],
    *,
    probe_distance: int = 4,
    probe_horizon: int = 1500,
    seed: int = 0,
    device: str = "cpu",
    well: Tuple[int, int] = (8, 8),
    attn_cfg: Optional[AttentionConfig] = None,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    A0: Optional[float] = None,
    placement_angle: Optional[float] = None,
) -> Dict:
    """
    Empirical residual probe: keep living S/R from the preamble (relocate S to
    probe_distance), matched wear. Front vs back *can* diverge via different
    physical S/R left by the schedule. This is Markovian residual path
    dependence — not non-Markovian memory beyond (S,R,wear,A).
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    H, W = world_cfg.H, world_cfg.W
    new_c = segment_center(
        probe_distance,
        well=well,
        world_cfg=world_cfg,
        placement_angle=placement_angle,
    )
    S_rel, _, _ = relocate_pattern_state(S, new_c, mass_threshold=attn_cfg.mass_threshold)

    world = ResourceGrid(nca, config=world_cfg, S0=S_rel.detach().clone(), seed=seed)
    world.S = S_rel.detach().clone()
    world.R = R.detach().clone()
    world.config.gen_rate = wear_cfg.gen_rate_base
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    wear.wear = [float(w) for w in inherited_wear]

    ctrl = WearAwareAttentionController(attn_cfg)
    if A0 is not None:
        ctrl.A = float(A0)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1009)
    S_ref = world.S.detach().clone()
    history: List[Dict[str, float]] = []

    def record(t: int) -> None:
        mass = self_mass(world.S)
        coh = coherence(world.S, S_ref, mass=mass)
        mults = wear.multipliers()
        history.append(
            {
                "t": float(t),
                "self_mass": mass,
                "coherence": (
                    float("nan")
                    if (isinstance(coh, float) and math.isnan(coh))
                    else float(coh)
                ),
                "A": ctrl.A,
                "alive": 1.0 if mass > DEATH_MASS_EPS else 0.0,
                "wear_0": wear.wear[0],
                "wear_1": wear.wear[1],
                "mean_mult": float(sum(mults) / len(mults)),
            }
        )

    record(0)
    ctrl.maybe_decide(0, world.S, world.R, model, policy, device, rng, wear)
    died = False
    tod: Optional[int] = None
    for t in range(1, probe_horizon + 1):
        ctrl.regen_step()
        boost_map = None
        if ctrl.active_boost_map is not None and ctrl.boost_remaining > 0:
            boost_map = ctrl.active_boost_map
        charge = False
        loc = None
        if (
            ctrl.pending_direct_charge
            and cost_cfg.charge_on_apply_only
            and boost_map is not None
            and cost_cfg.direct_boost_cost > 0
        ):
            charge = True
            loc = ctrl.last_applied_boost_location
            ctrl.pending_direct_charge = False
        elif boost_map is not None:
            loc = ctrl.last_applied_boost_location
        step_with_perwell_wear(
            world,
            wear,
            boost_map=boost_map,
            gate_mode="gated",
            direct_boost_cost=cost_cfg.direct_boost_cost,
            boost_location=loc,
            charge_direct_cost=charge,
        )
        ctrl.tick_boost()
        if _is_dead(world.S):
            died = True
            tod = t
            record(t)
            break
        if t % attn_cfg.K == 0:
            ctrl.maybe_decide(t, world.S, world.R, model, policy, device, rng, wear)
        record(t)

    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    final = history[-1]
    return {
        "history": history,
        "policy": policy,
        "probe_distance": probe_distance,
        "seed": seed,
        "start_wear": list(inherited_wear),
        "mean_coherence": (
            float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
        ),
        "final_coherence": final["coherence"],
        "time_of_death": tod,
        "survived": not died,
        "steps_run": int(final["t"]),
        "n_boosts": ctrl.total_boosts,
        "mode": "residual_SR_wear_matched",
        "placement_angle": placement_angle,
        "probe_center": new_c,
    }


@torch.no_grad()
def run_chained_sequence(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: Optional[SelfModelV35],
    policy: str,
    segments: Sequence[SegmentSpec],
    attn_cfg: Optional[AttentionConfig] = None,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    seed: int = 0,
    device: str = "cpu",
    base_center: Tuple[int, int] = (16, 16),
    well: Tuple[int, int] = (8, 8),
    fixed_sector_preset: Optional[int] = None,
    # Calibration probes (Check 2): evaluate model error on segment index >= min
    calibrate: bool = False,
    calibrate_min_segment: int = 1,  # 0-indexed; 1 ⇒ segment 2+
    pred_horizon: int = 200,
    placement_angles: Optional[Sequence[Optional[float]]] = None,
) -> Dict:
    """
    One continuous run over ordered (distance, duration) segments.

    No reset of S / R / wear / A between segments. Death ends the sequence.
    Segment-boundary markers are logged in history and segment_boundaries.
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    segments = list(segments)
    if not segments:
        raise ValueError("segments must be non-empty")

    H, W = world_cfg.H, world_cfg.W
    d0, _ = segments[0]
    center0 = segment_center(
        d0,
        well=well,
        world_cfg=world_cfg,
        placement_angle=(
            placement_angles[0] if placement_angles is not None else None
        ),
    )
    S0 = place_pattern(base_pattern, center0, base_center=base_center)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center0, seed=seed
    )
    world.config.gen_rate = wear_cfg.gen_rate_base

    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    ctrl = WearAwareAttentionController(
        attn_cfg, fixed_sector_preset=fixed_sector_preset
    )
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1009)

    # Identity reference: chain-start pattern (enduring observer frame)
    S_ref_chain = world.S.detach().clone()
    # Per-segment reference (reset only on relocate for local coherence track)
    S_ref_seg = world.S.detach().clone()

    history: List[Dict[str, float]] = []
    segment_boundaries: List[Dict] = []
    segment_summaries: List[Dict] = []
    calibration_samples: List[Dict] = []
    decision_details_all: List[Dict] = []

    global_t = 0
    died = False
    death_t: Optional[int] = None
    death_segment: Optional[int] = None

    def record(t: int, seg_idx: int, seg_local_t: int, boundary: float = 0.0) -> None:
        mass = self_mass(world.S)
        coh_chain = coherence(world.S, S_ref_chain, mass=mass)
        coh_seg = coherence(world.S, S_ref_seg, mass=mass)
        lr = local_resource_at_pattern_centroid(world.R, world.S)
        mults = wear.multipliers()
        # Mean effective gen multiplier (mechanical consequence of wear)
        mean_mult = float(sum(mults) / max(1, len(mults)))
        history.append(
            {
                "t": float(t),
                "segment": float(seg_idx),
                "seg_local_t": float(seg_local_t),
                "boundary": float(boundary),
                "self_mass": mass,
                "coherence": (
                    float("nan")
                    if (isinstance(coh_chain, float) and math.isnan(coh_chain))
                    else float(coh_chain)
                ),
                "coherence_seg": (
                    float("nan")
                    if (isinstance(coh_seg, float) and math.isnan(coh_seg))
                    else float(coh_seg)
                ),
                "local_R": (
                    float("nan")
                    if (isinstance(lr, float) and math.isnan(lr))
                    else float(lr)
                ),
                "A": ctrl.A,
                "boost_remaining": float(ctrl.boost_remaining),
                "alive": 1.0 if mass > DEATH_MASS_EPS else 0.0,
                "wear_0": wear.wear[0],
                "wear_1": wear.wear[1],
                "mult_0": mults[0],
                "mult_1": mults[1],
                "mean_mult": mean_mult,
                "all_floored": 1.0 if wear.all_floored() else 0.0,
                "distance": float(segments[seg_idx][0]),
            }
        )

    def maybe_calibrate(seg_idx: int, log) -> None:
        if not calibrate or model is None or policy != "selected":
            return
        if seg_idx < calibrate_min_segment:
            return
        # Only on actual decision ticks already processed
        if log is None:
            return
        boosted = bool(log.boosted)
        sample = evaluate_model_at_state(
            model=model,
            S=world.S,
            R=world.R,
            wear_state=wear,
            chosen_sector=log.selected_sector,
            boosted=boosted,
            boost_loc=log.selected if boosted else None,
            magnitude=attn_cfg.boost_magnitude,
            device=device,
            sigma=attn_cfg.boost_sigma,
            pred_horizon=pred_horizon,
            boost_duration=attn_cfg.boost_duration,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            nca=nca,
            world_cfg=world_cfg,
        )
        sample["t"] = float(global_t)
        sample["segment"] = float(seg_idx)
        sample["wear_0"] = wear.wear[0]
        sample["wear_1"] = wear.wear[1]
        sample["mult_0"] = wear.multipliers()[0]
        sample["mult_1"] = wear.multipliers()[1]
        sample["distance"] = float(segments[seg_idx][0])
        calibration_samples.append(sample)

    # ---- Segment loop ----
    for seg_idx, (distance, duration) in enumerate(segments):
        if died:
            break

        # Relocate for segments after the first (keep S structure, R, wear, A)
        if seg_idx > 0:
            ang = (
                placement_angles[seg_idx]
                if placement_angles is not None and seg_idx < len(placement_angles)
                else None
            )
            new_c = segment_center(
                distance, well=well, world_cfg=world_cfg, placement_angle=ang
            )
            old_S_cent = mass_weighted_centroid(world.S)
            world.S, old_c, _ = relocate_pattern_state(
                world.S, new_c, mass_threshold=attn_cfg.mass_threshold
            )
            # Clear active boost (location no longer valid); keep A budget
            ctrl.active_boost_map = None
            ctrl.boost_remaining = 0
            ctrl.pending_direct_charge = False
            ctrl.last_applied_boost_location = None
            # Segment-local coherence reference after relocate
            S_ref_seg = world.S.detach().clone()

            boundary_info = {
                "global_t": global_t,
                "from_segment": seg_idx - 1,
                "to_segment": seg_idx,
                "from_distance": segments[seg_idx - 1][0],
                "to_distance": distance,
                "old_centroid": old_c,
                "new_center": new_c,
                "pre_relocate_centroid": (
                    (float(old_S_cent[0]), float(old_S_cent[1]))
                    if old_S_cent is not None
                    else None
                ),
                "A": ctrl.A,
                "wear": list(wear.wear),
                "multipliers": wear.multipliers(),
                "alive": not _is_dead(world.S),
                "kind": "chained_boundary",
            }
            segment_boundaries.append(boundary_info)
            record(global_t, seg_idx, 0, boundary=1.0)
        else:
            segment_boundaries.append(
                {
                    "global_t": 0,
                    "from_segment": -1,
                    "to_segment": 0,
                    "from_distance": None,
                    "to_distance": distance,
                    "A": ctrl.A,
                    "wear": list(wear.wear),
                    "multipliers": wear.multipliers(),
                    "alive": True,
                    "kind": "chain_start",
                }
            )
            record(0, 0, 0, boundary=1.0)

        # Decision at segment entry (t=0 of segment, or immediate after relocate)
        log = ctrl.maybe_decide(
            global_t, world.S, world.R, model, policy, device, rng, wear
        )
        maybe_calibrate(seg_idx, log)
        if ctrl.decision_details:
            d = dict(ctrl.decision_details[-1])
            d["segment"] = seg_idx
            d["global_t"] = global_t
            decision_details_all.append(d)

        seg_start_t = global_t
        seg_start_wear = list(wear.wear)
        seg_start_mult = wear.multipliers()
        seg_alive_start = not _is_dead(world.S)
        n_boosts_start = ctrl.total_boosts

        for step_i in range(1, duration + 1):
            if died:
                break
            ctrl.regen_step()

            boost_map = None
            if ctrl.active_boost_map is not None and ctrl.boost_remaining > 0:
                boost_map = ctrl.active_boost_map
                ctrl.total_boost_steps += 1

            charge = False
            loc = None
            if (
                ctrl.pending_direct_charge
                and cost_cfg.charge_on_apply_only
                and boost_map is not None
                and cost_cfg.direct_boost_cost > 0
            ):
                charge = True
                loc = ctrl.last_applied_boost_location
                ctrl.pending_direct_charge = False
                ctrl.total_direct_charges += 1
            elif boost_map is not None:
                loc = ctrl.last_applied_boost_location

            step_with_perwell_wear(
                world,
                wear,
                boost_map=boost_map,
                gate_mode="gated",
                direct_boost_cost=cost_cfg.direct_boost_cost,
                boost_location=loc,
                charge_direct_cost=charge,
            )
            ctrl.tick_boost()
            global_t += 1

            if _is_dead(world.S):
                died = True
                death_t = global_t
                death_segment = seg_idx
                record(global_t, seg_idx, step_i, boundary=0.0)
                break

            if global_t % attn_cfg.K == 0:
                log = ctrl.maybe_decide(
                    global_t, world.S, world.R, model, policy, device, rng, wear
                )
                maybe_calibrate(seg_idx, log)
                if ctrl.decision_details:
                    d = dict(ctrl.decision_details[-1])
                    d["segment"] = seg_idx
                    d["global_t"] = global_t
                    decision_details_all.append(d)

            record(global_t, seg_idx, step_i, boundary=0.0)

        # Segment summary
        seg_hist = [h for h in history if int(h["segment"]) == seg_idx]
        alive_coh = [
            h["coherence_seg"]
            for h in seg_hist
            if h["alive"] > 0.5 and not math.isnan(h["coherence_seg"])
        ]
        final = seg_hist[-1] if seg_hist else history[-1]
        completed = (not died) or (death_segment is not None and death_segment > seg_idx)
        # completed this segment fully if we ran all steps without dying in it
        survived_segment = not (
            died and death_segment == seg_idx
        )
        segment_summaries.append(
            {
                "segment": seg_idx,
                "distance": distance,
                "duration": duration,
                "global_t_start": seg_start_t,
                "global_t_end": global_t,
                "steps_run": global_t - seg_start_t,
                "survived_segment": survived_segment,
                "completed_full_duration": (global_t - seg_start_t) >= duration
                and survived_segment,
                "died_mid_segment": died and death_segment == seg_idx,
                "mean_coherence_seg": (
                    float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
                ),
                "final_coherence_seg": final.get("coherence_seg", float("nan")),
                "final_coherence_chain": final.get("coherence", float("nan")),
                "final_self_mass": final["self_mass"],
                "start_wear": seg_start_wear,
                "end_wear": list(wear.wear),
                "start_multipliers": seg_start_mult,
                "end_multipliers": wear.multipliers(),
                "start_mean_mult": float(sum(seg_start_mult) / len(seg_start_mult)),
                "end_mean_mult": float(
                    sum(wear.multipliers()) / len(wear.multipliers())
                ),
                "n_boosts_in_segment": ctrl.total_boosts - n_boosts_start,
                "alive_at_start": seg_alive_start,
                "alive_at_end": not _is_dead(world.S) if not died else False,
            }
        )

    alive_coh_chain = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    final = history[-1] if history else {}

    return {
        "protocol": "chained",
        "policy": policy,
        "seed": seed,
        "segments": [(int(d), int(dur)) for d, dur in segments],
        "history": history,
        "segment_boundaries": segment_boundaries,
        "segment_summaries": segment_summaries,
        "decision_details": decision_details_all,
        "selection_logs": ctrl.logs,
        "geometry_logs": ctrl.geometry_logs,
        "calibration_samples": calibration_samples,
        "time_of_death": death_t,
        "death_segment": death_segment,
        "survived_full_chain": not died,
        "n_segments_reached": len(segment_summaries),
        "n_boosts": ctrl.total_boosts,
        "n_decisions": ctrl.total_decisions,
        "n_skipped_voluntary": ctrl.total_skipped_voluntary,
        "n_skipped_budget": ctrl.total_skipped_budget,
        "final_wear": list(wear.wear),
        "final_multipliers": wear.multipliers(),
        "final_A": ctrl.A,
        "mean_coherence_chain": (
            float(sum(alive_coh_chain) / len(alive_coh_chain))
            if alive_coh_chain
            else float("nan")
        ),
        "final_coherence_chain": final.get("coherence", float("nan")),
        "final_self_mass": final.get("self_mass", 0.0),
        "total_steps": global_t,
    }


def _angle_abs_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def aimed_well_index(
    cy: float,
    cx: float,
    choice_angle: float,
    wells: Sequence[Tuple[int, int]],
) -> int:
    """
    Which well the boost *direction* from the pattern centroid points toward.

    Important: do NOT use nearest_well_index(boost_cell) — sector candidates sit
    on the pattern body, so nearest-well-of-cell almost always equals the well
    the pattern is near, even when the chosen sector aims at the other well.
    """
    best_i = 0
    best_err = float("inf")
    for i, (wy, wx) in enumerate(wells):
        wa = math.atan2(float(wy) - cy, float(wx) - cx)
        err = _angle_abs_diff(choice_angle, wa)
        if err < best_err:
            best_err = err
            best_i = i
    return best_i


def analyze_pathological_decisions(
    run: Dict,
    wear_cfg: Optional[PerWellWearConfig] = None,
    wear_gap_threshold: float = 0.25,
) -> Dict:
    """
    Check 3 helper: flag decisions that are clearly bad given true wear state.

    Pathological cases:
      1. Boost while both wells are at the gen floor (skip is strictly better).
      2. Boost whose *aim direction* points at the more-worn well when wear gap
         exceeds `wear_gap_threshold` (fresher well available).

    Aim is measured by choice_angle from geometry_logs (centroid → sector cell),
    not by nearest-well of the boost cell (that metric is almost always the well
    the pattern sits near, independent of sector choice).

    wear_aware_heuristic reads current wear by design and should never do (2)
    and should skip under (1). selected may under OOD inherited starts.
    """
    wear_cfg = wear_cfg or PerWellWearConfig()
    wells = list(wear_cfg.wells)
    details = run.get("decision_details") or []
    geom_by_t: Dict[int, Dict] = {}
    for g in run.get("geometry_logs") or []:
        geom_by_t[int(g.get("t", -1))] = g

    flags: List[Dict] = []
    n_boost = 0
    n_pathological = 0
    n_boost_while_floored = 0
    n_boost_worn_well = 0
    n_scored_aim = 0

    for d in details:
        if not d.get("boosted"):
            continue
        n_boost += 1
        wears = d.get("wear") or []
        if len(wears) < 2:
            continue
        mults = d.get("multipliers") or []
        all_floored = bool(d.get("all_floored"))
        sector = d.get("selected_sector")
        t = d.get("t", d.get("global_t"))
        t_int = int(t) if t is not None else -1

        if all_floored:
            flags.append(
                {
                    "t": t,
                    "segment": d.get("segment"),
                    "reason": "boost_while_both_floored",
                    "wear": list(wears),
                    "multipliers": list(mults),
                    "sector": sector,
                    "policy": d.get("policy"),
                }
            )
            n_pathological += 1
            n_boost_while_floored += 1
            continue

        w0, w1 = float(wears[0]), float(wears[1])
        wear_gap = abs(w0 - w1)
        if wear_gap < wear_gap_threshold:
            continue
        fresher = 0 if w0 < w1 else 1
        more_worn = 1 - fresher

        g = geom_by_t.get(t_int)
        if g is None:
            continue
        ca = g.get("choice_angle")
        cy, cx = g.get("centroid_y"), g.get("centroid_x")
        if ca is None or cy is None or math.isnan(float(cy)):
            continue
        target_well = aimed_well_index(float(cy), float(cx), float(ca), wells)
        n_scored_aim += 1

        if int(target_well) == more_worn:
            flags.append(
                {
                    "t": t,
                    "segment": d.get("segment"),
                    "reason": "boost_toward_more_worn_well",
                    "wear": list(wears),
                    "fresher_well": fresher,
                    "more_worn_well": more_worn,
                    "target_well": int(target_well),
                    "wear_gap": wear_gap,
                    "sector": sector,
                    "policy": d.get("policy"),
                    "choice_angle": float(ca),
                }
            )
            n_pathological += 1
            n_boost_worn_well += 1

    return {
        "n_boost_decisions": n_boost,
        "n_pathological": n_pathological,
        "n_boost_while_both_floored": n_boost_while_floored,
        "n_boost_toward_more_worn": n_boost_worn_well,
        "n_aim_scored": n_scored_aim,
        "pathological_rate": (
            float(n_pathological / n_boost) if n_boost > 0 else 0.0
        ),
        "flags": flags[:50],
        "n_flags_total": len(flags),
    }
