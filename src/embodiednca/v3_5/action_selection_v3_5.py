"""7-way argmax (6 sectors + skip) over self_model_v3_5; skip can win with a full attention budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from embodiednca.v2.action_selection import (
    AttentionConfig,
    Candidate,
    SelectionLog,
    generate_candidates,
    geometry_of_choice,
    mask_candidates,
    mass_weighted_centroid,
    select_wellward_candidate,
    _angle_diff,
)
from embodiednca.v3.boost_cost import BoostCostConfig, read_R_at
from embodiednca.v2.boost_rollout_dataset import make_boost_map, make_perturbation_map
from embodiednca.v0.grid_world import DEFAULT_WELLS, ResourceGrid, WorldConfig, place_pattern
from embodiednca.v0.metrics import (
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from embodiednca.v3_5.self_model_v3_5 import SelfModelV35, predict_coherence_future
from embodiednca.v3_5.wear_perwell import (
    PerWellWearConfig,
    PerWellWearState,
    nearest_well_index,
    step_with_perwell_wear,
)

# Sector id reserved for the explicit skip / no-boost candidate
SKIP_SECTOR = -1


@dataclass
class ScoredCandidate:
    candidate: Optional[Candidate]  # None ⇔ skip
    is_skip: bool
    score: float
    sector: int  # SKIP_SECTOR for skip


@torch.no_grad()
def score_candidates_v35(
    model: SelfModelV35,
    S: torch.Tensor,
    R: torch.Tensor,
    wear_features: torch.Tensor,
    candidates: List[Candidate],
    magnitude: float,
    device: str,
    sigma: float = 1.5,
    include_skip: bool = True,
) -> List[ScoredCandidate]:
    """
    Score each sector candidate under boost-at-c encoding, plus optional skip
    (zero pert_map). wear_features: [n_wells] multipliers or raw wear.
    """
    if S.dim() == 3:
        S = S.unsqueeze(0)
    if R.dim() == 3:
        R = R.unsqueeze(0)
    if R.dim() == 2:
        R = R.view(1, 1, *R.shape)
    if wear_features.dim() == 1:
        wear_features = wear_features.unsqueeze(0)

    _, _, H, W = S.shape
    abs_mag = abs(float(magnitude))
    maps: List[torch.Tensor] = []
    meta: List[Tuple[Optional[Candidate], bool, int]] = []

    for c in candidates:
        p = make_perturbation_map(H, W, c.y, c.x, abs_mag, device=device, sigma=sigma)
        maps.append(-p)  # negative ⇔ boost
        meta.append((c, False, c.sector))

    if include_skip:
        maps.append(torch.zeros(1, 1, H, W, device=device))
        meta.append((None, True, SKIP_SECTOR))

    if not maps:
        return []

    P = torch.cat(maps, dim=0)
    B = P.shape[0]
    S_b = S.expand(B, -1, -1, -1).contiguous()
    R_b = R.expand(B, -1, -1, -1).contiguous()
    W_b = wear_features.expand(B, -1).contiguous()
    pred = predict_coherence_future(model, S_b, R_b, P, W_b, device=device)

    out: List[ScoredCandidate] = []
    for i, (c, is_skip, sector) in enumerate(meta):
        out.append(
            ScoredCandidate(
                candidate=c,
                is_skip=is_skip,
                score=float(pred[i].item()),
                sector=sector,
            )
        )
    return out


def select_wear_aware_heuristic(
    candidates: List[Candidate],
    S: torch.Tensor,
    wear_state: PerWellWearState,
    wells: Sequence[Tuple[int, int]],
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Tuple[Optional[Candidate], bool]:
    """
    Hand-coded, no model:
      - if both wells' effective_multipliers are at the floor → skip
      - else boost the sector nearest the well with *lower* wear
        (fresher / less degraded well)

    Returns (chosen_candidate_or_None, is_skip).
    """
    if wear_state.all_floored():
        return None, True
    if not candidates:
        return None, True

    wears = wear_state.wear
    # Lower wear = preferred well
    best_well_idx = int(min(range(len(wears)), key=lambda i: wears[i]))
    # If equal wear, prefer nearest geometrically (fallback)
    if len(set(round(w, 8) for w in wears)) == 1:
        cent = mass_weighted_centroid(S, threshold=mass_threshold)
        if cent is not None:
            cy, cx = cent
            best_well_idx = nearest_well_index(cy, cx, wells)

    target_well = wells[best_well_idx]
    cent = mass_weighted_centroid(S, threshold=mass_threshold)
    if cent is None:
        return min(candidates, key=lambda c: c.sector), False
    cy, cx = cent
    well_angle = math.atan2(target_well[0] - cy, target_well[1] - cx)

    def ang_score(c: Candidate) -> float:
        return _angle_diff(math.atan2(c.y - cy, c.x - cx), well_angle)

    return min(candidates, key=ang_score), False


def select_among_scored(
    scored: List[ScoredCandidate],
    policy: str,
    rng: torch.Generator,
    rr_index: int,
    fixed_sector: Optional[int],
    S: Optional[torch.Tensor],
    wear_state: Optional[PerWellWearState],
    wells: Sequence[Tuple[int, int]],
    candidates: List[Candidate],
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Tuple[Optional[Candidate], bool, int, Optional[int], Optional[float]]:
    """
    Returns (chosen, is_skip, new_rr_index, new_fixed_sector, score_of_selected).

    policy:
      selected — argmax over scored (incl. skip)
      wear_aware_heuristic — hand-coded wear rule
      wellward_heuristic — geometry only (never skip unless no candidates)
      never_boost — always skip
      always_boost — never skip; pick best non-skip (or wellward if no scores)
      random / round_robin / fixed — sector only, never skip voluntarily
    """
    if policy == "never_boost":
        return None, True, rr_index, fixed_sector, None

    if policy in ("wear_aware_heuristic", "wear_aware"):
        if wear_state is None or S is None:
            raise ValueError("wear_aware_heuristic requires wear_state and S")
        chosen, is_skip = select_wear_aware_heuristic(
            candidates, S, wear_state, wells, mass_threshold=mass_threshold
        )
        return chosen, is_skip, rr_index, fixed_sector, None

    if policy == "selected":
        if not scored:
            return None, True, rr_index, fixed_sector, None
        best = max(scored, key=lambda s: s.score)
        return (
            best.candidate,
            best.is_skip,
            rr_index,
            fixed_sector,
            best.score,
        )

    if policy == "always_boost":
        # Extreme: always spend A on a sector (wellward geometry), never skip.
        # Does not use the self-model — pure "greedy action" control.
        if S is not None and candidates:
            c = select_wellward_candidate(
                candidates, S, wells, mass_threshold=mass_threshold
            )
            return c, False, rr_index, fixed_sector, None
        if candidates:
            return candidates[0], False, rr_index, fixed_sector, None
        return None, True, rr_index, fixed_sector, None

    # Sector-only baselines: never voluntary skip
    if not candidates:
        return None, True, rr_index, fixed_sector, None

    if policy in ("wellward_heuristic", "wellward"):
        if S is None:
            raise ValueError("wellward requires S")
        c = select_wellward_candidate(
            candidates, S, wells, mass_threshold=mass_threshold
        )
        return c, False, rr_index, fixed_sector, None

    if policy == "random":
        i = int(torch.randint(0, len(candidates), (1,), generator=rng).item())
        return candidates[i], False, rr_index, fixed_sector, None

    if policy in ("round_robin", "round-robin"):
        i = rr_index % len(candidates)
        return candidates[i], False, rr_index + 1, fixed_sector, None

    if policy == "fixed":
        if fixed_sector is None:
            fixed_sector = int(min(c.sector for c in candidates))
        exact = [c for c in candidates if c.sector == fixed_sector]
        if exact:
            return exact[0], False, rr_index, fixed_sector, None

        def circ_dist(a: int, b: int, m: int = 6) -> int:
            d = abs(a - b) % m
            return min(d, m - d)

        best_c = min(
            candidates,
            key=lambda c: (circ_dist(c.sector, fixed_sector), c.sector),
        )
        return best_c, False, rr_index, fixed_sector, None

    raise ValueError(f"Unknown policy: {policy!r}")


class WearAwareAttentionController:
    """
    Attention controller with 7-way selection (6 sectors + skip), per-well wear
    state wiring, and direct boost cost flags.
    """

    def __init__(
        self,
        cfg: Optional[AttentionConfig] = None,
        fixed_sector_preset: Optional[int] = None,
    ):
        self.cfg = cfg or AttentionConfig()
        self.A = float(self.cfg.A0)
        self.boost_remaining = 0
        self.active_boost_map: Optional[torch.Tensor] = None
        self.rr_index = 0
        self.fixed_sector: Optional[int] = (
            int(fixed_sector_preset) if fixed_sector_preset is not None else None
        )
        self.logs: List[SelectionLog] = []
        self.geometry_logs: List[Dict] = []
        self.decision_details: List[Dict] = []  # skip/boost + wear at decision

        self.total_masked_selected = 0
        self.total_candidates_dropped = 0
        self.total_boosts = 0
        self.total_decisions = 0
        self.total_skipped_budget = 0
        self.total_skipped_voluntary = 0  # chose skip with budget available
        self.total_direct_charges = 0
        self.total_boost_steps = 0
        self.last_applied_boost_location: Optional[Tuple[int, int]] = None
        self.pending_direct_charge = False

    def regen_step(self) -> None:
        self.A = min(self.cfg.A_max, self.A + self.cfg.regen_rate)

    def tick_boost(self) -> None:
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            if self.boost_remaining <= 0:
                self.active_boost_map = None

    def maybe_decide(
        self,
        t: int,
        S: torch.Tensor,
        R: torch.Tensor,
        model: Optional[SelfModelV35],
        policy: str,
        device: str,
        rng: torch.Generator,
        wear_state: PerWellWearState,
    ) -> SelectionLog:
        cfg = self.cfg
        is_decision = (t == 0) or (t % cfg.K == 0)
        log = SelectionLog(
            t=t,
            policy=policy,
            n_raw=0,
            n_surviving=0,
            n_dropped_mask=0,
            selected=None,
            selected_sector=None,
            selected_was_masked=False,
            scores=[],
            score_of_selected=None,
            A_before=self.A,
            A_after=self.A,
            boosted=False,
        )
        if not is_decision:
            return log

        self.total_decisions += 1
        raw = generate_candidates(
            S, n_candidates=cfg.n_candidates, mass_threshold=cfg.mass_threshold
        )
        surviving, dropped, n_drop = mask_candidates(
            raw, S, wells=cfg.wells, mass_threshold=cfg.mass_threshold
        )
        self.total_candidates_dropped += n_drop
        log.n_raw = len(raw)
        log.n_surviving = len(surviving)
        log.n_dropped_mask = n_drop

        # Wear features for model (multipliers — matches training default)
        mults = torch.tensor(
            wear_state.multipliers(), dtype=torch.float32, device=device
        )

        qmag = (
            cfg.query_magnitude
            if cfg.query_magnitude is not None
            else cfg.boost_magnitude
        )

        scored: List[ScoredCandidate] = []
        if policy == "selected":
            if model is None:
                raise ValueError("policy='selected' requires a self-model")
            scored = score_candidates_v35(
                model,
                S,
                R,
                mults,
                surviving,
                magnitude=qmag,
                device=device,
                sigma=cfg.boost_sigma,
                include_skip=True,
            )
            log.scores = [s.score for s in scored]

        chosen, is_skip, self.rr_index, self.fixed_sector, sel_score = (
            select_among_scored(
                scored,
                policy=policy,
                rng=rng,
                rr_index=self.rr_index,
                fixed_sector=self.fixed_sector,
                S=S,
                wear_state=wear_state,
                wells=cfg.wells,
                candidates=surviving,
                mass_threshold=cfg.mass_threshold,
            )
        )
        log.score_of_selected = sel_score

        if chosen is not None and chosen.masked:
            self.total_masked_selected += 1
            log.selected_was_masked = True

        if chosen is not None and not is_skip:
            log.selected = (chosen.y, chosen.x)
            log.selected_sector = int(chosen.sector)
            geom = geometry_of_choice(
                S,
                chosen,
                wells=cfg.wells,
                n_candidates=cfg.n_candidates,
                mass_threshold=cfg.mass_threshold,
            )
            geom["t"] = t
            geom["policy"] = policy
            geom["chosen_sector"] = int(chosen.sector)
            self.geometry_logs.append(geom)
        else:
            log.selected = None
            log.selected_sector = SKIP_SECTOR

        # Arm boost?
        budget_ok = self.A >= cfg.boost_cost
        want_boost = (not is_skip) and chosen is not None
        # always_boost / sector baselines still need budget
        if want_boost and budget_ok:
            H, W = S.shape[-2], S.shape[-1]
            self.active_boost_map = make_boost_map(
                H,
                W,
                chosen.y,
                chosen.x,
                cfg.boost_magnitude,
                device=S.device,
                sigma=cfg.boost_sigma,
            )
            self.boost_remaining = cfg.boost_duration
            self.A -= cfg.boost_cost
            log.boosted = True
            self.total_boosts += 1
            self.last_applied_boost_location = (chosen.y, chosen.x)
            self.pending_direct_charge = True
        else:
            if want_boost and not budget_ok:
                self.total_skipped_budget += 1
            if is_skip and budget_ok and policy == "selected":
                self.total_skipped_voluntary += 1
            if is_skip and policy != "selected":
                # never_boost / wear_aware skip / etc.
                pass

        detail = {
            "t": t,
            "policy": policy,
            "is_skip": is_skip or not log.boosted,
            "voluntary_skip": bool(is_skip and budget_ok),
            "budget_skip": bool(want_boost and not budget_ok),
            "boosted": log.boosted,
            "selected_sector": log.selected_sector,
            "score": sel_score,
            "A_before": log.A_before,
            "A_after": self.A,
            "wear": list(wear_state.wear),
            "multipliers": wear_state.multipliers(),
            "all_floored": wear_state.all_floored(),
            "well0_floored": wear_state.is_floored(0),
            "well1_floored": wear_state.is_floored(1),
        }
        self.decision_details.append(detail)

        log.A_after = self.A
        self.logs.append(log)
        return log


@torch.no_grad()
def run_v35_episode(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: Optional[SelfModelV35],
    policy: str,
    distance: int,
    horizon: int = 5000,
    well: Tuple[int, int] = (8, 8),
    attn_cfg: Optional[AttentionConfig] = None,
    wear_cfg: Optional[PerWellWearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    seed: int = 0,
    device: str = "cpu",
    base_center: Tuple[int, int] = (16, 16),
    fixed_sector_preset: Optional[int] = None,
    # Pre-degrade one well before the comparison horizon (Check 3)
    pre_degrade_well_idx: Optional[int] = None,
    pre_degrade_boosts: int = 0,
    pre_degrade_steps: int = 0,
    initial_wear: Optional[Sequence[float]] = None,
    # Optional explicit placement (overrides well+distance default)
    center: Optional[Tuple[int, int]] = None,
    placement_angle: Optional[float] = None,
) -> Dict:
    """
    Full attention loop under per-well wear + direct cost + skip-capable selection.

    Placement: if `center` given, use it; elif `placement_angle` (radians) given,
    place at `distance` from `well` along that angle; else default
    (well[0], well[1] + distance) — east of the well.
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()

    if center is not None:
        pass  # use as given
    elif placement_angle is not None:
        cy = int(round(well[0] + distance * math.sin(placement_angle)))
        cx = int(round(well[1] + distance * math.cos(placement_angle)))
        # clamp to grid with margin
        H, W = world_cfg.H, world_cfg.W
        cy = max(2, min(H - 3, cy))
        cx = max(2, min(W - 3, cx))
        center = (cy, cx)
    else:
        center = (well[0], well[1] + distance)
    S0 = place_pattern(base_pattern, center, base_center=base_center)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed
    )
    world.config.gen_rate = wear_cfg.gen_rate_base

    H, W = world_cfg.H, world_cfg.W
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    if initial_wear is not None:
        wear.wear = [float(w) for w in initial_wear]

    # Optional asymmetric pre-degradation (boost near one well repeatedly)
    pre_history: List[Dict] = []
    if pre_degrade_well_idx is not None and (
        pre_degrade_boosts > 0 or pre_degrade_steps > 0
    ):
        wy, wx = wear_cfg.wells[pre_degrade_well_idx]
        # Place boost near that well but on the pattern if possible
        bmap = make_boost_map(
            H,
            W,
            wy,
            wx,
            attn_cfg.boost_magnitude,
            device=device,
            sigma=attn_cfg.boost_sigma,
        )
        steps = max(pre_degrade_steps, pre_degrade_boosts * attn_cfg.boost_duration)
        for s in range(steps):
            # Accrue ambient + boost on target well every step for `steps`
            boosted = s < pre_degrade_boosts * attn_cfg.boost_duration
            step_with_perwell_wear(
                world,
                wear,
                boost_map=bmap if boosted else None,
                gate_mode="gated",
                direct_boost_cost=0.0,  # pre-phase: wear only
                boost_location=(wy, wx) if boosted else None,
                charge_direct_cost=False,
            )
            pre_history.append(
                {
                    "t": -(steps - s),
                    "wear": list(wear.wear),
                    "multipliers": wear.multipliers(),
                }
            )
        # Reset pattern after pre-degrade so comparison starts from a fresh
        # pattern placement on the degraded resource field — keep R/wear.
        S0_fresh = place_pattern(base_pattern, center, base_center=base_center)
        world.S = S0_fresh
        # Re-settle briefly without boost
        for _ in range(10):
            step_with_perwell_wear(
                world,
                wear,
                boost_map=None,
                gate_mode="gated",
                direct_boost_cost=0.0,
                boost_location=None,
                charge_direct_cost=False,
            )

    ctrl = WearAwareAttentionController(
        attn_cfg, fixed_sector_preset=fixed_sector_preset
    )
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
                "boost_remaining": float(ctrl.boost_remaining),
                "alive": 1.0 if mass > DEATH_MASS_EPS else 0.0,
                "wear_0": wear.wear[0],
                "wear_1": wear.wear[1],
                "mult_0": mults[0],
                "mult_1": mults[1],
                "all_floored": 1.0 if wear.all_floored() else 0.0,
            }
        )

    record(0)
    ctrl.maybe_decide(0, world.S, world.R, model, policy, device, rng, wear)

    for t in range(1, horizon + 1):
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

        if t % attn_cfg.K == 0:
            ctrl.maybe_decide(
                t, world.S, world.R, model, policy, device, rng, wear
            )
        record(t)

    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    alive_mass = [h["self_mass"] for h in history if h["alive"] > 0.5]
    final = history[-1]
    tod = next((int(h["t"]) for h in history if h["alive"] < 0.5), None)

    # Floor-crossing times per well
    floor_t = []
    for wi in range(wear.n_wells):
        key = f"mult_{wi}"
        t_cross = next(
            (
                int(h["t"])
                for h in history
                if h[key] <= wear_cfg.gen_floor_frac + 1e-9
            ),
            None,
        )
        floor_t.append(t_cross)

    return {
        "history": history,
        "policy": policy,
        "distance": distance,
        "seed": seed,
        "mean_coherence": (
            float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
        ),
        "final_coherence": final["coherence"],
        "mean_self_mass": (
            float(sum(alive_mass) / len(alive_mass)) if alive_mass else 0.0
        ),
        "final_self_mass": final["self_mass"],
        "time_of_death": tod,
        "n_boosts": ctrl.total_boosts,
        "n_boost_steps": ctrl.total_boost_steps,
        "n_decisions": ctrl.total_decisions,
        "n_skipped_budget": ctrl.total_skipped_budget,
        "n_skipped_voluntary": ctrl.total_skipped_voluntary,
        "n_candidates_dropped_mask": ctrl.total_candidates_dropped,
        "n_masked_selected": ctrl.total_masked_selected,
        "n_direct_charges": ctrl.total_direct_charges,
        "final_wear": list(wear.wear),
        "final_multipliers": wear.multipliers(),
        "floor_crossing_t": floor_t,
        "selection_logs": ctrl.logs,
        "geometry_logs": ctrl.geometry_logs,
        "decision_details": ctrl.decision_details,
        "fixed_sector_locked": ctrl.fixed_sector,
        "pre_degrade_wear_end": (
            list(pre_history[-1]["wear"]) if pre_history else None
        ),
        "wear_cfg": {
            "wear_ambient_rate": wear_cfg.wear_ambient_rate,
            "wear_boost_rate": wear_cfg.wear_boost_rate,
            "wear_decay_coeff": wear_cfg.wear_decay_coeff,
            "gen_floor_frac": wear_cfg.gen_floor_frac,
            "gen_rate_base": wear_cfg.gen_rate_base,
        },
        "cost_cfg": {
            "direct_boost_cost": cost_cfg.direct_boost_cost,
        },
        "attn_cfg": {
            "K": attn_cfg.K,
            "boost_cost": attn_cfg.boost_cost,
            "regen_rate": attn_cfg.regen_rate,
            "A_max": attn_cfg.A_max,
            "boost_magnitude": attn_cfg.boost_magnitude,
            "n_candidates": attn_cfg.n_candidates,
        },
    }
