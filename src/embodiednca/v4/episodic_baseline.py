"""Same challenge sequence as chained, with a full S/R/wear/A reset at every segment boundary."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from embodiednca.v2.action_selection import AttentionConfig
from embodiednca.v3_5.action_selection_v3_5 import WearAwareAttentionController
from embodiednca.v3.boost_cost import BoostCostConfig
from embodiednca.v0.grid_world import ResourceGrid, WorldConfig, place_pattern
from embodiednca.v0.metrics import (
    DEATH_MASS_EPS,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from embodiednca.v3_5.self_model_v3_5 import SelfModelV35
from embodiednca.v3_5.wear_perwell import (
    PerWellWearConfig,
    PerWellWearState,
    step_with_perwell_wear,
)

from embodiednca.v4.chained_protocol import (
    SegmentSpec,
    evaluate_model_at_state,
    segment_center,
)


@torch.no_grad()
def run_episodic_sequence(
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
    calibrate: bool = False,
    calibrate_min_segment: int = 1,
    pred_horizon: int = 200,
    placement_angles: Optional[Sequence[Optional[float]]] = None,
) -> Dict:
    """
    Same segment list as chained, but full reset of S/R/wear/A each segment.

    Death within a segment ends that segment; subsequent segments still run
    (fresh start) — matching prior-stage "each scenario independent" design.
    Global history is concatenated with boundary markers for plotting parity.
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or PerWellWearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()
    segments = list(segments)
    if not segments:
        raise ValueError("segments must be non-empty")

    H, W = world_cfg.H, world_cfg.W
    history: List[Dict[str, float]] = []
    segment_boundaries: List[Dict] = []
    segment_summaries: List[Dict] = []
    calibration_samples: List[Dict] = []
    decision_details_all: List[Dict] = []
    all_logs = []
    all_geom = []

    global_t = 0
    total_boosts = 0
    total_decisions = 0
    total_vol_skip = 0
    total_budget_skip = 0
    any_death = False
    first_death_t: Optional[int] = None
    first_death_segment: Optional[int] = None

    for seg_idx, (distance, duration) in enumerate(segments):
        ang = (
            placement_angles[seg_idx]
            if placement_angles is not None and seg_idx < len(placement_angles)
            else None
        )
        center = segment_center(
            distance, well=well, world_cfg=world_cfg, placement_angle=ang
        )
        S0 = place_pattern(base_pattern, center, base_center=base_center)
        # Independent seed offset per segment for R settle noise parity
        world = ResourceGrid(
            nca,
            config=world_cfg,
            S0=S0,
            pattern_center=center,
            seed=seed + 17 * seg_idx,
        )
        world.config.gen_rate = wear_cfg.gen_rate_base
        wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
        # Full reset of attention budget
        ctrl = WearAwareAttentionController(
            attn_cfg, fixed_sector_preset=fixed_sector_preset
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed + 1009 + 31 * seg_idx)

        S_ref = world.S.detach().clone()

        segment_boundaries.append(
            {
                "global_t": global_t,
                "from_segment": seg_idx - 1,
                "to_segment": seg_idx,
                "from_distance": (
                    segments[seg_idx - 1][0] if seg_idx > 0 else None
                ),
                "to_distance": distance,
                "A": ctrl.A,
                "wear": list(wear.wear),
                "multipliers": wear.multipliers(),
                "alive": True,
                "kind": "episodic_reset",
            }
        )

        def record(t: int, seg_local_t: int, boundary: float = 0.0) -> None:
            mass = self_mass(world.S)
            coh = coherence(world.S, S_ref, mass=mass)
            lr = local_resource_at_pattern_centroid(world.R, world.S)
            mults = wear.multipliers()
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
                        if (isinstance(coh, float) and math.isnan(coh))
                        else float(coh)
                    ),
                    "coherence_seg": (
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
                    "mean_mult": mean_mult,
                    "all_floored": 1.0 if wear.all_floored() else 0.0,
                    "distance": float(distance),
                }
            )

        record(global_t, 0, boundary=1.0)

        def maybe_calibrate(log) -> None:
            if not calibrate or model is None or policy != "selected":
                return
            if seg_idx < calibrate_min_segment:
                return
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
            sample["distance"] = float(distance)
            calibration_samples.append(sample)

        log = ctrl.maybe_decide(
            0, world.S, world.R, model, policy, device, rng, wear
        )
        # re-stamp t as global for logs
        if ctrl.logs:
            ctrl.logs[-1].t = global_t
        maybe_calibrate(log)
        if ctrl.decision_details:
            d = dict(ctrl.decision_details[-1])
            d["segment"] = seg_idx
            d["global_t"] = global_t
            d["t"] = global_t
            decision_details_all.append(d)

        seg_start_t = global_t
        seg_start_wear = list(wear.wear)
        seg_start_mult = wear.multipliers()
        n_boosts_start = ctrl.total_boosts
        died_this = False
        death_local_t: Optional[int] = None

        for step_i in range(1, duration + 1):
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

            mass = self_mass(world.S)
            if mass <= DEATH_MASS_EPS:
                died_this = True
                death_local_t = step_i
                any_death = True
                if first_death_t is None:
                    first_death_t = global_t
                    first_death_segment = seg_idx
                record(global_t, step_i, boundary=0.0)
                break

            # decision on local step matching K; use local t for interval
            if step_i % attn_cfg.K == 0:
                log = ctrl.maybe_decide(
                    step_i, world.S, world.R, model, policy, device, rng, wear
                )
                if ctrl.logs:
                    ctrl.logs[-1].t = global_t
                maybe_calibrate(log)
                if ctrl.decision_details:
                    d = dict(ctrl.decision_details[-1])
                    d["segment"] = seg_idx
                    d["global_t"] = global_t
                    d["t"] = global_t
                    decision_details_all.append(d)

            record(global_t, step_i, boundary=0.0)

        seg_hist = [h for h in history if int(h["segment"]) == seg_idx]
        alive_coh = [
            h["coherence_seg"]
            for h in seg_hist
            if h["alive"] > 0.5 and not math.isnan(h["coherence_seg"])
        ]
        final = seg_hist[-1] if seg_hist else {}
        segment_summaries.append(
            {
                "segment": seg_idx,
                "distance": distance,
                "duration": duration,
                "global_t_start": seg_start_t,
                "global_t_end": global_t,
                "steps_run": global_t - seg_start_t,
                "survived_segment": not died_this,
                "completed_full_duration": (not died_this)
                and (global_t - seg_start_t) >= duration,
                "died_mid_segment": died_this,
                "time_of_death_local": death_local_t,
                "mean_coherence_seg": (
                    float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
                ),
                "final_coherence_seg": final.get("coherence_seg", float("nan")),
                "final_coherence_chain": final.get("coherence", float("nan")),
                "final_self_mass": final.get("self_mass", 0.0),
                "start_wear": seg_start_wear,
                "end_wear": list(wear.wear),
                "start_multipliers": seg_start_mult,
                "end_multipliers": wear.multipliers(),
                "start_mean_mult": float(sum(seg_start_mult) / len(seg_start_mult)),
                "end_mean_mult": float(
                    sum(wear.multipliers()) / len(wear.multipliers())
                ),
                "n_boosts_in_segment": ctrl.total_boosts - n_boosts_start,
                "alive_at_start": True,
                "alive_at_end": not died_this,
            }
        )
        total_boosts += ctrl.total_boosts
        total_decisions += ctrl.total_decisions
        total_vol_skip += ctrl.total_skipped_voluntary
        total_budget_skip += ctrl.total_skipped_budget
        all_logs.extend(ctrl.logs)
        all_geom.extend(ctrl.geometry_logs)

    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    final = history[-1] if history else {}

    return {
        "protocol": "episodic",
        "policy": policy,
        "seed": seed,
        "segments": [(int(d), int(dur)) for d, dur in segments],
        "history": history,
        "segment_boundaries": segment_boundaries,
        "segment_summaries": segment_summaries,
        "decision_details": decision_details_all,
        "selection_logs": all_logs,
        "geometry_logs": all_geom,
        "calibration_samples": calibration_samples,
        "time_of_death": first_death_t,
        "death_segment": first_death_segment,
        "survived_full_chain": not any_death,
        "n_segments_reached": len(segment_summaries),
        "n_boosts": total_boosts,
        "n_decisions": total_decisions,
        "n_skipped_voluntary": total_vol_skip,
        "n_skipped_budget": total_budget_skip,
        "final_wear": segment_summaries[-1]["end_wear"] if segment_summaries else [0, 0],
        "final_multipliers": (
            segment_summaries[-1]["end_multipliers"] if segment_summaries else [1, 1]
        ),
        "final_A": None,
        "mean_coherence_chain": (
            float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan")
        ),
        "final_coherence_chain": final.get("coherence", float("nan")),
        "final_self_mass": final.get("self_mass", 0.0),
        "total_steps": global_t,
    }
