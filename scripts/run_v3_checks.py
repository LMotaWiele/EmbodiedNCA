#!/usr/bin/env python3
"""v3 acceptance: global wear + direct boost cost, frozen v2 self-model. Writes results/v3/."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from embodiednca.paths import results_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from embodiednca.v2.action_selection import (
    AttentionConfig,
    AttentionController,
    SelectionLog,
    generate_candidates,
    geometry_of_choice,
    mask_candidates,
    score_candidates,
    select_candidate,
)
from embodiednca.v3.boost_cost import BoostCostConfig, apply_direct_boost_cost, read_R_at
from embodiednca.v2.boost_rollout_dataset import make_boost_map
from embodiednca.v0.grid_world import (
    DEFAULT_GEN_RATE,
    ResourceGrid,
    WorldConfig,
    place_pattern,
)
from embodiednca.v0.metrics import (
    DEATH_MASS_EPS,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from embodiednca.v1.rollout_dataset import DEFAULT_CKPT, load_v0_world
from embodiednca.v2.self_model_v2 import load_model
from embodiednca.v3.wear_dynamics import (
    WearConfig,
    WearState,
    closed_form_ambient_trajectory,
    effective_gen_rate_formula,
    step_with_wear,
)

RESULTS = results_dir("v3")

# Best fixed sector per distance from v2 6-way sweep (structural best).
# Used for the fixed-best-sector control in Check 2.
V2_BEST_FIXED_SECTOR: Dict[int, int] = {2: 4, 4: 2, 8: 0}

# Minimum |Δ mean_coherence| to call selected "underperforms" a control.
# Sign-only gaps smaller than this are noise / micro-effects, not evidence.
MEANINGFUL_COH_DELTA = 0.01

POLICIES = (
    "selected",
    "random",
    "round_robin",
    "fixed",
    "wellward_heuristic",
    "never_boost",
)

# Colours shared by Check 2 plots
POLICY_COLORS = {
    "selected": "C0",
    "random": "C1",
    "round_robin": "C2",
    "fixed": "C3",
    "wellward_heuristic": "C4",
    "never_boost": "C5",
}


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _finite_mean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite_std(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return float(math.sqrt(var))


# ---------------------------------------------------------------------------
# Wear-aware attention controller (extends v2 with never_boost + cost flags)
# ---------------------------------------------------------------------------
class WearAttentionController(AttentionController):
    """
    Same as v2 AttentionController, plus:
      - policy 'never_boost': decide (for logging) but never spend A / arm boost
      - tracks last_applied_boost_location for direct cost charging
      - total_direct_charges for Check 3
    """

    def __init__(
        self,
        cfg: Optional[AttentionConfig] = None,
        fixed_sector_preset: Optional[int] = None,
    ):
        super().__init__(cfg, fixed_sector_preset=fixed_sector_preset)
        self.last_applied_boost_location: Optional[Tuple[int, int]] = None
        self.pending_direct_charge: bool = False  # True for one step after arming
        self.total_direct_charges: int = 0
        self.total_boost_steps: int = 0  # steps where boost_map was applied

    def maybe_decide(
        self,
        t: int,
        S: torch.Tensor,
        R: torch.Tensor,
        model,
        policy: str,
        device: str,
        rng: torch.Generator,
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

        if not surviving:
            log.A_after = self.A
            self.logs.append(log)
            return log

        qmag = (
            cfg.query_magnitude
            if cfg.query_magnitude is not None
            else cfg.boost_magnitude
        )
        if policy == "selected":
            if model is None:
                raise ValueError("policy='selected' requires a self-model")
            scores = score_candidates(
                model,
                S,
                R,
                surviving,
                magnitude=qmag,
                device=device,
                sigma=cfg.boost_sigma,
            )
        else:
            scores = [0.0] * len(surviving)
        log.scores = scores

        # never_boost still picks a would-have candidate for logging, but never arms
        select_policy = "fixed" if policy == "never_boost" else policy
        if policy == "never_boost":
            # log-only selection: lowest sector (deterministic, no model)
            chosen, self.rr_index, self.fixed_sector = select_candidate(
                surviving,
                scores,
                policy="fixed",
                rng=rng,
                rr_index=self.rr_index,
                fixed_sector=self.fixed_sector,
                S=S,
                wells=cfg.wells,
                mass_threshold=cfg.mass_threshold,
            )
        else:
            chosen, self.rr_index, self.fixed_sector = select_candidate(
                surviving,
                scores,
                policy=select_policy,
                rng=rng,
                rr_index=self.rr_index,
                fixed_sector=self.fixed_sector,
                S=S,
                wells=cfg.wells,
                mass_threshold=cfg.mass_threshold,
            )

        if chosen is not None and chosen.masked:
            self.total_masked_selected += 1
            log.selected_was_masked = True

        if chosen is not None:
            log.selected = (chosen.y, chosen.x)
            log.selected_sector = int(chosen.sector)
            try:
                ci = next(
                    i
                    for i, c in enumerate(surviving)
                    if c.y == chosen.y
                    and c.x == chosen.x
                    and c.sector == chosen.sector
                )
                log.score_of_selected = float(scores[ci])
            except StopIteration:
                log.score_of_selected = None
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

        # Arm boost only if policy allows and budget allows
        can_boost = (
            policy != "never_boost"
            and chosen is not None
            and self.A >= cfg.boost_cost
        )
        if can_boost:
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
            self.pending_direct_charge = True  # charge on next step that applies it
        else:
            self.total_skipped_budget += 1

        log.A_after = self.A
        self.logs.append(log)
        return log


# ---------------------------------------------------------------------------
# Wear-aware episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_wear_episode(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model,
    policy: str,
    distance: int,
    horizon: int = 5000,
    well: Tuple[int, int] = (8, 8),
    attn_cfg: Optional[AttentionConfig] = None,
    wear_cfg: Optional[WearConfig] = None,
    cost_cfg: Optional[BoostCostConfig] = None,
    seed: int = 0,
    device: str = "cpu",
    base_center: Tuple[int, int] = (16, 16),
    fixed_sector_preset: Optional[int] = None,
    disable_boosting: bool = False,
) -> Dict:
    """
    Full attention loop under wear + direct boost cost.

    If disable_boosting=True (Check 1), force never_boost behaviour so only
    ambient wear accrues.
    """
    attn_cfg = attn_cfg or AttentionConfig()
    wear_cfg = wear_cfg or WearConfig()
    cost_cfg = cost_cfg or BoostCostConfig()

    if disable_boosting:
        policy = "never_boost"

    center = (well[0], well[1] + distance)
    S0 = place_pattern(base_pattern, center, base_center=base_center)
    # Settle R at full gen_rate_base (pre-wear world); wear then degrades over time.
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed
    )
    # Ensure world.config.gen_rate matches base (for any code that reads it);
    # runtime gen always comes from WearState.effective_gen_rate.
    world.config.gen_rate = wear_cfg.gen_rate_base

    ctrl = WearAttentionController(attn_cfg, fixed_sector_preset=fixed_sector_preset)
    wear = WearState(wear_cfg)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1009)

    S_ref = world.S.detach().clone()
    history: List[Dict[str, float]] = []
    egr_log: List[float] = []
    wear_log: List[float] = []
    direct_charge_events: List[Dict] = []

    def record(t: int, egr: float, mag_mean: float = 0.0) -> None:
        mass = self_mass(world.S)
        coh = coherence(world.S, S_ref, mass=mass)
        lr = local_resource_at_pattern_centroid(world.R, world.S)
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
                "wear": wear.wear,
                "effective_gen_rate": egr,
                "gen_rate_base": wear_cfg.gen_rate_base,
            }
        )
        egr_log.append(egr)
        wear_log.append(wear.wear)

    # t=0 state: no wear accrued yet
    record(0, egr=wear.effective_gen_rate())
    ctrl.maybe_decide(0, world.S, world.R, model, policy, device, rng)

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
            if loc is not None:
                r_before = read_R_at(world.R, loc[0], loc[1])
                direct_charge_events.append(
                    {
                        "t": t,
                        "y": loc[0],
                        "x": loc[1],
                        "cost": cost_cfg.direct_boost_cost,
                        "R_before": r_before,
                    }
                )

        _, mag, egr = step_with_wear(
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
            ctrl.maybe_decide(t, world.S, world.R, model, policy, device, rng)

        record(t, egr=egr, mag_mean=float(mag.mean().item()))

    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    alive_mass = [h["self_mass"] for h in history if h["alive"] > 0.5]
    final = history[-1]
    tod = next((int(h["t"]) for h in history if h["alive"] < 0.5), None)

    return {
        "history": history,
        "policy": policy if not disable_boosting else "never_boost",
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
        "n_candidates_dropped_mask": ctrl.total_candidates_dropped,
        "n_masked_selected": ctrl.total_masked_selected,
        "n_direct_charges": ctrl.total_direct_charges,
        "direct_charge_events": direct_charge_events,
        "final_wear": wear.wear,
        "final_effective_gen_rate": wear.effective_gen_rate(),
        "mean_effective_gen_rate": _finite_mean(egr_log),
        "min_effective_gen_rate": min(egr_log) if egr_log else float("nan"),
        "floor_gen_rate": wear.floor_gen_rate(),
        "selection_logs": ctrl.logs,
        "geometry_logs": ctrl.geometry_logs,
        "fixed_sector_locked": ctrl.fixed_sector,
        "wear_cfg": {
            "wear_ambient_rate": wear_cfg.wear_ambient_rate,
            "wear_boost_rate": wear_cfg.wear_boost_rate,
            "wear_decay_coeff": wear_cfg.wear_decay_coeff,
            "gen_floor_frac": wear_cfg.gen_floor_frac,
            "gen_rate_base": wear_cfg.gen_rate_base,
        },
        "cost_cfg": {
            "direct_boost_cost": cost_cfg.direct_boost_cost,
            "charge_on_apply_only": cost_cfg.charge_on_apply_only,
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


# ---------------------------------------------------------------------------
# Check 1 — mechanic sanity + parameter sweep
# ---------------------------------------------------------------------------
def sweep_wear_params(
    horizon: int = 5000,
) -> Tuple[List[Dict], WearConfig]:
    """
    Sweep ambient / boost / decay / floor. Pick an operating point where:
      - ambient-only reaches floor in ~mid horizon (not instantly, not never)
      - boost rate is meaningfully larger than ambient (so choice can matter)
      - floor is in [0.3, 0.5]
    """
    print("\n=== Wear parameter sweep (closed-form ambient) ===")
    rows: List[Dict] = []
    for ambient in (0.0005, 0.001, 0.002, 0.003, 0.005):
        for boost in (0.002, 0.004, 0.008, 0.012, 0.02):
            for coeff in (0.05, 0.10, 0.15, 0.20):
                for floor in (0.30, 0.40, 0.50):
                    cfg = WearConfig(
                        wear_ambient_rate=ambient,
                        wear_boost_rate=boost,
                        wear_decay_coeff=coeff,
                        gen_floor_frac=floor,
                        gen_rate_base=DEFAULT_GEN_RATE,
                    )
                    st = WearState(cfg)
                    t_floor = st.ambient_steps_to_floor()
                    # mid-horizon egr fraction of base
                    wear_mid = ambient * (horizon // 2)
                    egr_mid = effective_gen_rate_formula(
                        wear_mid, DEFAULT_GEN_RATE, coeff, floor
                    )
                    rows.append(
                        {
                            "wear_ambient_rate": ambient,
                            "wear_boost_rate": boost,
                            "wear_decay_coeff": coeff,
                            "gen_floor_frac": floor,
                            "t_floor_ambient": t_floor,
                            "egr_mid_frac": egr_mid / DEFAULT_GEN_RATE,
                            "boost_over_ambient": boost / max(ambient, 1e-12),
                            "wear_at_floor": st.wear_at_floor(),
                        }
                    )

    # Prefer: t_floor in [0.4*H, 0.9*H], boost/ambient in [3, 8], floor=0.4
    def score(r: Dict) -> float:
        tf = r["t_floor_ambient"]
        if math.isinf(tf):
            return 1e18
        # want ~0.6 * horizon
        target_t = 0.6 * horizon
        t_pen = abs(tf - target_t) / horizon
        ba = r["boost_over_ambient"]
        ba_pen = abs(ba - 4.0) / 4.0
        floor_pen = abs(r["gen_floor_frac"] - 0.40) * 2.0
        return t_pen + 0.3 * ba_pen + floor_pen

    best = min(rows, key=score)
    chosen = WearConfig(
        wear_ambient_rate=best["wear_ambient_rate"],
        wear_boost_rate=best["wear_boost_rate"],
        wear_decay_coeff=best["wear_decay_coeff"],
        gen_floor_frac=best["gen_floor_frac"],
        gen_rate_base=DEFAULT_GEN_RATE,
    )
    print(
        f"  chosen: ambient={chosen.wear_ambient_rate}  "
        f"boost={chosen.wear_boost_rate}  "
        f"coeff={chosen.wear_decay_coeff}  "
        f"floor={chosen.gen_floor_frac}"
    )
    print(
        f"  t_floor_ambient≈{best['t_floor_ambient']:.0f}  "
        f"boost/ambient={best['boost_over_ambient']:.1f}  "
        f"egr_mid_frac={best['egr_mid_frac']:.3f}"
    )
    # Neighbors
    ranked = sorted(rows, key=score)[:8]
    print("  nearby candidates:")
    for r in ranked:
        print(
            f"    amb={r['wear_ambient_rate']:.4f} bst={r['wear_boost_rate']:.4f} "
            f"c={r['wear_decay_coeff']:.2f} fl={r['gen_floor_frac']:.2f}  "
            f"t_floor={r['t_floor_ambient']:.0f}  "
            f"bst/amb={r['boost_over_ambient']:.1f}"
        )
    return rows, chosen


def check1_mechanic_sanity(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    wear_cfg: WearConfig,
    device: str,
    horizon: int = 5000,
    distance: int = 4,
    seed: int = 0,
) -> Dict:
    """
    Boosting disabled. Confirm:
      - wear increases monotonically from ambient alone
      - effective_gen_rate decreases and asymptotes ≥ floor * base
      - report timescale to floor
    """
    print("\n=== Check 1: mechanic sanity (boosting disabled) ===")
    attn_cfg = AttentionConfig()  # unused for boost; never_boost forced
    cost_cfg = BoostCostConfig(direct_boost_cost=0.15)  # should never fire

    ep = run_wear_episode(
        nca=nca,
        base_pattern=base_pattern,
        world_cfg=world_cfg,
        model=None,
        policy="never_boost",
        distance=distance,
        horizon=horizon,
        attn_cfg=attn_cfg,
        wear_cfg=wear_cfg,
        cost_cfg=cost_cfg,
        seed=seed,
        device=device,
        disable_boosting=True,
    )

    wears = [h["wear"] for h in ep["history"]]
    egrs = [h["effective_gen_rate"] for h in ep["history"]]
    ts = [h["t"] for h in ep["history"]]

    # Monotonic wear
    deltas = [wears[i + 1] - wears[i] for i in range(len(wears) - 1)]
    mono = all(d >= -1e-12 for d in deltas)
    # Ambient-only: every step should increase by exactly ambient_rate
    ambient = wear_cfg.wear_ambient_rate
    ambient_ok = all(abs(d - ambient) < 1e-9 for d in deltas)

    floor = wear_cfg.gen_rate_base * wear_cfg.gen_floor_frac
    min_egr = min(egrs)
    never_below_floor = min_egr >= floor - 1e-9
    # Decreasing until floor
    egr_nonincreasing = all(
        egrs[i + 1] <= egrs[i] + 1e-12 for i in range(len(egrs) - 1)
    )

    # Closed-form comparison
    cf_wear, cf_egr = closed_form_ambient_trajectory(wear_cfg, horizon)
    # After step t, wear should be wear0 + t * ambient (our record at t includes
    # post-step wear for t>=1; at t=0 wear=wear0)
    max_wear_err = max(abs(wears[i] - cf_wear[i]) for i in range(len(wears)))
    max_egr_err = max(abs(egrs[i] - cf_egr[i]) for i in range(len(egrs)))

    st = WearState(wear_cfg)
    t_floor = st.ambient_steps_to_floor()
    # Empirical first t where egr is within 1e-6 of floor
    t_floor_emp = next(
        (int(ts[i]) for i in range(len(egrs)) if egrs[i] <= floor + 1e-6),
        None,
    )

    # No boosts / no direct charges under disable_boosting
    no_boosts = ep["n_boosts"] == 0 and ep["n_boost_steps"] == 0
    no_charges = ep["n_direct_charges"] == 0

    print(f"  horizon={horizon}  dist={distance}")
    print(
        f"  wear mono↑={mono}  ambient-exact={ambient_ok}  "
        f"final_wear={wears[-1]:.6f}"
    )
    print(
        f"  egr nonincreasing={egr_nonincreasing}  "
        f"min_egr={min_egr:.6f}  floor={floor:.6f}  "
        f"never_below_floor={never_below_floor}"
    )
    print(
        f"  t_floor (closed-form ambient)={t_floor:.1f}  "
        f"t_floor (empirical)={t_floor_emp}  "
        f"final_egr={egrs[-1]:.6f}  "
        f"final_egr/base={egrs[-1] / wear_cfg.gen_rate_base:.3f}"
    )
    print(
        f"  closed-form match: max|wear err|={max_wear_err:.2e}  "
        f"max|egr err|={max_egr_err:.2e}"
    )
    print(
        f"  n_boosts={ep['n_boosts']}  n_boost_steps={ep['n_boost_steps']}  "
        f"n_direct_charges={ep['n_direct_charges']}  "
        f"(expect 0/0/0)"
    )
    print(
        f"  mean_coh={ep['mean_coherence']:.4f}  "
        f"final_mass={ep['final_self_mass']:.3f}  "
        f"tod={ep['time_of_death']}"
    )

    # Plot wear + egr over time
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(ts, wears, color="C3", lw=1.5, label="wear (sim)")
    axes[0].plot(ts, cf_wear[: len(ts)], "k--", lw=1, alpha=0.6, label="wear (closed-form)")
    axes[0].axhline(
        st.wear_at_floor(), color="0.5", ls=":", label=f"wear@floor={st.wear_at_floor():.3f}"
    )
    axes[0].set_ylabel("wear")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Check 1: ambient wear only  "
        f"amb={wear_cfg.wear_ambient_rate} coeff={wear_cfg.wear_decay_coeff} "
        f"floor={wear_cfg.gen_floor_frac}"
    )

    axes[1].plot(ts, egrs, color="C0", lw=1.5, label="effective_gen_rate (sim)")
    axes[1].plot(
        ts, cf_egr[: len(ts)], "k--", lw=1, alpha=0.6, label="egr (closed-form)"
    )
    axes[1].axhline(floor, color="C1", ls=":", label=f"floor={floor:.4f}")
    axes[1].axhline(
        wear_cfg.gen_rate_base, color="0.5", ls="--", alpha=0.5, label="gen_rate_base"
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("effective_gen_rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_plot = RESULTS / "check1_wear_egr.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # CSV
    csv_path = RESULTS / "check1_wear_egr.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "wear",
                "effective_gen_rate",
                "self_mass",
                "coherence",
                "local_R",
                "alive",
            ]
        )
        for h in ep["history"]:
            w.writerow(
                [
                    int(h["t"]),
                    h["wear"],
                    h["effective_gen_rate"],
                    h["self_mass"],
                    h["coherence"],
                    h["local_R"],
                    h["alive"],
                ]
            )

    passed = (
        mono
        and ambient_ok
        and never_below_floor
        and egr_nonincreasing
        and no_boosts
        and no_charges
        and max_wear_err < 1e-6
    )
    print(f"  Check 1 PASS={passed}")

    return {
        "pass": passed,
        "monotonic_wear": mono,
        "ambient_exact": ambient_ok,
        "never_below_floor": never_below_floor,
        "egr_nonincreasing": egr_nonincreasing,
        "no_boosts": no_boosts,
        "no_direct_charges": no_charges,
        "t_floor_closed_form": t_floor,
        "t_floor_empirical": t_floor_emp,
        "final_wear": wears[-1],
        "final_effective_gen_rate": egrs[-1],
        "min_effective_gen_rate": min_egr,
        "floor_gen_rate": floor,
        "gen_rate_base": wear_cfg.gen_rate_base,
        "max_wear_err_vs_closed_form": max_wear_err,
        "max_egr_err_vs_closed_form": max_egr_err,
        "mean_coherence": ep["mean_coherence"],
        "final_self_mass": ep["final_self_mass"],
        "time_of_death": ep["time_of_death"],
        "wear_cfg": ep["wear_cfg"],
        "plot": str(out_plot),
        "csv": str(csv_path),
        "horizon": horizon,
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# Check 2 — policy comparison under wear
# ---------------------------------------------------------------------------
def _window_mean_coherence(
    history: List[Dict[str, float]],
    t_lo: float,
    t_hi: float,
) -> float:
    """Mean coherence over alive steps with t in [t_lo, t_hi)."""
    vals = [
        h["coherence"]
        for h in history
        if t_lo <= h["t"] < t_hi
        and h.get("alive", 1.0) > 0.5
        and not math.isnan(h["coherence"])
    ]
    return _finite_mean(vals)


def check2_policy_comparison(
    model,
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    device: str,
    attn_cfg: AttentionConfig,
    wear_cfg: WearConfig,
    cost_cfg: BoostCostConfig,
    distances: Tuple[int, ...] = (2, 4, 8),
    horizon: int = 5000,
    seed: int = 0,
    condition_name: str = "combined",
    file_tag: str = "",
    save_csvs: bool = True,
) -> Dict:
    """
    Policy comparison under a given (wear_cfg, cost_cfg) pair.

    condition_name / file_tag distinguish the main combined run from ablations:
      combined     — both mechanics on (file_tag="")
      wear_only    — direct_boost_cost=0
      direct_only  — wear_boost_rate=0 (ambient wear still shared by all)
    """
    tag_label = condition_name
    print(
        f"\n=== Check 2 [{tag_label}]: policy comparison "
        "(existing v2 self-model, no retrain) ==="
    )
    print(f"  policies={POLICIES}")
    print(
        f"  wear: amb={wear_cfg.wear_ambient_rate}  "
        f"boost_rate={wear_cfg.wear_boost_rate}  "
        f"coeff={wear_cfg.wear_decay_coeff}  "
        f"floor={wear_cfg.gen_floor_frac}"
    )
    print(f"  direct_boost_cost={cost_cfg.direct_boost_cost}")
    print(f"  horizon={horizon}  distances={distances}")

    # Ambient-only floor time: after this, gen_rate is at floor for never_boost;
    # greedy policies hit the floor earlier, but *after* this t all policies
    # share the same effective_gen_rate (clamped). Useful for attribution.
    t_floor_ambient = WearState(wear_cfg).ambient_steps_to_floor()
    if math.isinf(t_floor_ambient):
        t_floor_ambient = float(horizon + 1)
    print(
        f"  ambient t_floor≈{t_floor_ambient:.0f}  "
        f"(after this, egr is floor-clamped for all policies that reached it)"
    )

    results: Dict = {}
    total_masked = 0
    total_dropped = 0

    for d in distances:
        results[d] = {}
        for pol in POLICIES:
            fixed_preset = None
            if pol == "fixed":
                fixed_preset = V2_BEST_FIXED_SECTOR.get(d, 0)
            print(
                f"  dist={d} policy={pol}"
                + (f" (fixed_sector={fixed_preset})" if fixed_preset is not None else "")
                + "…"
            )
            ep = run_wear_episode(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                model=model if pol == "selected" else None,
                policy=pol,
                distance=d,
                horizon=horizon,
                attn_cfg=attn_cfg,
                wear_cfg=wear_cfg,
                cost_cfg=cost_cfg,
                seed=seed,
                device=device,
                fixed_sector_preset=fixed_preset,
            )
            results[d][pol] = ep
            total_masked += ep["n_masked_selected"]
            total_dropped += ep["n_candidates_dropped_mask"]
            print(
                f"    mean_coh={ep['mean_coherence']:.4f}  "
                f"final_coh={ep['final_coherence'] if not math.isnan(ep['final_coherence']) else float('nan'):.4f}  "
                f"mean_mass={ep['mean_self_mass']:.3f}  "
                f"final_mass={ep['final_self_mass']:.3f}  "
                f"tod={ep['time_of_death']}  "
                f"boosts={ep['n_boosts']}/{ep['n_decisions']}  "
                f"boost_steps={ep['n_boost_steps']}  "
                f"final_wear={ep['final_wear']:.4f}  "
                f"final_egr={ep['final_effective_gen_rate']:.5f}  "
                f"charges={ep['n_direct_charges']}"
            )

    # Plots: coherence | self_mass | wear | egr
    n_d = len(distances)
    fig, axes = plt.subplots(n_d, 4, figsize=(16, 3.2 * n_d), sharex=False)
    if n_d == 1:
        axes = axes.reshape(1, 4)
    colors = POLICY_COLORS
    for i, d in enumerate(distances):
        for pol in POLICIES:
            hist = results[d][pol]["history"]
            ts = [h["t"] for h in hist]
            axes[i, 0].plot(
                ts,
                [h["coherence"] for h in hist],
                color=colors[pol],
                label=pol,
                alpha=0.85,
                lw=1.1,
            )
            axes[i, 1].plot(
                ts,
                [h["self_mass"] for h in hist],
                color=colors[pol],
                label=pol,
                alpha=0.85,
                lw=1.1,
            )
            axes[i, 2].plot(
                ts,
                [h["wear"] for h in hist],
                color=colors[pol],
                label=pol,
                alpha=0.85,
                lw=1.1,
            )
            axes[i, 3].plot(
                ts,
                [h["effective_gen_rate"] for h in hist],
                color=colors[pol],
                label=pol,
                alpha=0.85,
                lw=1.1,
            )
        # Mark ambient floor time on coh / egr (attribution guide)
        if t_floor_ambient <= horizon:
            for j in (0, 3):
                axes[i, j].axvline(
                    t_floor_ambient,
                    color="k",
                    ls="--",
                    alpha=0.45,
                    lw=0.9,
                    label="ambient t_floor" if j == 0 else None,
                )
        axes[i, 0].set_ylabel("coherence")
        axes[i, 1].set_ylabel("self_mass")
        axes[i, 2].set_ylabel("wear")
        axes[i, 3].set_ylabel("eff. gen_rate")
        axes[i, 0].set_title(f"dist={d} coherence")
        axes[i, 1].set_title(f"dist={d} self_mass")
        axes[i, 2].set_title(f"dist={d} wear")
        axes[i, 3].set_title(f"dist={d} effective_gen_rate")
        for j in range(4):
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend(fontsize=6)
            axes[i, j].set_xlabel("t")
        floor = wear_cfg.gen_rate_base * wear_cfg.gen_floor_frac
        axes[i, 3].axhline(floor, color="k", ls=":", alpha=0.5, lw=0.8)
    fig.suptitle(
        f"Check 2 [{tag_label}]  H={horizon}  "
        f"amb={wear_cfg.wear_ambient_rate} bst_w={wear_cfg.wear_boost_rate} "
        f"coeff={wear_cfg.wear_decay_coeff} floor={wear_cfg.gen_floor_frac}  "
        f"direct_cost={cost_cfg.direct_boost_cost}"
    )
    fig.tight_layout()
    if file_tag:
        out_plot = RESULTS / f"check2_policy_comparison_{file_tag}.png"
    else:
        out_plot = RESULTS / "check2_policy_comparison.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Summary + central predictions
    summary = {}
    comparisons = {}
    for d in distances:
        summary[d] = {}
        for pol in POLICIES:
            ep = results[d][pol]
            summary[d][pol] = {
                "mean_coherence": ep["mean_coherence"],
                "final_coherence": ep["final_coherence"],
                "mean_self_mass": ep["mean_self_mass"],
                "final_self_mass": ep["final_self_mass"],
                "time_of_death": ep["time_of_death"],
                "n_boosts": ep["n_boosts"],
                "n_boost_steps": ep["n_boost_steps"],
                "n_decisions": ep["n_decisions"],
                "n_direct_charges": ep["n_direct_charges"],
                "final_wear": ep["final_wear"],
                "final_effective_gen_rate": ep["final_effective_gen_rate"],
                "mean_effective_gen_rate": ep["mean_effective_gen_rate"],
                "min_effective_gen_rate": ep["min_effective_gen_rate"],
                "n_masked_selected": ep["n_masked_selected"],
                "fixed_sector_locked": ep.get("fixed_sector_locked"),
            }

        # Boost frequency vs outcomes
        boost_rows = [
            (
                pol,
                summary[d][pol]["n_boosts"],
                summary[d][pol]["n_boost_steps"],
                summary[d][pol]["mean_coherence"],
                summary[d][pol]["final_self_mass"],
                summary[d][pol]["time_of_death"],
                summary[d][pol]["final_effective_gen_rate"],
                summary[d][pol]["final_wear"],
            )
            for pol in POLICIES
        ]
        # Spearman-ish: does more boosting → worse final egr / mass / earlier death?
        # Simple pairwise: rank by n_boost_steps vs rank by mean_coh (negative corr?)
        by_boost = sorted(boost_rows, key=lambda r: r[2])  # boost_steps asc
        by_coh = sorted(
            boost_rows,
            key=lambda r: (
                r[3] if not math.isnan(r[3]) else -1e9
            ),
            reverse=True,
        )

        nb = summary[d]["never_boost"]
        sel = summary[d]["selected"]
        wh = summary[d]["wellward_heuristic"]
        fx = summary[d]["fixed"]
        rnd = summary[d]["random"]

        def beats(a, b, key="mean_coherence"):
            va, vb = a[key], b[key]
            if math.isnan(va) or math.isnan(vb):
                return False
            return va > vb

        def tod_later(a, b):
            """True if a dies later (or never) than b."""
            ta, tb = a["time_of_death"], b["time_of_death"]
            if ta is None and tb is None:
                return False  # tie
            if ta is None:
                return True
            if tb is None:
                return False
            return ta > tb

        never_boost_outlives = {
            pol: tod_later(nb, summary[d][pol])
            for pol in POLICIES
            if pol != "never_boost"
        }
        never_boost_higher_coh = {
            pol: beats(nb, summary[d][pol], "mean_coherence")
            for pol in POLICIES
            if pol != "never_boost"
        }
        # selected (wear-blind) vs wellward / fixed
        sel_vs_wellward_coh = (
            sel["mean_coherence"] - wh["mean_coherence"]
            if (
                not math.isnan(sel["mean_coherence"])
                and not math.isnan(wh["mean_coherence"])
            )
            else float("nan")
        )
        sel_vs_fixed_coh = (
            sel["mean_coherence"] - fx["mean_coherence"]
            if (
                not math.isnan(sel["mean_coherence"])
                and not math.isnan(fx["mean_coherence"])
            )
            else float("nan")
        )
        # Thresholded flags (authoritative for downstream use).
        # Raw sign-only flags kept for diagnostics; do not treat raw as evidence.
        sel_underperforms_wellward = (
            (not math.isnan(sel_vs_wellward_coh))
            and sel_vs_wellward_coh <= -MEANINGFUL_COH_DELTA
        )
        sel_underperforms_fixed = (
            (not math.isnan(sel_vs_fixed_coh))
            and sel_vs_fixed_coh <= -MEANINGFUL_COH_DELTA
        )
        sel_underperforms_wellward_raw = (
            (not math.isnan(sel_vs_wellward_coh)) and sel_vs_wellward_coh < 0.0
        )
        sel_underperforms_fixed_raw = (
            (not math.isnan(sel_vs_fixed_coh)) and sel_vs_fixed_coh < 0.0
        )

        # Pre/post ambient-floor coherence (attribution of late never_boost lead)
        pre_floor = {
            pol: _window_mean_coherence(
                results[d][pol]["history"], 0.0, t_floor_ambient
            )
            for pol in POLICIES
        }
        post_floor = {
            pol: _window_mean_coherence(
                results[d][pol]["history"], t_floor_ambient, float(horizon + 1)
            )
            for pol in POLICIES
        }
        # Empirical t when this policy's egr first hits floor
        t_floor_emp = {}
        floor_egr = wear_cfg.gen_rate_base * wear_cfg.gen_floor_frac
        for pol in POLICIES:
            hist = results[d][pol]["history"]
            t_hit = next(
                (
                    int(h["t"])
                    for h in hist
                    if h["effective_gen_rate"] <= floor_egr + 1e-6
                ),
                None,
            )
            t_floor_emp[pol] = t_hit

        # Correlation: boost_steps vs mean_coh (Pearson over policies)
        xs = [r[2] for r in boost_rows]
        ys = [r[3] if not math.isnan(r[3]) else 0.0 for r in boost_rows]
        if len(xs) >= 3 and _finite_std(xs) > 1e-12 and _finite_std(ys) > 1e-12:
            mx, my = _finite_mean(xs), _finite_mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = math.sqrt(
                sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
            )
            boost_coh_corr = num / den if den > 0 else float("nan")
        else:
            boost_coh_corr = float("nan")

        nb_vs_sel_pre = (
            pre_floor["never_boost"] - pre_floor["selected"]
            if (
                not math.isnan(pre_floor["never_boost"])
                and not math.isnan(pre_floor["selected"])
            )
            else float("nan")
        )
        nb_vs_sel_post = (
            post_floor["never_boost"] - post_floor["selected"]
            if (
                not math.isnan(post_floor["never_boost"])
                and not math.isnan(post_floor["selected"])
            )
            else float("nan")
        )

        comparisons[d] = {
            "never_boost_outlives": never_boost_outlives,
            "never_boost_higher_coh": never_boost_higher_coh,
            "never_boost_outlives_any": any(never_boost_outlives.values()),
            "never_boost_higher_coh_any": any(never_boost_higher_coh.values()),
            "selected_vs_wellward_delta_coh": sel_vs_wellward_coh,
            "selected_vs_fixed_delta_coh": sel_vs_fixed_coh,
            "selected_underperforms_wellward": sel_underperforms_wellward,
            "selected_underperforms_fixed": sel_underperforms_fixed,
            "selected_underperforms_wellward_raw": sel_underperforms_wellward_raw,
            "selected_underperforms_fixed_raw": sel_underperforms_fixed_raw,
            "meaningful_coh_delta": MEANINGFUL_COH_DELTA,
            "boost_steps_vs_mean_coh_pearson": boost_coh_corr,
            "t_floor_ambient": t_floor_ambient,
            "t_floor_empirical_by_policy": t_floor_emp,
            "mean_coh_pre_ambient_floor": pre_floor,
            "mean_coh_post_ambient_floor": post_floor,
            "never_boost_minus_selected_pre_floor": nb_vs_sel_pre,
            "never_boost_minus_selected_post_floor": nb_vs_sel_post,
            "boost_ranking_asc": [
                {"policy": r[0], "n_boost_steps": r[2], "mean_coh": r[3]}
                for r in by_boost
            ],
            "coh_ranking_desc": [
                {"policy": r[0], "mean_coh": r[3], "n_boost_steps": r[2]}
                for r in by_coh
            ],
        }

        print(f"\n  --- dist={d} central predictions [{tag_label}] ---")
        print(
            f"    never_boost mean_coh={nb['mean_coherence']:.4f}  "
            f"tod={nb['time_of_death']}  final_egr={nb['final_effective_gen_rate']:.5f}  "
            f"boosts={nb['n_boosts']}"
        )
        print(
            f"    selected  mean_coh={sel['mean_coherence']:.4f}  "
            f"tod={sel['time_of_death']}  final_egr={sel['final_effective_gen_rate']:.5f}  "
            f"boosts={sel['n_boosts']}"
        )
        print(
            f"    wellward  mean_coh={wh['mean_coherence']:.4f}  "
            f"tod={wh['time_of_death']}  final_egr={wh['final_effective_gen_rate']:.5f}  "
            f"boosts={wh['n_boosts']}"
        )
        print(
            f"    fixed(s{fx.get('fixed_sector_locked')}) mean_coh={fx['mean_coherence']:.4f}  "
            f"tod={fx['time_of_death']}  final_egr={fx['final_effective_gen_rate']:.5f}"
        )
        print(
            f"    never_boost higher_coh than any greedy? "
            f"{comparisons[d]['never_boost_higher_coh_any']}  "
            f"{never_boost_higher_coh}"
        )
        print(
            f"    never_boost outlives any greedy? "
            f"{comparisons[d]['never_boost_outlives_any']}  "
            f"{never_boost_outlives}"
        )
        print(
            f"    selected underperforms wellward "
            f"(|Δ|≥{MEANINGFUL_COH_DELTA})? {sel_underperforms_wellward}  "
            f"Δcoh(sel−wh)={sel_vs_wellward_coh:.5f}  "
            f"(raw_sign={sel_underperforms_wellward_raw})"
        )
        print(
            f"    selected underperforms fixed "
            f"(|Δ|≥{MEANINGFUL_COH_DELTA})? {sel_underperforms_fixed}  "
            f"Δcoh(sel−fx)={sel_vs_fixed_coh:.5f}  "
            f"(raw_sign={sel_underperforms_fixed_raw})"
        )
        print(
            f"    Pearson(boost_steps, mean_coh)={boost_coh_corr:.4f}  "
            f"(negative ⇒ more boosting ↔ worse coherence)"
        )
        print(
            f"    pre-floor (t<{t_floor_ambient:.0f}) mean_coh: "
            f"nb={pre_floor['never_boost']:.4f}  sel={pre_floor['selected']:.4f}  "
            f"Δ(nb−sel)={nb_vs_sel_pre:.4f}"
        )
        print(
            f"    post-floor (t≥{t_floor_ambient:.0f}) mean_coh: "
            f"nb={post_floor['never_boost']:.4f}  sel={post_floor['selected']:.4f}  "
            f"Δ(nb−sel)={nb_vs_sel_post:.4f}"
        )
        print(
            f"    t_floor empirical: "
            + ", ".join(f"{p}={t_floor_emp[p]}" for p in ("never_boost", "selected", "wellward_heuristic"))
        )

    # Save CSVs
    csv_prefix = f"check2_{file_tag}_" if file_tag else "check2_"
    if save_csvs:
        for d in distances:
            for pol in POLICIES:
                path = RESULTS / f"{csv_prefix}d{d}_{pol}.csv"
                hist = results[d][pol]["history"]
                with path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "t",
                            "self_mass",
                            "coherence",
                            "local_R",
                            "A",
                            "boost_remaining",
                            "alive",
                            "wear",
                            "effective_gen_rate",
                        ]
                    )
                    for h in hist:
                        w.writerow(
                            [
                                int(h["t"]),
                                h["self_mass"],
                                h["coherence"],
                                h["local_R"],
                                h["A"],
                                h["boost_remaining"],
                                h["alive"],
                                h["wear"],
                                h["effective_gen_rate"],
                            ]
                        )
                spath = RESULTS / f"{csv_prefix}d{d}_{pol}_selections.csv"
                with spath.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "t",
                            "selected_y",
                            "selected_x",
                            "selected_sector",
                            "boosted",
                            "score_of_selected",
                            "n_raw",
                            "n_surviving",
                            "n_dropped_mask",
                            "A_before",
                            "A_after",
                        ]
                    )
                    for log in results[d][pol]["selection_logs"]:
                        sy = log.selected[0] if log.selected else ""
                        sx = log.selected[1] if log.selected else ""
                        sec = (
                            log.selected_sector
                            if log.selected_sector is not None
                            else ""
                        )
                        sc = (
                            f"{log.score_of_selected:.6f}"
                            if log.score_of_selected is not None
                            else ""
                        )
                        w.writerow(
                            [
                                log.t,
                                sy,
                                sx,
                                sec,
                                int(log.boosted),
                                sc,
                                log.n_raw,
                                log.n_surviving,
                                log.n_dropped_mask,
                                f"{log.A_before:.4f}",
                                f"{log.A_after:.4f}",
                            ]
                        )

    # Aggregate verdict (honest)
    any_never_matters = any(
        comparisons[d]["never_boost_higher_coh_any"]
        or comparisons[d]["never_boost_outlives_any"]
        for d in distances
    )
    # Only score selected-vs-control on clean distances (2, 4). Dist 8 is a
    # dying regime in v0–v2 and is not diagnostic for wear-blindness.
    # Per-distance underperforms_* flags already use MEANINGFUL_COH_DELTA.
    clean_ds = [d for d in (2, 4) if d in comparisons]
    any_sel_under = any(
        comparisons[d]["selected_underperforms_wellward"]
        or comparisons[d]["selected_underperforms_fixed"]
        for d in clean_ds
    )
    any_sel_under_raw = any(
        comparisons[d]["selected_underperforms_wellward_raw"]
        or comparisons[d]["selected_underperforms_fixed_raw"]
        for d in clean_ds
    )
    # Negative boost-coh correlation at clean distances
    neg_corr_clean = all(
        (not math.isnan(comparisons[d]["boost_steps_vs_mean_coh_pearson"]))
        and comparisons[d]["boost_steps_vs_mean_coh_pearson"] < -0.1
        for d in (2, 4)
        if d in comparisons
    )
    # Do greedy policies share boost counts with selected? (global wear → where≈irrelevant)
    sel_boost_tied = {}
    for d in distances:
        s = summary[d]
        sel_b = s["selected"]["n_boosts"]
        sel_boost_tied[d] = {
            "selected_n_boosts": sel_b,
            "wellward_n_boosts": s["wellward_heuristic"]["n_boosts"],
            "fixed_n_boosts": s["fixed"]["n_boosts"],
            "never_boost_n_boosts": s["never_boost"]["n_boosts"],
            "selected_matches_wellward_boost_count": (
                sel_b == s["wellward_heuristic"]["n_boosts"]
            ),
        }

    print(f"\n  === Check 2 aggregate [{tag_label}] ===")
    print(f"  never_boost wins on coh/tod at any distance? {any_never_matters}")
    print(
        f"  selected underperforms wellward/fixed "
        f"(raw sign Δ<0, clean dist)? {any_sel_under_raw}"
    )
    print(
        f"  selected underperforms wellward/fixed "
        f"(|Δcoh|≥{MEANINGFUL_COH_DELTA}, clean dist)? {any_sel_under}"
    )
    print(
        f"  boost_steps negatively correlated with mean_coh at dist 2/4? {neg_corr_clean}"
    )
    for d in distances:
        print(f"  boost counts dist={d}: {sel_boost_tied[d]}")

    # Compact never_boost advantage table (for ablation comparison)
    nb_advantage = {}
    for d in distances:
        s = summary[d]
        nb_advantage[d] = {
            "delta_mean_coh_vs_selected": (
                s["never_boost"]["mean_coherence"] - s["selected"]["mean_coherence"]
                if (
                    not math.isnan(s["never_boost"]["mean_coherence"])
                    and not math.isnan(s["selected"]["mean_coherence"])
                )
                else float("nan")
            ),
            "never_boost_mean_coh": s["never_boost"]["mean_coherence"],
            "selected_mean_coh": s["selected"]["mean_coherence"],
            "never_boost_tod": s["never_boost"]["time_of_death"],
            "selected_tod": s["selected"]["time_of_death"],
            "never_boost_n_boosts": s["never_boost"]["n_boosts"],
            "selected_n_boosts": s["selected"]["n_boosts"],
            "delta_pre_floor": comparisons[d]["never_boost_minus_selected_pre_floor"],
            "delta_post_floor": comparisons[d]["never_boost_minus_selected_post_floor"],
        }
        print(
            f"  nb advantage dist={d}: Δmean_coh(nb−sel)="
            f"{nb_advantage[d]['delta_mean_coh_vs_selected']:.4f}  "
            f"pre_floor Δ={nb_advantage[d]['delta_pre_floor']:.4f}  "
            f"post_floor Δ={nb_advantage[d]['delta_post_floor']:.4f}"
        )

    if condition_name == "combined":
        if not any_never_matters and not any_sel_under:
            print(
                "  READING: combined mechanics may be too weak to change "
                "policy ranking at this OP — report numbers."
            )
        elif any_never_matters:
            print(
                "  READING (combined, pre-ablation): never_boost advantage is "
                "real, but attribution is ambiguous — wear and direct R cost "
                "have different time-courses. See Check 2b ablations. "
                "selected≈wellward/fixed is *structurally expected* under "
                "global wear (sector-blind by design), not an empirical "
                "resolution of self-model blindness."
            )
    else:
        print(
            f"  READING [{tag_label}]: never_boost_matters={any_never_matters}; "
            f"use ablation cross-compare for mechanism attribution."
        )

    return {
        "condition_name": condition_name,
        "file_tag": file_tag,
        "distances": list(distances),
        "policies": list(POLICIES),
        "horizon": horizon,
        "t_floor_ambient": t_floor_ambient,
        "summary": {str(d): summary[d] for d in distances},
        "comparisons": {str(d): comparisons[d] for d in distances},
        "never_boost_advantage": {str(d): nb_advantage[d] for d in distances},
        "any_never_boost_matters": any_never_matters,
        "any_selected_underperforms_controls": any_sel_under,
        "any_selected_underperforms_controls_raw": any_sel_under_raw,
        "meaningful_coh_delta": MEANINGFUL_COH_DELTA,
        "boost_coh_neg_corr_dist_2_4": neg_corr_clean,
        "boost_count_parity": {str(d): sel_boost_tied[d] for d in distances},
        "total_masked_selected": total_masked,
        "total_candidates_dropped_mask": total_dropped,
        "plot": str(out_plot),
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
            "regen_rate": attn_cfg.regen_rate,
            "boost_cost": attn_cfg.boost_cost,
            "A_max": attn_cfg.A_max,
            "boost_magnitude": attn_cfg.boost_magnitude,
        },
        "fixed_best_sectors": {
            str(d): V2_BEST_FIXED_SECTOR.get(d) for d in distances
        },
    }


def check2_mechanism_ablations(
    model,
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    device: str,
    attn_cfg: AttentionConfig,
    wear_cfg: WearConfig,
    cost_cfg: BoostCostConfig,
    distances: Tuple[int, ...] = (2, 4, 8),
    horizon: int = 5000,
    seed: int = 0,
) -> Dict:
    """
    Check 2b — attribute never_boost advantage to wear vs direct R cost.

      wear_only:   direct_boost_cost = 0  (boost still accelerates wear)
      direct_only: wear_boost_rate = 0    (ambient wear shared; no action-driven
                   gen_rate degradation; only Option A R charge differentiates)

    Ambient wear remains in both so the shared gen_rate floor still exists;
    what changes is whether *boosting* accelerates wear (wear_only) or only
    charges R (direct_only).
    """
    print("\n=== Check 2b: mechanism ablations (wear-only vs direct-cost-only) ===")

    wear_only_cfg = WearConfig(
        wear_ambient_rate=wear_cfg.wear_ambient_rate,
        wear_boost_rate=wear_cfg.wear_boost_rate,
        wear_decay_coeff=wear_cfg.wear_decay_coeff,
        gen_floor_frac=wear_cfg.gen_floor_frac,
        gen_rate_base=wear_cfg.gen_rate_base,
    )
    wear_only_cost = BoostCostConfig(direct_boost_cost=0.0)

    direct_only_cfg = WearConfig(
        wear_ambient_rate=wear_cfg.wear_ambient_rate,
        wear_boost_rate=0.0,  # action does not accelerate wear
        wear_decay_coeff=wear_cfg.wear_decay_coeff,
        gen_floor_frac=wear_cfg.gen_floor_frac,
        gen_rate_base=wear_cfg.gen_rate_base,
    )
    direct_only_cost = BoostCostConfig(
        direct_boost_cost=cost_cfg.direct_boost_cost
    )

    print(
        f"  wear_only: amb={wear_only_cfg.wear_ambient_rate}  "
        f"boost_rate={wear_only_cfg.wear_boost_rate}  "
        f"direct_cost={wear_only_cost.direct_boost_cost}"
    )
    print(
        f"  direct_only: amb={direct_only_cfg.wear_ambient_rate}  "
        f"boost_rate={direct_only_cfg.wear_boost_rate}  "
        f"direct_cost={direct_only_cost.direct_boost_cost}"
    )

    c2_wear = check2_policy_comparison(
        model,
        nca,
        base_pattern,
        world_cfg,
        device=device,
        attn_cfg=attn_cfg,
        wear_cfg=wear_only_cfg,
        cost_cfg=wear_only_cost,
        distances=distances,
        horizon=horizon,
        seed=seed,
        condition_name="wear_only",
        file_tag="wear_only",
        save_csvs=True,
    )
    c2_direct = check2_policy_comparison(
        model,
        nca,
        base_pattern,
        world_cfg,
        device=device,
        attn_cfg=attn_cfg,
        wear_cfg=direct_only_cfg,
        cost_cfg=direct_only_cost,
        distances=distances,
        horizon=horizon,
        seed=seed,
        condition_name="direct_only",
        file_tag="direct_only",
        save_csvs=True,
    )

    # Cross-condition comparison of never_boost advantage
    print("\n  === Ablation attribution (never_boost − selected mean_coh) ===")
    attribution = {}
    for d in distances:
        # Will be filled by caller for combined; here compare wear vs direct
        dw = c2_wear["never_boost_advantage"][str(d)]["delta_mean_coh_vs_selected"]
        dd = c2_direct["never_boost_advantage"][str(d)]["delta_mean_coh_vs_selected"]
        attribution[d] = {
            "wear_only_delta_nb_sel": dw,
            "direct_only_delta_nb_sel": dd,
            "wear_only_never_boost_matters_dist": (
                c2_wear["comparisons"][str(d)]["never_boost_higher_coh_any"]
                or c2_wear["comparisons"][str(d)]["never_boost_outlives_any"]
            ),
            "direct_only_never_boost_matters_dist": (
                c2_direct["comparisons"][str(d)]["never_boost_higher_coh_any"]
                or c2_direct["comparisons"][str(d)]["never_boost_outlives_any"]
            ),
            "wear_only_pre_floor_delta": c2_wear["never_boost_advantage"][str(d)][
                "delta_pre_floor"
            ],
            "wear_only_post_floor_delta": c2_wear["never_boost_advantage"][str(d)][
                "delta_post_floor"
            ],
            "direct_only_pre_floor_delta": c2_direct["never_boost_advantage"][str(d)][
                "delta_pre_floor"
            ],
            "direct_only_post_floor_delta": c2_direct["never_boost_advantage"][str(d)][
                "delta_post_floor"
            ],
        }
        print(
            f"  dist={d}: wear_only Δ(nb−sel)={dw:.4f}  "
            f"direct_only Δ(nb−sel)={dd:.4f}"
        )
        print(
            f"    wear_only pre/post floor Δ: "
            f"{attribution[d]['wear_only_pre_floor_delta']:.4f} / "
            f"{attribution[d]['wear_only_post_floor_delta']:.4f}"
        )
        print(
            f"    direct_only pre/post floor Δ: "
            f"{attribution[d]['direct_only_pre_floor_delta']:.4f} / "
            f"{attribution[d]['direct_only_post_floor_delta']:.4f}"
        )

    # Clean-distance verdict
    clean = [d for d in (2, 4) if d in attribution]
    wear_repro = any(
        (not math.isnan(attribution[d]["wear_only_delta_nb_sel"]))
        and attribution[d]["wear_only_delta_nb_sel"] > 0.01
        for d in clean
    )
    direct_repro = any(
        (not math.isnan(attribution[d]["direct_only_delta_nb_sel"]))
        and attribution[d]["direct_only_delta_nb_sel"] > 0.01
        for d in clean
    )
    # Which is larger on average over clean distances?
    wear_avg = _finite_mean(
        [attribution[d]["wear_only_delta_nb_sel"] for d in clean]
    )
    direct_avg = _finite_mean(
        [attribution[d]["direct_only_delta_nb_sel"] for d in clean]
    )

    if wear_repro and not direct_repro:
        reading = (
            "never_boost advantage is reproduced by wear-only (boost-accelerated "
            "gen_rate degradation); direct cost alone does not drive it at this OP."
        )
    elif direct_repro and not wear_repro:
        reading = (
            "never_boost advantage is reproduced by direct-cost-only; "
            "boost-accelerated wear alone does not drive the frugality gap at "
            "this OP. Headline should NOT credit 'wear' for the late advantage."
        )
    elif wear_repro and direct_repro:
        reading = (
            "both mechanics independently produce a never_boost advantage on "
            f"clean distances (mean Δ wear_only={wear_avg:.4f}, "
            f"direct_only={direct_avg:.4f}). Report both; do not collapse into "
            "a single 'wear' claim."
        )
    else:
        reading = (
            "neither ablation alone reproduces a clear never_boost advantage "
            "on clean distances — interaction effect, or combined OP is weak "
            "once split. Report numbers."
        )

    print(f"\n  ATTRIBUTION VERDICT: {reading}")
    print(
        f"  wear_only reproduces nb lead (clean, Δ>0.01)? {wear_repro}  "
        f"avg Δ={wear_avg:.4f}"
    )
    print(
        f"  direct_only reproduces nb lead (clean, Δ>0.01)? {direct_repro}  "
        f"avg Δ={direct_avg:.4f}"
    )

    # Bar chart: Δ(nb−sel) by condition × distance
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    # Combined will be overlaid by caller if available; plot ablations here
    x = list(range(len(distances)))
    w = 0.35
    wear_vals = [
        attribution[d]["wear_only_delta_nb_sel"] for d in distances
    ]
    dir_vals = [
        attribution[d]["direct_only_delta_nb_sel"] for d in distances
    ]
    ax.bar(
        [i - w / 2 for i in x],
        wear_vals,
        width=w,
        label="wear_only (direct_cost=0)",
        color="C0",
        alpha=0.85,
    )
    ax.bar(
        [i + w / 2 for i in x],
        dir_vals,
        width=w,
        label="direct_only (wear_boost_rate=0)",
        color="C3",
        alpha=0.85,
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(
        MEANINGFUL_COH_DELTA,
        color="0.5",
        ls=":",
        lw=0.8,
        label=f"±{MEANINGFUL_COH_DELTA} threshold",
    )
    ax.axhline(-MEANINGFUL_COH_DELTA, color="0.5", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"dist={d}" for d in distances])
    ax.set_ylabel("Δ mean_coh (never_boost − selected)")
    ax.set_title(
        "Check 2b: which mechanic reproduces the never_boost advantage?\n"
        f"H={horizon}  amb={wear_cfg.wear_ambient_rate}  "
        f"bst_w(combined)={wear_cfg.wear_boost_rate}  "
        f"direct(combined)={cost_cfg.direct_boost_cost}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    abl_plot = RESULTS / "check2_ablation_attribution.png"
    fig.savefig(abl_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {abl_plot}")

    return {
        "wear_only": c2_wear,
        "direct_only": c2_direct,
        "attribution_by_distance": {str(d): attribution[d] for d in distances},
        "wear_only_reproduces_nb_advantage_clean": wear_repro,
        "direct_only_reproduces_nb_advantage_clean": direct_repro,
        "wear_only_avg_delta_clean": wear_avg,
        "direct_only_avg_delta_clean": direct_avg,
        "reading": reading,
        "plot": str(abl_plot),
    }


# ---------------------------------------------------------------------------
# Check 3 — consistency
# ---------------------------------------------------------------------------
def check3_consistency(
    c1: Dict,
    c2: Dict,
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    wear_cfg: WearConfig,
    cost_cfg: BoostCostConfig,
    attn_cfg: AttentionConfig,
    device: str,
    seed: int = 0,
) -> Dict:
    """
    Confirm:
      - effective_gen_rate is used consistently (no stale gen_rate reads in logs)
      - direct_boost_cost only charged when boost is applied
      - never_boost has zero charges and zero boosts
    """
    print("\n=== Check 3: mask / gen_rate / direct-cost consistency ===")

    issues: List[str] = []

    # From Check 1: no charges under never_boost
    if c1.get("no_direct_charges") is not True:
        issues.append("Check1: expected zero direct charges under disable_boosting")
    if c1.get("no_boosts") is not True:
        issues.append("Check1: expected zero boosts under disable_boosting")
    if c1.get("never_below_floor") is not True:
        issues.append("Check1: effective_gen_rate dipped below floor")

    # From Check 2 summaries
    for d_str, pol_sum in c2.get("summary", {}).items():
        nb = pol_sum.get("never_boost", {})
        if nb.get("n_boosts", -1) != 0:
            issues.append(f"dist={d_str} never_boost n_boosts={nb.get('n_boosts')}")
        if nb.get("n_direct_charges", -1) != 0:
            issues.append(
                f"dist={d_str} never_boost n_direct_charges={nb.get('n_direct_charges')}"
            )
        for pol, st in pol_sum.items():
            # n_direct_charges must equal n_boosts (charge-on-apply-only)
            if st.get("n_direct_charges") != st.get("n_boosts"):
                issues.append(
                    f"dist={d_str} {pol}: n_direct_charges="
                    f"{st.get('n_direct_charges')} != n_boosts={st.get('n_boosts')}"
                )
            # egr never below floor
            floor = wear_cfg.gen_rate_base * wear_cfg.gen_floor_frac
            if st.get("min_effective_gen_rate", 0) < floor - 1e-9:
                issues.append(
                    f"dist={d_str} {pol}: min_egr={st.get('min_effective_gen_rate')} "
                    f"< floor={floor}"
                )
            # more boost_steps ⇒ more wear (monotonic relationship across
            # never_boost vs greedy) — soft check
            if pol != "never_boost":
                if (
                    st.get("n_boost_steps", 0) > 0
                    and st.get("final_wear", 0) + 1e-9
                    < nb.get("final_wear", 0)
                ):
                    # boosting should not *decrease* wear; ambient is same horizon
                    # so more boost steps ⇒ weakly more wear
                    issues.append(
                        f"dist={d_str} {pol}: final_wear={st.get('final_wear')} "
                        f"< never_boost final_wear={nb.get('final_wear')} "
                        f"despite n_boost_steps={st.get('n_boost_steps')}"
                    )

    # Short instrumented episode: charge only when boost applied
    print("  instrumented short episode (selected, dist=4, H=200)…")
    short = run_wear_episode(
        nca=nca,
        base_pattern=base_pattern,
        world_cfg=world_cfg,
        model=None,  # use random for speed if model not needed
        policy="random",
        distance=4,
        horizon=200,
        attn_cfg=attn_cfg,
        wear_cfg=wear_cfg,
        cost_cfg=cost_cfg,
        seed=seed,
        device=device,
    )
    # Every charge event must have matching boost
    n_boosts = short["n_boosts"]
    n_charges = short["n_direct_charges"]
    if n_charges != n_boosts:
        issues.append(
            f"short episode: n_direct_charges={n_charges} != n_boosts={n_boosts}"
        )
    # History egr must match WearState formula given wear
    formula_ok = True
    for h in short["history"]:
        expected = effective_gen_rate_formula(
            h["wear"],
            wear_cfg.gen_rate_base,
            wear_cfg.wear_decay_coeff,
            wear_cfg.gen_floor_frac,
        )
        if abs(h["effective_gen_rate"] - expected) > 1e-6:
            formula_ok = False
            issues.append(
                f"t={h['t']}: logged egr={h['effective_gen_rate']} "
                f"!= formula={expected}"
            )
            break
    # world.config.gen_rate must not be silently used in place of egr:
    # verify egr declines below base when wear > 0
    if short["final_wear"] > wear_cfg.wear_ambient_rate:
        if short["final_effective_gen_rate"] >= wear_cfg.gen_rate_base - 1e-9:
            # only fail if wear is large enough to matter
            if short["final_wear"] > 0.1:
                issues.append(
                    "final_egr still at base despite substantial wear — "
                    "possible stale gen_rate"
                )

    mask_ok = c2.get("total_masked_selected", 0) == 0
    if not mask_ok:
        issues.append(
            f"masked candidates selected: {c2.get('total_masked_selected')}"
        )

    passed = len(issues) == 0 and formula_ok and mask_ok
    print(f"  short episode: boosts={n_boosts} charges={n_charges} formula_ok={formula_ok}")
    print(f"  mask compliance: total_masked_selected={c2.get('total_masked_selected', 0)}")
    if issues:
        print(f"  ISSUES ({len(issues)}):")
        for iss in issues[:20]:
            print(f"    - {iss}")
    print(f"  Check 3 PASS={passed}")

    return {
        "pass": passed,
        "issues": issues,
        "formula_ok": formula_ok,
        "mask_ok": mask_ok,
        "short_n_boosts": n_boosts,
        "short_n_direct_charges": n_charges,
        "total_masked_selected": c2.get("total_masked_selected", 0),
        "total_candidates_dropped_mask": c2.get("total_candidates_dropped_mask", 0),
    }


# ---------------------------------------------------------------------------
# Observations + main
# ---------------------------------------------------------------------------
def write_observations(
    sweep_best: WearConfig,
    sweep_rows: List[Dict],
    c1: Dict,
    c2: Dict,
    c3: Dict,
    attn_cfg: AttentionConfig,
    cost_cfg: BoostCostConfig,
    args,
    device: str,
    ablations: Optional[Dict] = None,
) -> Path:
    obs: List[str] = []
    obs.append("# v3 wear-coupled environmental consequence — observations\n")
    obs.append(f"device: {device}\nseed: {args.seed}\n")
    obs.append(f"horizon: {args.horizon}\n")
    obs.append(f"v0_ckpt: {args.ckpt}\n")
    obs.append(f"v2_self_model: {args.model}\n")
    obs.append(
        "NOTE: self-model is the existing v2 checkpoint — NOT retrained on wear.\n"
    )

    obs.append("\n## Operating point (from Check 1 sweep)\n")
    obs.append(
        f"- wear_ambient_rate = {sweep_best.wear_ambient_rate}\n"
        f"- wear_boost_rate = {sweep_best.wear_boost_rate}\n"
        f"- wear_decay_coeff = {sweep_best.wear_decay_coeff}\n"
        f"- gen_floor_frac = {sweep_best.gen_floor_frac}\n"
        f"- gen_rate_base = {sweep_best.gen_rate_base}\n"
        f"- direct_boost_cost = {cost_cfg.direct_boost_cost}\n"
    )
    st = WearState(sweep_best)
    t_floor = st.ambient_steps_to_floor()
    obs.append(
        f"- closed-form ambient steps to floor ≈ {t_floor:.0f}\n"
        f"- wear at floor = {st.wear_at_floor():.4f}\n"
        f"- boost/ambient ratio = "
        f"{sweep_best.wear_boost_rate / max(sweep_best.wear_ambient_rate, 1e-12):.1f}\n"
    )
    obs.append(
        "- Selection criterion: t_floor near 0.6×horizon, boost/ambient ~4, "
        "floor ~0.4. First-guess parameters were not assumed correct.\n"
    )
    obs.append(
        f"- **Two cost mechanics with different time-courses:**\n"
        f"  1. **Wear → gen_rate:** ambient (+ boost-accelerated) wear drives "
        f"`effective_gen_rate` down to the floor by t≈{t_floor:.0f} even with "
        f"*zero* boosting. After every policy has hit the floor, additional "
        f"wear has **no further effect on gen_rate** (clamped). Greedy "
        f"policies hit the floor earlier; after max(t_floor_i) all policies "
        f"share identical egr.\n"
        f"  2. **Direct R charge (Option A):** paid every time a boost is "
        f"armed, for the whole horizon — never stops compounding for greedy "
        f"policies, costs `never_boost` nothing.\n"
    )

    obs.append("\n## Attention budget (reused from v2 operating point)\n")
    obs.append(
        f"- K={attn_cfg.K}  regen_rate={attn_cfg.regen_rate}  "
        f"boost_cost={attn_cfg.boost_cost}  A_max={attn_cfg.A_max}  "
        f"boost_magnitude={attn_cfg.boost_magnitude}\n"
    )

    obs.append("\n## Check 1 — mechanic sanity (boosting disabled)\n")
    obs.append(f"- pass: {c1['pass']}\n")
    obs.append(f"- monotonic wear: {c1['monotonic_wear']}\n")
    obs.append(f"- ambient-exact accrual: {c1['ambient_exact']}\n")
    obs.append(f"- egr never below floor: {c1['never_below_floor']}\n")
    obs.append(f"- egr nonincreasing: {c1['egr_nonincreasing']}\n")
    obs.append(
        f"- t_floor closed-form: {c1['t_floor_closed_form']:.1f}\n"
        f"- t_floor empirical: {c1['t_floor_empirical']}\n"
    )
    obs.append(
        f"- final_wear={c1['final_wear']:.6f}  "
        f"final_egr={c1['final_effective_gen_rate']:.6f}  "
        f"floor={c1['floor_gen_rate']:.6f}\n"
    )
    obs.append(
        f"- timescale: ambient wear reaches the gen_rate floor in ~"
        f"{c1['t_floor_closed_form']:.0f} steps. After that time, **wear can "
        f"no longer differentiate policies on gen_rate** (all at floor).\n"
    )
    obs.append(f"- plot: {c1['plot']}\n")

    thr = c2.get("meaningful_coh_delta", MEANINGFUL_COH_DELTA)
    obs.append("\n## Check 2 — combined mechanics (wear + direct cost)\n")
    obs.append(
        f"- Horizon: {c2['horizon']} (longer than v2's 2000)\n"
        f"- Distances: {c2['distances']}\n"
        f"- Policies: {c2['policies']}\n"
        f"- fixed = best sector from v2 6-way sweep: {c2.get('fixed_best_sectors')}\n"
        f"- meaningful_coh_delta threshold: {thr} "
        f"(underperforms_* flags use this; raw_sign is diagnostics only)\n"
    )
    for d in c2["distances"]:
        s = c2["summary"][str(d)]
        cmp_ = c2["comparisons"][str(d)]
        obs.append(f"- dist={d}:\n")
        for pol in c2["policies"]:
            st_p = s[pol]
            fc = st_p["final_coherence"]
            fc_s = (
                f"{fc:.4f}"
                if not (isinstance(fc, float) and math.isnan(fc))
                else "nan"
            )
            obs.append(
                f"  - {pol}: mean_coh={st_p['mean_coherence']:.4f}  "
                f"final_coh={fc_s}  "
                f"mean_mass={st_p['mean_self_mass']:.3f}  "
                f"final_mass={st_p['final_self_mass']:.3f}  "
                f"tod={st_p['time_of_death']}  "
                f"boosts={st_p['n_boosts']}/{st_p['n_decisions']}  "
                f"boost_steps={st_p['n_boost_steps']}  "
                f"final_wear={st_p['final_wear']:.4f}  "
                f"final_egr={st_p['final_effective_gen_rate']:.5f}  "
                f"charges={st_p['n_direct_charges']}"
                + (
                    f"  locked={st_p.get('fixed_sector_locked')}"
                    if pol == "fixed"
                    else ""
                )
                + "\n"
            )
        obs.append(
            f"  - never_boost higher_coh any greedy: "
            f"{cmp_['never_boost_higher_coh_any']}\n"
        )
        obs.append(
            f"  - never_boost outlives any greedy: "
            f"{cmp_['never_boost_outlives_any']}\n"
        )
        obs.append(
            f"  - Δcoh(selected − wellward) = "
            f"{cmp_['selected_vs_wellward_delta_coh']:.5f}  "
            f"underperforms_wellward={cmp_['selected_underperforms_wellward']}  "
            f"(raw_sign={cmp_.get('selected_underperforms_wellward_raw')})\n"
        )
        obs.append(
            f"  - Δcoh(selected − fixed) = "
            f"{cmp_['selected_vs_fixed_delta_coh']:.5f}  "
            f"underperforms_fixed={cmp_['selected_underperforms_fixed']}  "
            f"(raw_sign={cmp_.get('selected_underperforms_fixed_raw')})\n"
        )
        obs.append(
            f"  - Pearson(boost_steps, mean_coh) = "
            f"{cmp_['boost_steps_vs_mean_coh_pearson']:.4f}\n"
        )
        obs.append(
            f"  - pre-floor (t<{cmp_.get('t_floor_ambient', t_floor):.0f}) "
            f"Δ mean_coh(nb−sel) = "
            f"{cmp_.get('never_boost_minus_selected_pre_floor', float('nan')):.4f}\n"
        )
        obs.append(
            f"  - post-floor (t≥{cmp_.get('t_floor_ambient', t_floor):.0f}) "
            f"Δ mean_coh(nb−sel) = "
            f"{cmp_.get('never_boost_minus_selected_post_floor', float('nan')):.4f}\n"
        )
        obs.append(
            f"  - t_floor empirical by policy: "
            f"{cmp_.get('t_floor_empirical_by_policy')}\n"
        )

    obs.append("\n### Combined-run frugality (not yet attributed)\n")
    obs.append(
        f"- any_never_boost_matters: {c2['any_never_boost_matters']}\n"
        f"- boost_coh_neg_corr_dist_2_4: {c2['boost_coh_neg_corr_dist_2_4']}\n"
    )
    if c2["any_never_boost_matters"] or c2["boost_coh_neg_corr_dist_2_4"]:
        obs.append(
            "- **Combined finding:** `never_boost` out-coheres / outlives "
            "greedy policies at dist 2/4 under *both* mechanics on. Real, but "
            "must not be credited to a single mechanism without Check 2b. "
            f"Ambient wear alone hits the gen_rate floor by t≈{t_floor:.0f}; "
            "after every policy is floor-clamped, wear can no longer "
            "differentiate egr. Pre/post ambient-floor Δ(nb−sel) and the "
            "wear-only / direct-only ablations attribute which cost drives "
            "the full-horizon mean_coh gap (and longevity).\n"
        )
    else:
        obs.append(
            "- Combined run did not show a clear never_boost advantage; "
            "see ablation numbers anyway.\n"
        )

    obs.append("\n### selected vs wellward/fixed — design foreclosure, not data\n")
    obs.append(
        f"- any_selected_underperforms_controls (|Δ|≥{thr}, clean dist 2/4): "
        f"{c2['any_selected_underperforms_controls']}\n"
        f"- any_selected_underperforms_controls_raw (sign only): "
        f"{c2.get('any_selected_underperforms_controls_raw')}\n"
    )
    obs.append(
        "- **This arm of Check 2 was foreclosed by the Stage A design choice "
        "(global wear), not resolved by the data.** Global wear depends only "
        "on *whether* a boost happened, never *where*. No amount of "
        "self-model sophistication about sector choice can show up in the "
        "wear → gen_rate term. The only location-sensitive cost channel is "
        "the direct R charge at the boost cell, and v2 already established "
        "that `selected` locks nearly the same sector as wellward/fixed on "
        "this map — so no new discrimination is expected. Writing this up as "
        "'selected does not underperform under wear' would misread a "
        "structural non-result as an empirical finding. Per-well wear "
        "(deferred) is what would open this arm.\n"
    )
    bcp = c2.get("boost_count_parity", {})
    if bcp:
        obs.append("- Boost-count parity (same frequency ⇒ same global wear):\n")
        for d_str, row in bcp.items():
            obs.append(
                f"  - dist={d_str}: selected={row.get('selected_n_boosts')}  "
                f"wellward={row.get('wellward_n_boosts')}  "
                f"fixed={row.get('fixed_n_boosts')}  "
                f"never_boost={row.get('never_boost_n_boosts')}  "
                f"sel==wellward? {row.get('selected_matches_wellward_boost_count')}\n"
            )
    obs.append(f"- plot: {c2['plot']}\n")

    # ---- Check 2b ablations ----
    obs.append("\n## Check 2b — mechanism ablations (attribution)\n")
    if ablations is None:
        obs.append("- (not run)\n")
    else:
        obs.append(
            "- **wear_only:** direct_boost_cost=0; boost still accelerates wear.\n"
            "- **direct_only:** wear_boost_rate=0; ambient wear still shared by "
            "all policies (gen_rate still degrades to the floor on the ambient "
            "schedule); only the Option A R charge differentiates boosters "
            "from never_boost.\n"
        )
        obs.append(
            f"- wear_only reproduces nb advantage on clean dist (Δ>0.01)? "
            f"{ablations.get('wear_only_reproduces_nb_advantage_clean')}\n"
            f"- direct_only reproduces nb advantage on clean dist (Δ>0.01)? "
            f"{ablations.get('direct_only_reproduces_nb_advantage_clean')}\n"
            f"- wear_only avg Δ(nb−sel) clean: "
            f"{ablations.get('wear_only_avg_delta_clean')}\n"
            f"- direct_only avg Δ(nb−sel) clean: "
            f"{ablations.get('direct_only_avg_delta_clean')}\n"
        )
        # Include combined deltas for side-by-side
        obs.append("- Per-distance Δ mean_coh (never_boost − selected):\n")
        for d in c2["distances"]:
            comb = c2.get("never_boost_advantage", {}).get(str(d), {})
            att = ablations.get("attribution_by_distance", {}).get(str(d), {})
            obs.append(
                f"  - dist={d}: combined={comb.get('delta_mean_coh_vs_selected', float('nan')):.4f}  "
                f"wear_only={att.get('wear_only_delta_nb_sel', float('nan')):.4f}  "
                f"direct_only={att.get('direct_only_delta_nb_sel', float('nan')):.4f}\n"
            )
            obs.append(
                f"    wear_only pre/post floor Δ: "
                f"{att.get('wear_only_pre_floor_delta', float('nan')):.4f} / "
                f"{att.get('wear_only_post_floor_delta', float('nan')):.4f}\n"
            )
            obs.append(
                f"    direct_only pre/post floor Δ: "
                f"{att.get('direct_only_pre_floor_delta', float('nan')):.4f} / "
                f"{att.get('direct_only_post_floor_delta', float('nan')):.4f}\n"
            )
            # Summaries for each ablation condition
            for name, key in (("wear_only", "wear_only"), ("direct_only", "direct_only")):
                sub = ablations.get(key, {})
                sm = sub.get("summary", {}).get(str(d), {})
                if not sm:
                    continue
                obs.append(f"    [{name}] key policies:\n")
                for pol in ("never_boost", "selected", "wellward_heuristic"):
                    if pol not in sm:
                        continue
                    p = sm[pol]
                    obs.append(
                        f"      {pol}: mean_coh={p['mean_coherence']:.4f}  "
                        f"tod={p['time_of_death']}  "
                        f"boosts={p['n_boosts']}  "
                        f"final_wear={p['final_wear']:.4f}  "
                        f"final_egr={p['final_effective_gen_rate']:.5f}  "
                        f"charges={p['n_direct_charges']}\n"
                    )
        obs.append(f"- **ATTRIBUTION:** {ablations.get('reading')}\n")
        obs.append(f"- ablation plot: {ablations.get('plot')}\n")
        obs.append(
            f"- wear_only plot: "
            f"{ablations.get('wear_only', {}).get('plot')}\n"
            f"- direct_only plot: "
            f"{ablations.get('direct_only', {}).get('plot')}\n"
        )

    obs.append("\n## Check 3 — consistency\n")
    obs.append(f"- pass: {c3['pass']}\n")
    obs.append(f"- formula_ok (logged egr == wear formula): {c3['formula_ok']}\n")
    obs.append(f"- mask_ok: {c3['mask_ok']}\n")
    obs.append(
        f"- short episode boosts={c3['short_n_boosts']}  "
        f"charges={c3['short_n_direct_charges']} (must match)\n"
    )
    if c3.get("issues"):
        obs.append("- issues:\n")
        for iss in c3["issues"]:
            obs.append(f"  - {iss}\n")
    else:
        obs.append("- no issues\n")

    obs.append("\n## Stage A verdict\n")
    obs.append(
        "Stage A asks whether ignoring long-horizon consequence costs "
        "something measurable *before* retraining. Two mechanisms were on; "
        "attribution (Check 2b) decides what the headline may claim.\n"
    )

    # Craft next-step based on ablation attribution
    if ablations is not None:
        wear_r = ablations.get("wear_only_reproduces_nb_advantage_clean")
        dir_r = ablations.get("direct_only_reproduces_nb_advantage_clean")
        if dir_r and not wear_r:
            obs.append(
                "**Finding (precise):** frugality matters primarily because "
                "each boost has an **upfront R cost** that compounds for the "
                "whole horizon. Boost-accelerated wear alone does not "
                "reproduce the never_boost advantage at this OP (ambient wear "
                "already drives everyone to the gen_rate floor). Do **not** "
                "headline this as 'frugality under wear.'\n\n"
                "**Next:** a self-model earns complexity if it can decide "
                "*whether* to spend A (trade immediate coherence for R), not "
                "merely *where*. Options: (1) add a no-boost action to the "
                "decision set; (2) per-well wear so *where* becomes "
                "consequential (opens the foreclosed selected-vs-wellward "
                "arm); (3) stop — Stage A shows direct cost is enough for "
                "frugality without retrain. v3.5 retrain on wear-aware "
                "rollouts is **not** the natural next step if wear is not "
                "what drove the gap.\n"
            )
        elif wear_r and not dir_r:
            obs.append(
                "**Finding (precise):** frugality matters because "
                "boost-accelerated **wear** degrades future gen_rate; the "
                "direct R charge alone does not reproduce the gap at this OP.\n\n"
                "**Next:** wear-aware self-model (v3.5) with a no-boost option "
                "is a candidate; per-well wear still needed if the goal is "
                "sector-level discrimination.\n"
            )
        elif wear_r and dir_r:
            obs.append(
                "**Finding (precise):** **both** mechanics independently "
                "produce a never_boost advantage. Report both; do not "
                "collapse into a single 'wear' claim. Late-horizon gaps are "
                "still more naturally explained by direct cost (post-floor "
                "egr is identical).\n\n"
                "**Next:** if the interesting decision is *whether* to boost, "
                "either mechanic motivates a no-boost option in the selector. "
                "Per-well wear remains required to make *where* matter for "
                "environmental degradation. Do not retrain solely to re-rank "
                "sectors under global wear.\n"
            )
        else:
            obs.append(
                "**Finding:** neither ablation alone clearly reproduces the "
                "combined never_boost advantage — interaction or weak split. "
                "See numbers; do not overclaim mechanism.\n"
            )
    elif c2["any_never_boost_matters"]:
        obs.append(
            "**Finding (unattributed):** never_boost wins under combined "
            "mechanics; run Check 2b ablations before deciding v3.5 vs "
            "per-well wear.\n"
        )
    else:
        obs.append(
            "**Finding:** no clear frugality signal; do not force parameters.\n"
        )

    path = RESULTS / "observations.txt"
    path.write_text("".join(obs))
    print(f"\n  wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="v3 wear-coupled environmental consequence checks"
    )
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--ckpt", type=str, default=str(DEFAULT_CKPT), help="v0 trained NCA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(results_dir("v2") / "self_model_v2.pt"),
        help="existing v2 self-model (not retrained)",
    )
    parser.add_argument(
        "--direct_boost_cost",
        type=float,
        default=0.15,
        help="Option A direct R charge at boost target",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shorter horizon / smoke test",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default="2,4,8",
        help="comma-separated distances",
    )
    # Optional overrides for wear params (skip sweep)
    parser.add_argument("--wear_ambient_rate", type=float, default=None)
    parser.add_argument("--wear_boost_rate", type=float, default=None)
    parser.add_argument("--wear_decay_coeff", type=float, default=None)
    parser.add_argument("--gen_floor_frac", type=float, default=None)
    parser.add_argument(
        "--skip_ablations",
        action="store_true",
        help="skip Check 2b wear-only / direct-only ablations",
    )
    args = parser.parse_args()

    if args.quick:
        args.horizon = min(args.horizon, 800)

    distances = tuple(int(x) for x in args.distances.split(","))
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = pick_device(prefer_cuda=not args.cpu)
    print(f"device={device}  horizon={args.horizon}  distances={distances}")
    print(f"v0 ckpt: {args.ckpt}")
    print(f"v2 model: {args.model}")

    # ---- load frozen world + self-model ----
    nca, base_pattern, _ = load_v0_world(args.ckpt, device=device)
    world_cfg = WorldConfig(device=device)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"v2 self-model not found: {model_path}. "
            "Run v2 checks first, or pass --model."
        )
    model = load_model(model_path, device=device)
    print(f"  loaded self-model from {model_path} (no retrain)")

    # ---- wear param sweep / choose OP ----
    sweep_rows, wear_cfg = sweep_wear_params(horizon=args.horizon)
    # Apply CLI overrides if given
    wear_cfg = WearConfig(
        wear_ambient_rate=(
            args.wear_ambient_rate
            if args.wear_ambient_rate is not None
            else wear_cfg.wear_ambient_rate
        ),
        wear_boost_rate=(
            args.wear_boost_rate
            if args.wear_boost_rate is not None
            else wear_cfg.wear_boost_rate
        ),
        wear_decay_coeff=(
            args.wear_decay_coeff
            if args.wear_decay_coeff is not None
            else wear_cfg.wear_decay_coeff
        ),
        gen_floor_frac=(
            args.gen_floor_frac
            if args.gen_floor_frac is not None
            else wear_cfg.gen_floor_frac
        ),
        gen_rate_base=DEFAULT_GEN_RATE,
    )
    print(
        f"\n  operating point: ambient={wear_cfg.wear_ambient_rate}  "
        f"boost_rate={wear_cfg.wear_boost_rate}  "
        f"coeff={wear_cfg.wear_decay_coeff}  "
        f"floor={wear_cfg.gen_floor_frac}"
    )

    # Save sweep
    with (RESULTS / "check1_wear_sweep.json").open("w") as f:
        json.dump(
            {
                "chosen": {
                    "wear_ambient_rate": wear_cfg.wear_ambient_rate,
                    "wear_boost_rate": wear_cfg.wear_boost_rate,
                    "wear_decay_coeff": wear_cfg.wear_decay_coeff,
                    "gen_floor_frac": wear_cfg.gen_floor_frac,
                    "gen_rate_base": wear_cfg.gen_rate_base,
                },
                "rows": sweep_rows,
            },
            f,
            indent=2,
        )

    cost_cfg = BoostCostConfig(direct_boost_cost=args.direct_boost_cost)

    # v2 attention OP (scarce): regen 0.035, cost 1.0, A_max 1.5 from v2 defaults
    attn_cfg = AttentionConfig(
        K=20,
        regen_rate=0.035,
        boost_cost=1.0,
        A_max=1.5,
        A0=1.0,
        boost_magnitude=0.40,
        boost_duration=20,
        n_candidates=6,
    )

    # ---- Check 1 ----
    c1 = check1_mechanic_sanity(
        nca,
        base_pattern,
        world_cfg,
        wear_cfg,
        device=device,
        horizon=args.horizon,
        distance=4,
        seed=args.seed,
    )

    # ---- Check 2 (combined) ----
    c2 = check2_policy_comparison(
        model,
        nca,
        base_pattern,
        world_cfg,
        device=device,
        attn_cfg=attn_cfg,
        wear_cfg=wear_cfg,
        cost_cfg=cost_cfg,
        distances=distances,
        horizon=args.horizon,
        seed=args.seed,
        condition_name="combined",
        file_tag="",
        save_csvs=True,
    )

    # ---- Check 2b (ablations) ----
    ablations = None
    if not args.skip_ablations:
        ablations = check2_mechanism_ablations(
            model,
            nca,
            base_pattern,
            world_cfg,
            device=device,
            attn_cfg=attn_cfg,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            distances=distances,
            horizon=args.horizon,
            seed=args.seed,
        )
        # Overlay combined Δ on attribution bar chart for side-by-side
        try:
            comb_deltas = [
                c2["never_boost_advantage"][str(d)]["delta_mean_coh_vs_selected"]
                for d in distances
            ]
            fig, ax = plt.subplots(figsize=(9, 4.2))
            x = list(range(len(distances)))
            w = 0.25
            wear_vals = [
                ablations["attribution_by_distance"][str(d)]["wear_only_delta_nb_sel"]
                for d in distances
            ]
            dir_vals = [
                ablations["attribution_by_distance"][str(d)][
                    "direct_only_delta_nb_sel"
                ]
                for d in distances
            ]
            ax.bar(
                [i - w for i in x],
                comb_deltas,
                width=w,
                label="combined",
                color="0.45",
                alpha=0.9,
            )
            ax.bar(
                x,
                wear_vals,
                width=w,
                label="wear_only",
                color="C0",
                alpha=0.85,
            )
            ax.bar(
                [i + w for i in x],
                dir_vals,
                width=w,
                label="direct_only",
                color="C3",
                alpha=0.85,
            )
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(MEANINGFUL_COH_DELTA, color="0.5", ls=":", lw=0.8)
            ax.axhline(-MEANINGFUL_COH_DELTA, color="0.5", ls=":", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"dist={d}" for d in distances])
            ax.set_ylabel("Δ mean_coh (never_boost − selected)")
            ax.set_title(
                "Check 2b attribution: combined vs wear-only vs direct-only\n"
                f"H={args.horizon}"
            )
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            abl_plot = RESULTS / "check2_ablation_attribution.png"
            fig.savefig(abl_plot, dpi=140)
            plt.close(fig)
            ablations["plot"] = str(abl_plot)
            print(f"  updated attribution plot with combined: {abl_plot}")
        except Exception as e:
            print(f"  (attribution replot skipped: {e})")

    # ---- Check 3 ----
    c3 = check3_consistency(
        c1,
        c2,
        nca,
        base_pattern,
        world_cfg,
        wear_cfg,
        cost_cfg,
        attn_cfg,
        device=device,
        seed=args.seed,
    )

    write_observations(
        wear_cfg,
        sweep_rows,
        c1,
        c2,
        c3,
        attn_cfg,
        cost_cfg,
        args,
        device,
        ablations=ablations,
    )

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    # Slim ablations for JSON (drop full per-step history already in CSVs)
    abl_slim = None
    if ablations is not None:
        abl_slim = {
            k: v
            for k, v in ablations.items()
            if k not in ("wear_only", "direct_only")
        }
        for name in ("wear_only", "direct_only"):
            sub = ablations[name]
            abl_slim[name] = {
                kk: sub[kk]
                for kk in (
                    "condition_name",
                    "summary",
                    "comparisons",
                    "never_boost_advantage",
                    "any_never_boost_matters",
                    "any_selected_underperforms_controls",
                    "boost_coh_neg_corr_dist_2_4",
                    "wear_cfg",
                    "cost_cfg",
                    "plot",
                    "t_floor_ambient",
                )
                if kk in sub
            }

    summary = {
        "device": device,
        "seed": args.seed,
        "horizon": args.horizon,
        "wear_cfg": {
            "wear_ambient_rate": wear_cfg.wear_ambient_rate,
            "wear_boost_rate": wear_cfg.wear_boost_rate,
            "wear_decay_coeff": wear_cfg.wear_decay_coeff,
            "gen_floor_frac": wear_cfg.gen_floor_frac,
            "gen_rate_base": wear_cfg.gen_rate_base,
        },
        "cost_cfg": {"direct_boost_cost": cost_cfg.direct_boost_cost},
        "attn_cfg": {
            "K": attn_cfg.K,
            "regen_rate": attn_cfg.regen_rate,
            "boost_cost": attn_cfg.boost_cost,
            "A_max": attn_cfg.A_max,
            "boost_magnitude": attn_cfg.boost_magnitude,
        },
        "check1": c1,
        "check2": c2,
        "check2b_ablations": abl_slim,
        "check3": c3,
    }
    with (RESULTS / "summary.json").open("w") as f:
        json.dump(_sanitize(summary), f, indent=2)
    print(f"  wrote {RESULTS / 'summary.json'}")

    print("\n=== v3 checks complete ===")
    abl_msg = ""
    if ablations is not None:
        abl_msg = (
            f"  ablations: wear_repro={ablations['wear_only_reproduces_nb_advantage_clean']}  "
            f"direct_repro={ablations['direct_only_reproduces_nb_advantage_clean']}"
        )
    print(
        f"Check 1 pass={c1['pass']}  "
        f"Check 2 never_boost_matters={c2['any_never_boost_matters']}  "
        f"selected_underperforms={c2['any_selected_underperforms_controls']}  "
        f"Check 3 pass={c3['pass']}"
        f"{abl_msg}"
    )


if __name__ == "__main__":
    main()
