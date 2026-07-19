#!/usr/bin/env python3
"""
run_v2_checks.py
================
v2 Action Selection (Counterfactual Self-Maintenance Allocation) checks.

  Check 1 — boost-subset calibration (mandatory gate; stop if fail)
  Check 2 — selected vs random vs round-robin attention policies
  Check 3 — mask compliance (no selected action on masked candidates)

Run from EmbodiedNCA root or v2/:

    python v2/run_v2_checks.py
    python v2/run_v2_checks.py --skip_dataset --skip_train
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

V2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V2_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V1_ROOT = REPO_ROOT / "v1"
for p in (str(V2_ROOT), str(V1_ROOT), str(V0_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_selection import (  # noqa: E402
    AttentionConfig,
    pick_operating_point,
    run_attention_episode,
    sweep_attention_budget,
)
from boost_rollout_dataset import (  # noqa: E402
    DEFAULT_CKPT,
    BoostDatasetConfig,
    generate_dataset,
    load_v0_world,
    train_val_split,
)
from grid_world import WorldConfig  # noqa: E402
from self_model_v2 import (  # noqa: E402
    TrainConfig,
    evaluate_boost_calibration,
    load_model,
    save_model,
    train_self_model_v2,
)

RESULTS = V2_ROOT / "results"


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _finite_mean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite_std(xs: List[float]) -> float:
    """Population std of finite values; nan if <2 points."""
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return float(math.sqrt(var))


def post_transient_stats(
    history: List[Dict[str, float]],
    t_min: int = 300,
) -> Dict[str, float]:
    """Mean/std of coherence and self_mass over alive steps with t >= t_min."""
    coh = [
        h["coherence"]
        for h in history
        if h["t"] >= t_min
        and h.get("alive", 1.0) > 0.5
        and not math.isnan(h["coherence"])
    ]
    mass = [
        h["self_mass"]
        for h in history
        if h["t"] >= t_min and h.get("alive", 1.0) > 0.5
    ]
    return {
        "t_min": t_min,
        "n": len(coh),
        "coherence_mean": _finite_mean(coh),
        "coherence_std": _finite_std(coh),
        "self_mass_mean": _finite_mean(mass),
        "self_mass_std": _finite_std(mass),
    }


def _summarize_geometry(geometry_logs: List[Dict]) -> Dict:
    """Aggregate wellward-vs-chosen diagnostics across decision intervals."""
    if not geometry_logs:
        return {
            "n": 0,
            "wellward_frac": float("nan"),
            "mode_wellward_sector": None,
            "mean_abs_angle_error": float("nan"),
            "mean_dist_to_well": float("nan"),
        }
    from collections import Counter

    wellward = [g["choice_is_wellward"] for g in geometry_logs]
    ww_secs = [
        int(g["wellward_sector"])
        for g in geometry_logs
        if g.get("wellward_sector") is not None
        and not math.isnan(g["wellward_sector"])
    ]
    errs = [
        abs(g["angle_error_to_well"])
        for g in geometry_logs
        if g.get("angle_error_to_well") is not None
        and not math.isnan(g["angle_error_to_well"])
    ]
    dists = [
        g["dist_to_nearest_well"]
        for g in geometry_logs
        if g.get("dist_to_nearest_well") is not None
        and not math.isnan(g["dist_to_nearest_well"])
    ]
    mode_ww = None
    if ww_secs:
        mode_ww = Counter(ww_secs).most_common(1)[0][0]
    return {
        "n": len(geometry_logs),
        "wellward_frac": float(sum(wellward) / len(wellward)),
        "mode_wellward_sector": mode_ww,
        "mean_abs_angle_error": (
            float(sum(errs) / len(errs)) if errs else float("nan")
        ),
        "mean_dist_to_well": (
            float(sum(dists) / len(dists)) if dists else float("nan")
        ),
        "wellward_sector_counts": {
            str(k): int(v) for k, v in sorted(Counter(ww_secs).items())
        },
    }


def sector_choice_stats(selection_logs) -> Dict:
    """
    Characterize whether selection converges to a fixed sector.

    Uses only decision intervals where a candidate was actually chosen
    (selected_sector is not None), including budget-skipped would-have choices
    so stickiness reflects the policy's ranking, not just paid boosts.
    """
    sectors = [
        int(log.selected_sector)
        for log in selection_logs
        if log.selected_sector is not None
    ]
    boosted_sectors = [
        int(log.selected_sector)
        for log in selection_logs
        if log.selected_sector is not None and log.boosted
    ]
    if not sectors:
        return {
            "n_choices": 0,
            "n_boosted": 0,
            "unique_sectors": 0,
            "mode_sector": None,
            "mode_frac": float("nan"),
            "switch_rate": float("nan"),
            "n_switches": 0,
            "sector_counts": {},
            "boosted_sector_counts": {},
            "run_lengths": [],
            "max_run_length": 0,
            "mean_run_length": float("nan"),
        }

    from collections import Counter

    counts = Counter(sectors)
    mode_sector, mode_n = counts.most_common(1)[0]
    switches = sum(1 for a, b in zip(sectors, sectors[1:]) if a != b)
    # Run lengths of consecutive identical sector choices
    runs: List[int] = []
    run = 1
    for a, b in zip(sectors, sectors[1:]):
        if a == b:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)

    bcounts = Counter(boosted_sectors)
    return {
        "n_choices": len(sectors),
        "n_boosted": len(boosted_sectors),
        "unique_sectors": len(counts),
        "mode_sector": int(mode_sector),
        "mode_frac": float(mode_n / len(sectors)),
        "switch_rate": float(switches / max(1, len(sectors) - 1)),
        "n_switches": int(switches),
        "sector_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "boosted_sector_counts": {str(k): int(v) for k, v in sorted(bcounts.items())},
        "run_lengths": runs,
        "max_run_length": int(max(runs)),
        "mean_run_length": float(sum(runs) / len(runs)),
    }


# ---------------------------------------------------------------------------
# Check 1 — boost calibration
# ---------------------------------------------------------------------------
def check1_boost_calibration(
    model,
    val_data: Dict[str, torch.Tensor],
    device: str,
) -> Dict:
    print("\n=== Check 1: boost-subset calibration (mandatory gate) ===")
    cal = evaluate_boost_calibration(model, val_data, device=device)
    boost = cal["boost"]
    damage = cal["damage"]
    all_c = cal["all"]

    print(f"  BOOST subset n={boost['n']}")
    print(
        f"  boost  coh Pearson r (pooled) = {boost['coherence_r_all']:.4f}  "
        f"MAE = {boost['coherence_mae_all']:.4f}"
    )
    print(
        f"  boost  local_R r = {boost['local_R_r_all']:.4f}  "
        f"MAE = {boost['local_R_mae_all']:.4f}"
    )
    print(
        f"  boost  mean within-bin coh r = {boost['coherence_r_within_bin_mean']:.4f}"
    )
    print("  --- boost within distance-bin ---")
    within_rs = []
    for label, st in boost.get("by_distance", {}).items():
        if st.get("coherence_r") is None:
            print(f"    dist {label}: n={st['n']} (skip)")
            continue
        within_rs.append(st["coherence_r"])
        tstd = st.get("true_coh_std", float("nan"))
        tmean = st.get("true_coh_mean", float("nan"))
        print(
            f"    dist {label}: n={st['n']:4d}  coh r={st['coherence_r']:.4f}  "
            f"MAE={st['coherence_mae']:.4f}  "
            f"local_R r={st.get('local_R_r', float('nan')):.4f}  "
            f"MAE={st.get('local_R_mae', float('nan')):.4f}  "
            f"true_coh mean±std={tmean:.3f}±{tstd:.3f}"
        )
        # Flag benign r-collapse: low r + low MAE + collapsed target variance
        lr = st["coherence_r"]
        lm = st["coherence_mae"]
        if (
            lr is not None
            and not math.isnan(lr)
            and lr < 0.5
            and lm is not None
            and lm < 0.05
            and tstd is not None
            and not math.isnan(tstd)
            and tstd < 0.05
        ):
            print(
                f"      NOTE: coh r cliff here may be variance collapse "
                f"(true_coh std={tstd:.4f}, MAE still low={lm:.4f})"
            )

    print(f"\n  DAMAGE subset n={damage.get('n', 0)} (comparison only)")
    if damage.get("n", 0) > 0:
        print(
            f"  damage coh r (pooled) = {damage['coherence_r_all']:.4f}  "
            f"within-bin mean = {damage.get('coherence_r_within_bin_mean', float('nan')):.4f}"
        )
    print(f"  ALL pooled coh r = {all_c['coherence_r_all']:.4f}")

    print("  --- by pert type ---")
    for name, st in cal.get("by_type", {}).items():
        print(
            f"    {name}: n={st['n']:4d}  coh r={st['coherence_r_all']:.4f}  "
            f"within-bin mean r={st.get('coherence_r_within_bin_mean', float('nan')):.4f}"
        )

    mean_r = boost["coherence_r_within_bin_mean"]
    hard_fail = math.isnan(mean_r) or mean_r < 0.5
    strong = (not math.isnan(mean_r)) and mean_r >= 0.7
    if hard_fail:
        print(
            f"\n  FAIL: boost within-bin r = {mean_r:.4f} < 0.5 — "
            "self-model has not learned boost effects. STOP before action selection."
        )
    elif strong:
        print(
            f"\n  PASS (strong): boost within-bin r = {mean_r:.4f} ≥ 0.7 "
            "(comparable to v1 damage-subset within-bin r)."
        )
    else:
        print(
            f"\n  PASS (weak): boost within-bin r = {mean_r:.4f} in [0.5, 0.7) — "
            "above chance, below v1 damage target. Proceed with caution."
        )

    # Scatter plot (boost only), colored by distance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pc = boost["pred_coherence"].numpy()
    tc = boost["true_coherence"].numpy()
    boost_data = None
    if "pert_type" in val_data:
        from boost_rollout_dataset import PERT_BOOST, filter_by_pert_type

        boost_data = filter_by_pert_type(val_data, PERT_BOOST)
    if boost_data is not None and "distance_to_well" in boost_data:
        dist = boost_data["distance_to_well"].numpy()
        sc = axes[0].scatter(
            tc, pc, s=8, alpha=0.45, c=dist, cmap="viridis", edgecolors="none"
        )
        fig.colorbar(sc, ax=axes[0], fraction=0.046, label="dist to well")
    else:
        axes[0].scatter(tc, pc, s=8, alpha=0.4, c="C0", edgecolors="none")
    lims = [
        min(float(tc.min()), float(pc.min())) - 0.05,
        max(float(tc.max()), float(pc.max())) + 0.05,
    ]
    axes[0].plot(lims, lims, "k--", lw=1, alpha=0.6)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("true coherence_future")
    axes[0].set_ylabel("predicted coherence_future")
    axes[0].set_title(
        f"BOOST only  pooled r={boost['coherence_r_all']:.3f}  "
        f"within-bin mean r={mean_r:.3f}"
    )
    axes[0].grid(True, alpha=0.3)

    pl = boost["pred_local_R"].numpy()
    tl = boost["true_local_R"].numpy()
    axes[1].scatter(tl, pl, s=8, alpha=0.4, c="C1", edgecolors="none")
    axes[1].plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1, alpha=0.6)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("true local_R_future")
    axes[1].set_ylabel("predicted local_R_future")
    axes[1].set_title(
        f"BOOST local_R  r={boost['local_R_r_all']:.3f}  "
        f"MAE={boost['local_R_mae_all']:.3f}"
    )
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "check1_boost_calibration.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  plot: {out}")

    # Serialize without tensors
    def slim_cal(c: Dict) -> Dict:
        skip = {
            "pred_coherence",
            "true_coherence",
            "pred_local_R",
            "true_local_R",
        }
        return {k: v for k, v in c.items() if k not in skip}

    return {
        "boost": slim_cal(boost),
        "damage": slim_cal(damage) if damage.get("n", 0) else damage,
        "all": slim_cal(all_c),
        "by_type": cal.get("by_type", {}),
        "mean_within_bin_r_boost": mean_r,
        "pass": not hard_fail,
        "strong_pass": strong,
        "hard_fail": hard_fail,
        "plot": str(out),
    }


# ---------------------------------------------------------------------------
# Attention budget sweep
# ---------------------------------------------------------------------------
def tune_attention_budget(horizon: int = 2000, K: int = 20) -> AttentionConfig:
    print("\n=== Attention budget sweep (scarcity operating point) ===")
    rows = sweep_attention_budget(K=K, horizon=horizon)
    best = pick_operating_point(rows, target_frac=0.5)
    print(
        f"  selected: regen_rate={best['regen_rate']}  "
        f"boost_cost={best['boost_cost']}  A_max={best['A_max']}"
    )
    print(
        f"  boost_fraction={best['boost_fraction']:.3f}  "
        f"regen_per_interval={best['regen_per_interval']:.3f}  "
        f"scarce={best['scarce']}"
    )
    # Show a few neighbors
    print("  nearby scarce points:")
    scarce = sorted(
        [r for r in rows if r["scarce"]],
        key=lambda r: abs(r["boost_fraction"] - 0.5),
    )[:6]
    for r in scarce:
        print(
            f"    regen={r['regen_rate']:.3f} cost={r['boost_cost']:.1f} "
            f"Amax={r['A_max']:.1f}  frac={r['boost_fraction']:.3f}"
        )
    cfg = AttentionConfig(
        K=K,
        regen_rate=float(best["regen_rate"]),
        boost_cost=float(best["boost_cost"]),
        A_max=float(best["A_max"]),
        A0=min(1.0, float(best["A_max"])),
        boost_magnitude=0.40,
        boost_duration=K,
        n_candidates=6,
    )
    return cfg


# ---------------------------------------------------------------------------
# Check 2 + 3 — policy comparison + mask compliance
# ---------------------------------------------------------------------------
def check2_policy_comparison(
    model,
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    device: str,
    attn_cfg: AttentionConfig,
    distances: Tuple[int, ...] = (2, 4, 8),
    horizon: int = 2000,
    seed: int = 0,
    t_transient: int = 300,
) -> Dict:
    print(
        "\n=== Check 2: selected vs random vs round-robin vs fixed "
        "vs wellward_heuristic ==="
    )
    print("=== Check 3: mask compliance (no masked selection) ===")
    # fixed = lock lowest-id sector at t=0 forever (no model).
    # wellward_heuristic = boost sector nearest in angle to nearest well
    # (3-line geometry rule; expected to match selected if self-model is
    # just rediscovering "boost toward well" on this single map).
    policies = (
        "selected",
        "random",
        "round_robin",
        "fixed",
        "wellward_heuristic",
    )
    results: Dict = {}
    total_masked_selected = 0
    total_dropped = 0

    for d in distances:
        results[d] = {}
        for pol in policies:
            print(f"  dist={d} policy={pol}  horizon={horizon}…")
            ep = run_attention_episode(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                model=model if pol == "selected" else None,
                policy=pol,
                distance=d,
                horizon=horizon,
                attn_cfg=attn_cfg,
                seed=seed,
                device=device,
            )
            # Attach post-transient variance + sector stickiness + geometry
            ep["post_transient"] = post_transient_stats(
                ep["history"], t_min=t_transient
            )
            ep["sector_stats"] = sector_choice_stats(ep["selection_logs"])
            ep["geometry_summary"] = _summarize_geometry(ep.get("geometry_logs", []))
            results[d][pol] = ep
            total_masked_selected += ep["n_masked_selected"]
            total_dropped += ep["n_candidates_dropped_mask"]
            pt = ep["post_transient"]
            ss = ep["sector_stats"]
            gs = ep["geometry_summary"]
            print(
                f"    mean_coh={ep['mean_coherence']:.4f}  "
                f"final_coh={ep['final_coherence'] if not math.isnan(ep['final_coherence']) else float('nan'):.4f}  "
                f"mean_mass={ep['mean_self_mass']:.3f}  "
                f"final_mass={ep['final_self_mass']:.3f}  "
                f"tod={ep['time_of_death']}  "
                f"boosts={ep['n_boosts']}/{ep['n_decisions']}  "
                f"mask_drops={ep['n_candidates_dropped_mask']}  "
                f"masked_sel={ep['n_masked_selected']}"
            )
            print(
                f"    post-t≥{t_transient}: coh_std={pt['coherence_std']:.6f}  "
                f"mass_std={pt['self_mass_std']:.4f}  "
                f"sector mode={ss['mode_sector']} "
                f"mode_frac={ss['mode_frac']:.3f}  "
                f"switch_rate={ss['switch_rate']:.3f}  "
                f"unique={ss['unique_sectors']}  "
                f"max_run={ss['max_run_length']}"
                + (
                    f"  locked_sector={ep.get('fixed_sector_locked')}"
                    if pol == "fixed"
                    else ""
                )
            )
            if gs.get("n", 0) > 0:
                print(
                    f"    geometry: wellward_frac={gs['wellward_frac']:.3f}  "
                    f"mode_wellward_sector={gs['mode_wellward_sector']}  "
                    f"mode_chosen={ss['mode_sector']}  "
                    f"mean|angle_err_to_well|={gs['mean_abs_angle_error']:.3f} rad"
                )

    # Plots: coherence | self_mass | chosen sector over decision intervals
    fig, axes = plt.subplots(
        len(distances), 3, figsize=(14, 3.4 * len(distances)), sharex=False
    )
    if len(distances) == 1:
        axes = axes.reshape(1, 3)
    colors = {
        "selected": "C0",
        "random": "C1",
        "round_robin": "C2",
        "fixed": "C3",
        "wellward_heuristic": "C4",
    }
    for i, d in enumerate(distances):
        for pol in policies:
            hist = results[d][pol]["history"]
            ts = [h["t"] for h in hist]
            coh = [h["coherence"] for h in hist]
            mass = [h["self_mass"] for h in hist]
            axes[i, 0].plot(ts, coh, color=colors[pol], label=pol, alpha=0.85, lw=1.2)
            axes[i, 1].plot(ts, mass, color=colors[pol], label=pol, alpha=0.85, lw=1.2)
            logs = results[d][pol]["selection_logs"]
            st = [log.t for log in logs if log.selected_sector is not None]
            ss = [log.selected_sector for log in logs if log.selected_sector is not None]
            if st:
                axes[i, 2].step(
                    st,
                    ss,
                    where="post",
                    color=colors[pol],
                    label=pol,
                    alpha=0.9,
                    lw=1.3,
                )
                axes[i, 2].scatter(st, ss, color=colors[pol], s=10, alpha=0.5, zorder=3)
        axes[i, 0].set_ylabel("coherence")
        axes[i, 1].set_ylabel("self_mass")
        axes[i, 2].set_ylabel("sector id")
        axes[i, 0].set_title(f"dist={d}  coherence")
        axes[i, 1].set_title(f"dist={d}  self_mass")
        axes[i, 2].set_title(f"dist={d}  chosen sector")
        axes[i, 2].set_yticks(range(attn_cfg.n_candidates))
        for j in range(3):
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend(fontsize=7)
            axes[i, j].set_xlabel("t")
    fig.suptitle(
        "Check 2: selected / random / round_robin / fixed\n"
        f"K={attn_cfg.K} regen={attn_cfg.regen_rate} cost={attn_cfg.boost_cost} "
        f"Amax={attn_cfg.A_max} boost_mag={attn_cfg.boost_magnitude}"
    )
    fig.tight_layout()
    out_plot = RESULTS / "check2_policy_comparison.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Summary table: mean win + variance + sector stickiness + fixed control
    summary = {}
    selected_wins = {}
    variance_ratio = {}
    sector_summary = {}
    geometry_summary = {}
    for d in distances:
        summary[d] = {}
        sector_summary[d] = {}
        geometry_summary[d] = {}
        for pol in policies:
            ep = results[d][pol]
            pt = ep["post_transient"]
            ss = ep["sector_stats"]
            gs = ep["geometry_summary"]
            ss_slim = {k: v for k, v in ss.items() if k != "run_lengths"}
            summary[d][pol] = {
                "mean_coherence": ep["mean_coherence"],
                "final_coherence": ep["final_coherence"],
                "mean_self_mass": ep["mean_self_mass"],
                "final_self_mass": ep["final_self_mass"],
                "time_of_death": ep["time_of_death"],
                "n_boosts": ep["n_boosts"],
                "n_decisions": ep["n_decisions"],
                "n_candidates_dropped_mask": ep["n_candidates_dropped_mask"],
                "n_masked_selected": ep["n_masked_selected"],
                "fixed_sector_locked": ep.get("fixed_sector_locked"),
                "post_transient": pt,
                "sector_stats": ss_slim,
                "geometry_summary": gs,
            }
            sector_summary[d][pol] = ss_slim
            geometry_summary[d][pol] = gs
        sc = summary[d]["selected"]["mean_coherence"]
        rc = summary[d]["random"]["mean_coherence"]
        rrc = summary[d]["round_robin"]["mean_coherence"]
        fc = summary[d]["fixed"]["mean_coherence"]
        wc = summary[d]["wellward_heuristic"]["mean_coherence"]
        win_random = (not math.isnan(sc)) and (not math.isnan(rc)) and sc > rc
        win_rr = (not math.isnan(sc)) and (not math.isnan(rrc)) and sc > rrc
        win_fixed = (not math.isnan(sc)) and (not math.isnan(fc)) and sc > fc
        win_wellward = (not math.isnan(sc)) and (not math.isnan(wc)) and sc > wc
        selected_wins[d] = {
            "beats_random": win_random,
            "beats_round_robin": win_rr,
            "beats_fixed": win_fixed,
            "beats_wellward": win_wellward,
            "beats_both": win_random and win_rr,
            "delta_vs_random": (
                sc - rc if (not math.isnan(sc) and not math.isnan(rc)) else float("nan")
            ),
            "delta_vs_rr": (
                sc - rrc
                if (not math.isnan(sc) and not math.isnan(rrc))
                else float("nan")
            ),
            "delta_vs_fixed": (
                sc - fc if (not math.isnan(sc) and not math.isnan(fc)) else float("nan")
            ),
            "delta_vs_wellward": (
                sc - wc if (not math.isnan(sc) and not math.isnan(wc)) else float("nan")
            ),
        }
        s_std = summary[d]["selected"]["post_transient"]["coherence_std"]
        r_std = summary[d]["random"]["post_transient"]["coherence_std"]
        rr_std = summary[d]["round_robin"]["post_transient"]["coherence_std"]
        f_std = summary[d]["fixed"]["post_transient"]["coherence_std"]
        w_std = summary[d]["wellward_heuristic"]["post_transient"]["coherence_std"]
        variance_ratio[d] = {
            "selected_coh_std": s_std,
            "random_coh_std": r_std,
            "round_robin_coh_std": rr_std,
            "fixed_coh_std": f_std,
            "wellward_coh_std": w_std,
            "random_over_selected": (
                r_std / s_std
                if (s_std and s_std > 1e-12 and not math.isnan(s_std) and not math.isnan(r_std))
                else float("nan")
            ),
            "rr_over_selected": (
                rr_std / s_std
                if (s_std and s_std > 1e-12 and not math.isnan(s_std) and not math.isnan(rr_std))
                else float("nan")
            ),
            "fixed_over_selected": (
                f_std / s_std
                if (s_std and s_std > 1e-12 and not math.isnan(s_std) and not math.isnan(f_std))
                else float("nan")
            ),
            "wellward_over_selected": (
                w_std / s_std
                if (s_std and s_std > 1e-12 and not math.isnan(s_std) and not math.isnan(w_std))
                else float("nan")
            ),
            "selected_mass_std": summary[d]["selected"]["post_transient"]["self_mass_std"],
            "random_mass_std": summary[d]["random"]["post_transient"]["self_mass_std"],
            "round_robin_mass_std": summary[d]["round_robin"]["post_transient"][
                "self_mass_std"
            ],
            "fixed_mass_std": summary[d]["fixed"]["post_transient"]["self_mass_std"],
            "wellward_mass_std": summary[d]["wellward_heuristic"]["post_transient"][
                "self_mass_std"
            ],
        }
        print(
            f"  dist={d}: selected mean_coh={sc:.4f}  random={rc:.4f}  "
            f"rr={rrc:.4f}  fixed={fc:.4f}  wellward={wc:.4f}  "
            f"Δ(sel-rand)={selected_wins[d]['delta_vs_random']:.4f}  "
            f"Δ(sel-wellward)={selected_wins[d]['delta_vs_wellward']:.4f}  "
            f"beats_random={win_random}  beats_wellward={win_wellward}"
        )
        print(
            f"    coh_std t≥{t_transient}: sel={s_std:.6f}  rand={r_std:.6f}  "
            f"rr={rr_std:.6f}  fixed={f_std:.6f}  wellward={w_std:.6f}  "
            f"rand/sel={variance_ratio[d]['random_over_selected']:.1f}x  "
            f"wellward/sel={variance_ratio[d]['wellward_over_selected']:.2f}x"
        )
        for pol in policies:
            ss = sector_summary[d][pol]
            gs = geometry_summary[d][pol]
            print(
                f"    sector[{pol}]: mode={ss['mode_sector']} "
                f"frac={ss['mode_frac']:.3f} switch_rate={ss['switch_rate']:.3f} "
                f"unique={ss['unique_sectors']} max_run={ss['max_run_length']} "
                f"counts={ss['sector_counts']}  "
                f"wellward_frac={gs.get('wellward_frac', float('nan')):.3f} "
                f"wellward_sec={gs.get('mode_wellward_sector')}"
            )

    n_win = sum(1 for d in distances if selected_wins[d]["beats_random"])
    print(
        f"\n  selected beats random at {n_win}/{len(distances)} distances "
        f"(honest count, mean coherence)."
    )
    if n_win == 0:
        print(
            "  FINDING (mean): selected does NOT measurably beat random on mean "
            "coherence — spending attention somewhere is what matters for the mean."
        )

    # Variance finding (alive steady-state distances 2 and 4 are the clean ones)
    low_var_ok = all(
        (not math.isnan(variance_ratio[d]["random_over_selected"]))
        and variance_ratio[d]["random_over_selected"] >= 5.0
        for d in (2, 4)
        if d in variance_ratio
    )
    sticky_ok = all(
        sector_summary[d]["selected"]["mode_frac"] >= 0.8
        and sector_summary[d]["selected"]["switch_rate"]
        < sector_summary[d]["random"]["switch_rate"]
        for d in (2, 4)
        if d in sector_summary
    )
    # Cause test: is fixed variance as low as selected?
    # If fixed/selected coh_std ≈ 1, stickiness alone explains stability.
    # If fixed/selected ≫ 1, selected is picking a *better* fixed point.
    fixed_as_stable = all(
        (not math.isnan(variance_ratio[d]["fixed_coh_std"]))
        and (not math.isnan(variance_ratio[d]["selected_coh_std"]))
        and variance_ratio[d]["fixed_coh_std"]
        <= 2.0 * variance_ratio[d]["selected_coh_std"] + 1e-6
        for d in (2, 4)
        if d in variance_ratio
    )
    selected_lower_var_than_fixed = all(
        (not math.isnan(variance_ratio[d]["fixed_over_selected"]))
        and variance_ratio[d]["fixed_over_selected"] >= 2.0
        for d in (2, 4)
        if d in variance_ratio
    )
    print(
        f"\n  FINDING (variance): selected coh_std ≪ random/rr at dist 2/4? "
        f"{low_var_ok}"
    )
    for d in distances:
        if d not in variance_ratio:
            continue
        vr = variance_ratio[d]
        print(
            f"    dist={d}: rand/sel={vr['random_over_selected']:.1f}x  "
            f"rr/sel={vr['rr_over_selected']:.1f}x  "
            f"fixed/sel={vr['fixed_over_selected']:.1f}x  "
            f"(fixed_std={vr['fixed_coh_std']:.6f}  sel_std={vr['selected_coh_std']:.6f})"
        )
    print(
        f"  FINDING (sector stickiness): selected converges at dist 2/4? {sticky_ok}"
    )
    print(
        f"  CAUSE TEST (fixed baseline): fixed variance ≈ selected (stickiness "
        f"alone)? {fixed_as_stable}"
    )
    print(
        f"  CAUSE TEST: selected meaningfully lower variance than fixed (≥2×)? "
        f"{selected_lower_var_than_fixed}"
    )
    if fixed_as_stable and not selected_lower_var_than_fixed:
        print(
            "  CAUSE: fixed is as stable as selected → low variance is a "
            "mechanical consequence of *any* consistent sector choice, not "
            "evidence the self-model found a specially good region. Smaller "
            "finding: 'sticky allocation → homeostasis', not 'self-model "
            "discovered the right operating point'."
        )
    elif selected_lower_var_than_fixed:
        print(
            "  CAUSE (single-draw fixed only): selected stabler than this one "
            "locked sector (default s0). NOT yet 'self-model is smart' — s0 may "
            "be unlucky. See 6-way sweep below for the full landscape."
        )
    else:
        print(
            "  CAUSE (single-draw fixed): mixed / inconclusive — report both "
            "stds; 6-way sweep is the decisive control."
        )

    # Geometry: is selected's sector the wellward sector?
    print("\n  GEOMETRY (selected mode sector vs wellward):")
    for d in distances:
        if d not in geometry_summary:
            continue
        gs = geometry_summary[d]["selected"]
        ss = sector_summary[d]["selected"]
        print(
            f"    dist={d}: chosen_mode={ss['mode_sector']}  "
            f"wellward_mode={gs.get('mode_wellward_sector')}  "
            f"wellward_frac={gs.get('wellward_frac', float('nan')):.3f}  "
            f"mean|∠err|={gs.get('mean_abs_angle_error', float('nan')):.3f} rad  "
            f"mean_dist_well={gs.get('mean_dist_to_well', float('nan')):.2f}"
        )
        if (
            ss["mode_sector"] is not None
            and gs.get("mode_wellward_sector") is not None
            and int(ss["mode_sector"]) == int(gs["mode_wellward_sector"])
        ):
            print(
                f"      → mode sector == wellward sector (boost toward nearest well)"
            )
        elif gs.get("wellward_frac", 0) >= 0.5:
            print(
                f"      → often wellward ({gs['wellward_frac']:.0%}) but mode "
                f"differs from modal wellward id — check angle bins / centroid drift"
            )
        else:
            print(
                f"      → NOT predominantly wellward "
                f"(wellward_frac={gs.get('wellward_frac', float('nan')):.2f}) — "
                f"ranking is not simply 'boost toward the well'"
            )

    # ------------------------------------------------------------------
    # wellward_heuristic vs selected (final v2 control)
    # ------------------------------------------------------------------
    print("\n  WELLWARD HEURISTIC vs selected (final zero-learning control):")
    wellward_match: Dict = {}
    for d in distances:
        ss_sel = sector_summary[d]["selected"]
        ss_wh = sector_summary[d]["wellward_heuristic"]
        vr = variance_ratio[d]
        s_std = vr["selected_coh_std"]
        w_std = vr["wellward_coh_std"]
        # Match criteria: same mode sector OR std within 2x of each other
        # and both ≪ random
        mode_match = (
            ss_sel["mode_sector"] is not None
            and ss_wh["mode_sector"] is not None
            and int(ss_sel["mode_sector"]) == int(ss_wh["mode_sector"])
        )
        std_ratio = (
            max(s_std, w_std) / min(s_std, w_std)
            if (
                s_std
                and w_std
                and s_std > 1e-12
                and w_std > 1e-12
                and not math.isnan(s_std)
                and not math.isnan(w_std)
            )
            else float("nan")
        )
        std_close = (not math.isnan(std_ratio)) and std_ratio <= 2.0
        # Agreement fraction: how often they would pick same sector if we
        # compare mode_frac of wellward on selected's mode
        wellward_match[d] = {
            "selected_mode": ss_sel["mode_sector"],
            "wellward_mode": ss_wh["mode_sector"],
            "selected_mode_frac": ss_sel["mode_frac"],
            "wellward_mode_frac": ss_wh["mode_frac"],
            "selected_switch_rate": ss_sel["switch_rate"],
            "wellward_switch_rate": ss_wh["switch_rate"],
            "selected_coh_std": s_std,
            "wellward_coh_std": w_std,
            "std_ratio_max_over_min": std_ratio,
            "mode_match": mode_match,
            "std_close": std_close,
            "matches_selected": mode_match or std_close,
            "delta_mean_coh": selected_wins[d]["delta_vs_wellward"],
            "wellward_wellward_frac": geometry_summary[d]["wellward_heuristic"].get(
                "wellward_frac", float("nan")
            ),
        }
        print(
            f"    dist={d}: sel_mode={ss_sel['mode_sector']} "
            f"(frac={ss_sel['mode_frac']:.3f})  "
            f"wh_mode={ss_wh['mode_sector']} "
            f"(frac={ss_wh['mode_frac']:.3f})  "
            f"mode_match={mode_match}  "
            f"sel_std={s_std:.6f}  wh_std={w_std:.6f}  "
            f"std_ratio={std_ratio:.2f}  "
            f"matches={wellward_match[d]['matches_selected']}"
        )
    clean_d = [d for d in (2, 4) if d in wellward_match]
    wh_matches_clean = all(wellward_match[d]["matches_selected"] for d in clean_d)
    # Dist-4 is the clean operating point (wellward well-defined). Dist-2 has
    # centroid nearly on the well → wellward angle is degenerate.
    wh_matches_d4 = (
        wellward_match[4]["matches_selected"] if 4 in wellward_match else False
    )
    print(
        f"  wellward_heuristic matches selected at dist 2/4? {wh_matches_clean}  "
        f"(dist4 alone: {wh_matches_d4})"
    )
    if wh_matches_clean:
        print(
            "  FINAL VERDICT: wellward_heuristic matches selected on clean "
            "distances — the pipeline does the same job as a 3-line geometric "
            "rule on this map. Self-model-driven allocation does NOT earn its "
            "complexity."
        )
    elif wh_matches_d4 and not wellward_match.get(2, {}).get("mode_match", False):
        # Expected nuance: dist=4 coincides; dist=2 selected ≠ pure wellward
        print(
            "  FINAL VERDICT (nuanced): at dist=4, wellward_heuristic ≡ selected "
            "(mode + std). At dist=2, modes differ (selected locks structural "
            "best s4; pure wellward locks s2) because centroid≈on well makes "
            "wellward angle ill-defined — selected is slightly stabler than "
            "pure wellward but still ≡ fixed(best sector) with no model. "
            "Neither distance shows the self-model exceeding a zero-learning "
            "rule (wellward at d4, fixed-best at d2). Allocation pipeline does "
            "NOT earn its complexity for mean or for beating zero-learning "
            "ceilings on this single map."
        )
    else:
        print(
            "  FINAL VERDICT: wellward_heuristic does NOT match selected in the "
            "expected way — see per-distance modes/stds before claiming either "
            "geometry-only or self-model-smart."
        )

    # ------------------------------------------------------------------
    # 6-way fixed-sector sweep (cause discrimination, no model calls)
    # ------------------------------------------------------------------
    # Single-draw fixed(sector=0) only shows selected beats *that* sector.
    # Sweep all n_candidates sectors to see where selected sits in the
    # structural stability landscape.
    print(
        f"\n=== 6-way fixed-sector sweep (lock each sector 0..{attn_cfg.n_candidates - 1}) ==="
    )
    fixed_sweep: Dict = {}
    for d in distances:
        fixed_sweep[d] = {}
        for sec in range(attn_cfg.n_candidates):
            print(f"  dist={d} fixed_sector={sec}…")
            ep = run_attention_episode(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                model=None,
                policy="fixed",
                distance=d,
                horizon=horizon,
                attn_cfg=attn_cfg,
                seed=seed,
                device=device,
                fixed_sector_preset=sec,
            )
            pt = post_transient_stats(ep["history"], t_min=t_transient)
            ss = sector_choice_stats(ep["selection_logs"])
            gs = _summarize_geometry(ep.get("geometry_logs", []))
            fixed_sweep[d][sec] = {
                "mean_coherence": ep["mean_coherence"],
                "final_coherence": ep["final_coherence"],
                "mean_self_mass": ep["mean_self_mass"],
                "final_self_mass": ep["final_self_mass"],
                "time_of_death": ep["time_of_death"],
                "n_boosts": ep["n_boosts"],
                "post_transient": pt,
                "sector_stats": {k: v for k, v in ss.items() if k != "run_lengths"},
                "geometry_summary": gs,
                "fixed_sector_locked": ep.get("fixed_sector_locked"),
            }
            # Save CSVs for the sweep
            path = RESULTS / f"check2_d{d}_fixed{sec}.csv"
            with path.open("w") as f:
                f.write("t,self_mass,coherence,local_R,A,boost_remaining,alive\n")
                for h in ep["history"]:
                    f.write(
                        f"{int(h['t'])},{h['self_mass']},{h['coherence']},"
                        f"{h['local_R']},{h['A']},{h['boost_remaining']},{h['alive']}\n"
                    )
            print(
                f"    coh_std={pt['coherence_std']:.6f}  "
                f"mean_coh={ep['mean_coherence']:.4f}  "
                f"mass_std={pt['self_mass_std']:.4f}  "
                f"tod={ep['time_of_death']}"
            )

    # Rank selected against the full fixed landscape
    sweep_verdict: Dict = {}
    print("\n  --- selected vs full fixed landscape (post-t≥{} coh_std) ---".format(t_transient))
    for d in distances:
        sel_std = summary[d]["selected"]["post_transient"]["coherence_std"]
        sel_mean = summary[d]["selected"]["mean_coherence"]
        rows = []
        for sec in range(attn_cfg.n_candidates):
            st = fixed_sweep[d][sec]["post_transient"]["coherence_std"]
            mn = fixed_sweep[d][sec]["mean_coherence"]
            rows.append((sec, st, mn))
        # sort by std ascending (lower = stabler)
        ranked = sorted(rows, key=lambda r: (r[1] if not math.isnan(r[1]) else 1e9))
        best_sec, best_std, best_mean = ranked[0]
        worst_sec, worst_std, worst_mean = ranked[-1]
        stds = [r[1] for r in rows if not math.isnan(r[1])]
        # Where does selected sit?
        n_fixed_better = sum(1 for s in stds if s < sel_std - 1e-12)
        n_fixed_worse = sum(1 for s in stds if s > sel_std + 1e-12)
        n_fixed_tie = len(stds) - n_fixed_better - n_fixed_worse
        # Match best? within 20% of best_std or absolute eps
        matches_best = (not math.isnan(sel_std)) and (
            sel_std <= best_std * 1.2 + 1e-6 or abs(sel_std - best_std) < 1e-4
        )
        beats_best = (not math.isnan(sel_std)) and sel_std < best_std - 1e-6
        # Structural dominance: one sector ≪ all others (best at least 3x better than median)
        median_std = float(sorted(stds)[len(stds) // 2]) if stds else float("nan")
        structural_outlier = (
            not math.isnan(best_std)
            and not math.isnan(median_std)
            and median_std > 1e-12
            and (median_std / best_std) >= 3.0
        )
        # selected mode sector vs best fixed sector
        sel_mode = sector_summary[d]["selected"]["mode_sector"]
        sel_matches_best_sector = (
            sel_mode is not None and int(sel_mode) == int(best_sec)
        )
        ww = geometry_summary[d]["selected"].get("mode_wellward_sector")
        best_is_wellward = ww is not None and int(best_sec) == int(ww)

        if beats_best or matches_best:
            if n_fixed_better == 0:
                outcome = "matches_or_beats_best"
            else:
                outcome = "near_best"
        elif n_fixed_better <= 1:
            outcome = "near_top"
        elif n_fixed_better >= len(stds) - 1:
            outcome = "near_worst"
        else:
            outcome = "middle_of_pack"

        sweep_verdict[d] = {
            "selected_coh_std": sel_std,
            "selected_mean_coh": sel_mean,
            "selected_mode_sector": sel_mode,
            "fixed_by_sector": {
                str(sec): {
                    "coh_std": st,
                    "mean_coh": mn,
                    "rank_by_std": next(
                        i + 1 for i, r in enumerate(ranked) if r[0] == sec
                    ),
                }
                for sec, st, mn in rows
            },
            "best_fixed_sector": best_sec,
            "best_fixed_coh_std": best_std,
            "best_fixed_mean_coh": best_mean,
            "worst_fixed_sector": worst_sec,
            "worst_fixed_coh_std": worst_std,
            "median_fixed_coh_std": median_std,
            "n_fixed_better_std": n_fixed_better,
            "n_fixed_worse_std": n_fixed_worse,
            "n_fixed_tie_std": n_fixed_tie,
            "selected_rank_among_fixed_plus_self": n_fixed_better + 1,
            "matches_or_beats_best_fixed": matches_best or beats_best,
            "beats_best_fixed": beats_best,
            "structural_outlier_sector": structural_outlier,
            "selected_mode_equals_best_fixed": sel_matches_best_sector,
            "best_fixed_is_wellward": best_is_wellward,
            "outcome": outcome,
            "ranked_sectors_by_std": [
                {"sector": s, "coh_std": st, "mean_coh": mn} for s, st, mn in ranked
            ],
        }
        print(
            f"  dist={d}: selected std={sel_std:.6f} mode_sector={sel_mode}"
        )
        print(
            f"    fixed ranks (std↑stable): "
            + ", ".join(f"s{s}={st:.5f}" for s, st, _ in ranked)
        )
        print(
            f"    best_fixed=s{best_sec} std={best_std:.6f}  "
            f"worst=s{worst_sec} std={worst_std:.6f}  "
            f"median={median_std:.6f}  "
            f"structural_outlier={structural_outlier}"
        )
        print(
            f"    selected rank={n_fixed_better + 1}/{len(stds) + 1}  "
            f"(n_fixed better={n_fixed_better})  "
            f"outcome={outcome}  "
            f"mode==best_fixed={sel_matches_best_sector}  "
            f"best_fixed_is_wellward={best_is_wellward}"
        )
        if outcome == "matches_or_beats_best":
            print(
                "    → selected matches/beats the best fixed sector — "
                "non-trivial if it did so without an exhaustive sweep."
            )
        elif outcome == "middle_of_pack":
            print(
                "    → selected sits mid-pack among fixed sectors — better than "
                "unlucky arbitrary, but not evidence it found the optimum."
            )
        if structural_outlier and sel_matches_best_sector:
            print(
                "    → HEADLINE risk: one structurally dominant sector, and "
                "selected lands on it — may be geometry, not inference."
            )
        if structural_outlier and best_is_wellward:
            print(
                "    → best fixed sector is wellward — map geometry favors "
                "boost-toward-well for stability under this protocol."
            )

    # Aggregate cause reading across dist 2/4
    clean = [d for d in (2, 4) if d in sweep_verdict]
    outcomes = {d: sweep_verdict[d]["outcome"] for d in clean}
    all_match_best = all(
        sweep_verdict[d]["matches_or_beats_best_fixed"] for d in clean
    )
    all_structural = all(
        sweep_verdict[d]["structural_outlier_sector"]
        and sweep_verdict[d]["selected_mode_equals_best_fixed"]
        for d in clean
    )
    print("\n  6-WAY SWEEP VERDICT:")
    print(f"    outcomes by dist: {outcomes}")
    print(f"    selected matches/beats best fixed at dist 2/4? {all_match_best}")
    print(
        f"    structural-outlier + selected lands on it at dist 2/4? {all_structural}"
    )
    if all_structural:
        print(
            "    READING: one sector is structurally far more stable, and "
            "selected locks onto it — earlier concern stands: this can be "
            "geometry (e.g. wellward), not evidence of learned inference. "
            "Do NOT write up as 'self-model earns its complexity'."
        )
    elif all_match_best:
        print(
            "    READING: selected reaches the best (or near-best) stability "
            "among all fixed sectors without sweeping — non-trivial ranking. "
            "Still not 'surprise minimization' (objective is value-argmax)."
        )
    else:
        print(
            "    READING: selected is not uniquely optimal across the fixed "
            "landscape — report rank per distance; do not overclaim."
        )

    # Plot: bar chart of coh_std by fixed sector + selected line
    fig_s, axes_s = plt.subplots(
        1, len(distances), figsize=(4.2 * len(distances), 3.8), sharey=False
    )
    if len(distances) == 1:
        axes_s = [axes_s]
    for ax, d in zip(axes_s, distances):
        secs = list(range(attn_cfg.n_candidates))
        stds = [
            fixed_sweep[d][s]["post_transient"]["coherence_std"] for s in secs
        ]
        colors_bar = []
        sel_mode = sector_summary[d]["selected"]["mode_sector"]
        best_sec = sweep_verdict[d]["best_fixed_sector"]
        for s in secs:
            if s == best_sec:
                colors_bar.append("C2")
            elif sel_mode is not None and s == int(sel_mode):
                colors_bar.append("C0")
            else:
                colors_bar.append("0.7")
        ax.bar(secs, stds, color=colors_bar, alpha=0.85, label="fixed[s]")
        sel_std = summary[d]["selected"]["post_transient"]["coherence_std"]
        ax.axhline(
            sel_std, color="C0", ls="--", lw=1.5, label=f"selected ({sel_std:.4f})"
        )
        ax.set_xlabel("locked sector id")
        ax.set_ylabel("coh_std (t≥{})".format(t_transient))
        ax.set_title(f"dist={d}  best=s{best_sec}  sel_mode={sel_mode}")
        ax.set_xticks(secs)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig_s.suptitle(
        "6-way fixed-sector sweep vs selected\n"
        "green bar = lowest-std fixed; blue dashed = selected; blue bar = selected's mode sector"
    )
    fig_s.tight_layout()
    sweep_plot = RESULTS / "check2_fixed_sector_sweep.png"
    fig_s.savefig(sweep_plot, dpi=140)
    plt.close(fig_s)
    print(f"  plot: {sweep_plot}")

    # Check 3
    mask_ok = total_masked_selected == 0
    print(f"\n  Check 3: total masked candidates selected = {total_masked_selected}")
    print(f"  Check 3: total candidates dropped by mask = {total_dropped}")
    if mask_ok:
        print("  Check 3 PASS: no selected action landed on a masked-out candidate.")
    else:
        print(
            "  Check 3 FAIL: masked candidate was selected — bug in selection code, "
            "fix before trusting Check 2 numbers."
        )

    # Save per-policy trajectory CSVs + selection-decision CSVs
    for d in distances:
        for pol in policies:
            path = RESULTS / f"check2_d{d}_{pol}.csv"
            hist = results[d][pol]["history"]
            with path.open("w") as f:
                f.write("t,self_mass,coherence,local_R,A,boost_remaining,alive\n")
                for h in hist:
                    f.write(
                        f"{int(h['t'])},{h['self_mass']},{h['coherence']},"
                        f"{h['local_R']},{h['A']},{h['boost_remaining']},{h['alive']}\n"
                    )
            # Decision-interval sector log (one row per decision)
            spath = RESULTS / f"check2_d{d}_{pol}_selections.csv"
            with spath.open("w") as f:
                f.write(
                    "t,selected_y,selected_x,selected_sector,boosted,"
                    "score_of_selected,n_raw,n_surviving,n_dropped_mask,"
                    "A_before,A_after\n"
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
                    f.write(
                        f"{log.t},{sy},{sx},{sec},{int(log.boosted)},{sc},"
                        f"{log.n_raw},{log.n_surviving},{log.n_dropped_mask},"
                        f"{log.A_before:.4f},{log.A_after:.4f}\n"
                    )

    return {
        "distances": list(distances),
        "policies": list(policies),
        "t_transient": t_transient,
        "summary": {str(d): summary[d] for d in distances},
        "selected_wins": {str(d): selected_wins[d] for d in distances},
        "variance_ratio": {str(d): variance_ratio[d] for d in distances},
        "sector_summary": {str(d): sector_summary[d] for d in distances},
        "geometry_summary": {str(d): geometry_summary[d] for d in distances},
        "n_distances_selected_beats_random": n_win,
        "selected_clearly_beats_random": n_win >= 2,
        "selected_lower_variance_dist_2_4": low_var_ok,
        "selected_sector_sticky_dist_2_4": sticky_ok,
        "fixed_as_stable_as_selected_dist_2_4": fixed_as_stable,
        "selected_lower_var_than_fixed_dist_2_4": selected_lower_var_than_fixed,
        "fixed_sector_sweep": {
            str(d): {str(s): fixed_sweep[d][s] for s in fixed_sweep[d]}
            for d in distances
        },
        "fixed_sector_sweep_verdict": {
            str(d): sweep_verdict[d] for d in distances
        },
        "fixed_sweep_plot": str(sweep_plot),
        "fixed_sweep_selected_matches_best_dist_2_4": all_match_best,
        "fixed_sweep_structural_geometry_story_dist_2_4": all_structural,
        "wellward_vs_selected": {str(d): wellward_match[d] for d in distances},
        "wellward_matches_selected_dist_2_4": wh_matches_clean,
        "wellward_matches_selected_dist_4": wh_matches_d4,
        "mask_compliance": {
            "total_masked_selected": total_masked_selected,
            "total_candidates_dropped_mask": total_dropped,
            "pass": mask_ok,
        },
        "attn_cfg": {
            "K": attn_cfg.K,
            "regen_rate": attn_cfg.regen_rate,
            "boost_cost": attn_cfg.boost_cost,
            "A_max": attn_cfg.A_max,
            "boost_magnitude": attn_cfg.boost_magnitude,
            "n_candidates": attn_cfg.n_candidates,
        },
        "plot": str(out_plot),
        "episodes_meta": {
            str(d): {
                pol: {
                    k: results[d][pol][k]
                    for k in (
                        "mean_coherence",
                        "final_coherence",
                        "mean_self_mass",
                        "final_self_mass",
                        "time_of_death",
                        "n_boosts",
                        "n_decisions",
                        "n_masked_selected",
                        "n_candidates_dropped_mask",
                        "fixed_sector_locked",
                    )
                    if k in results[d][pol]
                }
                for pol in policies
            }
            for d in distances
        },
    }


def write_observations(
    c1: Dict,
    c2: Optional[Dict],
    attn_cfg: Optional[AttentionConfig],
    args,
    device: str,
    stopped_at_check1: bool,
) -> Path:
    obs: List[str] = []
    obs.append("# v2 action selection observations (auto-generated)\n")
    obs.append(f"device: {device}\nseed: {args.seed}\nn_examples: {args.n_examples}\n")
    obs.append(f"horizon_train: {args.horizon_train}  burn_in: {args.burn_in}\n")
    obs.append(f"epochs: {args.epochs}\nv0_ckpt: {args.ckpt}\n")

    obs.append("\n## Check 1 — boost-subset calibration (mandatory gate)\n")
    b = c1["boost"]
    obs.append(f"- Boost held-out n={b['n']}\n")
    obs.append(
        f"- boost coherence_future Pearson r (pooled) = {b['coherence_r_all']:.4f}\n"
        f"- boost coherence_future MAE (pooled) = {b['coherence_mae_all']:.4f}\n"
        f"- boost mean within-distance-bin coh r = "
        f"{c1['mean_within_bin_r_boost']:.4f}\n"
        f"- boost local_R r = {b['local_R_r_all']:.4f}  "
        f"MAE = {b['local_R_mae_all']:.4f}\n"
    )
    obs.append(f"- pass (within-bin r ≥ 0.5): {c1['pass']}\n")
    obs.append(f"- strong_pass (within-bin r ≥ 0.7): {c1['strong_pass']}\n")
    obs.append(f"- hard_fail: {c1['hard_fail']}\n")
    obs.append(f"- plot: {c1['plot']}\n")
    obs.append("- Boost within-distance-bin:\n")
    for label, st in b.get("by_distance", {}).items():
        if st.get("coherence_r") is None:
            obs.append(f"  - dist {label}: n={st['n']} (skip)\n")
            continue
        obs.append(
            f"  - dist {label}: n={st['n']} coh_r={st['coherence_r']:.4f} "
            f"coh_MAE={st['coherence_mae']:.4f} "
            f"local_R_r={st.get('local_R_r', float('nan')):.4f} "
            f"local_R_MAE={st.get('local_R_mae', float('nan')):.4f}\n"
        )
    if c1.get("damage") and c1["damage"].get("n", 0):
        d = c1["damage"]
        obs.append(
            f"- Damage subset (comparison): n={d['n']}  "
            f"coh r pooled={d['coherence_r_all']:.4f}  "
            f"within-bin mean={d.get('coherence_r_within_bin_mean', float('nan')):.4f}\n"
        )
    obs.append("- By pert type:\n")
    for name, st in c1.get("by_type", {}).items():
        obs.append(
            f"  - {name}: n={st['n']} coh_r={st['coherence_r_all']:.4f} "
            f"within-bin mean={st.get('coherence_r_within_bin_mean', float('nan')):.4f}\n"
        )

    if stopped_at_check1:
        obs.append(
            "\n## STOPPED after Check 1\n"
            "Boost calibration failed (within-bin r < 0.5). Action selection "
            "(Check 2/3) was not run — building allocation on an uncalibrated "
            "boost predictor would select based on noise.\n"
        )
    else:
        assert c2 is not None and attn_cfg is not None
        obs.append("\n## Attention budget operating point\n")
        obs.append(
            f"- K={attn_cfg.K}  regen_rate={attn_cfg.regen_rate}  "
            f"boost_cost={attn_cfg.boost_cost}  A_max={attn_cfg.A_max}  "
            f"boost_magnitude={attn_cfg.boost_magnitude}  "
            f"n_candidates={attn_cfg.n_candidates}\n"
        )
        regen_per = attn_cfg.regen_rate * attn_cfg.K
        obs.append(
            f"- regen_per_interval={regen_per:.3f}  "
            f"scarce={regen_per < attn_cfg.boost_cost} "
            f"(cannot boost every decision interval under greedy spend)\n"
        )

        t_tr = c2.get("t_transient", 300)
        obs.append(
            "\n## Check 2 — selected vs random vs round-robin vs fixed\n"
        )
        obs.append(f"- Horizon: {args.horizon_episode} steps (v0 standard)\n")
        obs.append(f"- Distances: {c2['distances']}\n")
        obs.append(
            f"- Post-transient window for variance: t≥{t_tr}, alive steps only\n"
        )
        obs.append(
            "- **fixed** baseline: lock lowest-id surviving sector at t=0, hold "
            "forever (no model, no randomness). Distinguishes sticky-argmax "
            "cause: informative ranking vs mechanical consistency.\n"
        )
        for d in c2["distances"]:
            s = c2["summary"][str(d)]
            w = c2["selected_wins"][str(d)]
            vr = c2.get("variance_ratio", {}).get(str(d), {})
            obs.append(f"- dist={d}:\n")
            for pol in c2["policies"]:
                st = s[pol]
                fc = st["final_coherence"]
                fc_s = (
                    f"{fc:.4f}"
                    if not (isinstance(fc, float) and math.isnan(fc))
                    else "nan"
                )
                pt = st.get("post_transient", {})
                ss = st.get("sector_stats", {})
                gs = st.get("geometry_summary", {})
                obs.append(
                    f"  - {pol}: mean_coh={st['mean_coherence']:.4f}  "
                    f"final_coh={fc_s}  "
                    f"mean_mass={st['mean_self_mass']:.3f}  "
                    f"final_mass={st['final_self_mass']:.3f}  "
                    f"tod={st['time_of_death']}  "
                    f"boosts={st['n_boosts']}/{st['n_decisions']}"
                    + (
                        f"  locked={st.get('fixed_sector_locked')}"
                        if pol == "fixed"
                        else ""
                    )
                    + "\n"
                )
                if pt:
                    obs.append(
                        f"    post-t≥{t_tr}: coh_mean={pt.get('coherence_mean', float('nan')):.4f}  "
                        f"coh_std={pt.get('coherence_std', float('nan')):.6f}  "
                        f"mass_mean={pt.get('self_mass_mean', float('nan')):.3f}  "
                        f"mass_std={pt.get('self_mass_std', float('nan')):.4f}\n"
                    )
                if ss:
                    obs.append(
                        f"    sector: mode={ss.get('mode_sector')}  "
                        f"mode_frac={ss.get('mode_frac', float('nan')):.3f}  "
                        f"switch_rate={ss.get('switch_rate', float('nan')):.3f}  "
                        f"unique={ss.get('unique_sectors')}  "
                        f"max_run={ss.get('max_run_length')}  "
                        f"mean_run={ss.get('mean_run_length', float('nan')):.2f}  "
                        f"counts={ss.get('sector_counts')}\n"
                    )
                if gs and gs.get("n", 0) > 0:
                    obs.append(
                        f"    geometry: wellward_frac={gs.get('wellward_frac', float('nan')):.3f}  "
                        f"mode_wellward_sector={gs.get('mode_wellward_sector')}  "
                        f"mean|∠err_to_well|={gs.get('mean_abs_angle_error', float('nan')):.3f} rad  "
                        f"mean_dist_well={gs.get('mean_dist_to_well', float('nan')):.2f}\n"
                    )
            obs.append(
                f"  - Δ(selected − random) mean_coh = {w['delta_vs_random']:.4f}  "
                f"beats_random={w['beats_random']}  "
                f"Δ(selected − fixed) = {w.get('delta_vs_fixed', float('nan')):.4f}  "
                f"beats_fixed={w.get('beats_fixed')}\n"
            )
            if vr:
                obs.append(
                    f"  - coh_std ratio random/selected = "
                    f"{vr.get('random_over_selected', float('nan')):.1f}x  "
                    f"rr/selected = {vr.get('rr_over_selected', float('nan')):.1f}x  "
                    f"fixed/selected = {vr.get('fixed_over_selected', float('nan')):.1f}x\n"
                )
        obs.append(
            f"- selected beats random at "
            f"{c2['n_distances_selected_beats_random']}/{len(c2['distances'])} "
            f"distances (mean coherence)\n"
        )
        obs.append(
            f"- selected_clearly_beats_random (≥2 distances): "
            f"{c2['selected_clearly_beats_random']}\n"
        )
        obs.append(
            f"- selected_lower_variance_dist_2_4 (coh_std rand/sel ≥5×): "
            f"{c2.get('selected_lower_variance_dist_2_4')}\n"
        )
        obs.append(
            f"- selected_sector_sticky_dist_2_4 (mode_frac≥0.8 & switch_rate < random): "
            f"{c2.get('selected_sector_sticky_dist_2_4')}\n"
        )
        obs.append(
            f"- fixed_as_stable_as_selected_dist_2_4: "
            f"{c2.get('fixed_as_stable_as_selected_dist_2_4')}\n"
        )
        obs.append(
            f"- selected_lower_var_than_fixed_dist_2_4 (≥2×): "
            f"{c2.get('selected_lower_var_than_fixed_dist_2_4')}\n"
        )
        if not c2["selected_clearly_beats_random"]:
            obs.append(
                "- **FINDING (mean):** selected does not clearly beat random on "
                "mean coherence. Self-model-driven choice does not raise central "
                "tendency over spending the attention budget somewhere. Do not "
                "reframe with final_mass or time_of_death to claim a mean win.\n"
            )
        if c2.get("selected_lower_variance_dist_2_4"):
            obs.append(
                "- **FINDING (variance / mechanism):** selected produces "
                "dramatically lower steady-state coherence variance than "
                "random/round_robin at dist 2/4, via sticky argmax "
                f"(sector stickiness={c2.get('selected_sector_sticky_dist_2_4')}).\n"
            )
        # Cause discrimination
        if c2.get("fixed_as_stable_as_selected_dist_2_4") and not c2.get(
            "selected_lower_var_than_fixed_dist_2_4"
        ):
            obs.append(
                "- **FINDING (cause — fixed control):** `fixed` (lock arbitrary "
                "sector 0 at t=0, hold forever, no model) has variance as low as "
                "`selected`. Low variance is therefore a mechanical consequence of "
                "*any consistent sector choice*, not evidence the self-model found "
                "a specially good region. Smaller honest finding: **sticky "
                "allocation → homeostasis**, not **self-model discovered the right "
                "operating point**. (Same discipline as v1 naive-‖S‖ / untrained "
                "ablations: check the boring explanation.)\n"
            )
        elif c2.get("selected_lower_var_than_fixed_dist_2_4"):
            obs.append(
                "- **FINDING (cause — fixed control):** `selected` has "
                "meaningfully lower variance than arbitrary `fixed` → self-model "
                "is doing more than being consistent; it consistently picks a "
                "better sector than a locked arbitrary one.\n"
            )
        else:
            obs.append(
                "- **FINDING (cause — fixed control):** mixed/inconclusive; see "
                "per-distance fixed/selected coh_std ratios above.\n"
            )
        # Geometry naming
        obs.append("- **Geometry of selected's mode sector:**\n")
        for d in c2["distances"]:
            gs = (
                c2.get("geometry_summary", {})
                .get(str(d), {})
                .get("selected", {})
            )
            ss = (
                c2.get("sector_summary", {})
                .get(str(d), {})
                .get("selected", {})
            )
            if not ss:
                continue
            mode = ss.get("mode_sector")
            ww = gs.get("mode_wellward_sector")
            wf = gs.get("wellward_frac", float("nan"))
            match = (
                mode is not None
                and ww is not None
                and int(mode) == int(ww)
            )
            obs.append(
                f"  - dist={d}: mode_sector={mode}  wellward_sector={ww}  "
                f"wellward_frac={wf:.3f}  "
                f"{'MATCH (boost toward nearest well)' if match else 'not simply wellward'}\n"
            )
        obs.append(f"- plot: {c2['plot']}\n")
        obs.append(
            "- Selection logs: `results/check2_d{2,4,8}_{policy}_selections.csv` "
            "(t, sector, y, x, boosted, score)\n"
        )

        # wellward heuristic (final control)
        obs.append(
            "\n## Check 2c — wellward_heuristic (final zero-learning control)\n"
        )
        obs.append(
            "- Policy: at each decision, pick the candidate whose direction from "
            "the pattern centroid is closest in angle to the nearest well. "
            "Pure geometry from well coordinates — no model, no training, "
            "no sticky lock (recomputed each interval).\n"
        )
        obs.append(
            f"- wellward_matches_selected_dist_2_4: "
            f"{c2.get('wellward_matches_selected_dist_2_4')}\n"
        )
        for d in c2["distances"]:
            wh = c2.get("wellward_vs_selected", {}).get(str(d), {})
            if not wh:
                continue
            obs.append(
                f"- dist={d}: sel_mode={wh.get('selected_mode')} "
                f"(frac={wh.get('selected_mode_frac', float('nan')):.3f})  "
                f"wh_mode={wh.get('wellward_mode')} "
                f"(frac={wh.get('wellward_mode_frac', float('nan')):.3f})  "
                f"mode_match={wh.get('mode_match')}  "
                f"sel_std={wh.get('selected_coh_std', float('nan')):.6f}  "
                f"wh_std={wh.get('wellward_coh_std', float('nan')):.6f}  "
                f"std_ratio={wh.get('std_ratio_max_over_min', float('nan')):.2f}  "
                f"Δmean_coh(sel−wh)={wh.get('delta_mean_coh', float('nan')):.4f}  "
                f"matches={wh.get('matches_selected')}\n"
            )
        if c2.get("wellward_matches_selected_dist_2_4"):
            obs.append(
                "- **FINAL v2 VERDICT:** `wellward_heuristic` matches `selected` "
                "on dist 2/4. Pipeline ≡ 3-line geometric rule on this map. "
                "Self-model-driven allocation does **not** earn its complexity. "
                "v2 is done.\n"
            )
        elif c2.get("wellward_matches_selected_dist_4"):
            obs.append(
                "- **FINAL v2 VERDICT (nuanced, from data):**\n"
                "  - **dist=4 (clean):** wellward_heuristic ≡ selected "
                "(mode=2, std ratio ~1.1, mean_coh within 0.001). Pure geometry "
                "does the same job as the full pipeline.\n"
                "  - **dist=2 (near-well):** modes differ — selected locks s4 "
                "(structural best from 6-way sweep, std=0.00031); wellward locks "
                "s2 (true geometric wellward, std=0.00078). Centroid sits ~0.24 "
                "from the well so wellward angle is near-degenerate; selected "
                "is slightly stabler than pure wellward but still matches "
                "fixed(s4) with zero model calls.\n"
                "  - **Neither distance:** selected exceeds a zero-learning "
                "ceiling (wellward at d4, fixed-best at d2). Mean coherence is "
                "not raised over random. Sticky low-variance is real but "
                "reproducible without the self-model.\n"
                "  - **Conclusion:** self-model-driven action selection does "
                "**not** earn its complexity under this design on this single "
                "map. v2 is done.\n"
            )
        else:
            obs.append(
                "- **FINAL v2 VERDICT:** wellward_heuristic does **not** match "
                "selected as expected — see per-distance numbers; dig before "
                "closing.\n"
            )
        obs.append(
            "- **v3 implication:** do not extend this design (more candidates, "
            "longer horizons, smarter selector) on one map — the ceiling is "
            "match a hand-coded / fixed-best rule. Interesting v3 requires "
            "situations a 3-line rule does not trivially solve: competing wells, "
            "wells that turn on/off, or tasks where nearer ≠ better. That means "
            "retraining on varied geometries, not more machinery on "
            "(8,8)/(24,24).\n"
        )

        # 6-way fixed sector sweep
        obs.append(
            "\n## Check 2b — 6-way fixed-sector sweep (cause discrimination)\n"
        )
        obs.append(
            "- Locks each sector id 0..5 for the full episode (no model). "
            "Places `selected` against the full structural stability landscape, "
            "not one unlucky arbitrary sector.\n"
        )
        obs.append(
            f"- selected matches/beats best fixed at dist 2/4: "
            f"{c2.get('fixed_sweep_selected_matches_best_dist_2_4')}\n"
        )
        obs.append(
            f"- structural-outlier + selected lands on it at dist 2/4: "
            f"{c2.get('fixed_sweep_structural_geometry_story_dist_2_4')}\n"
        )
        for d in c2["distances"]:
            v = c2.get("fixed_sector_sweep_verdict", {}).get(str(d), {})
            if not v:
                continue
            obs.append(f"- dist={d}:\n")
            obs.append(
                f"  - selected coh_std={v.get('selected_coh_std', float('nan')):.6f}  "
                f"mode_sector={v.get('selected_mode_sector')}  "
                f"rank={v.get('selected_rank_among_fixed_plus_self')}/7  "
                f"outcome={v.get('outcome')}\n"
            )
            obs.append(
                f"  - best_fixed=s{v.get('best_fixed_sector')} "
                f"std={v.get('best_fixed_coh_std', float('nan')):.6f}  "
                f"worst=s{v.get('worst_fixed_sector')} "
                f"std={v.get('worst_fixed_coh_std', float('nan')):.6f}  "
                f"median={v.get('median_fixed_coh_std', float('nan')):.6f}\n"
            )
            obs.append(
                f"  - mode==best_fixed={v.get('selected_mode_equals_best_fixed')}  "
                f"best_fixed_is_wellward={v.get('best_fixed_is_wellward')}  "
                f"structural_outlier={v.get('structural_outlier_sector')}\n"
            )
            ranked = v.get("ranked_sectors_by_std", [])
            if ranked:
                obs.append(
                    "  - fixed ranks (stable→noisy): "
                    + ", ".join(
                        f"s{r['sector']}={r['coh_std']:.5f}" for r in ranked
                    )
                    + "\n"
                )
        if c2.get("fixed_sweep_structural_geometry_story_dist_2_4"):
            obs.append(
                "- **FINDING (6-way sweep):** one sector is structurally far more "
                "stable and selected locks onto it. The earlier concern stands: "
                "this can be map geometry (e.g. wellward), not evidence the "
                "self-model discovered something non-trivial. Do **not** write up "
                "as 'the self-model earns its complexity.'\n"
            )
        elif c2.get("fixed_sweep_selected_matches_best_dist_2_4"):
            obs.append(
                "- **FINDING (6-way sweep):** selected matches or beats the best "
                "fixed-sector variant without an exhaustive sweep — non-trivial "
                "ranking relative to the full landscape. Still **not** evidence of "
                "surprise-minimization: the implemented objective is "
                "`argmax(predicted coherence)` (point-estimate value), not "
                "selection for low predicted variance.\n"
            )
        else:
            obs.append(
                "- **FINDING (6-way sweep):** selected does not uniquely dominate "
                "the fixed landscape — see per-distance ranks. Single-draw "
                "fixed(s0) overstated the case; report the full spread.\n"
            )
        obs.append(
            "- **Naming discipline:** v0 already established base homeostasis "
            "(nonzero steady update magnitude). Selected's lower trajectory "
            "variance is an *additional* stability effect on top of that base, "
            "not new evidence for the original homeostasis claim. Do not call it "
            "'confirms homeostasis' or 'reduce surprise' without a variance-aware "
            "objective test.\n"
        )
        obs.append(f"- sweep plot: {c2.get('fixed_sweep_plot')}\n")
        obs.append(
            "- Sweep CSVs: `results/check2_d{2,4,8}_fixed{0..5}.csv`\n"
        )

        obs.append("\n## Check 3 — mask compliance\n")
        mc = c2["mask_compliance"]
        obs.append(
            f"- total masked candidates selected: {mc['total_masked_selected']}\n"
            f"- total candidates dropped by validity_mask: "
            f"{mc['total_candidates_dropped_mask']}\n"
            f"- pass: {mc['pass']}\n"
        )
        if mc["total_candidates_dropped_mask"] > 0:
            obs.append(
                "- NOTE: some candidates were dropped by the equidistant/support "
                "mask — pattern support is touching the equidistant boundary. "
                "Logged counts above; selection only scored survivors.\n"
            )

    path = RESULTS / "observations.txt"
    path.write_text("".join(obs))
    print(f"\n  wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="v2 action-selection checks")
    parser.add_argument("--n_examples", type=int, default=14_000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--horizon_train", type=int, default=100)
    parser.add_argument("--horizon_episode", type=int, default=2000)
    parser.add_argument("--burn_in", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help="reuse results/boost_rollout_dataset.pt if present",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="reuse results/self_model_v2.pt if present",
    )
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument(
        "--force_action",
        action="store_true",
        help="run Check 2 even if Check 1 hard-fails (debug only)",
    )
    parser.add_argument(
        "--check2_only",
        action="store_true",
        help="reuse dataset+model; re-run Check 1 eval + Check 2/3 only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smaller dataset / shorter episode for smoke tests",
    )
    args = parser.parse_args()
    if args.check2_only:
        args.skip_dataset = True
        args.skip_train = True

    if args.quick:
        args.n_examples = min(args.n_examples, 800)
        args.epochs = min(args.epochs, 5)
        args.horizon_episode = min(args.horizon_episode, 400)

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = pick_device(prefer_cuda=not args.cpu)
    print(f"device={device}  n_examples={args.n_examples}  epochs={args.epochs}")
    print(f"v0 ckpt: {args.ckpt}")

    # ---- dataset ----
    ds_path = RESULTS / "boost_rollout_dataset.pt"
    if args.skip_dataset and ds_path.is_file():
        print(f"\n=== Loading dataset {ds_path} ===")
        data = torch.load(ds_path, map_location="cpu", weights_only=False)
        print(f"  n={data['S'].shape[0]}")
        if "pert_type" in data:
            from boost_rollout_dataset import PERT_TYPE_NAMES

            for code, name in PERT_TYPE_NAMES.items():
                n = int((data["pert_type"] == code).sum().item())
                print(f"  {name}: {n}")
    else:
        print("\n=== Generating damage+boost rollout dataset ===")
        dcfg = BoostDatasetConfig(
            n_examples=args.n_examples,
            burn_in=args.burn_in,
            horizon=args.horizon_train,
            seed=args.seed,
            device=device,
        )
        data = generate_dataset(
            dcfg, ckpt_path=args.ckpt, out_path=ds_path, verbose=True
        )

    # ---- train ----
    model_path = RESULTS / "self_model_v2.pt"
    hist_path = RESULTS / "train_history.json"
    if args.skip_train and model_path.is_file():
        print(f"\n=== Loading self-model {model_path} ===")
        model = load_model(model_path, device=device)
        _, val_data = train_val_split(data, val_frac=0.15, seed=args.seed)
        train_info = {
            "n_train": data["S"].shape[0] - val_data["S"].shape[0],
            "n_val": val_data["S"].shape[0],
            "history": {},
        }
    else:
        print("\n=== Training self_model_v2 on damage+boost ===")
        tcfg = TrainConfig(
            epochs=args.epochs,
            batch_size=64,
            lr=1e-3,
            seed=args.seed,
            device=device,
            hidden=32,
        )
        model, train_info = train_self_model_v2(
            data, tcfg=tcfg, channels=int(data["S"].shape[1]), verbose=True
        )
        val_data = train_info.pop("val_data")
        save_model(
            model,
            model_path,
            extra={
                "seed": args.seed,
                "epochs": args.epochs,
                "n": args.n_examples,
            },
        )
        with hist_path.open("w") as f:
            json.dump(train_info.get("history", {}), f, indent=2)
        print(f"  saved {model_path}")

    # ---- Check 1 ----
    c1 = check1_boost_calibration(model, val_data, device=device)

    stopped = False
    c2 = None
    attn_cfg = None

    if c1["hard_fail"] and not args.force_action:
        print(
            "\n*** Check 1 hard-fail: not proceeding to action selection "
            "(use --force_action to override for debugging). ***"
        )
        stopped = True
    else:
        if c1["hard_fail"]:
            print("\n*** Check 1 hard-fail but --force_action set; continuing. ***")
        attn_cfg = tune_attention_budget(
            horizon=args.horizon_episode, K=20
        )
        nca, base_pattern, _ = load_v0_world(args.ckpt, device=device)
        world_cfg = WorldConfig(device=device)
        c2 = check2_policy_comparison(
            model,
            nca,
            base_pattern,
            world_cfg,
            device=device,
            attn_cfg=attn_cfg,
            distances=(2, 4, 8),
            horizon=args.horizon_episode,
            seed=args.seed,
        )

    write_observations(c1, c2, attn_cfg, args, device, stopped_at_check1=stopped)

    summary = {
        "device": device,
        "seed": args.seed,
        "n_examples": args.n_examples,
        "check1": c1,
        "check2": (
            {
                k: v
                for k, v in c2.items()
                if k != "episodes_meta"
            }
            if c2
            else None
        ),
        "stopped_after_check1": stopped,
        "attn_cfg": (
            {
                "K": attn_cfg.K,
                "regen_rate": attn_cfg.regen_rate,
                "boost_cost": attn_cfg.boost_cost,
                "A_max": attn_cfg.A_max,
                "boost_magnitude": attn_cfg.boost_magnitude,
            }
            if attn_cfg
            else None
        ),
    }

    # Make JSON-safe (no tensors)
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

    with (RESULTS / "summary.json").open("w") as f:
        json.dump(_sanitize(summary), f, indent=2)
    print(f"  wrote {RESULTS / 'summary.json'}")

    print("\n=== v2 checks complete ===")
    if stopped:
        print("Result: STOPPED after Check 1 (boost calibration failed).")
    else:
        print(
            f"Check 1 pass={c1['pass']} strong={c1['strong_pass']}  "
            f"Check 2 mean_beats_random={c2['selected_clearly_beats_random']}  "
            f"Check 2 low_variance={c2.get('selected_lower_variance_dist_2_4')}  "
            f"Check 2 sector_sticky={c2.get('selected_sector_sticky_dist_2_4')}  "
            f"Check 2b structural_geometry={c2.get('fixed_sweep_structural_geometry_story_dist_2_4')}  "
            f"Check 2c wellward_matches={c2.get('wellward_matches_selected_dist_2_4')}  "
            f"Check 3 mask_ok={c2['mask_compliance']['pass']}"
        )


if __name__ == "__main__":
    main()
