#!/usr/bin/env python3
"""v3.5 follow-ups (local_R bins, floor timing, multi-seed, noise-death). Writes results/v3_5/diagnostics/."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
from embodiednca.paths import results_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from embodiednca.v2.action_selection import AttentionConfig
from embodiednca.v3_5.action_selection_v3_5 import (
    SKIP_SECTOR,
    run_v35_episode,
)
from embodiednca.v3.boost_cost import BoostCostConfig
from embodiednca.v3_5.death_aware_rollout_dataset import (
    DeathAwareDatasetConfig,
    DeathAwareTensorDataset,
    PERT_NOISE,
    PERT_NULL,
    PERT_TYPE_NAMES,
    PERT_ZERO,
    filter_by_pert_type,
    generate_dataset,
    train_val_split,
)
from embodiednca.v0.grid_world import DEFAULT_GEN_RATE, DEFAULT_WELLS, WorldConfig
from embodiednca.v0.metrics import DEATH_MASS_EPS, DEFAULT_ACTIVATION_THRESHOLD
from embodiednca.v1.rollout_dataset import DEFAULT_CKPT, load_v0_world
from embodiednca.v1.self_model import mae, pearson_r
from embodiednca.v3_5.self_model_v3_5 import (
    SelfModelV35,
    TrainConfig,
    balanced_accuracy,
    death_class_metrics,
    evaluate_calibration_v35,
    load_model,
    train_self_model_v3_5,
)
from embodiednca.v3_5.wear_perwell import PerWellWearConfig

RESULTS = results_dir("v3_5") / "diagnostics"
MAIN_RESULTS = results_dir("v3_5")
MODEL_PATH = MAIN_RESULTS / "self_model_v3_5.pt"
DATASET_PATH = MAIN_RESULTS / "death_aware_dataset.pt"

DEFAULT_WEAR = PerWellWearConfig(
    wear_ambient_rate=0.001,
    wear_boost_rate=0.004,
    wear_decay_coeff=0.20,
    gen_floor_frac=0.40,
    gen_rate_base=DEFAULT_GEN_RATE,
    wells=DEFAULT_WELLS,
)
DEFAULT_COST = BoostCostConfig(direct_boost_cost=0.15)

DIST_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 14)]
MEANINGFUL_COH_DELTA = 0.01


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# D1 — local_R within-bin collapse
# ---------------------------------------------------------------------------
@torch.no_grad()
def diagnostic1_local_R(
    model: SelfModelV35,
    val_data: Dict[str, torch.Tensor],
    device: str,
) -> Dict:
    print("\n=== D1: local_R within-bin collapse diagnosis ===")
    model.eval()
    n = val_data["S"].shape[0]
    preds_lr, trues_lr = [], []
    preds_coh = []
    wear_asym = []
    nearest_well_asym_proxy = []

    batch = 128
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        S = val_data["S"][i0:i1].to(device)
        R = val_data["R"][i0:i1].to(device)
        P = val_data["pert_map"][i0:i1].to(device)
        Wf = val_data["multipliers"][i0:i1].to(device)
        p_coh, p_lr, _ = model(S, R, P, Wf)
        preds_lr.append(p_lr.cpu())
        preds_coh.append(p_coh.cpu())
        trues_lr.append(val_data["local_R_future"][i0:i1])
        # wear asymmetry |mult_0 - mult_1| and |wear_0 - wear_1|
        w = val_data["wear"][i0:i1]
        m = val_data["multipliers"][i0:i1]
        wear_asym.append((w[:, 0] - w[:, 1]).abs())
        nearest_well_asym_proxy.append((m[:, 0] - m[:, 1]).abs())

    pred_lr = torch.cat(preds_lr)
    true_lr = torch.cat(trues_lr)
    pred_coh = torch.cat(preds_coh)
    wear_a = torch.cat(wear_asym)
    mult_a = torch.cat(nearest_well_asym_proxy)
    dist = val_data["distance_to_well"]
    abs_err = (pred_lr - true_lr).abs()

    # Per-bin true local_R mean±std (v1 methodology) + r + MAE
    by_bin = {}
    print(
        f"  {'bin':>10s}  {'n':>5s}  {'true_mean':>9s}  {'true_std':>8s}  "
        f"{'pred_std':>8s}  {'r':>7s}  {'MAE':>7s}  diagnosis"
    )
    for lo, hi in DIST_BINS:
        m = (dist >= lo) & (dist < hi)
        n_bin = int(m.sum().item())
        if n_bin < 3:
            by_bin[f"[{lo},{hi})"] = {"n": n_bin}
            continue
        t = true_lr[m]
        p = pred_lr[m]
        t_mean = float(t.mean().item())
        t_std = float(t.std().item()) if n_bin > 1 else 0.0
        p_std = float(p.std().item()) if n_bin > 1 else 0.0
        r = pearson_r(p, t)
        m_ae = mae(p, t)
        # Diagnosis: variance-collapse if true_std is tiny relative to overall
        # or absolute; genuine failure if MAE is large AND r is near 0 with
        # non-trivial true variance.
        overall_std = float(true_lr.std().item())
        if t_std < 0.05 or (overall_std > 0 and t_std / overall_std < 0.25):
            diag = "VARIANCE_COLLAPSE"
        elif (r is None or abs(r) < 0.2) and m_ae > 0.15:
            diag = "GENUINE_FAILURE"
        elif r is None or abs(r) < 0.2:
            diag = "LOW_R_OK_MAE" if m_ae <= 0.15 else "GENUINE_FAILURE"
        else:
            diag = "OK"
        by_bin[f"[{lo},{hi})"] = {
            "n": n_bin,
            "true_mean": t_mean,
            "true_std": t_std,
            "pred_mean": float(p.mean().item()),
            "pred_std": p_std,
            "local_R_r": r,
            "local_R_mae": m_ae,
            "diagnosis": diag,
            "wear_asym_mean": float(wear_a[m].mean().item()),
            "mult_asym_mean": float(mult_a[m].mean().item()),
        }
        print(
            f"  [{lo:2d},{hi:2d})  {n_bin:5d}  {t_mean:9.4f}  {t_std:8.4f}  "
            f"{p_std:8.4f}  {r if r is not None else float('nan'):7.3f}  "
            f"{m_ae:7.4f}  {diag}"
        )

    # Correlation of |error| with wear asymmetry (within mid/far bins)
    mid_far = (dist >= 4.0) & (dist < 14.0)
    if int(mid_far.sum().item()) >= 20:
        # Pearson of abs_err vs wear_asym within mid/far
        ea = abs_err[mid_far]
        wa = wear_a[mid_far]
        ma = mult_a[mid_far]
        r_err_wear = pearson_r(ea, wa)
        r_err_mult = pearson_r(ea, ma)
        # Split high vs low asymmetry
        med = float(wa.median().item())
        high = wa >= med
        low = ~high
        mae_high = float(ea[high].mean().item()) if high.any() else float("nan")
        mae_low = float(ea[low].mean().item()) if low.any() else float("nan")
        r_high = pearson_r(pred_lr[mid_far][high], true_lr[mid_far][high])
        r_low = pearson_r(pred_lr[mid_far][low], true_lr[mid_far][low])
    else:
        r_err_wear = r_err_mult = mae_high = mae_low = r_high = r_low = float("nan")
        med = float("nan")

    print(
        f"  mid/far abs_err vs wear_asym r={r_err_wear}  "
        f"vs mult_asym r={r_err_mult}"
    )
    print(
        f"  mid/far split at wear_asym median={med:.3f}: "
        f"MAE high={mae_high:.4f} low={mae_low:.4f}  "
        f"r high={r_high} low={r_low}"
    )

    # Plots: scatter per collapsed bins + true std bars
    collapsed = ["[4,6)", "[6,8)", "[8,10)", "[10,14)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # row0: scatters for collapsed-ish bins
    for i, blabel in enumerate(collapsed[:3]):
        ax = axes[0, i]
        lo, hi = map(int, blabel.strip("[]()").split(","))
        m = (dist >= lo) & (dist < hi)
        if m.any():
            ax.scatter(
                true_lr[m].numpy(),
                pred_lr[m].numpy(),
                s=14,
                alpha=0.5,
                c="C0",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        info = by_bin.get(blabel, {})
        ax.set_title(
            f"{blabel}  n={info.get('n', 0)}\n"
            f"r={info.get('local_R_r', float('nan'))}  "
            f"true_std={info.get('true_std', float('nan')):.3f}\n"
            f"{info.get('diagnosis', '')}"
        )
        ax.set_xlabel("true local_R")
        ax.set_ylabel("pred local_R")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # [10,14) in remaining cell of row0 — use axes[0,2] already used; put in [1,0]
    ax = axes[1, 0]
    blabel = "[10,14)"
    lo, hi = 10, 14
    m = (dist >= lo) & (dist < hi)
    if m.any():
        ax.scatter(true_lr[m].numpy(), pred_lr[m].numpy(), s=10, alpha=0.35, c="C0")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    info = by_bin.get(blabel, {})
    ax.set_title(
        f"{blabel}  n={info.get('n', 0)}\n"
        f"r={info.get('local_R_r', float('nan'))}  "
        f"true_std={info.get('true_std', float('nan')):.3f}\n"
        f"{info.get('diagnosis', '')}"
    )
    ax.set_xlabel("true local_R")
    ax.set_ylabel("pred local_R")
    ax.grid(True, alpha=0.3)

    # Bar: true mean±std per bin
    ax = axes[1, 1]
    labels = [f"[{lo},{hi})" for lo, hi in DIST_BINS]
    means = [by_bin.get(l, {}).get("true_mean", float("nan")) for l in labels]
    stds = [by_bin.get(l, {}).get("true_std", float("nan")) for l in labels]
    xs = range(len(labels))
    ax.bar(xs, means, yerr=stds, capsize=4, color="C1", alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("true local_R mean±std")
    ax.set_title("True local_R by distance bin")
    ax.grid(True, alpha=0.3, axis="y")

    # abs_err vs wear asymmetry (mid/far)
    ax = axes[1, 2]
    if mid_far.any():
        ax.scatter(
            wear_a[mid_far].numpy(),
            abs_err[mid_far].numpy(),
            s=10,
            alpha=0.35,
            c="C3",
        )
    ax.set_xlabel("|wear_A − wear_B|")
    ax.set_ylabel("|pred − true| local_R")
    ax.set_title(f"mid/far abs_err vs wear_asym\nr={r_err_wear}")
    ax.grid(True, alpha=0.3)

    fig.suptitle("D1: local_R within-bin diagnosis", fontsize=12)
    fig.tight_layout()
    out_plot = RESULTS / "d1_local_R_bins.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Headline
    diagnoses = {
        k: v.get("diagnosis")
        for k, v in by_bin.items()
        if isinstance(v, dict) and "diagnosis" in v
    }
    n_collapse = sum(1 for d in diagnoses.values() if d == "VARIANCE_COLLAPSE")
    n_fail = sum(1 for d in diagnoses.values() if d == "GENUINE_FAILURE")
    n_low_ok = sum(1 for d in diagnoses.values() if d == "LOW_R_OK_MAE")
    headline = (
        f"{n_collapse} bin(s) VARIANCE_COLLAPSE, {n_fail} GENUINE_FAILURE, "
        f"{n_low_ok} LOW_R_OK_MAE"
    )
    print(f"  HEADLINE: {headline}")

    return {
        "by_bin": by_bin,
        "overall_local_R_r": pearson_r(pred_lr, true_lr),
        "overall_local_R_mae": mae(pred_lr, true_lr),
        "overall_true_std": float(true_lr.std().item()),
        "mid_far_r_err_vs_wear_asym": r_err_wear,
        "mid_far_r_err_vs_mult_asym": r_err_mult,
        "mid_far_mae_high_asym": mae_high,
        "mid_far_mae_low_asym": mae_low,
        "mid_far_r_high_asym": r_high,
        "mid_far_r_low_asym": r_low,
        "headline": headline,
        "plot": str(out_plot),
    }


# ---------------------------------------------------------------------------
# D2 — decoupled floor-crossing
# ---------------------------------------------------------------------------
@torch.no_grad()
def diagnostic2_decoupled_floor(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: SelfModelV35,
    wear_cfg: PerWellWearConfig,
    device: str,
    quick: bool = False,
) -> Dict:
    """
    Shallow pre-degrade so one well floors early, but pattern is near-well
    enough to survive 1000+ steps after floor — decoupling floor from death.
    """
    print("\n=== D2: decoupled floor-crossing (shallow pre-degrade) ===")
    # Decouple floor from death:
    #   - well A starts near floor (floors early via ambient)
    #   - well B is fresh
    #   - pattern sits at dist=2 from *well B* (the healthy well), so A's
    #     floor does not starve the pattern. Survival window after A's floor
    #     should be long; any skip-spike at A's floor is not near-death.
    horizon = 4000 if not quick else 2500
    distance = 2
    well_B = wear_cfg.wells[1]  # (24, 24)
    attn_cfg = AttentionConfig()
    cost_cfg = DEFAULT_COST
    initial_wear = (2.70, 0.30)
    floor_wear = (1.0 - wear_cfg.gen_floor_frac) / wear_cfg.wear_decay_coeff
    steps_A_to_floor = max(
        0.0, (floor_wear - initial_wear[0]) / wear_cfg.wear_ambient_rate
    )
    print(
        f"  initial_wear={initial_wear}  floor_wear={floor_wear:.3f}  "
        f"ambient steps for A→floor≈{steps_A_to_floor:.0f}"
    )
    print(
        f"  pattern placed near well B {well_B} (fresh) at dist={distance}  "
        f"— A's floor should not kill; look for skip-spike at A's floor"
    )

    ep = run_v35_episode(
        nca=nca,
        base_pattern=base_pattern,
        world_cfg=world_cfg,
        model=model,
        policy="selected",
        distance=distance,
        well=well_B,
        horizon=horizon,
        attn_cfg=attn_cfg,
        wear_cfg=wear_cfg,
        cost_cfg=cost_cfg,
        seed=7,
        device=device,
        pre_degrade_well_idx=None,
        pre_degrade_boosts=0,
        initial_wear=initial_wear,
    )

    hist = ep["history"]
    details = ep["decision_details"]
    tod = ep["time_of_death"]
    floor = wear_cfg.gen_floor_frac

    t_floor_0 = next(
        (int(h["t"]) for h in hist if h["mult_0"] <= floor + 1e-9), None
    )
    t_floor_1 = next(
        (int(h["t"]) for h in hist if h["mult_1"] <= floor + 1e-9), None
    )
    t_all_floor = next(
        (int(h["t"]) for h in hist if h["all_floored"] > 0.5), None
    )

    # Alive-only decisions
    if tod is not None:
        details_alive = [d for d in details if d["t"] < tod]
    else:
        details_alive = details

    # Survival after first floor
    t_first = None
    floors = [t for t in (t_floor_0, t_floor_1) if t is not None]
    if floors:
        t_first = min(floors)
    post_floor_life = None
    if t_first is not None:
        end = tod if tod is not None else horizon
        post_floor_life = end - t_first

    # Boost rates: before first floor / between first and all / after all
    # Also: window around first floor ±400 steps while alive
    def rate_in(t_lo, t_hi, key_boost="boosted"):
        sub = [
            d
            for d in details_alive
            if t_lo <= d["t"] < t_hi
        ]
        if not sub:
            return float("nan"), 0, 0
        nb = sum(1 for d in sub if d[key_boost])
        vs = sum(1 for d in sub if d.get("voluntary_skip"))
        return nb / len(sub), len(sub), vs / len(sub)

    windows = {}
    if t_first is not None:
        windows["pre_first_floor"] = rate_in(0, t_first)
        if t_all_floor is not None and t_all_floor > t_first:
            windows["partial_floor"] = rate_in(t_first, t_all_floor)
            windows["post_all_floor"] = rate_in(t_all_floor, horizon + 1)
        else:
            windows["post_first_floor"] = rate_in(t_first, horizon + 1)
        # tight window around floor crossing
        windows["around_floor_-400_0"] = rate_in(t_first - 400, t_first)
        windows["around_floor_0_400"] = rate_in(t_first, t_first + 400)
        windows["around_floor_400_800"] = rate_in(t_first + 400, t_first + 800)

    # Time bins absolute
    bin_w = 200
    max_t = max((d["t"] for d in details_alive), default=0)
    bins = []
    for t0 in range(0, int(max_t) + 1, bin_w):
        br, n, vs = rate_in(t0, t0 + bin_w)
        if n == 0:
            continue
        bins.append(
            {"t0": t0, "t1": t0 + bin_w, "n": n, "boost_rate": br, "vol_skip_rate": vs}
        )

    # Verdict on timing hypothesis under decoupled scenario
    decoupled_ok = (
        post_floor_life is not None and post_floor_life >= 1000 and (tod is None or tod > (t_first or 0) + 1000)
    )
    # Look for skip spike at floor: around_floor_0_400 vol_skip > pre window
    pre_vs = windows.get("around_floor_-400_0", (float("nan"), 0, float("nan")))[2]
    post_vs = windows.get("around_floor_0_400", (float("nan"), 0, float("nan")))[2]
    pre_br = windows.get("around_floor_-400_0", (float("nan"), 0, float("nan")))[0]
    post_br = windows.get("around_floor_0_400", (float("nan"), 0, float("nan")))[0]

    if not decoupled_ok:
        verdict = "SCENARIO_FAILED_NOT_DECOUPLED"
    elif math.isnan(pre_br) or math.isnan(post_br):
        verdict = "INSUFFICIENT_DATA"
    elif post_br < pre_br - 0.10 or (not math.isnan(post_vs) and post_vs > pre_vs + 0.10):
        verdict = "CONFIRMED_FLOOR_AWARE_SKIP"
    elif abs(post_br - pre_br) <= 0.10:
        verdict = "NO_FLOOR_TIMING_SIGNAL"
    else:
        verdict = "OPPOSITE_OF_PREDICTION"

    print(f"  distance={distance} horizon={horizon} tod={tod}")
    print(
        f"  t_floor A/B/all={t_floor_0}/{t_floor_1}/{t_all_floor}  "
        f"t_first={t_first}  post_floor_life={post_floor_life}  "
        f"decoupled_ok={decoupled_ok}"
    )
    print(f"  mean_coh={ep['mean_coherence']:.4f}  n_boosts={ep['n_boosts']}  "
          f"vol_skip={ep['n_skipped_voluntary']}")
    for name, (br, n, vs) in windows.items():
        print(f"  window {name}: boost_rate={br:.3f}  vol_skip={vs:.3f}  n={n}")
    print(f"  VERDICT: {verdict}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ts = [h["t"] for h in hist]
    axes[0].plot(ts, [h["coherence"] for h in hist], "C0", lw=1.2, label="coherence")
    axes[0].plot(ts, [h["alive"] for h in hist], "C2", lw=1, alpha=0.5, label="alive")
    if t_first is not None:
        axes[0].axvline(t_first, color="C3", ls="--", label=f"first_floor t={t_first}")
    if t_all_floor is not None:
        axes[0].axvline(t_all_floor, color="0.4", ls=":", label=f"all_floor t={t_all_floor}")
    if tod is not None:
        axes[0].axvline(tod, color="k", ls="-", alpha=0.4, label=f"death t={tod}")
    axes[0].set_ylabel("coherence / alive")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"D2 decoupled floor  dist={distance} initial_wear={initial_wear}  [{verdict}]"
    )

    axes[1].plot(ts, [h["mult_0"] for h in hist], "C0", label="mult_A")
    axes[1].plot(ts, [h["mult_1"] for h in hist], "C1", label="mult_B")
    axes[1].axhline(floor, color="0.5", ls=":")
    if t_first is not None:
        axes[1].axvline(t_first, color="C3", ls="--")
    axes[1].set_ylabel("effective_multiplier")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if bins:
        centers = [0.5 * (b["t0"] + b["t1"]) for b in bins]
        axes[2].plot(centers, [b["boost_rate"] for b in bins], "C0-o", label="boost rate")
        axes[2].plot(
            centers, [b["vol_skip_rate"] for b in bins], "C3-s", label="vol skip rate"
        )
    if t_first is not None:
        axes[2].axvline(t_first, color="C3", ls="--", label="first_floor")
    if t_all_floor is not None:
        axes[2].axvline(t_all_floor, color="0.4", ls=":", label="all_floor")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("rate (alive decisions)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    out_plot = RESULTS / "d2_decoupled_floor_timing.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    return {
        "pre_boosts": 0,
        "initial_wear": list(initial_wear),
        "pre_wear_preview": list(initial_wear),
        "pre_mult_preview": [
            max(
                wear_cfg.gen_floor_frac,
                1.0 - wear_cfg.wear_decay_coeff * initial_wear[i],
            )
            for i in range(2)
        ],
        "distance": distance,
        "horizon": horizon,
        "time_of_death": tod,
        "t_floor_0": t_floor_0,
        "t_floor_1": t_floor_1,
        "t_all_floor": t_all_floor,
        "t_first_floor": t_first,
        "post_floor_life": post_floor_life,
        "decoupled_ok": decoupled_ok,
        "mean_coherence": ep["mean_coherence"],
        "n_boosts": ep["n_boosts"],
        "n_skipped_voluntary": ep["n_skipped_voluntary"],
        "windows": {
            k: {"boost_rate": v[0], "n": v[1], "vol_skip_rate": v[2]}
            for k, v in windows.items()
        },
        "time_bins": bins,
        "verdict": verdict,
        "plot": str(out_plot),
    }


# ---------------------------------------------------------------------------
# D3 — selection log diff + multi-seed Check 3
# ---------------------------------------------------------------------------
@torch.no_grad()
def diagnostic3_selection_diff_and_seeds(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: SelfModelV35,
    wear_cfg: PerWellWearConfig,
    device: str,
    quick: bool = False,
) -> Dict:
    print("\n=== D3: selected vs wear_aware selection-log diff + multi-seed ===")
    pre_boosts = 80 if not quick else 40
    horizon = 3000 if not quick else 1200
    distances = (2, 4) if not quick else (2, 4)
    seeds = (0, 1, 2) if not quick else (0, 1, 2)
    attn_cfg = AttentionConfig()
    cost_cfg = DEFAULT_COST
    policies = ("selected", "wear_aware_heuristic")

    # --- Detailed log diff at seed=0, dist=2 ---
    print("  --- selection log diff (seed=0, dist=2) ---")
    eps = {}
    for pol in policies:
        eps[pol] = run_v35_episode(
            nca=nca,
            base_pattern=base_pattern,
            world_cfg=world_cfg,
            model=model if pol == "selected" else None,
            policy=pol,
            distance=2,
            horizon=horizon,
            attn_cfg=attn_cfg,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            seed=0 + 2 * 17,  # match Check 3 seed convention: seed + dist*17
            device=device,
            pre_degrade_well_idx=0,
            pre_degrade_boosts=pre_boosts,
        )

    def alive_boost_decisions(ep: Dict) -> List[Dict]:
        tod = ep["time_of_death"]
        out = []
        for d in ep["decision_details"]:
            if tod is not None and d["t"] >= tod:
                continue
            if d.get("boosted"):
                out.append(d)
        return out

    sel_boosts = alive_boost_decisions(eps["selected"])
    wah_boosts = alive_boost_decisions(eps["wear_aware_heuristic"])
    print(
        f"  alive boosts: selected={len(sel_boosts)}  "
        f"wear_aware={len(wah_boosts)}"
    )

    # Align by decision time (K-grid)
    sel_by_t = {d["t"]: d for d in eps["selected"]["decision_details"]}
    wah_by_t = {d["t"]: d for d in eps["wear_aware_heuristic"]["decision_details"]}
    common_t = sorted(set(sel_by_t) & set(wah_by_t))
    if eps["selected"]["time_of_death"] is not None:
        common_t = [t for t in common_t if t < eps["selected"]["time_of_death"]]
    # Also cap at wear_aware death
    if eps["wear_aware_heuristic"]["time_of_death"] is not None:
        common_t = [
            t for t in common_t if t < eps["wear_aware_heuristic"]["time_of_death"]
        ]

    both_act = []
    only_sel = []
    only_wah = []
    both_skip = []
    same_sector = 0
    diff_sector = 0
    sector_pairs = []

    for t in common_t:
        s = sel_by_t[t]
        w = wah_by_t[t]
        sb, wb = bool(s["boosted"]), bool(w["boosted"])
        if sb and wb:
            both_act.append(t)
            ss, ws = s.get("selected_sector"), w.get("selected_sector")
            sector_pairs.append(
                {
                    "t": t,
                    "sel_sector": ss,
                    "wah_sector": ws,
                    "same": ss == ws,
                    "sel_wear": s.get("wear"),
                    "wah_wear": w.get("wear"),
                    "mult": s.get("multipliers"),
                }
            )
            if ss == ws:
                same_sector += 1
            else:
                diff_sector += 1
        elif sb and not wb:
            only_sel.append(t)
        elif wb and not sb:
            only_wah.append(t)
        else:
            both_skip.append(t)

    print(
        f"  common alive decisions: {len(common_t)}  "
        f"both_act={len(both_act)}  only_sel={len(only_sel)}  "
        f"only_wah={len(only_wah)}  both_skip={len(both_skip)}"
    )
    print(
        f"  when both act: same_sector={same_sector}  "
        f"diff_sector={diff_sector}"
    )
    for p in sector_pairs[:20]:
        print(
            f"    t={p['t']:4d}  sel_sec={p['sel_sector']}  "
            f"wah_sec={p['wah_sector']}  same={p['same']}  "
            f"mult={p['mult']}"
        )

    # Well attribution: which well is nearest the chosen cell
    def well_of_decision(d: Dict, ep: Dict) -> Optional[int]:
        if not d.get("boosted"):
            return None
        # Use geometry logs if available
        for g in ep.get("geometry_logs", []):
            if g.get("t") == d["t"]:
                # nearest well from geometry
                ny, nx = g.get("nearest_well_y"), g.get("nearest_well_x")
                if ny is None:
                    return None
                for i, (wy, wx) in enumerate(DEFAULT_WELLS):
                    if abs(wy - ny) < 0.5 and abs(wx - nx) < 0.5:
                        return i
        # fallback: sector-based not available; use selected coords from logs
        return None

    # From selection logs, get (y,x) of chosen boost
    def well_from_sel_log(ep: Dict, t: int) -> Optional[int]:
        for log in ep.get("selection_logs", []):
            if log.t == t and log.selected is not None:
                y, x = log.selected
                from embodiednca.v3_5.wear_perwell import nearest_well_index

                return nearest_well_index(y, x, DEFAULT_WELLS)
        return None

    both_act_wells = []
    for t in both_act:
        sw = well_from_sel_log(eps["selected"], t)
        ww = well_from_sel_log(eps["wear_aware_heuristic"], t)
        both_act_wells.append({"t": t, "sel_well": sw, "wah_well": ww, "same": sw == ww})
    n_same_well = sum(1 for x in both_act_wells if x["same"])
    n_diff_well = sum(1 for x in both_act_wells if x["same"] is False)
    print(
        f"  when both act: same_well={n_same_well}  diff_well={n_diff_well}"
    )
    for x in both_act_wells[:15]:
        print(
            f"    t={x['t']:4d}  sel_well={x['sel_well']}  "
            f"wah_well={x['wah_well']}  same={x['same']}"
        )

    log_diff = {
        "n_common_alive": len(common_t),
        "both_act": len(both_act),
        "only_selected": len(only_sel),
        "only_wear_aware": len(only_wah),
        "both_skip": len(both_skip),
        "same_sector_when_both_act": same_sector,
        "diff_sector_when_both_act": diff_sector,
        "same_well_when_both_act": n_same_well,
        "diff_well_when_both_act": n_diff_well,
        "sector_pairs": sector_pairs,
        "both_act_wells": both_act_wells,
        "only_sel_ts": only_sel,
        "only_wah_ts": only_wah,
        "selected_mean_coh": eps["selected"]["mean_coherence"],
        "wear_aware_mean_coh": eps["wear_aware_heuristic"]["mean_coherence"],
        "selected_n_boosts_alive": len(sel_boosts),
        "wear_aware_n_boosts_alive": len(wah_boosts),
    }

    # --- Multi-seed Check 3 ---
    print(
        "  --- multi-seed Check 3 (selected vs wear_aware) ---\n"
        "  NOTE: dynamics are deterministic at a fixed center; seed varies "
        "placement *angle* around the well so runs actually differ."
    )
    multi = {}
    # Three distinct angles: east (default), NE, SE — same |distance|
    angles = {
        0: 0.0,  # east (+x)
        1: math.pi / 4,  # northeast
        2: -math.pi / 4,  # southeast
    }
    for dist in distances:
        multi[dist] = {"selected": [], "wear_aware_heuristic": [], "deltas": []}
        for seed in seeds:
            row = {}
            ang = angles.get(int(seed), float(seed) * 0.7)
            for pol in policies:
                ep = run_v35_episode(
                    nca=nca,
                    base_pattern=base_pattern,
                    world_cfg=world_cfg,
                    model=model if pol == "selected" else None,
                    policy=pol,
                    distance=dist,
                    horizon=horizon,
                    attn_cfg=attn_cfg,
                    wear_cfg=wear_cfg,
                    cost_cfg=cost_cfg,
                    seed=seed + dist * 17,
                    device=device,
                    pre_degrade_well_idx=0,
                    pre_degrade_boosts=pre_boosts,
                    placement_angle=ang,
                )
                row[pol] = ep["mean_coherence"]
                multi[dist][pol].append(
                    {
                        "seed": seed,
                        "placement_angle": ang,
                        "mean_coherence": ep["mean_coherence"],
                        "time_of_death": ep["time_of_death"],
                        "n_boosts": ep["n_boosts"],
                        "n_skipped_voluntary": ep["n_skipped_voluntary"],
                    }
                )
            delta = row["selected"] - row["wear_aware_heuristic"]
            multi[dist]["deltas"].append(delta)
            print(
                f"  dist={dist} seed={seed} ang={ang:.2f}: sel={row['selected']:.4f}  "
                f"wah={row['wear_aware_heuristic']:.4f}  Δ={delta:+.4f}"
            )

        deltas = multi[dist]["deltas"]
        mean_d = sum(deltas) / len(deltas)
        std_d = (
            math.sqrt(sum((d - mean_d) ** 2 for d in deltas) / len(deltas))
            if len(deltas) > 1
            else 0.0
        )
        # all same sign?
        signs = [1 if d > MEANINGFUL_COH_DELTA else (-1 if d < -MEANINGFUL_COH_DELTA else 0) for d in deltas]
        if all(s == 1 for s in signs):
            repro = "reproducibly_beats"
        elif all(s == -1 for s in signs):
            repro = "reproducibly_loses"
        elif all(s == 0 for s in signs):
            repro = "reproducibly_ties"
        else:
            repro = "seed_sensitive_mixed"
        multi[dist]["mean_delta"] = mean_d
        multi[dist]["std_delta"] = std_d
        multi[dist]["reproducibility"] = repro
        multi[dist]["mean_selected"] = sum(
            r["mean_coherence"] for r in multi[dist]["selected"]
        ) / len(seeds)
        multi[dist]["mean_wear_aware"] = sum(
            r["mean_coherence"] for r in multi[dist]["wear_aware_heuristic"]
        ) / len(seeds)
        print(
            f"  dist={dist} summary: mean Δ={mean_d:+.4f} ± {std_d:.4f}  "
            f"[{repro}]"
        )

    # Plot multi-seed
    fig, axes = plt.subplots(1, len(distances), figsize=(5 * len(distances), 4), squeeze=False)
    for i, dist in enumerate(distances):
        ax = axes[0, i]
        xs = list(range(len(seeds)))
        sel_ys = [r["mean_coherence"] for r in multi[dist]["selected"]]
        wah_ys = [r["mean_coherence"] for r in multi[dist]["wear_aware_heuristic"]]
        ax.plot(xs, sel_ys, "C0-o", label="selected")
        ax.plot(xs, wah_ys, "C2-s", label="wear_aware_heuristic")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("seed")
        ax.set_ylabel("mean coherence")
        ax.set_title(
            f"dist={dist}  mean Δ={multi[dist]['mean_delta']:+.4f}±"
            f"{multi[dist]['std_delta']:.4f}\n[{multi[dist]['reproducibility']}]"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_plot = RESULTS / "d3_multiseed_selected_vs_wear_aware.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Sector agreement plot for seed0 dist2
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for t in common_t:
        s = sel_by_t[t]
        w = wah_by_t[t]
        if s["boosted"]:
            ax.scatter(t, 1, c="C0", marker="o", s=30, zorder=3)
            ax.annotate(
                str(s.get("selected_sector")),
                (t, 1.05),
                fontsize=7,
                ha="center",
                color="C0",
            )
        else:
            ax.scatter(t, 1, c="C0", marker="x", s=20, alpha=0.4)
        if w["boosted"]:
            ax.scatter(t, 0, c="C2", marker="s", s=30, zorder=3)
            ax.annotate(
                str(w.get("selected_sector")),
                (t, -0.15),
                fontsize=7,
                ha="center",
                color="C2",
            )
        else:
            ax.scatter(t, 0, c="C2", marker="x", s=20, alpha=0.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wear_aware", "selected"])
    ax.set_xlabel("t (decision)")
    ax.set_title(
        f"D3 selection timeline dist=2 seed=0  "
        f"both_act same_sec={same_sector} diff_sec={diff_sector}"
    )
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out_plot2 = RESULTS / "d3_selection_timeline.png"
    fig.savefig(out_plot2, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot2}")

    return {
        "log_diff_dist2_seed0": log_diff,
        "multi_seed": {
            str(d): {
                k: v
                for k, v in multi[d].items()
            }
            for d in distances
        },
        "seeds": list(seeds),
        "pre_boosts": pre_boosts,
        "horizon": horizon,
        "plot_multiseed": str(out_plot),
        "plot_timeline": str(out_plot2),
    }


# ---------------------------------------------------------------------------
# D4 — noise-type death blind spot
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_noise_survival(
    model: SelfModelV35,
    data: Dict[str, torch.Tensor],
    device: str,
) -> Dict:
    noise = filter_by_pert_type(data, PERT_NOISE)
    if noise["S"].shape[0] < 5:
        return {"n": 0, "death_rate": float("nan"), "balanced_acc": float("nan")}
    cal = evaluate_calibration_v35(
        model, noise, device=device, label="noise", wear_input="multipliers"
    )
    return {
        "n": cal["n"],
        "death_rate": cal["death_rate_true"],
        "balanced_acc": cal["survival_balanced_acc"],
        "death_precision": cal["death_precision"],
        "death_recall": cal["death_recall"],
        "tp": cal["tp"],
        "fp": cal["fp"],
        "fn": cal["fn"],
        "tn": cal["tn"],
        "n_true_death": cal["n_true_death"],
    }


@torch.no_grad()
def _generate_late_noise_forks(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    wear_cfg: PerWellWearConfig,
    cost_cfg: BoostCostConfig,
    device: str,
    n_target: int = 800,
    pred_horizon: int = 400,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Force noise-type examples with nontrivial death rate by forking from
    *late* host states (high wear / low mass) at far distance, with a longer
    prediction horizon. This is the cheapest way to get noise-class death labels
    if mid-episode noise rarely kills within H=200.
    """
    from embodiednca.v0.metrics import (
        DEATH_MASS_EPS,
        DEFAULT_ACTIVATION_THRESHOLD,
        coherence,
        local_resource_at_pattern_centroid,
        self_mass,
    )
    from embodiednca.v0.grid_world import ResourceGrid, place_pattern
    from embodiednca.v3_5.wear_perwell import (
        PerWellWearState,
        nearest_well_distance,
        step_with_perwell_wear,
    )
    from embodiednca.v1.rollout_dataset import (
        BASE_CENTER,
        apply_perturbation,
        sample_center_at_distance,
    )
    from embodiednca.v2.boost_rollout_dataset import make_boost_map, make_perturbation_map
    from embodiednca.v3_5.death_aware_rollout_dataset import PERT_NOISE

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    H, W = world_cfg.H, world_cfg.W
    rows: List[Dict[str, torch.Tensor]] = []
    episode = 0
    max_episodes = max(30, n_target // 3 + 20)

    while len(rows) < n_target and episode < max_episodes:
        episode += 1
        dist = 6.0 + 7.0 * float(torch.rand(1, generator=rng).item())
        center = sample_center_at_distance(
            DEFAULT_WELLS, H, W, dist, rng
        )
        S0 = place_pattern(base_pattern, center, base_center=BASE_CENTER)
        world = ResourceGrid(
            nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed + episode
        )
        wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)
        ep_H = 2200
        # host with occasional boosts
        boost_rem = 0
        bfield = None
        bloc = None
        snapshots = []
        for t in range(1, ep_H + 1):
            if t % 20 == 0 and float(torch.rand(1, generator=rng).item()) < 0.4:
                mag = world.S.norm(dim=1)[0]
                active = (mag > DEFAULT_ACTIVATION_THRESHOLD).nonzero(as_tuple=False)
                if active.numel() > 0:
                    idx = int(torch.randint(0, active.shape[0], (1,), generator=rng).item())
                    by, bx = int(active[idx, 0]), int(active[idx, 1])
                    bfield = make_boost_map(H, W, by, bx, 0.4, device=device, sigma=1.5)
                    boost_rem = 20
                    bloc = (by, bx)
            bmap = None
            charge = False
            loc = None
            if boost_rem > 0 and bfield is not None:
                bmap = bfield
                loc = bloc
                charge = boost_rem == 20
                boost_rem -= 1
            step_with_perwell_wear(
                world, wear, boost_map=bmap, gate_mode="gated",
                direct_boost_cost=cost_cfg.direct_boost_cost if charge else 0.0,
                boost_location=loc, charge_direct_cost=charge,
            )
            mass = self_mass(world.S)
            # Late-stage: t>600, still alive but degraded, or high wear
            if t >= 600 and mass > DEATH_MASS_EPS and (
                mass < 40 or max(wear.wear) > 1.5 or t % 100 == 0
            ):
                snapshots.append(
                    (
                        world.S.detach().clone(),
                        world.R.detach().clone(),
                        list(wear.wear),
                        t,
                        dist,
                    )
                )
            if mass <= DEATH_MASS_EPS:
                break

        # Fork noise from up to 6 late snapshots
        for S_t, R_t, wear_snap, t_host, dist in snapshots[-6:]:
            if len(rows) >= n_target:
                break
            world_f = ResourceGrid(
                nca, config=world_cfg, S0=S_t.clone(), pattern_center=None, seed=0
            )
            world_f.S = S_t.clone()
            world_f.R = R_t.clone()
            wear_f = PerWellWearState(wear_cfg, H=H, W=W, device=device)
            wear_f.wear = list(wear_snap)
            S_ref = world_f.S.detach().clone()
            mults = wear_f.multipliers()

            mag_map = world_f.S.norm(dim=1)[0]
            active = (mag_map > DEFAULT_ACTIVATION_THRESHOLD).nonzero(as_tuple=False)
            if active.numel() > 0:
                idx = int(torch.randint(0, active.shape[0], (1,), generator=rng).item())
                py, px = int(active[idx, 0]), int(active[idx, 1])
            else:
                py = int(torch.randint(0, H, (1,), generator=rng).item())
                px = int(torch.randint(0, W, (1,), generator=rng).item())
            k = 3 if float(torch.rand(1, generator=rng).item()) < 0.5 else 5
            S_pert, mag = apply_perturbation(
                world_f.S, py, px, "noise", k, 1.5, rng
            )
            world_f.S = S_pert
            died_t = None
            if self_mass(world_f.S) <= DEATH_MASS_EPS:
                died_t = 0
            for h in range(1, pred_horizon + 1):
                step_with_perwell_wear(
                    world_f, wear_f, boost_map=None, gate_mode="gated"
                )
                if died_t is None and self_mass(world_f.S) <= DEATH_MASS_EPS:
                    died_t = h
            coh = coherence(world_f.S, S_ref)
            local_r = local_resource_at_pattern_centroid(world_f.R, world_f.S)
            died = 1.0 if died_t is not None else 0.0
            coh_t = 0.0 if (isinstance(coh, float) and math.isnan(coh)) else float(coh)
            lr_t = 0.0 if (isinstance(local_r, float) and math.isnan(local_r)) else float(local_r)
            pert_map = make_perturbation_map(H, W, py, px, abs(mag), device=device)
            # centroid
            mag_s = S_t.norm(dim=1)[0]
            mask = mag_s > DEFAULT_ACTIVATION_THRESHOLD
            if mask.any():
                ys, xs = torch.where(mask)
                w = mag_s[mask]
                cy = float((ys.float() * w).sum() / w.sum())
                cx = float((xs.float() * w).sum() / w.sum())
            else:
                cy, cx = H / 2.0, W / 2.0
            rows.append(
                {
                    "S": S_t[0].cpu().float(),
                    "R": R_t[0].cpu().float(),
                    "pert_map": pert_map[0].cpu().float(),
                    "wear": torch.tensor(wear_snap, dtype=torch.float32),
                    "multipliers": torch.tensor(mults, dtype=torch.float32),
                    "pert_type": torch.tensor(PERT_NOISE, dtype=torch.int64),
                    "coherence_future": torch.tensor(coh_t, dtype=torch.float32),
                    "local_R_future": torch.tensor(lr_t, dtype=torch.float32),
                    "survived": torch.tensor(1.0 - died, dtype=torch.float32),
                    "died_within_horizon": torch.tensor(died, dtype=torch.float32),
                    "distance_to_well": torch.tensor(
                        nearest_well_distance(cy, cx, DEFAULT_WELLS), dtype=torch.float32
                    ),
                }
            )
        if verbose and (episode % 10 == 0 or episode == 1):
            nd = sum(1 for r in rows if float(r["died_within_horizon"]) > 0.5)
            print(
                f"  late-noise episode {episode}: n={len(rows)}/{n_target}  "
                f"death_rate={nd / max(1, len(rows)):.3f}"
            )

    if not rows:
        return {"S": torch.zeros(0)}
    keys = rows[0].keys()
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}


def diagnostic4_noise_death(
    nca,
    base_pattern: torch.Tensor,
    device: str,
    quick: bool = False,
    wear_cfg: Optional[PerWellWearConfig] = None,
) -> Dict:
    """
    Scope the noise-type death blind spot: scarcity vs modeling gap.

    Protocol:
      1. Report baseline death scarcity under noise in the Stage A dataset.
      2. Generate late-state far-distance noise forks with longer pred_horizon
         (to actually produce death labels under noise).
      3. Train on original train + late-noise deaths; evaluate on held-out
         late-noise forks.
      4. Verdict: RESOLVED_BY_MORE_DATA vs PERSISTS_MODELING_GAP.
    """
    print("\n=== D4: noise-type death blind spot ===")
    wear_cfg = wear_cfg or DEFAULT_WEAR
    cost_cfg = DEFAULT_COST

    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    train_data, val_data = train_val_split(data, val_frac=0.15, seed=0)
    model0 = load_model(MODEL_PATH, device=device)

    base_noise_val = _eval_noise_survival(model0, val_data, device)
    noise_all = filter_by_pert_type(data, PERT_NOISE)
    n_dead_noise = int((noise_all["survived"] < 0.5).sum().item())
    # Compare death rates by type in full dataset
    print("  death rates by pert type (full Stage A dataset):")
    for code, name in PERT_TYPE_NAMES.items():
        sub = filter_by_pert_type(data, code)
        n = sub["S"].shape[0]
        nd = int((sub["survived"] < 0.5).sum().item()) if n else 0
        print(f"    {name:6s}: n={n:5d}  n_dead={nd:4d}  death_rate={nd/max(1,n):.4f}")
    print(
        f"  baseline noise-in-val: n={base_noise_val['n']}  "
        f"death_rate={base_noise_val['death_rate']:.4f}  "
        f"bal_acc={base_noise_val['balanced_acc']:.4f}  "
        f"TP/FN={base_noise_val.get('tp')}/{base_noise_val.get('fn')}"
    )
    print(
        f"  full dataset noise: n={noise_all['S'].shape[0]}  "
        f"n_dead={n_dead_noise}  "
        f"death_rate={n_dead_noise / max(1, noise_all['S'].shape[0]):.4f}"
    )
    print(
        "  NOTE: noise-type death is structurally rare under Stage A sampling "
        "(mid-episode noise rarely kills within pred_H=200). Late-state forks "
        "with longer H are needed to even *pose* the question."
    )

    # Generate late-noise train + holdout
    n_train_noise = 400 if quick else 1200
    n_eval_noise = 150 if quick else 400
    pred_H = 300 if quick else 400
    nca_d, base_pat, cfg = load_v0_world(DEFAULT_CKPT, device=device)
    world_cfg = WorldConfig(
        H=cfg.H, W=cfg.W, channels=cfg.channels, hidden_dim=cfg.hidden_dim,
        gen_rate=wear_cfg.gen_rate_base, consumption_rate=cfg.consumption_rate,
        decay_rate=cfg.decay_rate, diffusion_rate=cfg.diffusion_rate,
        resource_diffusion=cfg.resource_diffusion, well_sigma=cfg.well_sigma,
        wells=DEFAULT_WELLS, device=device,
    )

    print(f"  generating {n_train_noise + n_eval_noise} late-state noise forks "
          f"(pred_H={pred_H})...")
    late_all = _generate_late_noise_forks(
        nca_d, base_pat, world_cfg, wear_cfg, cost_cfg, device,
        n_target=n_train_noise + n_eval_noise,
        pred_horizon=pred_H,
        seed=77,
        verbose=True,
    )
    n_late = late_all["S"].shape[0]
    n_late_dead = int((late_all["survived"] < 0.5).sum().item()) if n_late else 0
    print(
        f"  late-noise total n={n_late}  n_dead={n_late_dead}  "
        f"death_rate={n_late_dead / max(1, n_late):.3f}"
    )

    if n_late < 20 or n_late_dead < 5:
        verdict = "STRUCTURALLY_RARE_NOISE_DEATH"
        print(
            f"  VERDICT: {verdict} — even late forks don't produce enough "
            f"noise-type deaths to train/eval. Not a model gap; noise rarely "
            f"kills within the prediction window under this physics."
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            list(PERT_TYPE_NAMES.values()),
            [
                float((filter_by_pert_type(data, c)["survived"] < 0.5).float().mean().item())
                if filter_by_pert_type(data, c)["S"].shape[0]
                else 0.0
                for c in PERT_TYPE_NAMES
            ],
            color=["C0", "C1", "C2", "C3"],
        )
        ax.set_ylabel("death rate")
        ax.set_title("D4: death rate by pert type (Stage A dataset)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out_plot = RESULTS / "d4_noise_death_confusion.png"
        fig.savefig(out_plot, dpi=140)
        plt.close(fig)
        return {
            "baseline_noise_val": base_noise_val,
            "full_dataset_noise_n": int(noise_all["S"].shape[0]),
            "full_dataset_noise_n_dead": n_dead_noise,
            "late_noise_n": n_late,
            "late_noise_n_dead": n_late_dead,
            "verdict": verdict,
            "clean_before": {},
            "clean_after": {},
            "plot": str(out_plot),
        }

    # Split late_all into train/holdout stratified by death
    g = torch.Generator().manual_seed(123)
    dead_idx = torch.where(late_all["survived"] < 0.5)[0]
    live_idx = torch.where(late_all["survived"] >= 0.5)[0]
    # 25% holdout of each class
    def split_idx(idx):
        if len(idx) == 0:
            return idx, idx
        perm = idx[torch.randperm(len(idx), generator=g)]
        n_h = max(1, len(perm) // 4)
        return perm[n_h:], perm[:n_h]

    tr_d, ho_d = split_idx(dead_idx)
    tr_l, ho_l = split_idx(live_idx)
    train_idx = torch.cat([tr_d, tr_l])
    hold_idx = torch.cat([ho_d, ho_l])
    train_idx = train_idx[torch.randperm(len(train_idx), generator=g)]
    hold_idx = hold_idx[torch.randperm(len(hold_idx), generator=g)]

    late_train = {k: v[train_idx] for k, v in late_all.items()}
    late_hold = {k: v[hold_idx] for k, v in late_all.items()}
    print(
        f"  late train n={late_train['S'].shape[0]}  "
        f"dead={int((late_train['survived']<0.5).sum())}  |  "
        f"hold n={late_hold['S'].shape[0]}  "
        f"dead={int((late_hold['survived']<0.5).sum())}"
    )

    # Baseline on clean late holdout
    clean_before = _eval_noise_survival(model0, late_hold, device)
    print(
        f"  model0 on late-hold: bal_acc={clean_before['balanced_acc']:.3f}  "
        f"death_rate={clean_before['death_rate']:.3f}  "
        f"P/R={clean_before['death_precision']:.3f}/{clean_before['death_recall']:.3f}  "
        f"TP/FN={clean_before['tp']}/{clean_before['fn']}"
    )

    # Merge late_train into original train; oversample late deaths
    def cat_dict(a: Dict, b: Dict) -> Dict:
        keys = [k for k in a if isinstance(a[k], torch.Tensor) and k in b]
        return {k: torch.cat([a[k], b[k]], dim=0) for k in keys}

    # Ensure late_train has all keys train_data needs
    for k in train_data:
        if isinstance(train_data[k], torch.Tensor) and k not in late_train:
            # pad missing keys with zeros of right shape
            shape = list(train_data[k].shape)
            shape[0] = late_train["S"].shape[0]
            late_train[k] = torch.zeros(*shape, dtype=train_data[k].dtype)

    train_aug = cat_dict(train_data, late_train)
    # Oversample noise deaths in train_aug
    noise_dead = (train_aug["pert_type"] == PERT_NOISE) & (train_aug["survived"] < 0.5)
    n_nd = int(noise_dead.sum().item())
    if n_nd > 0:
        nd_idx = torch.where(noise_dead)[0]
        reps = max(2, int(min(8, 400 / max(1, n_nd))))
        rep_idx = nd_idx.repeat(reps)
        g2 = torch.Generator().manual_seed(42)
        base_idx = torch.arange(train_aug["S"].shape[0])
        new_idx = torch.cat([base_idx, rep_idx])
        new_idx = new_idx[torch.randperm(new_idx.numel(), generator=g2)]
        train_aug = {
            k: v[new_idx] for k, v in train_aug.items() if isinstance(v, torch.Tensor)
        }
    print(
        f"  train_aug n={train_aug['S'].shape[0]}  "
        f"noise_dead={int(((train_aug['pert_type']==PERT_NOISE)&(train_aug['survived']<0.5)).sum())}"
    )

    from torch.utils.data import DataLoader
    from embodiednca.v3_5.self_model_v3_5 import self_model_loss, _batch_wear_features

    tcfg = TrainConfig(
        epochs=12 if quick else 25,
        batch_size=64,
        device=device,
        seed=0,
        wear_input="multipliers",
        hidden=48,
        surv_loss_scale=2.5,
    )
    n_dead = float((train_aug["survived"] < 0.5).sum().item())
    n_live = float((train_aug["survived"] >= 0.5).sum().item())
    death_weight = (
        float(min(max(n_live / max(n_dead, 1.0), 2.0), 12.0)) if n_dead > 0 else None
    )
    model1 = SelfModelV35(
        channels=32, hidden=tcfg.hidden, n_wells=2, wear_input="multipliers"
    ).to(device)
    model1.load_state_dict(model0.state_dict())
    opt = torch.optim.Adam(model1.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loader = DataLoader(
        DeathAwareTensorDataset(train_aug), batch_size=tcfg.batch_size, shuffle=True
    )
    print("  fine-tuning on train + late-noise deaths...")
    for epoch in range(1, tcfg.epochs + 1):
        model1.train()
        losses = []
        for batch in loader:
            S = batch["S"].to(device)
            R = batch["R"].to(device)
            P = batch["pert_map"].to(device)
            Wf = _batch_wear_features(batch, "multipliers", device)
            y_coh = batch["coherence_future"].to(device)
            y_lr = batch["local_R_future"].to(device)
            y_surv = batch["survived"].to(device)
            opt.zero_grad(set_to_none=True)
            p_coh, p_lr, p_surv = model1(S, R, P, Wf)
            loss, stats = self_model_loss(
                p_coh, p_lr, p_surv, y_coh, y_lr, y_surv,
                surv_pos_weight=death_weight, surv_loss_scale=tcfg.surv_loss_scale,
            )
            loss.backward()
            opt.step()
            losses.append(stats["loss"])
        if epoch == 1 or epoch % 5 == 0 or epoch == tcfg.epochs:
            nv = _eval_noise_survival(model1, late_hold, device)
            print(
                f"  epoch {epoch:3d}/{tcfg.epochs}  loss={sum(losses)/len(losses):.4f}  "
                f"late-hold bal_acc={nv['balanced_acc']:.3f}  "
                f"P/R={nv['death_precision']:.2f}/{nv['death_recall']:.2f}  "
                f"TP/FN={nv['tp']:.0f}/{nv['fn']:.0f}"
            )

    clean_after = _eval_noise_survival(model1, late_hold, device)
    print(
        f"  model1 on late-hold: bal_acc={clean_after['balanced_acc']:.3f}  "
        f"P/R={clean_after['death_precision']:.3f}/{clean_after['death_recall']:.3f}  "
        f"TP/FN={clean_after['tp']}/{clean_after['fn']}"
    )

    b0 = clean_before["balanced_acc"]
    b1 = clean_after["balanced_acc"]
    n_death_hold = clean_before["n_true_death"]
    if n_death_hold < 5:
        verdict = "INSUFFICIENT_EVAL_DEATHS"
    elif b1 >= 0.65 and (b1 - b0) >= 0.10:
        verdict = "RESOLVED_BY_MORE_DATA"
    elif b1 < 0.60 and b0 < 0.60:
        verdict = "PERSISTS_MODELING_GAP"
    elif b1 > b0 + 0.05:
        verdict = "IMPROVED_BUT_PARTIAL"
    else:
        verdict = "NO_CLEAR_IMPROVEMENT"
    print(f"  VERDICT: {verdict}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, lab, m in (
        (axes[0], "baseline model0", clean_before),
        (axes[1], "after late-noise data model1", clean_after),
    ):
        mat = [[m.get("tn", 0), m.get("fp", 0)], [m.get("fn", 0), m.get("tp", 0)]]
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred live", "pred death"])
        ax.set_yticklabels(["true live", "true death"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i][j]:.0f}", ha="center", va="center")
        ax.set_title(
            f"{lab}\nbal_acc={m.get('balanced_acc', float('nan')):.3f}  "
            f"n_death={m.get('n_true_death', 0):.0f}"
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"D4 noise-type death (late forks)  [{verdict}]")
    fig.tight_layout()
    out_plot = RESULTS / "d4_noise_death_confusion.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    torch.save(
        {
            "state_dict": model1.state_dict(),
            "channels": 32,
            "hidden": tcfg.hidden,
            "n_wells": 2,
            "wear_input": "multipliers",
            "version": "v3.5-noise-aug",
        },
        RESULTS / "self_model_v3_5_noise_aug.pt",
    )

    return {
        "baseline_noise_val": base_noise_val,
        "full_dataset_noise_n": int(noise_all["S"].shape[0]),
        "full_dataset_noise_n_dead": n_dead_noise,
        "late_noise_n": n_late,
        "late_noise_n_dead": n_late_dead,
        "late_hold_n": int(late_hold["S"].shape[0]),
        "late_hold_n_dead": int((late_hold["survived"] < 0.5).sum().item()),
        "clean_before": clean_before,
        "clean_after": clean_after,
        "verdict": verdict,
        "plot": str(out_plot),
    }


# ---------------------------------------------------------------------------
# Write observations
# ---------------------------------------------------------------------------
def write_diag_observations(d1: Dict, d2: Dict, d3: Dict, d4: Dict) -> None:
    lines: List[str] = []
    lines.append("# v3.5 post-stage diagnostics (D1–D4)\n")
    lines.append(
        "Follow-ups to the Stage A result that a hand-coded wear_aware_heuristic "
        "ties or slightly beats selected. Resolve what we can trust before any v4.\n"
    )

    lines.append("\n## D1 — local_R within-bin collapse\n")
    lines.append(f"- overall local_R r={d1['overall_local_R_r']:.4f}  "
                 f"MAE={d1['overall_local_R_mae']:.4f}  "
                 f"true_std={d1['overall_true_std']:.4f}\n")
    lines.append(f"- **headline: {d1['headline']}**\n")
    for blabel, info in d1["by_bin"].items():
        if "diagnosis" not in info:
            continue
        lines.append(
            f"- {blabel}: n={info['n']}  true_mean±std="
            f"{info['true_mean']:.4f}±{info['true_std']:.4f}  "
            f"pred_std={info['pred_std']:.4f}  "
            f"r={info['local_R_r']}  MAE={info['local_R_mae']:.4f}  "
            f"**{info['diagnosis']}**\n"
        )
    lines.append(
        f"- mid/far abs_err vs wear_asym r={d1['mid_far_r_err_vs_wear_asym']}\n"
    )
    lines.append(
        f"- mid/far MAE high_asym={d1['mid_far_mae_high_asym']}  "
        f"low_asym={d1['mid_far_mae_low_asym']}\n"
    )
    lines.append(
        f"- mid/far r high_asym={d1['mid_far_r_high_asym']}  "
        f"low_asym={d1['mid_far_r_low_asym']}\n"
    )
    # Interpretation block
    collapse_bins = [
        k for k, v in d1["by_bin"].items() if v.get("diagnosis") == "VARIANCE_COLLAPSE"
    ]
    fail_bins = [
        k for k, v in d1["by_bin"].items() if v.get("diagnosis") == "GENUINE_FAILURE"
    ]
    low_ok = [
        k for k, v in d1["by_bin"].items() if v.get("diagnosis") == "LOW_R_OK_MAE"
    ]
    lines.append("\n### D1 interpretation\n")
    if collapse_bins or low_ok:
        lines.append(
            f"- Bins {collapse_bins + low_ok}: low r is primarily a "
            f"**variance-collapse / uninformative-r** artifact (true local_R "
            f"has little within-bin spread; MAE remains small). Same pattern "
            f"as v1/v2 far-distance bins — not a new per-well-wear failure mode "
            f"by itself.\n"
        )
    if fail_bins:
        lines.append(
            f"- Bins {fail_bins}: **genuine prediction failure** "
            f"(non-trivial true variance + low r + high MAE). Investigate "
            f"further before trusting local_R-driven decisions in that regime.\n"
        )
    r_wa = d1["mid_far_r_err_vs_wear_asym"]
    if isinstance(r_wa, float) and not math.isnan(r_wa) and abs(r_wa) > 0.15:
        lines.append(
            f"- abs_err correlates with wear asymmetry (r={r_wa:.3f}): "
            f"input-representation gap for *which well's* multiplier drives "
            f"local R is a plausible, fixable issue.\n"
        )
    else:
        lines.append(
            f"- abs_err vs wear asymmetry r={r_wa}: no strong correlation — "
            f"does not specifically implicate asymmetric-wear encoding as the "
            f"driver of residual local_R error.\n"
        )
    lines.append(f"- plot: {d1['plot']}\n")

    lines.append("\n## D2 — decoupled floor-crossing\n")
    lines.append(
        f"- initial_wear (no physical pre-boost)={d2.get('initial_wear', d2.get('pre_wear_preview'))}  "
        f"mult≈{d2.get('pre_mult_preview')}\n"
    )
    lines.append(
        f"- dist={d2['distance']} horizon={d2['horizon']} tod={d2['time_of_death']}\n"
    )
    lines.append(
        f"- t_floor A/B/all={d2['t_floor_0']}/{d2['t_floor_1']}/{d2['t_all_floor']}  "
        f"post_floor_life={d2['post_floor_life']}  "
        f"decoupled_ok={d2['decoupled_ok']}\n"
    )
    for name, w in d2.get("windows", {}).items():
        lines.append(
            f"- window {name}: boost_rate={w['boost_rate']}  "
            f"vol_skip={w['vol_skip_rate']}  n={w['n']}\n"
        )
    lines.append(f"- **verdict: {d2['verdict']}**\n")
    lines.append("\n### D2 interpretation\n")
    if d2["verdict"] == "CONFIRMED_FLOOR_AWARE_SKIP":
        lines.append(
            "- Skip-spike reappears at floor-crossing with long post-floor "
            "survival → real evidence of learned floor-awareness. Timing "
            "hypothesis from v3 stays open and worth leveraging.\n"
        )
    elif d2["verdict"] == "NO_FLOOR_TIMING_SIGNAL":
        lines.append(
            "- No skip-spike at floor-crossing when death is decoupled → "
            "dist=4 late-skip in Stage A was near-death, not floor-timing. "
            "**Close the v3 timing hypothesis** as not supported by the "
            "self-model under this OP.\n"
        )
    elif d2["verdict"] == "SCENARIO_FAILED_NOT_DECOUPLED":
        lines.append(
            "- Scenario failed to produce 1000+ post-floor life; re-tune "
            "shallow pre-degrade / distance before concluding.\n"
        )
    else:
        lines.append(f"- Verdict {d2['verdict']}: see windows and plot.\n")
    lines.append(f"- plot: {d2['plot']}\n")

    lines.append("\n## D3 — selected vs wear_aware: what differs + multi-seed\n")
    ld = d3["log_diff_dist2_seed0"]
    lines.append(
        f"- dist=2 seed=0 (Check 3 seed convention): "
        f"sel mean_coh={ld['selected_mean_coh']:.4f}  "
        f"wah={ld['wear_aware_mean_coh']:.4f}\n"
    )
    lines.append(
        f"- alive boosts: selected={ld['selected_n_boosts_alive']}  "
        f"wear_aware={ld['wear_aware_n_boosts_alive']}\n"
    )
    lines.append(
        f"- common alive decisions: {ld['n_common_alive']}  "
        f"both_act={ld['both_act']}  only_sel={ld['only_selected']}  "
        f"only_wah={ld['only_wear_aware']}  both_skip={ld['both_skip']}\n"
    )
    lines.append(
        f"- when both act: same_sector={ld['same_sector_when_both_act']}  "
        f"diff_sector={ld['diff_sector_when_both_act']}  "
        f"same_well={ld['same_well_when_both_act']}  "
        f"diff_well={ld['diff_well_when_both_act']}\n"
    )
    lines.append("\n### Multi-seed reproducibility (3 seeds)\n")
    for dist, info in d3["multi_seed"].items():
        lines.append(
            f"- dist={dist}: mean Δ(sel−wah)={info['mean_delta']:+.4f} ± "
            f"{info['std_delta']:.4f}  "
            f"mean_sel={info['mean_selected']:.4f}  "
            f"mean_wah={info['mean_wear_aware']:.4f}  "
            f"**{info['reproducibility']}**\n"
        )
        for r in info["selected"]:
            # find matching wah
            pass
        for i, r in enumerate(info["selected"]):
            w = info["wear_aware_heuristic"][i]
            dlt = info["deltas"][i]
            lines.append(
                f"  - seed={r['seed']}: sel={r['mean_coherence']:.4f}  "
                f"wah={w['mean_coherence']:.4f}  Δ={dlt:+.4f}  "
                f"tod_sel/wah={r['time_of_death']}/{w['time_of_death']}\n"
            )
    lines.append("\n### D3 interpretation\n")
    # auto summary from multi_seed
    repros = {d: info["reproducibility"] for d, info in d3["multi_seed"].items()}
    lines.append(f"- Reproducibility labels: {repros}\n")
    if ld["diff_well_when_both_act"] > ld["same_well_when_both_act"]:
        lines.append(
            "- Residual gap is more about *which well* than about frequency "
            "(both policies take a similar number of boosts).\n"
        )
    elif ld["diff_sector_when_both_act"] > 0 and ld["same_well_when_both_act"] >= ld["diff_well_when_both_act"]:
        lines.append(
            "- When both act they usually target the same well; residual gap "
            "is more about sector/timing within a well than well choice.\n"
        )
    else:
        lines.append(
            "- Both-act agreement is high on well and/or sector; residual "
            "Δ may be mostly *when* each chooses to act (only_sel vs only_wah "
            "timings) rather than where.\n"
        )
    lines.append(f"- plots: {d3['plot_multiseed']}, {d3['plot_timeline']}\n")

    lines.append("\n## D4 — noise-type death blind spot\n")
    if d4.get("verdict") == "SKIPPED":
        lines.append("- skipped this run\n")
    else:
        bn = d4.get("baseline_noise_val") or {}
        lines.append(
            f"- baseline noise-in-val: n={bn.get('n')}  "
            f"death_rate={bn.get('death_rate')}  "
            f"bal_acc={bn.get('balanced_acc')}  "
            f"TP/FN={bn.get('tp')}/{bn.get('fn')}\n"
        )
        lines.append(
            f"- full Stage A noise: n={d4.get('full_dataset_noise_n')}  "
            f"n_dead={d4.get('full_dataset_noise_n_dead')}\n"
        )
        lines.append(
            f"- late-state noise forks: n={d4.get('late_noise_n')}  "
            f"n_dead={d4.get('late_noise_n_dead')}  "
            f"hold n/dead={d4.get('late_hold_n')}/{d4.get('late_hold_n_dead')}\n"
        )
        cb = d4.get("clean_before") or {}
        ca = d4.get("clean_after") or {}
        lines.append(
            f"- model0 late-hold bal_acc={cb.get('balanced_acc')}  "
            f"P/R={cb.get('death_precision')}/{cb.get('death_recall')}\n"
        )
        lines.append(
            f"- model1 late-hold bal_acc={ca.get('balanced_acc')}  "
            f"P/R={ca.get('death_precision')}/{ca.get('death_recall')}\n"
        )
        lines.append(f"- **verdict: {d4.get('verdict')}**\n")
        lines.append("\n### D4 interpretation\n")
        if d4.get("verdict") == "RESOLVED_BY_MORE_DATA":
            lines.append(
                "- Once late-state noise forks produce real death labels, more "
                "data fixes the blind spot. Not a modeling gap; Stage A simply "
                "had almost no noise-class deaths (noise rarely kills mid-episode "
                "within pred_H=200).\n"
            )
        elif d4.get("verdict") == "PERSISTS_MODELING_GAP":
            lines.append(
                "- Even with late-state noise deaths in training, held-out "
                "balanced accuracy stays near chance → real gap in noise×wear×death "
                "prediction. Low priority for Check 3 (which is about boost "
                "selection, not noise damage).\n"
            )
        elif d4.get("verdict") == "STRUCTURALLY_RARE_NOISE_DEATH":
            lines.append(
                "- Noise-type death is **structurally rare** under this physics "
                "even with late forks (0 deaths in 1498 late-state noise forks "
                "at pred_H=400). The 'blind spot' is a label scarcity / physics "
                "fact (noise is non-lethal within the prediction window), not a "
                "model failure. Do not chase.\n"
            )
        else:
            lines.append(
                f"- Verdict {d4.get('verdict')}: intermediate or thin eval. "
                f"Lower priority than D1–D3.\n"
            )
        lines.append(f"- plot: {d4.get('plot')}\n")

    lines.append("\n## Bigger picture (before v4)\n")
    lines.append(
        "v3.5's central finding stands: a properly strengthened heuristic ties "
        "or slightly beats the learned model under asymmetric wear — the second "
        "time in a row a hand-coded rule matches a trained self-model on this "
        "task family (after v2 geometry). Do not assume the next mechanic will "
        "be the one that finally makes the model pull ahead.\n"
    )
    lines.append(
        "Trust order for downstream work:\n"
        "1. D1 local_R diagnosis — only trust resource-state claims if "
        "collapse bins are variance artifacts (or fix genuine failures).\n"
        "2. D2 floor-timing — closed or confirmed; stop treating Stage A "
        "dist=4 skip spike as evidence of floor-awareness unless D2 confirms.\n"
        "3. D3 multi-seed — only treat selected vs wear_aware rankings as "
        "real if reproducible across seeds.\n"
        "4. D4 noise-death — scarcity vs modeling; low priority for Check 3 "
        "interpretation.\n"
    )

    path = RESULTS / "diagnostics_observations.txt"
    path.write_text("".join(lines))
    print(f"\n  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="v3.5 post-stage diagnostics D1–D4")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip_d2", action="store_true")
    p.add_argument("--skip_d3", action="store_true")
    p.add_argument("--skip_d4", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    print(f"device={device}  quick={args.quick}")

    if not MODEL_PATH.is_file() or not DATASET_PATH.is_file():
        raise SystemExit(
            f"Need {MODEL_PATH} and {DATASET_PATH}. Run python scripts/run_v3_5_checks.py first."
        )

    wear_cfg = DEFAULT_WEAR
    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    _, val_data = train_val_split(data, val_frac=0.15, seed=0)
    model = load_model(MODEL_PATH, device=device)

    nca, base_pattern, cfg = load_v0_world(DEFAULT_CKPT, device=device)
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
        wells=DEFAULT_WELLS,
        device=device,
    )

    d1 = diagnostic1_local_R(model, val_data, device)

    if args.skip_d2:
        d2 = {"verdict": "SKIPPED", "windows": {}, "plot": None}
    else:
        d2 = diagnostic2_decoupled_floor(
            nca, base_pattern, world_cfg, model, wear_cfg, device, quick=args.quick
        )

    if args.skip_d3:
        d3 = {
            "log_diff_dist2_seed0": {},
            "multi_seed": {},
            "plot_multiseed": None,
            "plot_timeline": None,
        }
    else:
        d3 = diagnostic3_selection_diff_and_seeds(
            nca, base_pattern, world_cfg, model, wear_cfg, device, quick=args.quick
        )

    if args.skip_d4:
        d4 = {"verdict": "SKIPPED", "baseline_noise_val": {}, "clean_before": {}, "clean_after": {}, "plot": None}
    else:
        d4 = diagnostic4_noise_death(
            nca, base_pattern, device, quick=args.quick, wear_cfg=wear_cfg
        )

    write_diag_observations(d1, d2, d3, d4)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    with (RESULTS / "diagnostics_summary.json").open("w") as f:
        json.dump(sanitize({"d1": d1, "d2": d2, "d3": d3, "d4": d4}), f, indent=2)
    print(f"  wrote {RESULTS / 'diagnostics_summary.json'}")
    print("\n=== v3.5 diagnostics done ===")
    print(f"  D1: {d1.get('headline')}")
    print(f"  D2: {d2.get('verdict')}")
    if d3.get("multi_seed"):
        for dist, info in d3["multi_seed"].items():
            print(
                f"  D3 dist={dist}: {info.get('reproducibility')}  "
                f"Δ={info.get('mean_delta')}"
            )
    print(f"  D4: {d4.get('verdict')}")


if __name__ == "__main__":
    main()
