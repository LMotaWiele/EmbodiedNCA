#!/usr/bin/env python3
"""
run_v3_5_checks.py
==================
v3.5 Per-Well Wear + Death-Aware Self-Model + No-Boost Option.

  Check 1 — per-well wear mechanic sanity (no cross-contamination)
  Check 2 — death-aware calibration (survival balanced accuracy gate)
  Check 3 — asymmetric-wear policy comparison (incl. wear_aware_heuristic)
  Check 4 — timing behavior (boost rate vs floor-crossing)

Does not modify v0 step, SimpleNCA, or frozen v1/v2 checkpoints.

    python v3_5/run_v3_5_checks.py
    python v3_5/run_v3_5_checks.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

V35_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V35_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V1_ROOT = REPO_ROOT / "v1"
V2_ROOT = REPO_ROOT / "v2"
V3_ROOT = REPO_ROOT / "v3"
for p in (str(V35_ROOT), str(V3_ROOT), str(V2_ROOT), str(V1_ROOT), str(V0_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_selection import AttentionConfig  # noqa: E402
from action_selection_v3_5 import run_v35_episode  # noqa: E402
from boost_cost import BoostCostConfig  # noqa: E402
from boost_rollout_dataset import make_boost_map  # noqa: E402
from death_aware_rollout_dataset import (  # noqa: E402
    DeathAwareDatasetConfig,
    PERT_TYPE_NAMES,
    generate_dataset,
)
from grid_world import (  # noqa: E402
    DEFAULT_GEN_RATE,
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    place_pattern,
    step_resource,
)
from metrics import self_mass  # noqa: E402
from rollout_dataset import DEFAULT_CKPT, load_v0_world  # noqa: E402
from self_model_v3_5 import (  # noqa: E402
    TrainConfig,
    evaluate_calibration_v35,
    load_model,
    save_model,
    train_self_model_v3_5,
)
from wear_perwell import (  # noqa: E402
    PerWellWearConfig,
    PerWellWearState,
    closed_form_ambient_perwell,
    effective_multiplier,
    mask_value_at_well,
    nearest_well_index,
    step_with_perwell_wear,
)

RESULTS = V35_ROOT / "results"
MODEL_PATH = RESULTS / "self_model_v3_5.pt"
DATASET_PATH = RESULTS / "death_aware_dataset.pt"

# v3 operating point (reuse; Stage A established these)
DEFAULT_WEAR = PerWellWearConfig(
    wear_ambient_rate=0.001,
    wear_boost_rate=0.004,
    wear_decay_coeff=0.20,
    gen_floor_frac=0.40,
    gen_rate_base=DEFAULT_GEN_RATE,
    wells=DEFAULT_WELLS,
)
DEFAULT_COST = BoostCostConfig(direct_boost_cost=0.15)

MEANINGFUL_COH_DELTA = 0.01

POLICIES_CHECK3 = (
    "selected",
    "wellward_heuristic",
    "wear_aware_heuristic",
    "never_boost",
    "always_boost",
)

POLICY_COLORS = {
    "selected": "C0",
    "wellward_heuristic": "C4",
    "wear_aware_heuristic": "C2",
    "never_boost": "C5",
    "always_boost": "C1",
}


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _finite_mean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Check 1 — per-well wear sanity
# ---------------------------------------------------------------------------
@torch.no_grad()
def check1_perwell_sanity(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    wear_cfg: PerWellWearConfig,
    device: str,
    horizon: int = 800,
    seed: int = 0,
) -> Dict:
    """
    Asymmetric usage: place pattern near well A, boost only near A.
    Confirm wear_B only moves via ambient; mask values match closed form.
    """
    print("\n=== Check 1: per-well wear mechanic sanity ===")
    H, W = world_cfg.H, world_cfg.W
    wells = wear_cfg.wells
    well_A, well_B = wells[0], wells[1]

    # Pattern near well A only
    center = (well_A[0], well_A[1] + 2)
    S0 = place_pattern(base_pattern, center, base_center=(16, 16))
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed
    )
    wear = PerWellWearState(wear_cfg, H=H, W=W, device=device)

    # Boost fixed at well A cell (nearest = A)
    boost_map = make_boost_map(
        H, W, well_A[0], well_A[1], 0.40, device=device, sigma=1.5
    )
    assert nearest_well_index(well_A[0], well_A[1], wells) == 0

    hist: List[Dict] = []
    for t in range(1, horizon + 1):
        step_with_perwell_wear(
            world,
            wear,
            boost_map=boost_map,
            gate_mode="gated",
            direct_boost_cost=0.0,
            boost_location=well_A,
            charge_direct_cost=False,
        )
        mults = wear.multipliers()
        eff = wear.effective_generation_mask()
        # Closed-form expected mask peak at each well:
        # per-well mask peaks at 1.0; other well's contribution at this cell
        # is small (Gaussian tail). Expected at well i:
        #   sum_j mask_j[well_i] * mult_j
        expected_at = []
        for wi, wpos in enumerate(wells):
            exp_v = 0.0
            for j, m in enumerate(wear.per_well_masks):
                exp_v += float(m[0, 0, wpos[0], wpos[1]].item()) * mults[j]
            expected_at.append(exp_v)
        actual_at = [mask_value_at_well(eff, wpos) for wpos in wells]
        hist.append(
            {
                "t": t,
                "wear_0": wear.wear[0],
                "wear_1": wear.wear[1],
                "mult_0": mults[0],
                "mult_1": mults[1],
                "mask_A": actual_at[0],
                "mask_B": actual_at[1],
                "exp_mask_A": expected_at[0],
                "exp_mask_B": expected_at[1],
            }
        )

    # Closed-form trajectories
    # well A: ambient + boost every step
    # well B: ambient only
    amb = wear_cfg.wear_ambient_rate
    bst = wear_cfg.wear_boost_rate
    w0_cf, w1_cf = [], []
    wa, wb = 0.0, 0.0
    for _ in range(horizon):
        wa = wa + amb + bst
        wb = wb + amb
        w0_cf.append(wa)
        w1_cf.append(wb)

    wear0 = [h["wear_0"] for h in hist]
    wear1 = [h["wear_1"] for h in hist]
    max_err_A = max(abs(wear0[i] - w0_cf[i]) for i in range(horizon))
    max_err_B = max(abs(wear1[i] - w1_cf[i]) for i in range(horizon))

    # Cross-contamination: well B must equal ambient-only closed form
    no_cross = max_err_B < 1e-9
    # well A must get ambient+boost
    boost_only_on_A = max_err_A < 1e-9

    # Mask match
    max_mask_err_A = max(abs(h["mask_A"] - h["exp_mask_A"]) for h in hist)
    max_mask_err_B = max(abs(h["mask_B"] - h["exp_mask_B"]) for h in hist)
    mask_ok = max_mask_err_A < 1e-6 and max_mask_err_B < 1e-6

    # Multipliers closed form
    mult0_ok = all(
        abs(
            h["mult_0"]
            - effective_multiplier(
                h["wear_0"], wear_cfg.wear_decay_coeff, wear_cfg.gen_floor_frac
            )
        )
        < 1e-9
        for h in hist
    )
    mult1_ok = all(
        abs(
            h["mult_1"]
            - effective_multiplier(
                h["wear_1"], wear_cfg.wear_decay_coeff, wear_cfg.gen_floor_frac
            )
        )
        < 1e-9
        for h in hist
    )

    # Ambient-only control (both wells equal)
    wear_amb = PerWellWearState(wear_cfg, H=H, W=W, device=device)
    world_amb = ResourceGrid(
        nca, config=world_cfg, S0=S0.clone(), pattern_center=center, seed=seed + 1
    )
    for t in range(horizon):
        step_with_perwell_wear(
            world_amb, wear_amb, boost_map=None, gate_mode="gated"
        )
    amb_equal = abs(wear_amb.wear[0] - wear_amb.wear[1]) < 1e-9
    amb_cf = amb * horizon
    amb_match = abs(wear_amb.wear[0] - amb_cf) < 1e-9

    passed = (
        no_cross
        and boost_only_on_A
        and mask_ok
        and mult0_ok
        and mult1_ok
        and amb_equal
        and amb_match
    )

    print(f"  horizon={horizon}  boost always at well A {well_A}")
    print(
        f"  wear_A final={wear0[-1]:.4f}  wear_B final={wear1[-1]:.4f}  "
        f"(cf A={w0_cf[-1]:.4f} B={w1_cf[-1]:.4f})"
    )
    print(
        f"  no_cross_contamination (B ambient-only)={no_cross}  "
        f"max|err_B|={max_err_B:.2e}"
    )
    print(
        f"  boost_only_on_A={boost_only_on_A}  max|err_A|={max_err_A:.2e}"
    )
    print(
        f"  mask closed-form match: max|err_A|={max_mask_err_A:.2e}  "
        f"max|err_B|={max_mask_err_B:.2e}"
    )
    print(
        f"  ambient-only both equal={amb_equal} match_cf={amb_match}  "
        f"wear={wear_amb.wear}"
    )
    print(f"  Check 1 PASS={passed}")

    # Plot
    ts = [h["t"] for h in hist]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(ts, wear0, "C0", lw=1.5, label="wear_A (boosted well)")
    axes[0].plot(ts, wear1, "C1", lw=1.5, label="wear_B (ambient only)")
    axes[0].plot(ts, w0_cf, "C0", ls="--", alpha=0.5, label="cf A")
    axes[0].plot(ts, w1_cf, "C1", ls="--", alpha=0.5, label="cf B")
    axes[0].axhline(
        wear.wear_at_floor(), color="0.5", ls=":", label="wear@floor"
    )
    axes[0].set_ylabel("wear")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        "Check 1: asymmetric boost near well A — no cross-contamination"
    )

    axes[1].plot(ts, [h["mult_0"] for h in hist], "C0", lw=1.5, label="mult_A")
    axes[1].plot(ts, [h["mult_1"] for h in hist], "C1", lw=1.5, label="mult_B")
    axes[1].axhline(wear_cfg.gen_floor_frac, color="0.5", ls=":", label="floor")
    axes[1].set_ylabel("effective_multiplier")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, [h["mask_A"] for h in hist], "C0", lw=1.5, label="mask@A")
    axes[2].plot(ts, [h["mask_B"] for h in hist], "C1", lw=1.5, label="mask@B")
    axes[2].plot(
        ts, [h["exp_mask_A"] for h in hist], "C0", ls="--", alpha=0.5, label="exp@A"
    )
    axes[2].plot(
        ts, [h["exp_mask_B"] for h in hist], "C1", ls="--", alpha=0.5, label="exp@B"
    )
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("gen mask at well")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    out_plot = RESULTS / "check1_perwell_wear_egr.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    csv_path = RESULTS / "check1_perwell_wear.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)

    return {
        "pass": passed,
        "no_cross_contamination": no_cross,
        "boost_only_on_A": boost_only_on_A,
        "mask_ok": mask_ok,
        "max_err_wear_A": max_err_A,
        "max_err_wear_B": max_err_B,
        "max_mask_err_A": max_mask_err_A,
        "max_mask_err_B": max_mask_err_B,
        "final_wear_A": wear0[-1],
        "final_wear_B": wear1[-1],
        "final_mult_A": hist[-1]["mult_0"],
        "final_mult_B": hist[-1]["mult_1"],
        "ambient_only_equal": amb_equal,
        "plot": str(out_plot),
    }


# ---------------------------------------------------------------------------
# Check 2 — death-aware calibration
# ---------------------------------------------------------------------------
def check2_death_aware_calibration(
    device: str,
    quick: bool = False,
    skip_dataset: bool = False,
    skip_train: bool = False,
    wear_cfg: Optional[PerWellWearConfig] = None,
) -> Dict:
    print("\n=== Check 2: death-aware calibration ===")
    wear_cfg = wear_cfg or DEFAULT_WEAR
    cost_cfg = DEFAULT_COST

    if quick:
        n_examples = 800
        episode_horizon = 1500
        epochs = 15
        pred_horizon = 150
        sample_interval = 80
    else:
        n_examples = 8000
        episode_horizon = 2800
        epochs = 40
        pred_horizon = 200
        sample_interval = 70

    # Dataset
    if skip_dataset and DATASET_PATH.is_file():
        print(f"  loading dataset from {DATASET_PATH}")
        data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    else:
        dcfg = DeathAwareDatasetConfig(
            n_examples=n_examples,
            episode_horizon=episode_horizon,
            pred_horizon=pred_horizon,
            sample_interval=sample_interval,
            far_frac=0.65,
            far_dist_min=6.0,
            dist_max=13.0,
            null_frac=0.20,
            seed=0,
            device=device,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            min_death_rate=0.10,
        )
        data = generate_dataset(dcfg, out_path=DATASET_PATH, verbose=True)

    meta = data.get("meta", {})
    death_rate = float(meta.get("death_rate", float("nan")))
    if math.isnan(death_rate) and "died_within_horizon" in data:
        death_rate = float(data["died_within_horizon"].float().mean().item())
    n = int(data["S"].shape[0])
    print(
        f"  dataset n={n}  death_rate={death_rate:.4f}  "
        f"type_counts={meta.get('type_counts')}"
    )

    nontrivial_death = 0.05 < death_rate < 0.95
    if not nontrivial_death:
        print(
            f"  WARNING: death_rate={death_rate:.4f} is extreme "
            f"(want nontrivial, not ~0 or ~1)"
        )

    # Train
    if skip_train and MODEL_PATH.is_file():
        print(f"  loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH, device=device)
        # Still need val_data for calibration — rebuild split
        from death_aware_rollout_dataset import train_val_split

        _, val_data = train_val_split(data, val_frac=0.15, seed=0)
        train_info = {"val_data": val_data, "history": {}, "n_train": -1, "n_val": val_data["S"].shape[0]}
    else:
        tcfg = TrainConfig(
            epochs=epochs,
            batch_size=64,
            device=device,
            seed=0,
            wear_input="multipliers",
            hidden=48,
            surv_loss_scale=2.5,
        )
        model, train_info = train_self_model_v3_5(data, tcfg=tcfg, verbose=True)
        save_model(
            model,
            MODEL_PATH,
            extra={
                "death_rate": death_rate,
                "n_examples": n,
                "history": train_info["history"],
            },
        )
        print(f"  saved {MODEL_PATH}")

    val_data = train_info["val_data"]
    cal = evaluate_calibration_v35(
        model, val_data, device=device, label="val", wear_input="multipliers"
    )

    # Per-type calibration
    by_type = {}
    if "pert_type" in val_data:
        for code, name in PERT_TYPE_NAMES.items():
            m = val_data["pert_type"] == code
            if int(m.sum().item()) >= 5:
                sub = {
                    k: v[m]
                    for k, v in val_data.items()
                    if isinstance(v, torch.Tensor)
                }
                by_type[name] = evaluate_calibration_v35(
                    model, sub, device=device, label=name, wear_input="multipliers"
                )

    print(f"  val n={cal['n']}  true_death_rate={cal['death_rate_true']:.4f}")
    print(
        f"  survival raw_acc={cal['survival_acc']:.4f}  "
        f"balanced_acc={cal['survival_balanced_acc']:.4f}"
    )
    print(
        f"  death precision={cal['death_precision']:.4f}  "
        f"recall={cal['death_recall']:.4f}  f1={cal['death_f1']:.4f}"
    )
    print(
        f"  confusion (death as +): TP={cal['tp']:.0f} FP={cal['fp']:.0f} "
        f"FN={cal['fn']:.0f} TN={cal['tn']:.0f}"
    )
    print(
        f"  coherence r={cal['coherence_r_all']:.4f}  "
        f"within_bin_mean={cal['coherence_r_within_bin_mean']:.4f}  "
        f"local_R r={cal['local_R_r_all']:.4f}"
    )
    print(f"  pass_survival (bal_acc>0.5)={cal['pass_survival']}")
    print(f"  pass_within_bin (coh r≥0.5)={cal['pass_within_bin']}")

    # Plots: calibration scatter + survival confusion
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # coh scatter
    ax = axes[0, 0]
    ax.scatter(
        cal["true_coherence"].numpy(),
        cal["pred_coherence"].numpy(),
        s=8,
        alpha=0.35,
        c="C0",
    )
    ax.plot([-1, 1], [-1, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("true coherence_future")
    ax.set_ylabel("pred coherence_future")
    ax.set_title(f"coherence  r={cal['coherence_r_all']:.3f}")
    ax.grid(True, alpha=0.3)

    # local_R scatter
    ax = axes[0, 1]
    ax.scatter(
        cal["true_local_R"].numpy(),
        cal["pred_local_R"].numpy(),
        s=8,
        alpha=0.35,
        c="C1",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("true local_R_future")
    ax.set_ylabel("pred local_R_future")
    ax.set_title(f"local_R  r={cal['local_R_r_all']:.3f}")
    ax.grid(True, alpha=0.3)

    # survival histogram by class
    ax = axes[1, 0]
    ps = cal["pred_survived"].numpy()
    ts = cal["true_survived"].numpy()
    ax.hist(ps[ts >= 0.5], bins=20, alpha=0.6, label="true survived", color="C2")
    ax.hist(ps[ts < 0.5], bins=20, alpha=0.6, label="true died", color="C3")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("pred P(survived)")
    ax.set_ylabel("count")
    ax.set_title(
        f"survival bal_acc={cal['survival_balanced_acc']:.3f}  "
        f"death_rate={cal['death_rate_true']:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # confusion matrix
    ax = axes[1, 1]
    cm = [
        [cal["tn"], cal["fp"]],
        [cal["fn"], cal["tp"]],
    ]
    # rows: true survived / died; cols: pred survived / died — reframe
    # death as positive: rows [true live, true death] x cols [pred live, pred death]
    # tn=live&pred_live, fp=live&pred_death, fn=death&pred_live, tp=death&pred_death
    mat = [[cal["tn"], cal["fp"]], [cal["fn"], cal["tp"]]]
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred live", "pred death"])
    ax.set_yticklabels(["true live", "true death"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i][j]:.0f}", ha="center", va="center", color="k")
    ax.set_title(
        f"death P/R={cal['death_precision']:.2f}/{cal['death_recall']:.2f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_plot = RESULTS / "check2_death_aware_calibration.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Serialize without tensors
    cal_serial = {
        k: v
        for k, v in cal.items()
        if not isinstance(v, torch.Tensor) and k != "by_distance"
    }
    cal_serial["by_distance"] = {
        bk: {
            kk: (None if vv is None else float(vv) if isinstance(vv, (int, float)) else vv)
            for kk, vv in bv.items()
        }
        for bk, bv in cal.get("by_distance", {}).items()
    }
    by_type_serial = {
        name: {
            k: v
            for k, v in c.items()
            if not isinstance(v, torch.Tensor) and k != "by_distance"
        }
        for name, c in by_type.items()
    }

    return {
        "dataset_n": n,
        "dataset_death_rate": death_rate,
        "dataset_type_counts": meta.get("type_counts"),
        "nontrivial_death": nontrivial_death,
        "calibration": cal_serial,
        "by_type": by_type_serial,
        "survival_balanced_acc": cal["survival_balanced_acc"],
        "survival_acc": cal["survival_acc"],
        "death_precision": cal["death_precision"],
        "death_recall": cal["death_recall"],
        "pass_survival": cal["pass_survival"],
        "pass_within_bin": cal["pass_within_bin"],
        "coherence_r_within_bin_mean": cal["coherence_r_within_bin_mean"],
        "plot": str(out_plot),
        "model_path": str(MODEL_PATH),
    }


# ---------------------------------------------------------------------------
# Check 3 — asymmetric-wear policy comparison
# ---------------------------------------------------------------------------
@torch.no_grad()
def check3_asymmetric_policy(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: SelfModelV35,
    wear_cfg: PerWellWearConfig,
    device: str,
    horizon: int = 3000,
    distances: Sequence[int] = (2, 4),
    seed: int = 0,
    quick: bool = False,
) -> Dict:
    """
    Pre-degrade well A, then compare selected vs wear_aware_heuristic vs
    wellward / never_boost / always_boost.
    """
    print("\n=== Check 3: asymmetric-wear policy comparison ===")
    if quick:
        horizon = 1200
        distances = (2,)
        pre_boosts = 40
    else:
        pre_boosts = 80  # enough to push well A well ahead of B

    attn_cfg = AttentionConfig()
    cost_cfg = DEFAULT_COST
    results: Dict[int, Dict[str, Dict]] = {}

    # How much pre-degrade moves wear
    wear_preview = PerWellWearState(wear_cfg, H=world_cfg.H, W=world_cfg.W, device="cpu")
    steps_preview = pre_boosts * attn_cfg.boost_duration
    for _ in range(steps_preview):
        wear_preview.step(boosted_this_step=True, boost_location=wear_cfg.wells[0])
    print(
        f"  pre-degrade: {pre_boosts} boost-durations near well A  "
        f"→ wear≈{wear_preview.wear}  mult≈{wear_preview.multipliers()}  "
        f"(floor wear={wear_preview.wear_at_floor():.3f})"
    )

    for dist in distances:
        print(f"\n  --- dist={dist} horizon={horizon} ---")
        results[dist] = {}
        for policy in POLICIES_CHECK3:
            ep = run_v35_episode(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                model=model if policy == "selected" else None,
                policy=policy,
                distance=dist,
                horizon=horizon,
                attn_cfg=attn_cfg,
                wear_cfg=wear_cfg,
                cost_cfg=cost_cfg,
                seed=seed + dist * 17,
                device=device,
                pre_degrade_well_idx=0,
                pre_degrade_boosts=pre_boosts,
            )
            results[dist][policy] = {
                "mean_coherence": ep["mean_coherence"],
                "final_coherence": ep["final_coherence"],
                "mean_self_mass": ep["mean_self_mass"],
                "final_self_mass": ep["final_self_mass"],
                "time_of_death": ep["time_of_death"],
                "n_boosts": ep["n_boosts"],
                "n_decisions": ep["n_decisions"],
                "n_skipped_voluntary": ep["n_skipped_voluntary"],
                "n_skipped_budget": ep["n_skipped_budget"],
                "n_boost_steps": ep["n_boost_steps"],
                "final_wear": ep["final_wear"],
                "final_multipliers": ep["final_multipliers"],
                "floor_crossing_t": ep["floor_crossing_t"],
                "history": ep["history"],
                "decision_details": ep["decision_details"],
            }
            r = results[dist][policy]
            print(
                f"    {policy:22s}  mean_coh={r['mean_coherence']:.4f}  "
                f"tod={r['time_of_death']}  boosts={r['n_boosts']}/{r['n_decisions']}  "
                f"vol_skip={r['n_skipped_voluntary']}  "
                f"wear={[round(w, 3) for w in r['final_wear']]}"
            )

    # Headlines per distance
    headlines = {}
    for dist, by_pol in results.items():
        sel = by_pol["selected"]["mean_coherence"]
        wah = by_pol["wear_aware_heuristic"]["mean_coherence"]
        ww = by_pol["wellward_heuristic"]["mean_coherence"]
        nb = by_pol["never_boost"]["mean_coherence"]
        ab = by_pol["always_boost"]["mean_coherence"]
        d_wah = sel - wah
        d_ww = sel - ww
        if abs(d_wah) < MEANINGFUL_COH_DELTA:
            vs_wah = "ties"
        elif d_wah > 0:
            vs_wah = "beats"
        else:
            vs_wah = "loses_to"
        headlines[dist] = {
            "selected_mean_coh": sel,
            "wear_aware_mean_coh": wah,
            "wellward_mean_coh": ww,
            "never_boost_mean_coh": nb,
            "always_boost_mean_coh": ab,
            "delta_sel_minus_wear_aware": d_wah,
            "delta_sel_minus_wellward": d_ww,
            "selected_vs_wear_aware": vs_wah,
            "selected_beats_wellward": d_ww > MEANINGFUL_COH_DELTA,
        }
        print(
            f"  dist={dist}: selected {vs_wah} wear_aware_heuristic  "
            f"Δ={d_wah:+.4f}  (vs wellward Δ={d_ww:+.4f})"
        )

    # Plot coherence trajectories + bar summary
    n_dist = len(distances)
    fig, axes = plt.subplots(n_dist, 2, figsize=(12, 4 * n_dist), squeeze=False)
    for i, dist in enumerate(distances):
        ax = axes[i, 0]
        for policy in POLICIES_CHECK3:
            hist = results[dist][policy]["history"]
            ts = [h["t"] for h in hist]
            coh = [h["coherence"] for h in hist]
            ax.plot(
                ts,
                coh,
                color=POLICY_COLORS.get(policy, "k"),
                lw=1.2,
                alpha=0.85,
                label=policy,
            )
        ax.set_ylabel("coherence")
        ax.set_xlabel("t")
        ax.set_title(f"dist={dist}  (well A pre-degraded)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        axb = axes[i, 1]
        names = list(POLICIES_CHECK3)
        vals = [results[dist][p]["mean_coherence"] for p in names]
        colors = [POLICY_COLORS.get(p, "0.5") for p in names]
        axb.bar(range(len(names)), vals, color=colors)
        axb.set_xticks(range(len(names)))
        axb.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axb.set_ylabel("mean coherence (alive)")
        axb.set_title(f"dist={dist} mean_coh summary")
        axb.grid(True, alpha=0.3, axis="y")
        # annotate
        for j, v in enumerate(vals):
            if not math.isnan(v):
                axb.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_plot = RESULTS / "check3_asymmetric_policy.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Wear trajectories for selected vs wear_aware (one dist)
    dist0 = distances[0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for policy, ls in (("selected", "-"), ("wear_aware_heuristic", "--")):
        hist = results[dist0][policy]["history"]
        ts = [h["t"] for h in hist]
        axes[0].plot(
            ts,
            [h["wear_0"] for h in hist],
            ls=ls,
            color="C0",
            label=f"{policy} wear_A",
        )
        axes[0].plot(
            ts,
            [h["wear_1"] for h in hist],
            ls=ls,
            color="C1",
            label=f"{policy} wear_B",
        )
    axes[0].set_ylabel("wear")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Check 3 wear trajectories (dist={dist0})")
    for policy, ls in (("selected", "-"), ("wear_aware_heuristic", "--")):
        hist = results[dist0][policy]["history"]
        ts = [h["t"] for h in hist]
        axes[1].plot(
            ts, [h["mult_0"] for h in hist], ls=ls, color="C0", label=f"{policy} mult_A"
        )
        axes[1].plot(
            ts, [h["mult_1"] for h in hist], ls=ls, color="C1", label=f"{policy} mult_B"
        )
    axes[1].axhline(wear_cfg.gen_floor_frac, color="0.5", ls=":")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("multiplier")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    wear_plot = RESULTS / "check3_wear_trajectories.png"
    fig.savefig(wear_plot, dpi=140)
    plt.close(fig)

    # Drop heavy history from serializable summary
    summary_results = {}
    for dist, by_pol in results.items():
        summary_results[dist] = {
            p: {k: v for k, v in r.items() if k not in ("history", "decision_details")}
            for p, r in by_pol.items()
        }

    return {
        "distances": list(distances),
        "horizon": horizon,
        "pre_degrade_boosts": pre_boosts,
        "pre_degrade_wear_preview": wear_preview.wear,
        "headlines": headlines,
        "results": summary_results,
        # keep decision_details for Check 4 from selected
        "selected_decision_details": {
            dist: results[dist]["selected"]["decision_details"] for dist in distances
        },
        "selected_histories": {
            dist: results[dist]["selected"]["history"] for dist in distances
        },
        "plot": str(out_plot),
        "wear_plot": str(wear_plot),
    }


# ---------------------------------------------------------------------------
# Check 4 — timing behavior
# ---------------------------------------------------------------------------
@torch.no_grad()
def check4_timing_behavior(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model,
    wear_cfg: PerWellWearConfig,
    device: str,
    horizon: int = 5000,
    distances: Sequence[int] = (2, 4),
    seed: int = 0,
    quick: bool = False,
) -> Dict:
    """
    Prediction from v3 analysis: selected's boost rate should be lower while
    wear is actively increasing toward the floor, and recover afterward.

    Runs dedicated *no pre-degrade* selected episodes so floor-crossing falls
    mid-horizon (Check 3's asymmetric pre-degrade floors well A at t=0).
    """
    print("\n=== Check 4: timing behavior (boost/skip vs floor-crossing) ===")
    if quick:
        horizon = 2000
        distances = (2,)

    floor = wear_cfg.gen_floor_frac
    attn_cfg = AttentionConfig()
    cost_cfg = DEFAULT_COST
    out = {}

    for dist in distances:
        ep = run_v35_episode(
            nca=nca,
            base_pattern=base_pattern,
            world_cfg=world_cfg,
            model=model,
            policy="selected",
            distance=dist,
            horizon=horizon,
            attn_cfg=attn_cfg,
            wear_cfg=wear_cfg,
            cost_cfg=cost_cfg,
            seed=seed + 300 + dist,
            device=device,
            pre_degrade_well_idx=None,
            pre_degrade_boosts=0,
        )
        details = ep["decision_details"]
        hist = ep["history"]
        tod = ep["time_of_death"]
        # Only decisions while the pattern is alive — post-death "skips" are
        # empty-candidate artifacts, not voluntary timing.
        if tod is not None:
            details = [d for d in details if d["t"] < tod]
            hist_alive = [h for h in hist if h["t"] < tod]
        else:
            hist_alive = hist

        t_floor_0 = next(
            (int(h["t"]) for h in hist if h["mult_0"] <= floor + 1e-9), None
        )
        t_floor_1 = next(
            (int(h["t"]) for h in hist if h["mult_1"] <= floor + 1e-9), None
        )
        # both wells at floor
        t_all_floor = next(
            (int(h["t"]) for h in hist if h["all_floored"] > 0.5), None
        )
        # first well to floor (start of "partial floor")
        first_floors = [t for t in (t_floor_0, t_floor_1) if t is not None]
        t_first_floor = min(first_floors) if first_floors else None

        pre_boost = pre_dec = 0
        post_boost = post_dec = 0
        for d in details:
            boosted = bool(d["boosted"])
            all_f = bool(d["all_floored"])
            if all_f:
                post_dec += 1
                if boosted:
                    post_boost += 1
            else:
                pre_dec += 1
                if boosted:
                    pre_boost += 1

        pre_rate = pre_boost / max(1, pre_dec)
        post_rate = post_boost / max(1, post_dec)

        bin_w = 250
        max_t = max((d["t"] for d in details), default=0)
        bins = []
        for t0 in range(0, int(max_t) + 1, bin_w):
            t1 = t0 + bin_w
            sub = [d for d in details if t0 <= d["t"] < t1]
            if not sub:
                continue
            bins.append(
                {
                    "t0": t0,
                    "t1": t1,
                    "n": len(sub),
                    "boost_rate": sum(1 for d in sub if d["boosted"]) / len(sub),
                    "voluntary_skip_rate": sum(
                        1 for d in sub if d.get("voluntary_skip")
                    )
                    / len(sub),
                }
            )

        rel_bins = []
        if t_all_floor is not None:
            for lo, hi, label in (
                (-1000, -500, "pre_floor_far"),
                (-500, 0, "pre_floor_near"),
                (0, 500, "post_floor_near"),
                (500, 1000, "post_floor_far"),
            ):
                sub = [
                    d for d in details if lo <= (d["t"] - t_all_floor) < hi
                ]
                if not sub:
                    rel_bins.append(
                        {"label": label, "n": 0, "boost_rate": float("nan")}
                    )
                    continue
                rel_bins.append(
                    {
                        "label": label,
                        "n": len(sub),
                        "boost_rate": sum(1 for d in sub if d["boosted"])
                        / len(sub),
                        "voluntary_skip_rate": sum(
                            1 for d in sub if d.get("voluntary_skip")
                        )
                        / len(sub),
                    }
                )

        if pre_dec >= 3 and post_dec >= 3:
            if pre_rate < post_rate - 0.05:
                verdict = "CONFIRMED"
            elif abs(pre_rate - post_rate) <= 0.05:
                verdict = "INCONCLUSIVE_SIMILAR"
            else:
                verdict = "DISCONFIRMED"
        else:
            verdict = "INSUFFICIENT_DATA"

        n_alive_dec = len(details)
        n_vol = sum(1 for d in details if d.get("voluntary_skip"))
        n_boost_alive = sum(1 for d in details if d["boosted"])
        print(f"  dist={dist} (no pre-degrade, horizon={horizon}):")
        print(
            f"    mean_coh={ep['mean_coherence']:.4f}  tod={tod}  "
            f"boosts_alive={n_boost_alive}/{n_alive_dec}  "
            f"vol_skip_alive={n_vol}  "
            f"(full-ep boosts={ep['n_boosts']}/{ep['n_decisions']})"
        )
        print(
            f"    t_floor well0={t_floor_0}  well1={t_floor_1}  "
            f"t_first={t_first_floor}  t_all_floor={t_all_floor}"
        )
        print(
            f"    pre-floor (actively degrading): boost_rate={pre_rate:.3f}  "
            f"({pre_boost}/{pre_dec})"
        )
        print(
            f"    post-floor: boost_rate={post_rate:.3f}  ({post_boost}/{post_dec})"
        )
        print(f"    prediction (pre < post): {verdict}")
        for b in bins:
            print(
                f"    bin [{b['t0']},{b['t1']}): boost={b['boost_rate']:.2f}  "
                f"vol_skip={b['voluntary_skip_rate']:.2f}  n={b['n']}"
            )

        out[dist] = {
            "t_floor_0": t_floor_0,
            "t_floor_1": t_floor_1,
            "t_first_floor": t_first_floor,
            "t_all_floor": t_all_floor,
            "pre_floor_boost_rate": pre_rate,
            "post_floor_boost_rate": post_rate,
            "pre_n": pre_dec,
            "post_n": post_dec,
            "verdict": verdict,
            "time_bins": bins,
            "rel_bins": rel_bins,
            "mean_coherence": ep["mean_coherence"],
            "n_boosts": ep["n_boosts"],
            "n_skipped_voluntary": ep["n_skipped_voluntary"],
            "time_of_death": ep["time_of_death"],
        }

    dists = list(out.keys())
    if dists:
        fig, axes = plt.subplots(
            len(dists), 1, figsize=(9, 3.5 * len(dists)), squeeze=False
        )
        for i, dist in enumerate(dists):
            ax = axes[i, 0]
            bins = out[dist]["time_bins"]
            if bins:
                centers = [0.5 * (b["t0"] + b["t1"]) for b in bins]
                rates = [b["boost_rate"] for b in bins]
                skips = [b["voluntary_skip_rate"] for b in bins]
                ax.plot(centers, rates, "C0-o", label="boost rate")
                ax.plot(centers, skips, "C3-s", label="voluntary skip rate")
            tf = out[dist]["t_all_floor"]
            if tf is not None:
                ax.axvline(tf, color="0.4", ls="--", label=f"all_floored t={tf}")
            for tlab, col, lab in (
                (out[dist]["t_floor_0"], "C0", "floor A"),
                (out[dist]["t_floor_1"], "C1", "floor B"),
            ):
                if tlab is not None:
                    ax.axvline(tlab, color=col, ls=":", alpha=0.6, label=lab)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("t")
            ax.set_ylabel("rate")
            ax.set_title(
                f"Check 4 dist={dist}: selected boost/skip timing  "
                f"[{out[dist]['verdict']}]  "
                f"pre={out[dist]['pre_floor_boost_rate']:.2f} "
                f"post={out[dist]['post_floor_boost_rate']:.2f}"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_plot = RESULTS / "check4_boost_timing.png"
        fig.savefig(out_plot, dpi=140)
        plt.close(fig)
        print(f"  plot: {out_plot}")
    else:
        out_plot = None

    return {"by_distance": out, "plot": str(out_plot) if out_plot else None}


# ---------------------------------------------------------------------------
# Observations / README
# ---------------------------------------------------------------------------
def write_observations(
    c1: Dict,
    c2: Dict,
    c3: Dict,
    c4: Dict,
    device: str,
    quick: bool,
) -> None:
    lines: List[str] = []
    lines.append("# v3.5 Per-Well Wear + Death-Aware Self-Model + No-Boost — observations\n")
    lines.append(f"device: {device}\n")
    lines.append(f"quick: {quick}\n")
    lines.append(f"v0_ckpt: {DEFAULT_CKPT}\n")
    lines.append(f"self_model: {MODEL_PATH}\n")
    lines.append("NOTE: new checkpoint self_model_v3_5.pt — does not edit v1/v2.\n")

    lines.append("\n## Operating point (reused from v3 Stage A)\n")
    lines.append(f"- wear_ambient_rate = {DEFAULT_WEAR.wear_ambient_rate}\n")
    lines.append(f"- wear_boost_rate = {DEFAULT_WEAR.wear_boost_rate}\n")
    lines.append(f"- wear_decay_coeff = {DEFAULT_WEAR.wear_decay_coeff}\n")
    lines.append(f"- gen_floor_frac = {DEFAULT_WEAR.gen_floor_frac}\n")
    lines.append(f"- direct_boost_cost = {DEFAULT_COST.direct_boost_cost}\n")
    lines.append("- Attention: K=20 regen=0.035 boost_cost=1.0 A_max=1.5 mag=0.4\n")

    lines.append("\n## Check 1 — per-well wear mechanic sanity\n")
    lines.append(f"- pass: {c1['pass']}\n")
    lines.append(f"- no_cross_contamination: {c1['no_cross_contamination']}\n")
    lines.append(f"- boost_only_on_A: {c1['boost_only_on_A']}\n")
    lines.append(f"- mask closed-form ok: {c1['mask_ok']}\n")
    lines.append(
        f"- final wear_A={c1['final_wear_A']:.4f} wear_B={c1['final_wear_B']:.4f}\n"
    )
    lines.append(
        f"- final mult_A={c1['final_mult_A']:.4f} mult_B={c1['final_mult_B']:.4f}\n"
    )
    lines.append(f"- plot: {c1['plot']}\n")

    lines.append("\n## Check 2 — death-aware calibration\n")
    lines.append(f"- dataset n: {c2['dataset_n']}\n")
    lines.append(f"- **death_rate (class balance): {c2['dataset_death_rate']:.4f}**\n")
    lines.append(f"- nontrivial death rate: {c2['nontrivial_death']}\n")
    lines.append(f"- type_counts: {c2['dataset_type_counts']}\n")
    lines.append(
        f"- **survival balanced accuracy: {c2['survival_balanced_acc']:.4f}**\n"
    )
    lines.append(f"- survival raw accuracy: {c2['survival_acc']:.4f}\n")
    lines.append(
        f"- death precision/recall: {c2['death_precision']:.4f} / "
        f"{c2['death_recall']:.4f}\n"
    )
    lines.append(f"- pass_survival (bal_acc > 0.5): {c2['pass_survival']}\n")
    lines.append(
        f"- coherence within-bin r mean: {c2['coherence_r_within_bin_mean']}\n"
    )
    lines.append(f"- pass_within_bin: {c2['pass_within_bin']}\n")
    lines.append(f"- plot: {c2['plot']}\n")

    lines.append("\n## Check 3 — asymmetric-wear policy comparison\n")
    lines.append(
        f"- pre_degrade_boosts near well A: {c3['pre_degrade_boosts']}  "
        f"preview wear≈{c3['pre_degrade_wear_preview']}\n"
    )
    lines.append(f"- horizon: {c3['horizon']}\n")
    lines.append(
        "- Policies: selected, wellward_heuristic, **wear_aware_heuristic**, "
        "never_boost, always_boost\n"
    )
    lines.append(
        "- **Central question:** does selected beat / match / lose to "
        "wear_aware_heuristic?\n"
    )
    for dist, h in c3["headlines"].items():
        lines.append(f"- dist={dist}:\n")
        lines.append(f"  - selected mean_coh = {h['selected_mean_coh']:.4f}\n")
        lines.append(
            f"  - wear_aware_heuristic mean_coh = {h['wear_aware_mean_coh']:.4f}\n"
        )
        lines.append(f"  - wellward_heuristic mean_coh = {h['wellward_mean_coh']:.4f}\n")
        lines.append(f"  - never_boost mean_coh = {h['never_boost_mean_coh']:.4f}\n")
        lines.append(f"  - always_boost mean_coh = {h['always_boost_mean_coh']:.4f}\n")
        lines.append(
            f"  - Δ(selected − wear_aware) = {h['delta_sel_minus_wear_aware']:+.4f}\n"
        )
        lines.append(
            f"  - **selected {h['selected_vs_wear_aware']} wear_aware_heuristic**\n"
        )
        if dist in c3["results"]:
            for pol, r in c3["results"][dist].items():
                lines.append(
                    f"  - [{pol}] boosts={r['n_boosts']}/{r['n_decisions']}  "
                    f"vol_skip={r['n_skipped_voluntary']}  "
                    f"tod={r['time_of_death']}  "
                    f"final_wear={r['final_wear']}\n"
                )
    lines.append(f"- plot: {c3['plot']}\n")

    # Overall central answer
    verdicts = [h["selected_vs_wear_aware"] for h in c3["headlines"].values()]
    lines.append("\n### Central outcome (selected vs wear_aware_heuristic)\n")
    if all(v == "beats" for v in verdicts):
        lines.append(
            "- **selected BEATS wear_aware_heuristic** on all reported distances. "
            "First demonstrated case where the learned self-model earns its "
            "complexity over a hand-coded state-tracking rule — worth probing "
            "what state it tracks beyond simple wear comparison.\n"
        )
    elif all(v == "loses_to" for v in verdicts):
        lines.append(
            "- **selected LOSES TO wear_aware_heuristic** on all reported distances. "
            "Honest conclusion: even asymmetric per-well wear is still simple "
            "enough for a slightly-smarter heuristic. Next lever likely needs "
            "multi-step lookahead the heuristic cannot approximate.\n"
        )
    elif all(v == "ties" for v in verdicts):
        lines.append(
            "- **selected TIES wear_aware_heuristic** (|Δ|<0.01) on all reported "
            "distances. The learned model does not earn complexity over the "
            "hand-coded wear rule at this OP.\n"
        )
    else:
        lines.append(
            f"- Mixed across distances: {dict((d, c3['headlines'][d]['selected_vs_wear_aware']) for d in c3['headlines'])}. "
            "Report per-distance; do not collapse to a single victory claim.\n"
        )

    lines.append("\n## Check 4 — timing behavior\n")
    lines.append(
        "- Prediction: boost rate lower while wear actively degrades toward "
        "floor; recovers after floor.\n"
    )
    for dist, info in c4["by_distance"].items():
        lines.append(f"- dist={dist}:\n")
        lines.append(
            f"  - t_floor well0/well1/all = {info['t_floor_0']}/"
            f"{info['t_floor_1']}/{info['t_all_floor']}\n"
        )
        lines.append(
            f"  - pre-floor boost_rate = {info['pre_floor_boost_rate']:.3f} "
            f"(n={info['pre_n']})\n"
        )
        lines.append(
            f"  - post-floor boost_rate = {info['post_floor_boost_rate']:.3f} "
            f"(n={info['post_n']})\n"
        )
        lines.append(f"  - **verdict: {info['verdict']}**\n")
    lines.append(f"- plot: {c4.get('plot')}\n")

    lines.append("\n## Honesty notes\n")
    lines.append(
        "- A correctly implemented result that does not favor the learned model "
        "is a valid, reportable outcome.\n"
    )
    lines.append(
        "- Unattributed ~−0.02 post-floor residual from v3 is not chased here; "
        "note if it reappears in selected vs never_boost post-floor windows.\n"
    )
    lines.append(
        "- selected vs wellward under *global* wear was design-foreclosed; "
        "per-well wear + wear_aware_heuristic is the fair comparison.\n"
    )

    obs_path = RESULTS / "observations.txt"
    obs_path.write_text("".join(lines))
    print(f"\n  wrote {obs_path}")

    # README
    readme = f"""# v3.5 — Per-Well Wear + Death-Aware Self-Model + No-Boost

Two independent per-well wear tracks (so *where* you boost has a wear
consequence); a retrained self-model that has seen deaths and wear state;
and an explicit **no-boost / skip** candidate in the same argmax as the 6
sectors.

Does **not** modify `v0` step logic, `SimpleNCA`, or frozen `v1`/`v2`
checkpoints. Trains a **new** `self_model_v3_5.pt`.

## Setup

```bash
cd EmbodiedNCA
source .venv/bin/activate
python v3_5/run_v3_5_checks.py
python v3_5/run_v3_5_checks.py --quick
python v3_5/run_v3_5_checks.py --skip_dataset --skip_train  # reuse artifacts
```

Requires `v0/results/trained_nca.pt`.

## Layout

```
v3_5/
├── wear_perwell.py                 # per-well wear + effective generation mask
├── death_aware_rollout_dataset.py  # long-horizon, death-containing dataset
├── self_model_v3_5.py              # wear-conditioned self-model
├── action_selection_v3_5.py        # 7-way selection (6 sectors + skip)
├── run_v3_5_checks.py
├── results/
│   ├── self_model_v3_5.pt
│   └── observations.txt
└── README.md
```

## Mechanics

1. **Per-well wear** — ambient on both wells; boost accrues only on the nearest
   well. Degradation folds into a spatial generation mask; `gen_rate_base`
   stays a scalar for v0's `step_resource`.
2. **Skip candidate** — 7th option in the argmax; costs no A, no R charge, no
   boost wear. Competitive even with full budget.
3. **Death-aware self-model** — inputs include both wells' effective
   multipliers; trained under full v3.5 physics with nontrivial death labels.

## Acceptance checks

| Check | What |
|------:|------|
| 1 | No cross-contamination; mask matches closed form under asymmetric boost |
| 2 | Survival **balanced** accuracy (not raw acc); coh/local_R calibration |
| 3 | Asymmetric pre-wear: selected vs **wear_aware_heuristic** (fair control) |
| 4 | Boost rate pre- vs post-floor (timing prediction from v3) |

## Measured results (latest run)

- Dataset death rate: **{c2['dataset_death_rate']:.4f}**
- Survival balanced accuracy: **{c2['survival_balanced_acc']:.4f}**
- selected vs wear_aware_heuristic: **{', '.join(f'dist{d}={c3["headlines"][d]["selected_vs_wear_aware"]}' for d in c3['headlines'])}**

Full write-up: `results/observations.txt`.
"""
    (V35_ROOT / "README.md").write_text(readme)
    print(f"  wrote {V35_ROOT / 'README.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="v3.5 acceptance checks")
    p.add_argument("--quick", action="store_true", help="smaller dataset / shorter horizon")
    p.add_argument("--skip_dataset", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_check3", action="store_true")
    p.add_argument("--skip_check4", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    print(f"device={device}  quick={args.quick}")

    wear_cfg = DEFAULT_WEAR
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

    # Check 1
    c1 = check1_perwell_sanity(
        nca,
        base_pattern,
        world_cfg,
        wear_cfg,
        device=device,
        horizon=600 if args.quick else 800,
        seed=args.seed,
    )

    # Check 2
    c2 = check2_death_aware_calibration(
        device=device,
        quick=args.quick,
        skip_dataset=args.skip_dataset,
        skip_train=args.skip_train,
        wear_cfg=wear_cfg,
    )
    model = load_model(MODEL_PATH, device=device)

    # Check 3
    if args.skip_check3:
        c3 = {
            "distances": [],
            "horizon": 0,
            "pre_degrade_boosts": 0,
            "pre_degrade_wear_preview": [0, 0],
            "headlines": {},
            "results": {},
            "selected_decision_details": {},
            "selected_histories": {},
            "plot": None,
            "wear_plot": None,
        }
    else:
        c3 = check3_asymmetric_policy(
            nca,
            base_pattern,
            world_cfg,
            model,
            wear_cfg,
            device=device,
            horizon=1500 if args.quick else 3000,
            distances=(2,) if args.quick else (2, 4),
            seed=args.seed,
            quick=args.quick,
        )

    # Check 4 (dedicated no-pre-degrade selected runs)
    if args.skip_check4:
        c4 = {"by_distance": {}, "plot": None}
    else:
        c4 = check4_timing_behavior(
            nca,
            base_pattern,
            world_cfg,
            model,
            wear_cfg,
            device=device,
            horizon=2500 if args.quick else 5000,
            distances=(2,) if args.quick else (2, 4),
            seed=args.seed,
            quick=args.quick,
        )

    write_observations(c1, c2, c3, c4, device=device, quick=args.quick)

    # Summary JSON (no heavy histories)
    summary = {
        "check1": {k: v for k, v in c1.items()},
        "check2": {
            k: v
            for k, v in c2.items()
            if k not in ("calibration",)  # still include serial cal
        },
        "check2_calibration": c2.get("calibration"),
        "check3_headlines": c3.get("headlines"),
        "check3_results": c3.get("results"),
        "check4": c4,
    }
    # Make JSON-safe
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

    with (RESULTS / "summary.json").open("w") as f:
        json.dump(sanitize(summary), f, indent=2)
    print(f"  wrote {RESULTS / 'summary.json'}")

    print("\n=== v3.5 done ===")
    print(f"  Check 1 pass: {c1['pass']}")
    print(
        f"  Check 2 survival bal_acc: {c2['survival_balanced_acc']:.4f}  "
        f"death_rate: {c2['dataset_death_rate']:.4f}"
    )
    for dist, h in c3.get("headlines", {}).items():
        print(
            f"  Check 3 dist={dist}: selected {h['selected_vs_wear_aware']} "
            f"wear_aware  Δ={h['delta_sel_minus_wear_aware']:+.4f}"
        )
    for dist, info in c4.get("by_distance", {}).items():
        print(f"  Check 4 dist={dist}: {info['verdict']}")


if __name__ == "__main__":
    main()
