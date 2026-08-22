"""Zero-shot checks of frozen self_model_v3_5 on a relocated two-well layout. No retrain."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from embodiednca.paths import results_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from embodiednca.v2.action_selection import AttentionConfig
from embodiednca.v3_5.action_selection_v3_5 import run_v35_episode
from embodiednca.v3.boost_cost import BoostCostConfig
from embodiednca.v3_5.death_aware_rollout_dataset import (
    DeathAwareDatasetConfig,
    generate_dataset,
    train_val_split,
)
from embodiednca.v0.grid_world import DEFAULT_GEN_RATE, DEFAULT_WELLS, WorldConfig
from embodiednca.v1.rollout_dataset import DEFAULT_CKPT, load_v0_world
from embodiednca.v3_5.self_model_v3_5 import (
    SelfModelV35,
    evaluate_calibration_v35,
    load_model,
)
from embodiednca.v3_5.wear_perwell import PerWellWearConfig, PerWellWearState

RESULTS = results_dir("v5")
MODEL_PATH = results_dir("v3_5") / "self_model_v3_5.pt"
ORIG_DATASET_PATH = results_dir("v3_5") / "death_aware_dataset.pt"
RELOC_DATASET_PATH = RESULTS / "relocated_eval_dataset.pt"

# Training layout (never modify — frozen reference)
ORIGINAL_WELLS: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS  # (8,8), (24,24)

# Relocated two-well layout: same inter-well distance as training
# (16√2 ≈ 22.63) so difficulty is not confounded — only geometry changes.
# Training vector was (16, 16); here (16, −16) from (12, 20) → (28, 4).
RELOCATED_WELLS: Tuple[Tuple[int, int], ...] = (
    (12, 20),
    (28, 4),
)

# v3.5 operating point (reused)
DEFAULT_COST = BoostCostConfig(direct_boost_cost=0.15)
MEANINGFUL_COH_DELTA = 0.01

# v3.5 original-layout Check 3 reference (from observations)
ORIG_POLICY_REF = {
    2: {
        "selected_mean_coh": 0.5124,
        "wear_aware_mean_coh": 0.5188,
        "wellward_mean_coh": 0.4840,
        "delta_sel_minus_wear_aware": -0.0064,
        "selected_vs_wear_aware": "ties",
    },
    4: {
        "selected_mean_coh": 0.3795,
        "wear_aware_mean_coh": 0.3915,
        "wellward_mean_coh": 0.3342,
        "delta_sel_minus_wear_aware": -0.0120,
        "selected_vs_wear_aware": "loses_to",
    },
}

POLICIES = (
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


def inter_well_distance(wells: Sequence[Tuple[int, int]]) -> float:
    (y0, x0), (y1, x1) = wells[0], wells[1]
    return float(math.hypot(y1 - y0, x1 - x0))


def make_wear_cfg(wells: Sequence[Tuple[int, int]]) -> PerWellWearConfig:
    return PerWellWearConfig(
        wear_ambient_rate=0.001,
        wear_boost_rate=0.004,
        wear_decay_coeff=0.20,
        gen_floor_frac=0.40,
        gen_rate_base=DEFAULT_GEN_RATE,
        wells=tuple(wells),
    )


def make_world_cfg(
    base_cfg: WorldConfig,
    wells: Sequence[Tuple[int, int]],
    device: str,
) -> WorldConfig:
    return WorldConfig(
        H=base_cfg.H,
        W=base_cfg.W,
        channels=base_cfg.channels,
        hidden_dim=base_cfg.hidden_dim,
        gen_rate=DEFAULT_GEN_RATE,
        consumption_rate=base_cfg.consumption_rate,
        decay_rate=base_cfg.decay_rate,
        diffusion_rate=base_cfg.diffusion_rate,
        resource_diffusion=base_cfg.resource_diffusion,
        well_sigma=base_cfg.well_sigma,
        wells=tuple(wells),
        device=device,
    )


def make_attn_cfg(wells: Sequence[Tuple[int, int]]) -> AttentionConfig:
    return AttentionConfig(wells=tuple(wells))


def _serializable_cal(cal: Dict) -> Dict:
    out: Dict[str, Any] = {}
    for k, v in cal.items():
        if isinstance(v, torch.Tensor):
            continue
        if k == "by_distance":
            out[k] = {
                bk: {
                    kk: (
                        None
                        if vv is None
                        else float(vv)
                        if isinstance(vv, (int, float))
                        else vv
                    )
                    for kk, vv in bv.items()
                }
                for bk, bv in v.items()
            }
        elif isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Check 1 — zero-shot calibration: original vs relocated
# ---------------------------------------------------------------------------
@torch.no_grad()
def check1_calibration_relocated(
    model: SelfModelV35,
    device: str,
    quick: bool = False,
    skip_dataset: bool = False,
    seed: int = 0,
) -> Dict:
    """
    Evaluate frozen self_model_v3_5 on:
      - original-layout held-out samples (from v3.5 training dataset if present)
      - newly generated relocated-layout samples (no training)

    Pass bar: relocated coh/local_R calibration comparable to original
    (not collapse to near-chance). Collapse is a valid headline finding.
    """
    print("\n=== Check 1: zero-shot calibration (original vs relocated) ===")
    RESULTS.mkdir(parents=True, exist_ok=True)

    d_orig = inter_well_distance(ORIGINAL_WELLS)
    d_reloc = inter_well_distance(RELOCATED_WELLS)
    print(
        f"  original wells={ORIGINAL_WELLS}  inter-well dist={d_orig:.3f}"
    )
    print(
        f"  relocated wells={RELOCATED_WELLS}  inter-well dist={d_reloc:.3f}  "
        f"(Δ dist={d_reloc - d_orig:+.3f}; should be ~0)"
    )
    print(
        "  NOTE: heuristics take well positions as parameters — updating them "
        "is one line. Self-model weights stay frozen; only runtime well coords "
        "for world physics / distance change."
    )

    if quick:
        n_reloc = 600
        episode_horizon = 1500
        pred_horizon = 150
        sample_interval = 80
    else:
        n_reloc = 2500
        episode_horizon = 2800
        pred_horizon = 200
        sample_interval = 70

    # --- Original layout calibration ---
    orig_source = "none"
    cal_orig: Optional[Dict] = None
    if ORIG_DATASET_PATH.is_file():
        print(f"  loading original-layout dataset: {ORIG_DATASET_PATH}")
        data_orig = torch.load(
            ORIG_DATASET_PATH, map_location="cpu", weights_only=False
        )
        _, val_orig = train_val_split(data_orig, val_frac=0.15, seed=0)
        cal_orig = evaluate_calibration_v35(
            model, val_orig, device=device, label="original_val", wear_input="multipliers"
        )
        orig_source = "v3_5_death_aware_val"
        print(
            f"  original val n={cal_orig['n']}  "
            f"coh_r={cal_orig['coherence_r_all']:.4f}  "
            f"within_bin={cal_orig['coherence_r_within_bin_mean']:.4f}  "
            f"local_R_r={cal_orig['local_R_r_all']:.4f}  "
            f"coh_mae={cal_orig['coherence_mae_all']:.4f}"
        )
    else:
        print(
            f"  WARNING: {ORIG_DATASET_PATH} missing — generating smaller "
            "original-layout eval set for fair comparison"
        )
        wear_o = make_wear_cfg(ORIGINAL_WELLS)
        dcfg_o = DeathAwareDatasetConfig(
            n_examples=n_reloc,
            episode_horizon=episode_horizon,
            pred_horizon=pred_horizon,
            sample_interval=sample_interval,
            far_frac=0.65,
            far_dist_min=6.0,
            dist_max=13.0,
            null_frac=0.20,
            seed=seed + 1,
            device=device,
            wells=ORIGINAL_WELLS,
            wear_cfg=wear_o,
            cost_cfg=DEFAULT_COST,
            min_death_rate=0.05,
        )
        data_orig = generate_dataset(dcfg_o, out_path=None, verbose=True)
        cal_orig = evaluate_calibration_v35(
            model, data_orig, device=device, label="original_gen", wear_input="multipliers"
        )
        orig_source = "generated_eval"
        print(
            f"  original gen n={cal_orig['n']}  "
            f"coh_r={cal_orig['coherence_r_all']:.4f}  "
            f"within_bin={cal_orig['coherence_r_within_bin_mean']:.4f}"
        )

    # --- Relocated layout eval dataset (zero-shot; never used for training) ---
    if skip_dataset and RELOC_DATASET_PATH.is_file():
        print(f"  loading relocated eval dataset: {RELOC_DATASET_PATH}")
        data_reloc = torch.load(
            RELOC_DATASET_PATH, map_location="cpu", weights_only=False
        )
    else:
        wear_r = make_wear_cfg(RELOCATED_WELLS)
        dcfg_r = DeathAwareDatasetConfig(
            n_examples=n_reloc,
            episode_horizon=episode_horizon,
            pred_horizon=pred_horizon,
            sample_interval=sample_interval,
            far_frac=0.65,
            far_dist_min=6.0,
            dist_max=13.0,
            null_frac=0.20,
            seed=seed + 7,
            device=device,
            wells=RELOCATED_WELLS,
            wear_cfg=wear_r,
            cost_cfg=DEFAULT_COST,
            min_death_rate=0.05,
        )
        print(
            f"  generating relocated-layout eval dataset n={n_reloc} "
            f"(no training — evaluation only)"
        )
        data_reloc = generate_dataset(
            dcfg_r, out_path=RELOC_DATASET_PATH, verbose=True
        )

    cal_reloc = evaluate_calibration_v35(
        model, data_reloc, device=device, label="relocated", wear_input="multipliers"
    )
    print(
        f"  relocated n={cal_reloc['n']}  "
        f"coh_r={cal_reloc['coherence_r_all']:.4f}  "
        f"within_bin={cal_reloc['coherence_r_within_bin_mean']:.4f}  "
        f"local_R_r={cal_reloc['local_R_r_all']:.4f}  "
        f"coh_mae={cal_reloc['coherence_mae_all']:.4f}  "
        f"death_rate={cal_reloc['death_rate_true']:.4f}"
    )

    # Verdict
    # Collapse = near-chance within-bin coherence (r < 0.5) or dramatic
    # relative drop vs original. Hold = still clearly above chance and
    # not a collapse relative to original.
    o_wb = float(cal_orig["coherence_r_within_bin_mean"])
    r_wb = float(cal_reloc["coherence_r_within_bin_mean"])
    o_r = float(cal_orig["coherence_r_all"])
    r_r = float(cal_reloc["coherence_r_all"])
    o_lr = float(cal_orig["local_R_r_all"])
    r_lr = float(cal_reloc["local_R_r_all"])
    o_mae = float(cal_orig["coherence_mae_all"])
    r_mae = float(cal_reloc["coherence_mae_all"])

    near_chance = (math.isnan(r_wb) or r_wb < 0.5) and (math.isnan(r_r) or r_r < 0.5)
    # Dramatic collapse: within-bin drops by >0.35 absolute or below half original
    dramatic_drop = (
        (not math.isnan(o_wb) and not math.isnan(r_wb) and (o_wb - r_wb) > 0.35)
        or (not math.isnan(o_wb) and not math.isnan(r_wb) and r_wb < 0.5 * o_wb)
    )
    # Still useful: within-bin ≥ 0.5 (same gate as v3.5 pass_within_bin)
    holds = (not math.isnan(r_wb) and r_wb >= 0.5) and not near_chance

    if near_chance or (dramatic_drop and not holds):
        verdict = "collapsed"
    elif holds:
        if (not math.isnan(o_wb)) and abs(o_wb - r_wb) <= 0.15:
            verdict = "held_comparable"
        else:
            verdict = "held_degraded_but_above_chance"
    else:
        verdict = "degraded_borderline"

    print(
        f"  Check 1 verdict: {verdict}  "
        f"(orig within_bin={o_wb:.4f} → reloc={r_wb:.4f}; "
        f"orig coh_r={o_r:.4f} → reloc={r_r:.4f})"
    )
    print(f"  pass_within_bin_relocated={cal_reloc['pass_within_bin']}")

    # Plot: side-by-side calibration (mirrors v3.5 Check 2 style)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, cal, title, color in (
        (
            axes[0, 0],
            cal_orig,
            f"original wells  coh r={cal_orig['coherence_r_all']:.3f}",
            "C0",
        ),
        (
            axes[0, 1],
            cal_reloc,
            f"relocated wells  coh r={cal_reloc['coherence_r_all']:.3f}",
            "C3",
        ),
    ):
        ax.scatter(
            cal["true_coherence"].numpy(),
            cal["pred_coherence"].numpy(),
            s=8,
            alpha=0.3,
            c=color,
        )
        ax.plot([-1, 1], [-1, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("true coherence_future")
        ax.set_ylabel("pred coherence_future")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for ax, cal, title, color in (
        (
            axes[1, 0],
            cal_orig,
            f"original  local_R r={cal_orig['local_R_r_all']:.3f}",
            "C1",
        ),
        (
            axes[1, 1],
            cal_reloc,
            f"relocated  local_R r={cal_reloc['local_R_r_all']:.3f}",
            "C3",
        ),
    ):
        ax.scatter(
            cal["true_local_R"].numpy(),
            cal["pred_local_R"].numpy(),
            s=8,
            alpha=0.3,
            c=color,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("true local_R_future")
        ax.set_ylabel("pred local_R_future")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Check 1: zero-shot calibration  [{verdict}]\n"
        f"orig wells {ORIGINAL_WELLS} → reloc {RELOCATED_WELLS}",
        fontsize=11,
    )
    fig.tight_layout()
    out_plot = RESULTS / "check1_calibration_original_vs_relocated.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Within-bin bar comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = [
        ("coh r (all)", o_r, r_r),
        ("coh r (within-bin)", o_wb, r_wb),
        ("local_R r", o_lr, r_lr),
        ("coh MAE (↓)", o_mae, r_mae),
    ]
    x = range(len(metrics))
    w = 0.35
    ax.bar(
        [i - w / 2 for i in x],
        [m[1] for m in metrics],
        width=w,
        label="original layout",
        color="C0",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [m[2] for m in metrics],
        width=w,
        label="relocated layout",
        color="C3",
    )
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="chance gate (r=0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylabel("value")
    ax.set_title(f"Check 1 metric comparison  [{verdict}]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    bar_plot = RESULTS / "check1_calibration_metrics_bar.png"
    fig.savefig(bar_plot, dpi=140)
    plt.close(fig)

    return {
        "original_wells": list(ORIGINAL_WELLS),
        "relocated_wells": list(RELOCATED_WELLS),
        "inter_well_dist_original": d_orig,
        "inter_well_dist_relocated": d_reloc,
        "original_source": orig_source,
        "original": _serializable_cal(cal_orig),
        "relocated": _serializable_cal(cal_reloc),
        "verdict": verdict,
        "holds_up": holds and not near_chance,
        "collapsed": verdict == "collapsed",
        "pass_within_bin_relocated": bool(cal_reloc["pass_within_bin"]),
        "plot": str(out_plot),
        "bar_plot": str(bar_plot),
        "delta_within_bin": (
            float(r_wb - o_wb)
            if not (math.isnan(r_wb) or math.isnan(o_wb))
            else float("nan")
        ),
        "delta_coh_r": (
            float(r_r - o_r)
            if not (math.isnan(r_r) or math.isnan(o_r))
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Check 2 — policy comparison on relocated layout
# ---------------------------------------------------------------------------
@torch.no_grad()
def check2_policy_relocated(
    nca,
    base_pattern: torch.Tensor,
    world_cfg_base: WorldConfig,
    model: SelfModelV35,
    device: str,
    horizon: int = 3000,
    distances: Sequence[int] = (2, 4),
    seed: int = 0,
    quick: bool = False,
) -> Dict:
    """
    Same structure as v3.5 Check 3, on RELOCATED_WELLS.
    Heuristics get new coordinates only; selected uses frozen weights.
    """
    print("\n=== Check 2: policy comparison on relocated layout ===")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if quick:
        horizon = 1200
        distances = (2,)
        pre_boosts = 40
    else:
        pre_boosts = 80

    wells = RELOCATED_WELLS
    wear_cfg = make_wear_cfg(wells)
    world_cfg = make_world_cfg(world_cfg_base, wells, device)
    attn_cfg = make_attn_cfg(wells)
    cost_cfg = DEFAULT_COST

    print(
        f"  wells={wells}  (heuristics pointed at new coords; "
        f"self_model_v3_5 unmodified)"
    )
    print(
        f"  ease-of-update asymmetry: wellward/wear_aware take wells as a "
        f"parameter — one-line change. Self-model never trained on this map."
    )

    wear_preview = PerWellWearState(
        wear_cfg, H=world_cfg.H, W=world_cfg.W, device="cpu"
    )
    steps_preview = pre_boosts * attn_cfg.boost_duration
    for _ in range(steps_preview):
        wear_preview.step(boosted_this_step=True, boost_location=wells[0])
    print(
        f"  pre-degrade: {pre_boosts} boost-durations near well A {wells[0]}  "
        f"→ wear≈{wear_preview.wear}  mult≈{wear_preview.multipliers()}"
    )

    results: Dict[int, Dict[str, Dict]] = {}
    for dist in distances:
        print(f"\n  --- dist={dist} horizon={horizon} well={wells[0]} ---")
        results[dist] = {}
        for policy in POLICIES:
            ep = run_v35_episode(
                nca=nca,
                base_pattern=base_pattern,
                world_cfg=world_cfg,
                model=model if policy == "selected" else None,
                policy=policy,
                distance=dist,
                horizon=horizon,
                well=wells[0],  # place relative to relocated well A
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
                f"tod={r['time_of_death']}  "
                f"boosts={r['n_boosts']}/{r['n_decisions']}  "
                f"vol_skip={r['n_skipped_voluntary']}"
            )

    headlines: Dict[int, Dict] = {}
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

    # Pattern classification vs original-layout Check 3
    pattern_outcome = _classify_check2_pattern(headlines)
    print(f"  Check 2 pattern outcome: {pattern_outcome['label']}")
    print(f"  {pattern_outcome['summary']}")

    # Policy comparison plot (mirrors v3.5 Check 3)
    n_dist = len(distances)
    fig, axes = plt.subplots(n_dist, 2, figsize=(12, 4 * n_dist), squeeze=False)
    for i, dist in enumerate(distances):
        ax = axes[i, 0]
        for policy in POLICIES:
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
        ax.set_title(
            f"relocated wells  dist={dist}  (well A pre-degraded)"
        )
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        axb = axes[i, 1]
        names = list(POLICIES)
        vals = [results[dist][p]["mean_coherence"] for p in names]
        colors = [POLICY_COLORS.get(p, "0.5") for p in names]
        axb.bar(range(len(names)), vals, color=colors)
        axb.set_xticks(range(len(names)))
        axb.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axb.set_ylabel("mean coherence (alive)")
        axb.set_title(f"dist={dist} mean_coh summary")
        axb.grid(True, alpha=0.3, axis="y")
        for j, v in enumerate(vals):
            if not math.isnan(v):
                axb.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"Check 2: relocated policy comparison  [{pattern_outcome['label']}]\n"
        f"wells={wells}",
        fontsize=11,
    )
    fig.tight_layout()
    out_plot = RESULTS / "check2_policy_comparison_relocated.png"
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)
    print(f"  plot: {out_plot}")

    # Original vs relocated delta bars (selected − wear_aware)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dists_plot = [d for d in distances if d in ORIG_POLICY_REF]
    if not dists_plot:
        dists_plot = list(distances)
    x = range(len(dists_plot))
    w = 0.35
    orig_deltas = [
        ORIG_POLICY_REF[d]["delta_sel_minus_wear_aware"]
        if d in ORIG_POLICY_REF
        else float("nan")
        for d in dists_plot
    ]
    reloc_deltas = [
        headlines[d]["delta_sel_minus_wear_aware"] for d in dists_plot
    ]
    ax.bar(
        [i - w / 2 for i in x],
        orig_deltas,
        width=w,
        label="original layout (v3.5)",
        color="C0",
    )
    ax.bar(
        [i + w / 2 for i in x],
        reloc_deltas,
        width=w,
        label="relocated layout (v5)",
        color="C3",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(
        MEANINGFUL_COH_DELTA,
        color="0.5",
        ls="--",
        lw=1,
        label=f"±{MEANINGFUL_COH_DELTA} tie band",
    )
    ax.axhline(-MEANINGFUL_COH_DELTA, color="0.5", ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"dist={d}" for d in dists_plot])
    ax.set_ylabel("Δ mean_coh (selected − wear_aware)")
    ax.set_title("Check 2: selected vs wear_aware — original vs relocated")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    delta_plot = RESULTS / "check2_delta_original_vs_relocated.png"
    fig.savefig(delta_plot, dpi=140)
    plt.close(fig)

    summary_results = {}
    for dist, by_pol in results.items():
        summary_results[dist] = {
            p: {
                k: v
                for k, v in r.items()
                if k not in ("history", "decision_details")
            }
            for p, r in by_pol.items()
        }

    return {
        "wells": list(wells),
        "distances": list(distances),
        "horizon": horizon,
        "pre_degrade_boosts": pre_boosts,
        "pre_degrade_wear_preview": list(wear_preview.wear),
        "headlines": {int(k): v for k, v in headlines.items()},
        "results": {int(k): v for k, v in summary_results.items()},
        "pattern_outcome": pattern_outcome,
        "original_ref": ORIG_POLICY_REF,
        "plot": str(out_plot),
        "delta_plot": str(delta_plot),
    }


def _classify_check2_pattern(headlines: Dict[int, Dict]) -> Dict:
    """
    Three reportable outcomes from the spec:

    1. same_pattern — heuristic still ties/beats (as on original layout)
    2. self_model_clearly_loses — worse than original-layout margin
    3. self_model_wins_or_ties_more — surprising; flag for multi-seed check

    Original pattern: ties at d=2, loses_to at d=4 (heuristic ≥ selected).
    """
    verdicts = {
        int(d): h["selected_vs_wear_aware"] for d, h in headlines.items()
    }
    deltas = {
        int(d): h["delta_sel_minus_wear_aware"] for d, h in headlines.items()
    }

    # Heuristic still ≥ selected on every reported distance?
    heuristic_still_ge = all(v in ("ties", "loses_to") for v in verdicts.values())
    # Self-model clearly wins anywhere?
    model_wins_any = any(v == "beats" for v in verdicts.values())
    # All ties or better for model?
    model_all_ge = all(v in ("ties", "beats") for v in verdicts.values()) and any(
        v == "beats" for v in verdicts.values()
    )

    # Margin comparison on shared distances
    worse_margins = []
    better_margins = []
    for d, delta in deltas.items():
        if d not in ORIG_POLICY_REF:
            continue
        o_delta = ORIG_POLICY_REF[d]["delta_sel_minus_wear_aware"]
        # More negative = model worse relative to heuristic
        if delta < o_delta - MEANINGFUL_COH_DELTA:
            worse_margins.append(d)
        elif delta > o_delta + MEANINGFUL_COH_DELTA:
            better_margins.append(d)

    if model_wins_any and model_all_ge:
        label = "self_model_wins_or_ties_more"
        summary = (
            "Self-model now wins or ties more clearly than on the original "
            "layout — surprising given v2/v3.5; treat as provisional and "
            "multi-seed check before taking at face value."
        )
    elif model_wins_any:
        label = "self_model_wins_or_ties_more"
        summary = (
            f"Self-model beats wear_aware on some distances ({verdicts}). "
            "Mixed but more favorable than original ties/loses pattern — "
            "worth multi-seed confirmation."
        )
    elif heuristic_still_ge and worse_margins and not better_margins:
        label = "self_model_clearly_loses"
        summary = (
            "Self-model still loses/ties to the position-updated heuristic, "
            f"and the margin is worse than original on dist={worse_margins}. "
            "Evidence that apparent competence was more memorized than "
            "transferred."
        )
    elif heuristic_still_ge:
        label = "same_pattern"
        summary = (
            "Same pattern as original layout: heuristic ties or beats "
            "selected after a one-line well-coordinate update. Self-model "
            "competence was never coordinate-specific in a way that "
            "differentiated it — it still does not add value over a "
            "correctly-parameterized heuristic."
        )
    else:
        label = "mixed"
        summary = (
            f"Mixed verdicts {verdicts}; deltas={deltas}. Report per-distance."
        )

    return {
        "label": label,
        "summary": summary,
        "verdicts_by_dist": verdicts,
        "deltas_by_dist": deltas,
        "worse_margin_dists": worse_margins,
        "better_margin_dists": better_margins,
    }


# ---------------------------------------------------------------------------
# Check 3 — architectural support for N≠2 wells (no run if unsupported)
# ---------------------------------------------------------------------------
def check3_architecture_n_wells(model: SelfModelV35) -> Dict:
    """
    Inspect whether self_model_v3_5's input encoding and surrounding code
    support a well count other than 2 without modification.

    If not: report as architectural generalization limit and stop.
    Do not retrain to force three-well through.
    """
    print("\n=== Check 3: architectural support for N≠2 wells ===")

    findings: List[str] = []
    supports_variable = True

    # 1. Model n_wells is fixed at construction / checkpoint
    n_wells_ckpt = int(model.n_wells)
    findings.append(
        f"SelfModelV35.n_wells = {n_wells_ckpt} (fixed at init; checkpoint)"
    )
    if n_wells_ckpt != 2:
        findings.append(
            f"Unexpected: checkpoint n_wells={n_wells_ckpt} (expected 2)"
        )

    # 2. First conv input channels = C + 1 (R) + 1 (pert) + n_wells
    first_conv = model.enc[0]
    in_ch = int(first_conv.in_channels)
    expected_in = model.channels + 1 + 1 + n_wells_ckpt
    findings.append(
        f"First Conv2d in_channels={in_ch} "
        f"(S={model.channels} + R + pert + wear×{n_wells_ckpt} = {expected_in})"
    )
    if in_ch != expected_in:
        findings.append("WARNING: in_channels mismatch with formula")

    # 3. Forward requires wear_features [B, n_wells] matching n_wells
    #    Three-well input would be [B, 3] → wrong channel count on cat
    try:
        B, C, H, W = 1, model.channels, 32, 32
        S = torch.zeros(B, C, H, W)
        R = torch.zeros(B, 1, H, W)
        P = torch.zeros(B, 1, H, W)
        W2 = torch.ones(B, 2)
        W3 = torch.ones(B, 3)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            _ = model(S.to(device), R.to(device), P.to(device), W2.to(device))
        two_ok = True
        findings.append("Forward with wear_features [B,2]: OK")
    except Exception as e:
        two_ok = False
        findings.append(f"Forward with [B,2] FAILED: {e}")

    three_ok = False
    three_err = None
    try:
        with torch.no_grad():
            _ = model(S.to(device), R.to(device), P.to(device), W3.to(device))
        three_ok = True
        findings.append(
            "Forward with wear_features [B,3]: unexpectedly OK "
            "(would still be wrong semantics if weights assume 2)"
        )
    except Exception as e:
        three_err = f"{type(e).__name__}: {e}"
        findings.append(f"Forward with wear_features [B,3]: FAILS — {three_err}")
        supports_variable = False

    # 4. Wear / world mechanics support variable well count if wear0 matches
    try:
        wear_3 = PerWellWearConfig(
            wells=((8, 8), (16, 16), (24, 24)),
            wear0=(0.0, 0.0, 0.0),
        )
        st = PerWellWearState(wear_3, H=32, W=32, device="cpu")
        findings.append(
            f"PerWellWearState supports n_wells={st.n_wells} from len(wells) "
            f"when wear0 length matches (mechanics are variable-N)"
        )
        mechanics_ok = True
    except Exception as e:
        findings.append(f"PerWellWearState with 3 wells failed: {e}")
        mechanics_ok = False
    findings.append(
        "Self-model input is fixed-size two-well wear channels — not variable-N"
    )
    findings.append(
        "Training distribution and decision_details logging assume two wells "
        "(wear_0/wear_1, well0_floored/well1_floored)"
    )
    findings.append(
        "PerWellWearConfig default wear0=(0,0) requires explicit wear0 for N≠2; "
        "docstring says 'two wells' but loop bodies use len(wells)"
    )

    if three_ok and n_wells_ckpt == 2:
        # Even if shapes somehow worked, weights were trained for 2 channels
        supports_variable = False
        findings.append(
            "Even shape-compatible 3-well input would be semantically invalid: "
            "weights encode exactly two wear channels from training"
        )

    if not three_ok:
        supports_variable = False

    if supports_variable:
        verdict = "supported"
        summary = (
            "Architecture appears to support variable well count without "
            "modification — three-well Check 1/2 could be run as a bonus."
        )
    else:
        verdict = "not_supported"
        summary = (
            "self_model_v3_5 does NOT support N≠2 wells without modification: "
            f"fixed n_wells={n_wells_ckpt} wear input channels "
            f"(Conv2d in_channels={in_ch}), forward rejects [B,3] wear "
            f"({three_err}). Per-well wear *mechanics* are variable-N, but "
            "the frozen self-model's input encoding is pairwise/two-slot. "
            "This is a concrete architectural generalization limit — reported "
            "as the finding; no three-well run and no retrain."
        )

    print(f"  supports_variable_n_wells: {supports_variable}")
    for f in findings:
        print(f"  - {f}")
    print(f"  verdict: {verdict}")
    print(f"  {summary}")

    return {
        "supports_variable_n_wells": supports_variable,
        "n_wells_checkpoint": n_wells_ckpt,
        "conv_in_channels": in_ch,
        "forward_two_wells_ok": two_ok,
        "forward_three_wells_ok": three_ok,
        "forward_three_wells_error": three_err,
        "mechanics_variable_n": mechanics_ok,
        "findings": findings,
        "verdict": verdict,
        "summary": summary,
        "three_well_run_performed": False,
        "reason_stopped": (
            None
            if supports_variable
            else "architectural fixed n_wells=2; Stage A does not retrain"
        ),
    }


def layout_geometry_note() -> Dict:
    """Document layout choice for observations."""
    o = ORIGINAL_WELLS
    r = RELOCATED_WELLS
    return {
        "original": list(o),
        "relocated": list(r),
        "original_inter_well": inter_well_distance(o),
        "relocated_inter_well": inter_well_distance(r),
        "rationale": (
            "Relocated wells (12,20) and (28,4) keep inter-well distance "
            f"{inter_well_distance(r):.3f} equal to training "
            f"{inter_well_distance(o):.3f} (16√2). Spec example (12,20)/(20,6) "
            "has shorter separation (~16.1); we preserve geometry so the test "
            "is 'different layout' not 'harder map'."
        ),
    }
