#!/usr/bin/env python3
"""
run_sanity_checks.py
====================
Three acceptance-criteria experiments for v0 resource-gated embedded-pattern NCA.

  Check 1 — break-even near a well (with gen/consumption sweep if needed)
  Check 2 — distance-dependence of survival
  Check 3 — gating ablation (gated / ungated / always-zero)

Run from this directory:

    python run_sanity_checks.py

Outputs land in ./results/ (plots + CSVs + observations.txt).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from grid_world import (
    DEFAULT_PATTERN_RADIUS,
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    make_seed_pattern,
    place_pattern,
    train_self_maintenance,
)
from metrics import compute_all, history_column, save_history_csv
from nca_module import SimpleNCA


RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_CENTER = (16, 16)


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clone_nca(nca: SimpleNCA) -> SimpleNCA:
    return copy.deepcopy(nca)


@torch.no_grad()
def run_episode(
    nca: SimpleNCA,
    config: WorldConfig,
    pattern_center: Tuple[int, int],
    horizon: int,
    seed: int = 0,
    gate_mode: str = "gated",
    S0: Optional[torch.Tensor] = None,
    base_pattern: Optional[torch.Tensor] = None,
) -> List[Dict[str, float]]:
    if S0 is None and base_pattern is not None:
        S0 = place_pattern(base_pattern, pattern_center, BASE_CENTER)

    world = ResourceGrid(
        nca=nca,
        config=config,
        pattern_center=pattern_center,
        seed=seed,
        S0=S0,
    )
    history: List[Dict[str, float]] = [
        compute_all(
            world.S, world.S0, world.R, world.wells, t=0, mean_update_magnitude=0.0
        )
    ]
    for t in range(1, horizon + 1):
        _, mag = world.step(gate_mode=gate_mode)
        history.append(
            compute_all(
                world.S,
                world.S0,
                world.R,
                world.wells,
                t=t,
                mean_update_magnitude=float(mag.mean().item()),
            )
        )
    return history


def summarize_local_R(
    history: List[Dict[str, float]], burn_in: int = 400
) -> Dict[str, float]:
    series = history_column(history, "local_resource_at_pattern_centroid")
    tail = series[burn_in:] if len(series) > burn_in else series
    return {
        "mean": float(statistics.mean(tail)),
        "stdev": float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0,
        "min": float(min(tail)),
        "max": float(max(tail)),
        "final": float(series[-1]),
    }


def is_breakeven(stats: Dict[str, float]) -> bool:
    return (
        0.20 <= stats["mean"] <= 0.85
        and stats["min"] > 0.08
        and stats["max"] < 0.98
        and stats["stdev"] < 0.20
    )


def cfg_from_operating(
    operating: WorldConfig,
    device: str,
    gen_rate: Optional[float] = None,
    consumption_rate: Optional[float] = None,
) -> WorldConfig:
    return WorldConfig(
        H=operating.H,
        W=operating.W,
        channels=operating.channels,
        hidden_dim=operating.hidden_dim,
        R_max=operating.R_max,
        decay_rate=operating.decay_rate,
        gen_rate=operating.gen_rate if gen_rate is None else gen_rate,
        consumption_rate=(
            operating.consumption_rate if consumption_rate is None else consumption_rate
        ),
        diffusion_rate=operating.diffusion_rate,
        resource_diffusion=operating.resource_diffusion,
        well_sigma=operating.well_sigma,
        wells=operating.wells,
        device=device,
    )


# ---------------------------------------------------------------------------
# Check 1
# ---------------------------------------------------------------------------

def check1_breakeven(
    base_nca: SimpleNCA,
    base_pattern: torch.Tensor,
    device: str,
    horizon: int = 2000,
    seed: int = 0,
) -> Dict:
    print("\n=== Check 1: break-even near a well ===")
    well = DEFAULT_WELLS[0]

    candidates: List[Tuple[float, float, str]] = [
        # Spec-literal defaults first (expected to be too weak for multi-cell)
        (0.05, 0.01, "spec_default"),
        # Operating region around the tuned well field (gen≈0.2, wide wells)
        (0.20, 0.2, "gen0.2_cons0.2"),
        (0.20, 0.5, "gen0.2_cons0.5"),
        (0.20, 1.0, "gen0.2_cons1.0"),
        (0.20, 1.5, "gen0.2_cons1.5"),
        (0.20, 2.0, "gen0.2_cons2.0"),
        (0.20, 3.0, "gen0.2_cons3.0"),
        (0.15, 1.0, "gen0.15_cons1.0"),
        (0.25, 1.0, "gen0.25_cons1.0"),
        (0.25, 2.0, "gen0.25_cons2.0"),
        (0.10, 0.5, "gen0.1_cons0.5"),
        (0.30, 2.0, "gen0.3_cons2.0"),
    ]

    base_cfg = WorldConfig(device=device)
    sweep_rows = []
    chosen = None
    chosen_hist = None
    chosen_cfg = None

    for gen_rate, consumption_rate, label in candidates:
        cfg = cfg_from_operating(
            base_cfg, device, gen_rate=gen_rate, consumption_rate=consumption_rate
        )
        nca = clone_nca(base_nca).to(device)
        hist = run_episode(
            nca,
            cfg,
            pattern_center=well,
            horizon=horizon,
            seed=seed,
            gate_mode="gated",
            base_pattern=base_pattern,
        )
        stats = summarize_local_R(hist, burn_in=max(400, horizon // 4))
        row = {
            "label": label,
            "gen_rate": gen_rate,
            "consumption_rate": consumption_rate,
            **stats,
            "final_coherence": hist[-1]["coherence"],
            "final_self_mass": hist[-1]["self_mass"],
        }
        sweep_rows.append(row)
        print(
            f"  {label:22s} gen={gen_rate:.3f} cons={consumption_rate:.3f}  "
            f"localR mean={stats['mean']:.4f} std={stats['stdev']:.4f} "
            f"[{stats['min']:.4f},{stats['max']:.4f}]  "
            f"coh_final={row['final_coherence']:.3f}"
        )
        # Prefer break-even AND still-alive pattern
        if is_breakeven(stats) and row["final_coherence"] > 0.3 and chosen is None:
            chosen = row
            chosen_hist = hist
            chosen_cfg = cfg

    if chosen is None:
        # Closest to mean 0.5 among candidates that keep some coherence
        def score(r):
            pen = abs(r["mean"] - 0.5) + r["stdev"]
            if r["min"] <= 0.05:
                pen += 1.0
            if r["max"] >= 0.98:
                pen += 1.0
            if r["final_coherence"] < 0.2:
                pen += 1.0
            elif r["final_coherence"] < 0.5:
                pen += 0.3
            return pen

        ranked = sorted(sweep_rows, key=score)
        chosen = ranked[0]
        cfg = cfg_from_operating(
            base_cfg,
            device,
            gen_rate=chosen["gen_rate"],
            consumption_rate=chosen["consumption_rate"],
        )
        nca = clone_nca(base_nca).to(device)
        chosen_hist = run_episode(
            nca,
            cfg,
            pattern_center=well,
            horizon=horizon,
            seed=seed,
            gate_mode="gated",
            base_pattern=base_pattern,
        )
        chosen_cfg = cfg
        print(
            f"  NOTE: no candidate met strict break-even+alive band; "
            f"using closest '{chosen['label']}' (mean localR={chosen['mean']:.4f}, "
            f"coh={chosen['final_coherence']:.3f})"
        )
    else:
        print(f"  Selected operating point: {chosen['label']}")

    out_plot = RESULTS_DIR / "check1_breakeven.png"
    ts = history_column(chosen_hist, "t")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(
        ts,
        history_column(chosen_hist, "local_resource_at_pattern_centroid"),
        label="local R @ centroid",
    )
    axes[0].plot(
        ts,
        history_column(chosen_hist, "total_resource"),
        label="total R (sum)",
        alpha=0.7,
    )
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 reference")
    axes[0].set_ylabel("resource")
    axes[0].set_title(
        f"Check 1: break-even on well {well}\n"
        f"gen_rate={chosen['gen_rate']}, consumption_rate={chosen['consumption_rate']}"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, history_column(chosen_hist, "coherence"), label="coherence")
    axes[1].plot(ts, history_column(chosen_hist, "self_mass"), label="self_mass")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("pattern metrics")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    save_history_csv(chosen_hist, RESULTS_DIR / "check1_breakeven.csv")
    with (RESULTS_DIR / "check1_sweep.json").open("w") as f:
        json.dump({"selected": chosen, "sweep": sweep_rows}, f, indent=2)

    return {
        "selected": chosen,
        "sweep": sweep_rows,
        "config": chosen_cfg,
        "plot": str(out_plot),
    }


# ---------------------------------------------------------------------------
# Check 2
# ---------------------------------------------------------------------------

def check2_distance(
    base_nca: SimpleNCA,
    base_pattern: torch.Tensor,
    device: str,
    operating: WorldConfig,
    horizon: int = 2000,
    seed: int = 0,
    distances: Sequence[int] = (2, 4, 8),
) -> Dict:
    """
    Place pattern at increasing distance from well 0, along +x.
    Defaults 2 / 4 / 8 sit on the resource gradient (near well survives,
    mid degrades, far dies). Spec's 20 is off-grid relative to the second
    well at (24,24); 8 is already past the survival threshold.
    """
    print("\n=== Check 2: distance-dependence ===")
    well = DEFAULT_WELLS[0]
    placements = []
    for d in distances:
        center = (well[0], well[1] + d)
        # nearest-well distance (for logging)
        nd = min(
            ((center[0] - wy) ** 2 + (center[1] - wx) ** 2) ** 0.5
            for wy, wx in operating.wells
        )
        placements.append((d, center, nd))

    series = {}
    for d, center, nd in placements:
        cfg = cfg_from_operating(operating, device)
        nca = clone_nca(base_nca).to(device)
        hist = run_episode(
            nca,
            cfg,
            pattern_center=center,
            horizon=horizon,
            seed=seed,
            gate_mode="gated",
            base_pattern=base_pattern,
        )
        series[d] = hist
        save_history_csv(hist, RESULTS_DIR / f"check2_dist{d}.csv")
        print(
            f"  dist={d:2d} center={center} nearest_well_d={nd:.2f}  "
            f"final coh={hist[-1]['coherence']:.4f}  "
            f"mass={hist[-1]['self_mass']:.4f}  "
            f"localR_mean_tail={summarize_local_R(hist)['mean']:.4f}"
        )

    out_plot = RESULTS_DIR / "check2_distance.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for d, hist in series.items():
        ts = history_column(hist, "t")
        axes[0].plot(ts, history_column(hist, "coherence"), label=f"dist={d}")
        axes[1].plot(ts, history_column(hist, "self_mass"), label=f"dist={d}")
    axes[0].set_ylabel("coherence")
    axes[0].set_title("Check 2: distance from well vs pattern survival")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("self_mass")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    finals = {d: series[d][-1]["coherence"] for d in series}
    # Also compare area-under-curve as a smoother distance signal
    auc = {
        d: float(sum(history_column(series[d], "coherence"))) / len(series[d])
        for d in series
    }
    ordered = sorted(finals.keys())
    monotone_final = all(
        finals[ordered[i]] >= finals[ordered[i + 1]] - 1e-3
        for i in range(len(ordered) - 1)
    )
    monotone_auc = all(
        auc[ordered[i]] >= auc[ordered[i + 1]] - 1e-3
        for i in range(len(ordered) - 1)
    )
    print(f"  final coherences: {finals}")
    print(f"  mean coherence (AUC/T): {auc}")
    print(f"  farther => lower final coh? {monotone_final}")
    print(f"  farther => lower mean coh?  {monotone_auc}")

    return {
        "finals": finals,
        "mean_coherence": auc,
        "monotone_distance_effect": monotone_final or monotone_auc,
        "plot": str(out_plot),
        "placements": {str(d): c for d, c, _ in placements},
    }


# ---------------------------------------------------------------------------
# Check 3
# ---------------------------------------------------------------------------

def check3_gating(
    base_nca: SimpleNCA,
    base_pattern: torch.Tensor,
    device: str,
    operating: WorldConfig,
    horizon: int = 2000,
    seed: int = 0,
    distance: int = 2,
) -> Dict:
    """
    Mid-scarcity placement (default dist=2): enough resource that gated is not
    identical to zero, little enough that gated is not identical to ungated.
    """
    print("\n=== Check 3: gating ablation ===")
    well = DEFAULT_WELLS[0]
    center = (well[0], well[1] + distance)
    S0 = place_pattern(base_pattern, center, BASE_CENTER).to(device)

    modes = ("gated", "ungated", "zero")
    series = {}
    for mode in modes:
        cfg = cfg_from_operating(operating, device)
        nca = clone_nca(base_nca).to(device)
        hist = run_episode(
            nca,
            cfg,
            pattern_center=center,
            horizon=horizon,
            seed=seed,
            gate_mode=mode,
            S0=S0.clone(),
        )
        series[mode] = hist
        save_history_csv(hist, RESULTS_DIR / f"check3_{mode}.csv")
        print(
            f"  mode={mode:8s}  final coh={hist[-1]['coherence']:.4f}  "
            f"mass={hist[-1]['self_mass']:.4f}"
        )

    out_plot = RESULTS_DIR / "check3_gating.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    style = {"gated": "-", "ungated": "--", "zero": ":"}
    for mode, hist in series.items():
        ts = history_column(hist, "t")
        axes[0].plot(ts, history_column(hist, "coherence"), ls=style[mode], label=mode)
        axes[1].plot(ts, history_column(hist, "self_mass"), ls=style[mode], label=mode)
    axes[0].set_ylabel("coherence")
    axes[0].set_title(
        f"Check 3: gating ablation (pattern {distance} cells from well {well})"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("self_mass")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    c_g = series["gated"][-1]["coherence"]
    c_u = series["ungated"][-1]["coherence"]
    c_z = series["zero"][-1]["coherence"]
    mid = horizon // 2
    mid_vals = {m: series[m][mid]["coherence"] for m in modes}
    mean_vals = {
        m: float(sum(history_column(series[m], "coherence"))) / len(series[m])
        for m in modes
    }
    # Prefer mean-over-time ordering; final can be noisy if ungated slowly drifts
    ordering_ok = (mean_vals["ungated"] + 0.02 >= mean_vals["gated"]) and (
        mean_vals["gated"] + 0.02 >= mean_vals["zero"]
    )
    print(f"  final coherence: gated={c_g:.4f} ungated={c_u:.4f} zero={c_z:.4f}")
    print(f"  mid-horizon coherence: {mid_vals}")
    print(f"  mean coherence: {mean_vals}")
    print(f"  ordering ungated >= gated >= zero (on mean, slack 0.02)? {ordering_ok}")

    return {
        "final_coherence": {"gated": c_g, "ungated": c_u, "zero": c_z},
        "mid_coherence": mid_vals,
        "mean_coherence": mean_vals,
        "ordering_ok": ordering_ok,
        "plot": str(out_plot),
        "center": center,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v0 embodied NCA sanity checks")
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(prefer_cuda=not args.cpu)
    print(f"device={device}  horizon={args.horizon}  seed={args.seed}")

    cfg0 = WorldConfig(device=device)
    nca = SimpleNCA(
        channels=cfg0.channels,
        hidden_dim=cfg0.hidden_dim,
        grid_size=cfg0.H,
    ).to(device)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    base_pattern = make_seed_pattern(
        cfg0.channels,
        cfg0.H,
        cfg0.W,
        center=BASE_CENTER,
        radius=DEFAULT_PATTERN_RADIUS,
        device="cpu",
        generator=g,
    ).to(device)

    if not args.skip_train:
        print("\n=== Training local self-maintenance rule ===")
        print("  (curriculum + translations + spatial partial gates)")
        losses = train_self_maintenance(
            nca,
            base_pattern,
            diffusion_rate=cfg0.diffusion_rate,
            device=device,
            seed=args.seed,
            verbose=True,
        )
        with (RESULTS_DIR / "train_losses.json").open("w") as f:
            json.dump({"losses": losses, "final": losses[-1] if losses else None}, f)
        torch.save(
            {"nca": nca.state_dict(), "base_pattern": base_pattern.cpu()},
            RESULTS_DIR / "trained_nca.pt",
        )
        print(f"  final train loss: {losses[-1]:.6f}")
    else:
        ckpt = RESULTS_DIR / "trained_nca.pt"
        if ckpt.exists():
            data = torch.load(ckpt, map_location=device, weights_only=False)
            nca.load_state_dict(data["nca"])
            base_pattern = data["base_pattern"].to(device)
            print(f"Loaded checkpoint {ckpt}")
        else:
            print("WARNING: --skip_train and no checkpoint; using untrained NCA.")

    nca.eval()
    for p in nca.parameters():
        p.requires_grad_(False)

    c1 = check1_breakeven(
        nca, base_pattern, device, horizon=args.horizon, seed=args.seed
    )
    operating = c1["config"]
    c2 = check2_distance(
        nca, base_pattern, device, operating, horizon=args.horizon, seed=args.seed
    )
    c3 = check3_gating(
        nca, base_pattern, device, operating, horizon=args.horizon, seed=args.seed
    )

    observations = []
    observations.append("# v0 sanity-check observations (auto-generated)\n")
    observations.append(
        f"device: {device}\nhorizon: {args.horizon}\nseed: {args.seed}\n"
    )
    observations.append("\n## Check 1 — break-even near a well\n")
    sel = c1["selected"]
    observations.append(
        f"- Selected rates: gen_rate={sel['gen_rate']}, "
        f"consumption_rate={sel['consumption_rate']} ({sel['label']})\n"
        f"- Steady-state local R @ centroid (post burn-in): "
        f"mean={sel['mean']:.4f}, stdev={sel['stdev']:.4f}, "
        f"range=[{sel['min']:.4f}, {sel['max']:.4f}]\n"
        f"- Final coherence / self_mass: {sel.get('final_coherence')} / "
        f"{sel.get('final_self_mass')}\n"
        f"- Plot: {c1['plot']}\n"
        f"- Full sweep: results/check1_sweep.json\n"
    )
    observations.append("\n## Check 2 — distance-dependence\n")
    observations.append(
        f"- Final coherences: {c2['finals']}\n"
        f"- Mean coherences: {c2['mean_coherence']}\n"
        f"- Distance effect visible: {c2['monotone_distance_effect']}\n"
        f"- Placements: {c2['placements']}\n"
        f"- Plot: {c2['plot']}\n"
    )
    observations.append("\n## Check 3 — gating ablation\n")
    observations.append(
        f"- Final coherence: {c3['final_coherence']}\n"
        f"- Mid-horizon coherence: {c3['mid_coherence']}\n"
        f"- Mean coherence: {c3['mean_coherence']}\n"
        f"- Ordering ungated >= gated >= zero: {c3['ordering_ok']}\n"
        f"- Pattern center: {c3['center']}\n"
        f"- Plot: {c3['plot']}\n"
    )
    observations.append(
        "\n## World-physics notes\n"
        "- S diffusion: unopposed erosion when gate is low.\n"
        "- R diffusion + soft wells: distance-dependent resource field.\n"
        "- Self-maintenance weights trained offline (translations + partial\n"
        "  gates + curriculum); no external controller at runtime.\n"
    )
    obs_path = RESULTS_DIR / "observations.txt"
    obs_path.write_text("".join(observations))
    print(f"\nWrote {obs_path}")
    print("Done. Plots in", RESULTS_DIR)


if __name__ == "__main__":
    main()
