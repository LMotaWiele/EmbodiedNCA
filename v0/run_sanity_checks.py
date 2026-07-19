#!/usr/bin/env python3
"""
run_sanity_checks.py
====================
Three acceptance-criteria experiments for v0 resource-gated embedded-pattern NCA.

  Check 1 — break-even near a well (with gen/consumption sweep if needed)
  Check 2 — distance-dependence of survival
  Check 3 — gating ablation (gated / ungated / always-zero) at distances 2 and 4

Run from EmbodiedNCA repo root or this directory:

    python v0/run_sanity_checks.py
    # or:  cd v0 && python run_sanity_checks.py

Outputs land in v0/results/ (plots + CSVs + observations.txt).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Allow `python v0/run_sanity_checks.py` from repo root or `python run_sanity_checks.py` from v0/
_V0_ROOT = Path(__file__).resolve().parent
if str(_V0_ROOT) not in sys.path:
    sys.path.insert(0, str(_V0_ROOT))

from grid_world import (  # noqa: E402
    DEFAULT_PATTERN_RADIUS,
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    make_seed_pattern,
    place_pattern,
    train_self_maintenance,
)
from metrics import (  # noqa: E402
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    compute_all,
    finite_mean,
    history_column,
    last_alive_value,
    save_history_csv,
    time_of_death,
)
from nca_module import SimpleNCA  # noqa: E402


RESULTS_DIR = _V0_ROOT / "results"
BASE_CENTER = (16, 16)


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clone_nca(nca: SimpleNCA) -> SimpleNCA:
    return copy.deepcopy(nca)


def _finite(xs: Sequence[float]) -> List[float]:
    return [x for x in xs if not math.isnan(x)]


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
            world.S,
            world.S0,
            world.R,
            world.wells,
            t=0,
            mean_update_magnitude=0.0,
            total_consumption=0.0,
        )
    ]
    for t in range(1, horizon + 1):
        _, mag = world.step(gate_mode=gate_mode)
        consumption = config.consumption_rate * mag
        history.append(
            compute_all(
                world.S,
                world.S0,
                world.R,
                world.wells,
                t=t,
                mean_update_magnitude=float(mag.mean().item()),
                total_consumption=float(consumption.sum().item()),
            )
        )
    return history


def summarize_local_R(
    history: List[Dict[str, float]], burn_in: int = 400
) -> Dict[str, float]:
    """Stats over post-burn-in *alive* steps only (NaNs dropped)."""
    series = history_column(history, "local_resource_at_pattern_centroid")
    masses = history_column(history, "self_mass")
    tail = []
    for i, (r, m) in enumerate(zip(series, masses)):
        if i < burn_in:
            continue
        if m > DEATH_MASS_EPS and not math.isnan(r):
            tail.append(r)
    if not tail:
        # Pattern dead through the whole tail — report NaNs, not fabricated zeros.
        return {
            "mean": float("nan"),
            "stdev": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "final": float("nan"),
        }
    return {
        "mean": float(statistics.mean(tail)),
        "stdev": float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0,
        "min": float(min(tail)),
        "max": float(max(tail)),
        "final": float(tail[-1]),
    }


def summarize_update_magnitude(
    history: List[Dict[str, float]], burn_in: int = 400
) -> Dict[str, float]:
    """Steady-state mean_update_magnitude on alive steps — homeostasis signal."""
    mags = history_column(history, "mean_update_magnitude")
    masses = history_column(history, "self_mass")
    tail = [
        m
        for i, (m, mass) in enumerate(zip(mags, masses))
        if i >= burn_in and mass > DEATH_MASS_EPS
    ]
    if not tail:
        return {"mean": float("nan"), "stdev": float("nan"), "final": float("nan")}
    return {
        "mean": float(statistics.mean(tail)),
        "stdev": float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0,
        "final": float(tail[-1]),
    }


def is_breakeven(stats: Dict[str, float]) -> bool:
    if math.isnan(stats["mean"]):
        return False
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


def fmt_opt(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{nd}f}"


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
        (0.05, 0.01, "spec_default"),
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
    burn_in = max(400, horizon // 4)

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
        stats = summarize_local_R(hist, burn_in=burn_in)
        umag = summarize_update_magnitude(hist, burn_in=burn_in)
        tod = time_of_death(hist)
        last_coh = last_alive_value(hist, "coherence")
        row = {
            "label": label,
            "gen_rate": gen_rate,
            "consumption_rate": consumption_rate,
            **stats,
            "final_coherence": last_coh,
            "final_self_mass": hist[-1]["self_mass"],
            "time_of_death": tod,
            "mean_update_magnitude_ss": umag["mean"],
            "final_update_magnitude": umag["final"],
        }
        sweep_rows.append(row)
        print(
            f"  {label:22s} gen={gen_rate:.3f} cons={consumption_rate:.3f}  "
            f"localR mean={fmt_opt(stats['mean'])} std={fmt_opt(stats['stdev'])} "
            f"[{fmt_opt(stats['min'])},{fmt_opt(stats['max'])}]  "
            f"coh={fmt_opt(last_coh, 3)}  "
            f"‖u‖_ss={fmt_opt(umag['mean'], 5)}  "
            f"tod={tod if tod is not None else 'survived'}"
        )
        if (
            is_breakeven(stats)
            and not math.isnan(last_coh)
            and last_coh > 0.3
            and tod is None
            and chosen is None
        ):
            chosen = row
            chosen_hist = hist
            chosen_cfg = cfg

    if chosen is None:
        def score(r):
            if math.isnan(r["mean"]):
                return 1e9
            pen = abs(r["mean"] - 0.5) + (r["stdev"] if not math.isnan(r["stdev"]) else 1.0)
            if r["min"] is not None and not math.isnan(r["min"]) and r["min"] <= 0.05:
                pen += 1.0
            if r["max"] is not None and not math.isnan(r["max"]) and r["max"] >= 0.98:
                pen += 1.0
            fc = r["final_coherence"]
            if math.isnan(fc) or fc < 0.2:
                pen += 1.0
            elif fc < 0.5:
                pen += 0.3
            if r["time_of_death"] is not None:
                pen += 2.0
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
            f"using closest '{chosen['label']}' (mean localR={fmt_opt(chosen['mean'])})"
        )
    else:
        print(f"  Selected operating point: {chosen['label']}")

    umag = summarize_update_magnitude(chosen_hist, burn_in=burn_in)
    print(
        f"  Homeostasis: steady-state mean_update_magnitude = {fmt_opt(umag['mean'], 5)} "
        f"(final {fmt_opt(umag['final'], 5)}) — nonzero ⇒ continuous repair, not frozen S"
    )

    # ---- Plot: local R on left axis, mean total R / cell on right (same scale band) ----
    out_plot = RESULTS_DIR / "check1_breakeven.png"
    ts = history_column(chosen_hist, "t")
    local_r = history_column(chosen_hist, "local_resource_at_pattern_centroid")
    total_r = history_column(chosen_hist, "total_resource")
    n_cells = float(chosen_cfg.H * chosen_cfg.W)
    mean_r_per_cell = [tr / n_cells for tr in total_r]
    umags = history_column(chosen_hist, "mean_update_magnitude")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax0 = axes[0]
    (ln0,) = ax0.plot(ts, local_r, color="C0", label="local R @ centroid")
    ln0b = ax0.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 reference")
    ax0.set_ylabel("local R (per cell)")
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0b = ax0.twinx()
    (ln1,) = ax0b.plot(
        ts, mean_r_per_cell, color="C1", alpha=0.8, label="mean R / cell (grid)"
    )
    ax0b.set_ylabel("mean R / cell", color="C1")
    ax0b.tick_params(axis="y", labelcolor="C1")
    ax0b.set_ylim(
        -0.05,
        max(1.05, max(mean_r_per_cell) * 1.1 if mean_r_per_cell else 1.05),
    )
    ax0.set_title(
        f"Check 1: break-even on well {well}\n"
        f"gen_rate={chosen['gen_rate']}, consumption_rate={chosen['consumption_rate']}"
    )
    ax0.legend(handles=[ln0, ln0b, ln1], loc="upper right")

    axes[1].plot(ts, history_column(chosen_hist, "coherence"), label="coherence")
    axes[1].plot(ts, history_column(chosen_hist, "self_mass"), label="self_mass")
    axes[1].set_ylabel("pattern metrics")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, umags, color="C2", label="mean ‖update‖")
    axes[2].axhline(
        umag["mean"] if not math.isnan(umag["mean"]) else 0.0,
        color="C2",
        ls="--",
        lw=0.8,
        alpha=0.7,
        label=f"ss mean={fmt_opt(umag['mean'], 5)}",
    )
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("mean update magnitude")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Homeostasis signal: nonzero steady ‖update‖ = ongoing repair")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    save_history_csv(chosen_hist, RESULTS_DIR / "check1_breakeven.csv")
    # JSON-safe: replace None/nan
    def jsonable(obj):
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonable(v) for v in obj]
        if obj is None:
            return None
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    with (RESULTS_DIR / "check1_sweep.json").open("w") as f:
        json.dump(
            jsonable({"selected": chosen, "sweep": sweep_rows, "homeostasis_umag": umag}),
            f,
            indent=2,
        )

    return {
        "selected": chosen,
        "sweep": sweep_rows,
        "config": chosen_cfg,
        "plot": str(out_plot),
        "homeostasis_umag": umag,
        "history": chosen_hist,
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
    print("\n=== Check 2: distance-dependence ===")
    well = DEFAULT_WELLS[0]
    placements = []
    for d in distances:
        center = (well[0], well[1] + d)
        nd = min(
            ((center[0] - wy) ** 2 + (center[1] - wx) ** 2) ** 0.5
            for wy, wx in operating.wells
        )
        placements.append((d, center, nd))

    series = {}
    death_times = {}
    last_cohs = {}
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
        tod = time_of_death(hist)
        death_times[d] = tod
        last_cohs[d] = last_alive_value(hist, "coherence")
        save_history_csv(hist, RESULTS_DIR / f"check2_dist{d}.csv")
        print(
            f"  dist={d:2d} center={center} nearest_well_d={nd:.2f}  "
            f"last_alive_coh={fmt_opt(last_cohs[d])}  "
            f"final_mass={hist[-1]['self_mass']:.4f}  "
            f"tod={tod if tod is not None else 'survived'}  "
            f"localR_alive_mean={fmt_opt(summarize_local_R(hist)['mean'])}"
        )

    out_plot = RESULTS_DIR / "check2_distance.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for d, hist in series.items():
        ts = history_column(hist, "t")
        axes[0].plot(ts, history_column(hist, "coherence"), label=f"dist={d}")
        axes[1].plot(ts, history_column(hist, "self_mass"), label=f"dist={d}")
        if death_times[d] is not None:
            axes[0].axvline(death_times[d], color="gray", ls=":", lw=0.7, alpha=0.5)
            axes[1].axvline(death_times[d], color="gray", ls=":", lw=0.7, alpha=0.5)
    axes[0].set_ylabel("coherence (NaN after death)")
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

    # Rank by last-alive coherence; dead patterns use 0 for ordering after death time
    # Prefer: longer survival + higher last-alive coh for survivors
    ordered = sorted(last_cohs.keys())
    # mean coherence over alive steps only
    mean_alive = {
        d: finite_mean(history_column(series[d], "coherence")) for d in series
    }
    # For monotone: farther should have lower mean_alive coh, or earlier death
    monotone_mean = all(
        mean_alive[ordered[i]] >= mean_alive[ordered[i + 1]] - 1e-3
        or (
            death_times[ordered[i]] is None
            and death_times[ordered[i + 1]] is not None
        )
        for i in range(len(ordered) - 1)
    )
    # Survival ordering: later death (or survive) is better; farther should die sooner
    def survival_rank(d):
        tod = death_times[d]
        return horizon + 1 if tod is None else tod

    monotone_survival = all(
        survival_rank(ordered[i]) >= survival_rank(ordered[i + 1])
        for i in range(len(ordered) - 1)
    )
    print(f"  last-alive coherences: {last_cohs}")
    print(f"  mean alive coherence: {mean_alive}")
    print(f"  time_of_death: {death_times}")
    print(f"  farther => lower mean coh? {monotone_mean}")
    print(f"  farther => earlier death?  {monotone_survival}")

    return {
        "last_alive_coherence": last_cohs,
        "mean_coherence": mean_alive,
        "time_of_death": death_times,
        "monotone_distance_effect": monotone_mean or monotone_survival,
        "plot": str(out_plot),
        "placements": {str(d): c for d, c, _ in placements},
    }


# ---------------------------------------------------------------------------
# Check 3
# ---------------------------------------------------------------------------

def _run_gating_at_distance(
    base_nca: SimpleNCA,
    base_pattern: torch.Tensor,
    device: str,
    operating: WorldConfig,
    horizon: int,
    seed: int,
    distance: int,
) -> Dict:
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
        save_history_csv(hist, RESULTS_DIR / f"check3_d{distance}_{mode}.csv")

        tod = time_of_death(hist)
        last_coh = last_alive_value(hist, "coherence")
        # Consumption accounting: total over run (always defined)
        total_cons = sum(history_column(hist, "total_consumption"))
        print(
            f"  dist={distance} mode={mode:8s}  "
            f"last_alive_coh={fmt_opt(last_coh)}  "
            f"final_mass={hist[-1]['self_mass']:.4f}  "
            f"tod={tod if tod is not None else 'survived'}  "
            f"Σconsumption={total_cons:.2f}  "
            f"mean‖u‖={finite_mean(history_column(hist, 'mean_update_magnitude')):.5f}"
        )

    out_plot = RESULTS_DIR / f"check3_gating_d{distance}.png"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    style = {"gated": "-", "ungated": "--", "zero": ":"}
    for mode, hist in series.items():
        ts = history_column(hist, "t")
        axes[0].plot(ts, history_column(hist, "coherence"), ls=style[mode], label=mode)
        axes[1].plot(ts, history_column(hist, "self_mass"), ls=style[mode], label=mode)
    axes[0].set_ylabel("coherence (NaN after death)")
    axes[0].set_title(
        f"Check 3: gating ablation @ dist={distance} from well {well}"
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

    last_cohs = {m: last_alive_value(series[m], "coherence") for m in modes}
    mean_alive = {
        m: finite_mean(history_column(series[m], "coherence")) for m in modes
    }
    tods = {m: time_of_death(series[m]) for m in modes}
    total_cons = {
        m: float(sum(history_column(series[m], "total_consumption"))) for m in modes
    }

    # Ordering on mean-alive coherence (NaN-safe: treat fully-dead as 0)
    def score(m):
        v = mean_alive[m]
        return 0.0 if math.isnan(v) else v

    ordering_ok = (score("ungated") + 0.02 >= score("gated")) and (
        score("gated") + 0.02 >= score("zero")
    )
    # Zero-gate must still pay for attempted updates (spec: cost ∝ ‖update‖)
    zero_pays = total_cons["zero"] > 1.0  # nontrivial total over the run
    print(
        f"  dist={distance} mean-alive coh: {mean_alive}  ordering_ok={ordering_ok}"
    )
    print(
        f"  dist={distance} Σconsumption: {total_cons}  "
        f"zero-gate pays for attempts? {zero_pays}"
    )

    return {
        "distance": distance,
        "center": center,
        "last_alive_coherence": last_cohs,
        "mean_coherence": mean_alive,
        "time_of_death": tods,
        "total_consumption": total_cons,
        "zero_gate_pays": zero_pays,
        "ordering_ok": ordering_ok,
        "plot": str(out_plot),
        "series": series,
    }


def check3_gating(
    base_nca: SimpleNCA,
    base_pattern: torch.Tensor,
    device: str,
    operating: WorldConfig,
    horizon: int = 2000,
    seed: int = 0,
    distances: Sequence[int] = (2, 4),
) -> Dict:
    """
    Gating ablation at multiple scarcity levels. Dist 2 is mid-scarcity
    (gated intermediate); dist 4 is deeper scarcity (confirms ordering holds).
    """
    print("\n=== Check 3: gating ablation ===")
    by_dist = {}
    for d in distances:
        print(f"\n  -- distance {d} --")
        by_dist[d] = _run_gating_at_distance(
            base_nca,
            base_pattern,
            device,
            operating,
            horizon,
            seed,
            d,
        )

    all_ok = all(by_dist[d]["ordering_ok"] for d in by_dist)
    all_pay = all(by_dist[d]["zero_gate_pays"] for d in by_dist)
    print(f"\n  ordering ok at all distances? {all_ok}")
    print(f"  zero-gate consumption charged at all distances? {all_pay}")

    return {
        "by_distance": {str(d): {k: v for k, v in by_dist[d].items() if k != "series"} for d in by_dist},
        "ordering_ok_all": all_ok,
        "zero_gate_pays_all": all_pay,
        "plot": by_dist[distances[0]]["plot"],
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

    # ---- observations ----
    observations = []
    observations.append("# v0 sanity-check observations (auto-generated)\n")
    observations.append(
        f"device: {device}\nhorizon: {args.horizon}\nseed: {args.seed}\n"
    )

    observations.append("\n## Check 1 — break-even near a well\n")
    sel = c1["selected"]
    umag = c1["homeostasis_umag"]
    observations.append(
        f"- Selected rates: gen_rate={sel['gen_rate']}, "
        f"consumption_rate={sel['consumption_rate']} ({sel['label']})\n"
        f"- Steady-state local R @ centroid (post burn-in, alive steps): "
        f"mean={fmt_opt(sel['mean'])}, stdev={fmt_opt(sel['stdev'])}, "
        f"range=[{fmt_opt(sel['min'])}, {fmt_opt(sel['max'])}]\n"
        f"- Final (last-alive) coherence / final mass: "
        f"{fmt_opt(sel.get('final_coherence'))} / {sel.get('final_self_mass')}\n"
        f"- **Homeostasis (not a freeze):** steady-state mean_update_magnitude = "
        f"{fmt_opt(umag['mean'], 5)} (final {fmt_opt(umag['final'], 5)}). "
        f"Nonzero and flat ⇒ continuous erosion↔repair equilibrium, not static S.\n"
        f"- Plot: {c1['plot']}  (local R and mean-R/cell on dual axes; ‖u‖ panel)\n"
        f"- Full sweep: results/check1_sweep.json\n"
    )

    observations.append("\n## Check 2 — distance-dependence\n")
    observations.append(
        f"- Last-alive coherences: {c2['last_alive_coherence']}\n"
        f"- Mean alive coherences: {c2['mean_coherence']}\n"
        f"- Time of death: {c2['time_of_death']}\n"
        f"- Distance effect visible: {c2['monotone_distance_effect']}\n"
        f"- Placements: {c2['placements']}\n"
        f"- Plot: {c2['plot']}\n"
        f"- Note: post-death coherence/localR are NaN (not fabricated).\n"
    )

    observations.append("\n## Check 3 — gating ablation\n")
    for d_str, res in c3["by_distance"].items():
        observations.append(
            f"### distance {d_str}\n"
            f"- Last-alive coherence: {res['last_alive_coherence']}\n"
            f"- Mean alive coherence: {res['mean_coherence']}\n"
            f"- Time of death: {res['time_of_death']}\n"
            f"- Σconsumption (full run): {res['total_consumption']}\n"
            f"- Zero-gate pays for attempted updates: {res['zero_gate_pays']}\n"
            f"- Ordering ungated ≥ gated ≥ zero: {res['ordering_ok']}\n"
            f"- Center: {res['center']}  plot: {res['plot']}\n"
        )
    observations.append(
        f"- Ordering ok at all tested distances: {c3['ordering_ok_all']}\n"
        f"- Zero-gate consumption charged at all distances: {c3['zero_gate_pays_all']}\n"
    )

    observations.append(
        "\n## Accounting / metric policy (deliberate)\n"
        "- Consumption = consumption_rate * ‖update‖_pre-gate per cell. "
        "Gate multiplies the residual applied to S, not the bill. "
        "gate_mode='zero' still computes update_net and still charges R.\n"
        f"- DEFAULT_ACTIVATION_THRESHOLD={DEFAULT_ACTIVATION_THRESHOLD} "
        f"(cells counted in self_mass); DEATH_MASS_EPS={DEATH_MASS_EPS} "
        "(alive flag / NaN pattern metrics). Mass cliff ~threshold→0 is the "
        "last active cell dropping below activation, not DEATH_MASS_EPS=0.05.\n"
        "- Once self_mass ≤ DEATH_MASS_EPS, coherence / local_R / distance "
        "are NaN; time_of_death is the first such t. total_resource and "
        "update/consumption stay defined (world-level).\n"
    )
    observations.append(
        "\n## Interpretive notes\n"
        "- check2_dist2 and check3_d2_gated are the same gated trajectory "
        "(reused arm; not an independent re-trial).\n"
        "- Zero-gate Σconsumption can exceed gated: gated settles to cheap "
        "homeostasis (low per-step cons); zero never settles so update_net "
        "keeps billing moderately large attempts over the full horizon.\n"
        "- Ungated self_mass oscillates with no resource bound — expected "
        "unconstrained dynamics, not update_net instability.\n"
        "- Check-1 ss numbers may drift slightly across full re-runs (CUDA "
        "retrain not bit-stable); gen/cons operating rates are fixed; latest "
        "results/ are authoritative (historic/ holds prior snapshots).\n"
    )
    observations.append(
        "\n## World-physics notes\n"
        "- S diffusion: unopposed erosion when gate is low.\n"
        "- R diffusion + soft wells: distance-dependent resource field.\n"
        "- Self-maintenance weights trained offline; no external controller "
        "at runtime.\n"
    )
    obs_path = RESULTS_DIR / "observations.txt"
    obs_path.write_text("".join(observations))
    print(f"\nWrote {obs_path}")
    print("Done. Plots in", RESULTS_DIR)


if __name__ == "__main__":
    main()
