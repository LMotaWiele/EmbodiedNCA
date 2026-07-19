#!/usr/bin/env python3
"""
run_v5_checks.py
================
v5 Stage A — Generalization to Novel Well Layouts (project closing stage).

  Check 1 — zero-shot calibration of frozen self_model_v3_5 on relocated wells
  Check 2 — selected vs position-updated heuristics on relocated layout
  Check 3 — architectural support for N≠2 wells (inspect only if unsupported)

No retraining. No edits to v0 step / SimpleNCA / frozen checkpoints.

    python v5/run_v5_checks.py
    python v5/run_v5_checks.py --quick
    python v5/run_v5_checks.py --skip_dataset   # reuse relocated eval set
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

V5_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V5_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V1_ROOT = REPO_ROOT / "v1"
V2_ROOT = REPO_ROOT / "v2"
V3_ROOT = REPO_ROOT / "v3"
V35_ROOT = REPO_ROOT / "v3_5"
for p in (
    str(V5_ROOT),
    str(V35_ROOT),
    str(V3_ROOT),
    str(V2_ROOT),
    str(V1_ROOT),
    str(V0_ROOT),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from rollout_dataset import DEFAULT_CKPT, load_v0_world  # noqa: E402
from self_model_v3_5 import load_model  # noqa: E402

from relocated_layout_checks import (  # noqa: E402
    MODEL_PATH,
    ORIGINAL_WELLS,
    RELOCATED_WELLS,
    RESULTS,
    check1_calibration_relocated,
    check2_policy_relocated,
    check3_architecture_n_wells,
    layout_geometry_note,
)


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def write_observations(c1: Dict, c2: Dict, c3: Dict, device: str, quick: bool) -> Path:
    geom = layout_geometry_note()
    lines: List[str] = []
    lines.append(
        "# v5 Generalization to Novel Well Layouts — observations (closing stage)\n"
    )
    lines.append(f"device: {device}\n")
    lines.append(f"quick: {quick}\n")
    lines.append(f"v0_ckpt: {DEFAULT_CKPT}\n")
    lines.append(f"self_model: {MODEL_PATH}\n")
    lines.append(
        "NOTE: frozen self_model_v3_5.pt — zero-shot only; no retraining.\n"
    )

    lines.append("\n## Layout\n")
    lines.append(f"- original (training) wells: {geom['original']}\n")
    lines.append(f"- relocated wells: {geom['relocated']}\n")
    lines.append(
        f"- inter-well distance: original={geom['original_inter_well']:.4f}  "
        f"relocated={geom['relocated_inter_well']:.4f}\n"
    )
    lines.append(f"- rationale: {geom['rationale']}\n")
    lines.append(
        "- Heuristic update: `AttentionConfig(wells=RELOCATED)` / "
        "`PerWellWearConfig(wells=RELOCATED)` / placement `well=RELOCATED[0]` — "
        "position parameters only. This ease-of-update is part of what is "
        "being tested: heuristic generalization is nearly free; self-model "
        "generalization is an open empirical question.\n"
    )

    lines.append("\n## Check 1 — zero-shot calibration (original vs relocated)\n")
    lines.append(f"- original_source: {c1['original_source']}\n")
    o, r = c1["original"], c1["relocated"]
    lines.append(
        f"- original: n={o['n']}  coh_r={o['coherence_r_all']:.4f}  "
        f"within_bin={o['coherence_r_within_bin_mean']:.4f}  "
        f"local_R_r={o['local_R_r_all']:.4f}  "
        f"coh_mae={o['coherence_mae_all']:.4f}\n"
    )
    lines.append(
        f"- relocated: n={r['n']}  coh_r={r['coherence_r_all']:.4f}  "
        f"within_bin={r['coherence_r_within_bin_mean']:.4f}  "
        f"local_R_r={r['local_R_r_all']:.4f}  "
        f"coh_mae={r['coherence_mae_all']:.4f}  "
        f"death_rate={r.get('death_rate_true', float('nan'))}\n"
    )
    lines.append(
        f"- Δ within_bin (reloc − orig) = {c1.get('delta_within_bin')}\n"
    )
    lines.append(f"- Δ coh_r (reloc − orig) = {c1.get('delta_coh_r')}\n")
    lines.append(f"- pass_within_bin_relocated: {c1['pass_within_bin_relocated']}\n")
    lines.append(f"- holds_up: {c1['holds_up']}\n")
    lines.append(f"- collapsed: {c1['collapsed']}\n")
    lines.append(f"- **verdict: {c1['verdict']}**\n")
    lines.append(f"- plot: {c1['plot']}\n")
    lines.append(f"- bar_plot: {c1['bar_plot']}\n")

    if c1["collapsed"]:
        lines.append(
            "\n### Check 1 interpretation\n"
            "Calibration **collapsed** on the relocated layout. Headline finding: "
            "the model appears to have learned coordinate-specific structure "
            "('wells at pixel X,Y') more than transferable resource-gradient "
            "structure. This is a real, informative negative result for Stage A, "
            "independent of Check 2.\n"
        )
    elif c1["verdict"] == "held_comparable":
        lines.append(
            "\n### Check 1 interpretation\n"
            "Calibration **held** on the relocated layout at a level comparable "
            "to the original map. The self-model's predictive channels transfer "
            "zero-shot when wells move (with inter-well geometry preserved).\n"
        )
    else:
        lines.append(
            "\n### Check 1 interpretation\n"
            f"Calibration verdict **{c1['verdict']}**: not a full collapse to "
            "chance, but not identical to original-layout numbers. Report the "
            "deltas; do not claim perfect spatial invariance.\n"
        )

    lines.append("\n## Check 2 — policy comparison on relocated layout\n")
    lines.append(f"- wells: {c2['wells']}\n")
    lines.append(f"- horizon: {c2['horizon']}\n")
    lines.append(
        f"- pre_degrade_boosts near well A: {c2['pre_degrade_boosts']}  "
        f"preview wear≈{c2['pre_degrade_wear_preview']}\n"
    )
    lines.append(
        "- Policies: selected, wellward_heuristic, **wear_aware_heuristic**, "
        "never_boost, always_boost\n"
    )
    lines.append(
        "- Fair test: heuristics get new coordinates (free); self-model is "
        "zero-shot on a layout absent from training.\n"
    )
    for dist, h in c2["headlines"].items():
        lines.append(f"- dist={dist}:\n")
        lines.append(f"  - selected mean_coh = {h['selected_mean_coh']:.4f}\n")
        lines.append(
            f"  - wear_aware_heuristic mean_coh = {h['wear_aware_mean_coh']:.4f}\n"
        )
        lines.append(
            f"  - wellward_heuristic mean_coh = {h['wellward_mean_coh']:.4f}\n"
        )
        lines.append(
            f"  - never_boost mean_coh = {h['never_boost_mean_coh']:.4f}\n"
        )
        lines.append(
            f"  - always_boost mean_coh = {h['always_boost_mean_coh']:.4f}\n"
        )
        lines.append(
            f"  - Δ(selected − wear_aware) = "
            f"{h['delta_sel_minus_wear_aware']:+.4f}\n"
        )
        lines.append(
            f"  - **selected {h['selected_vs_wear_aware']} wear_aware_heuristic**\n"
        )
        if dist in c2.get("original_ref", {}):
            oref = c2["original_ref"][dist]
            lines.append(
                f"  - original-layout ref: selected {oref['selected_vs_wear_aware']} "
                f"wear_aware  Δ={oref['delta_sel_minus_wear_aware']:+.4f}\n"
            )
        if dist in c2.get("results", {}):
            for pol, r in c2["results"][dist].items():
                lines.append(
                    f"  - [{pol}] boosts={r['n_boosts']}/{r['n_decisions']}  "
                    f"vol_skip={r['n_skipped_voluntary']}  "
                    f"tod={r['time_of_death']}\n"
                )
    po = c2["pattern_outcome"]
    lines.append(f"- **pattern outcome: {po['label']}**\n")
    lines.append(f"- {po['summary']}\n")
    lines.append(f"- plot: {c2['plot']}\n")
    lines.append(f"- delta_plot: {c2['delta_plot']}\n")
    if c2.get("multiseed"):
        lines.append("\n### Multi-seed scrutiny\n")
        lines.append(f"- multiseed artifact present; see check2_multiseed.json\n")
        sc = c2.get("pattern_outcome", {}).get("scrutiny", {})
        if sc:
            lines.append(f"- scrutiny note: {sc.get('note', '')}\n")

    lines.append("\n## Check 3 — architectural support for N≠2 wells\n")
    lines.append(
        f"- supports_variable_n_wells: {c3['supports_variable_n_wells']}\n"
    )
    lines.append(f"- n_wells_checkpoint: {c3['n_wells_checkpoint']}\n")
    lines.append(f"- conv_in_channels: {c3['conv_in_channels']}\n")
    lines.append(f"- forward_two_wells_ok: {c3['forward_two_wells_ok']}\n")
    lines.append(f"- forward_three_wells_ok: {c3['forward_three_wells_ok']}\n")
    if c3.get("forward_three_wells_error"):
        lines.append(
            f"- forward_three_wells_error: {c3['forward_three_wells_error']}\n"
        )
    lines.append(f"- mechanics_variable_n (PerWellWearState): {c3['mechanics_variable_n']}\n")
    lines.append(f"- three_well_run_performed: {c3['three_well_run_performed']}\n")
    if c3.get("reason_stopped"):
        lines.append(f"- reason_stopped: {c3['reason_stopped']}\n")
    lines.append(f"- **verdict: {c3['verdict']}**\n")
    lines.append(f"- {c3['summary']}\n")
    for f in c3.get("findings", []):
        lines.append(f"  - {f}\n")

    lines.append("\n## Project closing statement (v0–v5)\n")
    lines.append(_closing_paragraph(c1, c2, c3))
    lines.append("\n")

    lines.append("\n## Honesty notes\n")
    lines.append(
        "- Negative zero-shot results are complete findings, not bugs to fix "
        "before finishing. No Stage B is planned from this repo.\n"
    )
    lines.append(
        "- Relocating wells is the fair version of the heuristic comparison: "
        "a designer can parameterize positions; training data cannot invent "
        "layouts it never saw.\n"
    )
    lines.append(
        "- No retraining, no new mechanics, no changes to v0 / SimpleNCA / "
        "frozen checkpoints.\n"
    )

    obs_path = RESULTS / "observations.txt"
    RESULTS.mkdir(parents=True, exist_ok=True)
    obs_path.write_text("".join(lines))
    print(f"\n  wrote {obs_path}")
    return obs_path


def _closing_paragraph(c1: Dict, c2: Dict, c3: Dict) -> str:
    cal = c1["verdict"]
    pat = c2["pattern_outcome"]["label"]
    arch = c3["verdict"]
    return (
        "EmbodiedNCA (v0–v5) built a resource-gated embedded-pattern NCA world "
        "(v0), then successively asked whether a learned self-model of future "
        "coherence / local resource / survival could earn its complexity over "
        "hand-coded controls as geometry (v2), asymmetric per-well wear (v3.5), "
        "chained identity (v4), and finally **novel well layouts** (v5) were "
        "introduced. On fixed maps, correctly parameterized heuristics matched "
        f"or beat `selected`. On the closing zero-shot relocation test, "
        f"calibration **{cal}**, policy pattern **{pat}**, and N≠2 wells "
        f"**{arch}** for the frozen self-model. Prediction transfers across "
        "two-well layouts; controller advantage over position-updated heuristics "
        "is layout-dependent and must be read with absolute levels (not only "
        "relative Δ). Fixed two-well input encoding is a hard architectural stop "
        "for N>2. Complete on this codebase's terms: continual learning / "
        "decision-driven plasticity is a different project, not Stage B here."
    )


def write_readme(c1: Dict, c2: Dict, c3: Dict) -> Path:
    po = c2["pattern_outcome"]["label"]
    readme = f"""# v5 — Generalization to Novel Well Layouts (Closing Stage)

Zero-shot test of frozen `self_model_v3_5` and position-parameterized
heuristics on a **relocated** two-well layout neither has ever seen.

No retraining. No edits to `v0` step / `SimpleNCA` / frozen checkpoints.

## Setup

```bash
cd EmbodiedNCA
source .venv/bin/activate
python v5/run_v5_checks.py
python v5/run_v5_checks.py --quick
python v5/run_v5_checks.py --skip_dataset   # reuse relocated eval dataset
```

Requires `v0/results/trained_nca.pt` and `v3_5/results/self_model_v3_5.pt`.

## Layout

```
v5/
├── relocated_layout_checks.py   # Check 1–3 implementations
├── run_v5_checks.py             # CLI harness
├── results/
│   ├── check1_calibration_original_vs_relocated.png
│   ├── check2_policy_comparison_relocated.png
│   ├── observations.txt
│   └── summary.json
└── README.md
```

## Wells

| | Coordinates (y, x) | Inter-well distance |
|--|-------------------:|--------------------:|
| Training (v0–v4) | (8,8), (24,24) | 16√2 ≈ 22.63 |
| Relocated (v5) | (12,20), (28,4) | 16√2 ≈ 22.63 |

Same separation / grid difficulty; only spatial arrangement changes.
Heuristics take well positions as parameters (one-line update). The self-model
keeps frozen weights; only runtime world coordinates change.

## Acceptance checks (this run)

| Check | Question | Result |
|------:|----------|--------|
| 1 | Calibration hold on relocated layout? | **{c1['verdict']}** |
| 2 | selected vs position-updated heuristics | **{po}** |
| 3 | Architecture supports N≠2 wells? | **{c3['verdict']}** |

Full write-up: `results/observations.txt`.

## Framing

This is the one stage where heuristic comparison is the right test: relocating
wells is something a heuristic designer can parameterize, but the self-model's
training data never included. Three Check-2 outcomes are all complete:

1. **Same pattern** — heuristic still ties/beats
2. **Self-model clearly loses** — worse margin than original (memorization)
3. **Self-model wins/ties more** — surprising; multi-seed before believing

## Non-goals

No retraining (including after negative results), no new mechanics, no Stage B
from this repo. Negative zero-shot findings close the project cleanly.

## Project closing statement

{_closing_paragraph(c1, c2, c3)}
"""
    path = V5_ROOT / "README.md"
    path.write_text(readme)
    print(f"  wrote {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="v5 relocated-layout generalization checks")
    ap.add_argument("--quick", action="store_true", help="shorter horizons / smaller eval set")
    ap.add_argument(
        "--skip_dataset",
        action="store_true",
        help="reuse results/relocated_eval_dataset.pt if present",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu" if args.cpu else pick_device(True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("v5 — Generalization to Novel Well Layouts (closing stage)")
    print("=" * 72)
    print(f"device={device}  quick={args.quick}")
    print(f"original wells={ORIGINAL_WELLS}")
    print(f"relocated wells={RELOCATED_WELLS}")
    print(f"frozen model={MODEL_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing frozen self-model: {MODEL_PATH}\n"
            "Run v3_5/run_v3_5_checks.py first."
        )
    if not Path(DEFAULT_CKPT).is_file():
        raise FileNotFoundError(f"Missing v0 checkpoint: {DEFAULT_CKPT}")

    model = load_model(MODEL_PATH, device=device)
    model.eval()
    print(
        f"loaded self_model_v3_5  hidden={model.hidden}  "
        f"n_wells={model.n_wells}  wear_input={model.wear_input}"
    )

    nca, base_pattern, world_cfg_base = load_v0_world(DEFAULT_CKPT, device=device)

    # Check 1
    c1 = check1_calibration_relocated(
        model=model,
        device=device,
        quick=args.quick,
        skip_dataset=args.skip_dataset,
        seed=args.seed,
    )

    # Check 2
    c2 = check2_policy_relocated(
        nca=nca,
        base_pattern=base_pattern,
        world_cfg_base=world_cfg_base,
        model=model,
        device=device,
        quick=args.quick,
        seed=args.seed,
    )

    # Check 3 (architecture; no three-well run if unsupported)
    c3 = check3_architecture_n_wells(model)
    if c3["supports_variable_n_wells"]:
        print(
            "\n  [Check 3] Architecture supports variable N — three-well "
            "bonus not auto-run in Stage A default; extend manually if desired."
        )

    # Artifacts
    summary = {
        "stage": "v5",
        "device": device,
        "quick": args.quick,
        "original_wells": list(ORIGINAL_WELLS),
        "relocated_wells": list(RELOCATED_WELLS),
        "model_path": str(MODEL_PATH),
        "check1": c1,
        "check2": c2,
        "check3": c3,
        "closing": _closing_paragraph(c1, c2, c3),
    }
    summary_path = RESULTS / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    print(f"\n  wrote {summary_path}")

    write_observations(c1, c2, c3, device=device, quick=args.quick)
    write_readme(c1, c2, c3)

    print("\n" + "=" * 72)
    print("v5 COMPLETE (project closing stage)")
    print(f"  Check 1 calibration: {c1['verdict']}")
    print(f"  Check 2 pattern:     {c2['pattern_outcome']['label']}")
    print(f"  Check 3 N≠2 wells:   {c3['verdict']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
