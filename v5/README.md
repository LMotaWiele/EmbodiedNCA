# v5 — Generalization to Novel Well Layouts (Closing Stage)

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
│   ├── check1_calibration_metrics_bar.png
│   ├── check2_policy_comparison_relocated.png
│   ├── check2_delta_original_vs_relocated.png
│   ├── check2_multiseed.json
│   ├── relocated_eval_dataset.pt
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
| 1 | Calibration hold on relocated layout? | **held_comparable** (within-bin r 0.845 → 0.851) |
| 2 | selected vs position-updated heuristics | **self_model_wins_or_ties_more** (Δ +0.027 / +0.051; multi-seed 5/5) |
| 3 | Architecture supports N≠2 wells? | **not_supported** (fixed 2 wear channels) |

### Check 2 caveat (required for positive-looking outcome)

Multi-seed (5 seeds) is deterministic for these policies and confirms the win.
Absolute levels: **selected holds** near original-layout mean_coh; **wear_aware
degrades** because fresher-well and nearest-well angles nearly coincide under
6-sector quantization (wear_aware ≡ wellward at dist=2). The relative win is
real but partly heuristic collapse, not a large new transfer bonus.

Full write-up: `results/observations.txt`.

## Framing

This is the one stage where heuristic comparison is the right test: relocating
wells is something a heuristic designer can parameterize, but the self-model's
training data never included. Three Check-2 outcomes are all complete:

1. **Same pattern** — heuristic still ties/beats
2. **Self-model clearly loses** — worse margin than original (memorization)
3. **Self-model wins/ties more** — surprising; multi-seed before believing ← **this run**

## Non-goals

No retraining (including after negative results), no new mechanics, no Stage B
from this repo. Results close the project cleanly either way.

## Project closing statement

EmbodiedNCA (v0–v5) built a resource-gated embedded-pattern NCA world (v0), then
successively asked whether a learned self-model of future coherence / local
resource / survival could earn its complexity over hand-coded controls as
geometry (v2), asymmetric per-well wear (v3.5), chained identity (v4), and
finally **novel well layouts** (v5) were introduced. On fixed maps, correctly
parameterized heuristics matched or beat `selected`. On the closing zero-shot
relocation test, calibration **held** (no coordinate-memorization collapse for
prediction), `selected` **beat** position-updated heuristics with multi-seed
deterministic confirmation — largely because absolute model performance held
while the wear heuristic lost sector resolution when fresher- and nearest-well
directions nearly coincided — and variable well-count (N≠2) is architecturally
**unsupported** for the frozen self-model. The project's complete answer: this
stack produces a useful predictive self-model that transfers across two-well
layouts, but does not establish a general consciousness-adjacent controller
advantage; hand-coded geometry/wear rules remain competitive whenever their
parameters can be set, and fixed two-well input encoding is a hard stop for N>2.
Further work (continual learning / decision-driven plasticity) is a different
codebase and question, not Stage B of this one.
