# EmbodiedNCA

A closed experiment series: one self-organizing pattern inside a Neural Cellular Automaton, whose continued existence is **resource-gated**. Local resource `R` decays everywhere and is replenished only near fixed wells. The pattern’s own self-maintenance residual is multiplied by local `R`, so scarcity is physically modeled degradation, not a scripted death.

There is no external controller in v0. Later stages add a read-only self-model, a scarce attention budget, irreversible wear, chained persistence, and finally a zero-shot test on wells the model never saw.

**v5 closes this project.** Continual learning / decision-driven plasticity would be a different codebase, not a v6 here.

Package: `src/embodiednca/` · layout: `AGENTS.md` · numbers: `docs/design/`.

## Why stages

Each stage adds one mechanic and asks a falsifiable question before the next is worth building:

1. **Self-maintenance** — can a pattern live on a well and die away from one, with gating (not just extra compute) doing the work?
2. **Self-model** — can a small predictor forecast future coherence / local `R` under local probes, and where is `identity_weight` trustworthy?
3. **Action** — given a scarce attention budget, does argmax over that predictor beat dumb allocation on a static map?
4. **Wear** — when boosting also degrades the wells, does frugality matter?
5. **Per-well wear + skip** — when *where* you boost has a wear consequence, does the learned chooser beat a wear-aware heuristic?
6. **Chained identity** — does inherited `S`/`R`/wear/`A` change outcomes versus episodic reset?
7. **Novel layouts** — do frozen weights transfer when the wells move?

## What each stage showed

| Stage | Mechanic | Finding |
|------:|----------|---------|
| **v0** | Resource-gated residual NCA, no controller | Break-even on a well; survival falls with distance; gating (not extra compute) is the load-bearing difference. |
| **v1** | Read-only self-model + `identity_weight` | Calibration is strong; the map has a **false-signal** equidistant locus — never raw full-grid argmax. |
| **v2** | Scarce attention, 6-sector argmax over a boost-trained self-model | On a static two-well map, `selected` matches wellward / best-fixed. Geometry is enough. |
| **v3** | Global wear (degrades `gen_rate`) + direct `R` boost cost | Frugality matters; Stage A, no retrain. |
| **v3.5** | Per-well wear, death-aware self-model, skip in the argmax | `selected` ties or loses to `wear_aware_heuristic`. A non-victory is a valid outcome. |
| **v4** | Chained segments: no reset of `S`/`R`/wear/`A` | Inherited state changes outcomes; calibration thins on later segments; residual path dependence. |
| **v5** | Relocated wells, frozen `self_model_v3_5` (project close) | Calibration holds. `selected` beats position-updated heuristics, partly because `wear_aware` collapses under 6-sector geometry. **N≠2 wells is unsupported** (fixed two wear channels). |

Measured tables, plots, and caveats: [`docs/design/v0.md`](docs/design/v0.md) … [`v5.md`](docs/design/v5.md). Spec for the v0 mechanic: [`docs/design/v0_embodied_nca_spec.md`](docs/design/v0_embodied_nca_spec.md).

## Setup

Python ≥ 3.12.

```bash
source .venv/bin/activate
pip install -e '.[dev]'
```

Checkpoints already live under `results/<stage>/` (e.g. `results/v0/trained_nca.pt`). Reuse them; do not retrain from scratch unless you mean to.

## Run (from repo root)

```bash
# v0 — train self-maintenance + three acceptance checks
python scripts/run_sanity_checks.py
python scripts/run_sanity_checks.py --skip_train   # reuse results/v0/trained_nca.pt

# v1 — requires results/v0/trained_nca.pt
python scripts/run_v1_checks.py
python scripts/run_v1_checks.py --skip_dataset --skip_train

# v2 — requires v0 checkpoint; trains/reuses self_model_v2
python scripts/run_v2_checks.py --skip_dataset --skip_train
python scripts/run_v2_checks.py --quick

# v3 — wear + direct boost cost; frozen v2 self-model (no retrain)
python scripts/run_v3_checks.py
python scripts/run_v3_checks.py --quick

# v3.5 — per-well wear, death-aware self-model, skip candidate
python scripts/run_v3_5_checks.py
python scripts/run_v3_5_checks.py --quick
python scripts/run_v3_5_checks.py --skip_dataset --skip_train
python scripts/run_v3_5_diagnostics.py --quick

# v4 — chained vs episodic persistence (no retrain)
python scripts/run_v4_checks.py
python scripts/run_v4_checks.py --quick

# v5 — novel well layouts (closing stage, no retrain)
python scripts/run_v5_checks.py
python scripts/run_v5_checks.py --quick
python scripts/run_v5_checks.py --skip_dataset
```

`--quick` shortens horizons / dataset size (smoke). `--skip_train` / `--skip_dataset` load existing artifacts under `results/<stage>/`.

```bash
pytest
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
