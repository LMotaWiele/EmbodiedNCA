# EmbodiedNCA

Resource-gated embedded-pattern Neural Cellular Automata experiments.

All stages live in this single repo for version control:

```text
EmbodiedNCA/
├── v0/                 # resource-gated self-maintenance (no controller)
├── v1/                 # read-only self-model + identity_weight
├── v2/                 # action selection (attention budget + self-model)
├── v3/                 # global wear-coupled environmental consequence
├── v3_5/               # per-well wear + death-aware self-model + skip
├── v4/                 # chained challenges (persistent S/R/wear/A)
├── v5/                 # novel well layouts (zero-shot closing stage)
├── requirements.txt
├── .venv/              # shared environment
└── README.md           # this file
```

## Setup

```bash
cd EmbodiedNCA
# recommended:
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

## Run

```bash
# v0 — train self-maintenance + three acceptance checks
python v0/run_sanity_checks.py
python v0/run_sanity_checks.py --skip_train   # reuse v0/results/trained_nca.pt

# v1 — requires v0/results/trained_nca.pt
python v1/run_v1_checks.py
python v1/run_v1_checks.py --skip_dataset --skip_train

# v2 — requires v0 checkpoint; trains/reuses self_model_v2
python v2/run_v2_checks.py --skip_dataset --skip_train

# v3 — wear + direct boost cost; uses existing v2 self-model (no retrain)
python v3/run_v3_checks.py
python v3/run_v3_checks.py --quick

# v3.5 — per-well wear, death-aware self-model, skip candidate
python v3_5/run_v3_5_checks.py
python v3_5/run_v3_5_checks.py --quick

# v4 — chained vs episodic persistence (Stage A, no retrain)
python v4/run_v4_checks.py
python v4/run_v4_checks.py --quick

# v5 — novel well layouts (closing stage, no retrain)
python v5/run_v5_checks.py
python v5/run_v5_checks.py --quick
python v5/run_v5_checks.py --skip_dataset
```

You can also `cd v0` / `cd v1` / … and run the scripts from those directories;
paths are resolved from each file’s location, not from the working directory.

## Stage summary

| Stage | What | Controller? | Key deliverable |
|------:|------|:-----------:|-----------------|
| **v0** | Resource-gated pattern dynamics | no | break-even, distance, gating checks |
| **v1** | Self-model of future coherence / local R under local probes; `identity_weight` map | no (read-only) | calibration r, structured weight maps |
| **v2** | Scarce attention + sector selection via self-model | attention budget | selected ≡ wellward/fixed on static map |
| **v3** | Global wear degrades `gen_rate` + direct R boost cost | same + wear wrapper | frugality matters; Stage A (no retrain) |
| **v3.5** | Per-well wear + death-aware self-model + skip candidate | 7-way argmax | selected vs wear_aware_heuristic |
| **v4** | Chained challenges; no reset of S/R/wear/A between segments | frozen v3.5 policies | chained vs episodic; order sensitivity |
| **v5** | Relocated wells; zero-shot frozen self-model (project close) | frozen v3.5 policies | calibration holds; selected beats updated heuristics; N≠2 unsupported |

Details and measured numbers: `v0/README.md` … `v5/README.md`. **v5 closes this project.**
