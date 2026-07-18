# v0 Resource-Gated Embedded-Pattern NCA

Standalone experiment: a single self-organizing pattern living inside an NCA
grid, with **no external controller**. Continued existence depends on a local
resource field that decays and is replenished only near fixed wells. The
pattern's own self-maintenance update is **multiplied by local resource**, so
scarcity causes physically modeled degradation rather than a scripted death.

This is intentionally **not** the TransNCA transformer/entity-tracking pipeline.
Only `SimpleNCA` (perception conv + update_net + residual clamp) is reused.

## Layout

```text
embodied_nca/
├── nca_module.py          # copied from TransNCA/shared/nca_module.py
├── grid_world.py          # ResourceGrid + gated dynamics + training
├── metrics.py             # self_mass, coherence, resource metrics
├── run_sanity_checks.py   # three acceptance-criteria experiments
├── results/               # plots, CSVs, trained weights, observations
└── README.md
```

## Setup

Uses the same stack as TransNCA (`torch`, `matplotlib`, `numpy`):

```bash
# reuse TransNCA venv, or create a new one
source ../TransNCA/.venv/bin/activate   # adjust path as needed
cd embodied_nca
python run_sanity_checks.py
```

Options:

```bash
python run_sanity_checks.py --horizon 2000          # default
python run_sanity_checks.py --cpu                   # force CPU
python run_sanity_checks.py --skip_train            # reuse results/trained_nca.pt
```

## Core mechanic

Each timestep (batch dim = 1, grid 32×32, C=32):

1. **Diffuse S** — mild ambient mixing (world physics).
2. **Gated NCA step** — wrap `SimpleNCA` without modifying it:
   ```text
   update = update_net(perception(S))
   S ← clamp(S + (R / R_max) · update, -10, 10)
   ```
3. **Consume** — `R ← max(0, R − consumption_rate · ‖update‖₂ per cell)`.
4. **Resource physics** — decay everywhere, generate at soft wells, diffuse R.

Gate modes for ablation: `gated` (default), `ungated` (`S += update`), `zero`
(no NCA update at all).

### Why ambient diffusion / soft wells / R diffusion?

- `SimpleNCA` is a pure residual. With `gate=0`, S would **freeze** forever.
  Ambient S diffusion is the unopposed erosion the gate is meant to fight
  (spec: “ordinary grid diffusion”).
- A one-pixel generation delta cannot sustain a multi-cell pattern under
  per-cell gating. Wells have a narrow Gaussian footprint (`well_sigma=2.5`)
  and R is mildly diffused so distance-to-well is a real gradient.
- These are **world physics**, not a scripted `if R≤0: reset()`. There is no
  reset path.

### Self-maintenance weights

`SimpleNCA` zero-inits its last layer, so a random network never updates.
Before the checks, `train_self_maintenance` teaches the local rule to hold one
base pattern against diffusion, with:

- random translations (position-equivariant),
- smooth spatial partial gates in ~[0.5, 1],
- curriculum unrolls up to 128 steps.

At runtime there is still **no controller** reading/writing the grid.

## Metrics (logged every step → CSV)

| Metric | Meaning |
|--------|---------|
| `self_mass` | Σ of per-cell ‖S‖ over cells above threshold |
| `coherence` | Cosine similarity to S₀ on active support |
| `total_resource` | Σ R |
| `local_resource_at_pattern_centroid` | R at mass-weighted centroid |
| `distance_to_nearest_well` | Euclidean distance, centroid → nearest well |

## Acceptance checks (results from seed=0, horizon=2000, CUDA)

Plots: `results/check1_breakeven.png`, `check2_distance.png`, `check3_gating.png`.

### Check 1 — break-even near a well

Pattern centered on well `(8,8)`. Sweep of `(gen_rate, consumption_rate)`:

| Setting | mean local R | coh final | Notes |
|---------|-------------:|----------:|-------|
| Spec default gen=0.05, cons=0.01 | ~1.00 | 0.96 | Well too generous once field is smooth |
| **Selected gen=0.20, cons=1.0** | **0.706 ± 0.000** | **0.95** | Narrow band, pattern alive |
| gen=0.20, cons=0.2–0.5 | 1.00 | ≥0.97 | Still free-climbing to R_max |

**Observation:** Spec-literal rates do **not** break even under a multi-cell
smooth resource field — local R sits on the rail at 1.0. Raising
`consumption_rate` to ~1.0 (with `gen_rate=0.20`, wide wells) yields a stable
~0.71 local R while the pattern remains coherent. Full sweep:
`results/check1_sweep.json`.

### Check 2 — distance-dependence

Same rates, pattern at distances **2 / 4 / 8** cells from well `(8,8)` along +x
(spec's 20 would sit near the second well at `(24,24)`; 8 is already past the
survival cliff):

| Distance | Final coherence | Mean coherence | Self-mass |
|---------:|----------------:|---------------:|----------:|
| 2 | 0.73 | 0.73 | 97 |
| 4 | 0.52 | 0.54 | 57 |
| 8 | ~0 | 0.10 | 0 |

**Observation:** Clear monotone decay with distance. Near-well patterns hold;
mid-range degrade; far patterns die. Effect is somewhat threshold-like past
~6 cells (resource falls below the maintenance floor), not a perfectly linear
gradient — but farther is strictly worse on both final and mean coherence.

### Check 3 — gating ablation

Same initial pattern at distance 2 (mid-scarcity so gated ≠ zero and ≠ ungated):

| Mode | Final coherence | Mean coherence |
|------|----------------:|---------------:|
| ungated | 0.97 | 0.97 |
| gated | 0.73 | 0.73 |
| zero | 0.08 | 0.16 |

**Observation:** Ordering `ungated ≥ gated ≥ zero` holds cleanly. Gated is a
genuine intermediate: resource scarcity weakens self-maintenance without
instant collapse. Zero-gate shows gradual diffusion-driven death (mass → ~1,
coherence → ~0.08). No scripted reset.

## Operating defaults (post-sweep)

```text
H=W=32, C=32, hidden=64
wells = (8,8), (24,24)
gen_rate=0.20, decay_rate=0.02, consumption_rate=1.0
well_sigma=2.5, resource_diffusion=0.40
diffusion_rate (S)=0.04
pattern radius=2.5
```

## Explicit non-goals (not in v0)

- No transformer / controller / action selection
- No confidence / volatility / identity_weight channels
- No multi-agent interaction
- No learned well placement

Do not start self-model / identity-weight work until these three checks are
reviewed.
