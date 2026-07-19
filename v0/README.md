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
EmbodiedNCA/
├── v0/                    # this package
│   ├── nca_module.py
│   ├── grid_world.py
│   ├── metrics.py
│   ├── run_sanity_checks.py
│   ├── results/           # latest plots, CSVs, trained weights, observations
│   │   └── historic/      # prior result snapshots
│   └── README.md
├── v1/                    # self-model + identity_weight (see v1/README.md)
├── requirements.txt
└── README.md              # repo overview
```

## Setup

Uses the repo-root venv (`torch`, `matplotlib`, `numpy`). Python 3.12 recommended:

```bash
cd EmbodiedNCA
source .venv/bin/activate   # or: uv venv .venv --python 3.12 && uv pip install -r requirements.txt

python v0/run_sanity_checks.py
```

```bash
python v0/run_sanity_checks.py --horizon 2000          # default
python v0/run_sanity_checks.py --cpu                   # force CPU
python v0/run_sanity_checks.py --skip_train            # reuse v0/results/trained_nca.pt
```

## Core mechanic

Each timestep (batch dim = 1, grid 32×32, C=32):

1. **Diffuse S** — mild ambient mixing (world physics).
2. **Gated NCA step** — wrap `SimpleNCA` without modifying it:
   ```text
   update = update_net(perception(S))          # always computed
   S ← clamp(S + gate · update, -10, 10)       # gate depends on mode
   ```
3. **Consume** — `R ← max(0, R − consumption_rate · ‖update‖₂ per cell)`.
4. **Resource physics** — decay everywhere, generate at soft wells, diffuse R.

Gate modes for ablation:

| Mode | Applied residual | Update computed? | Resource charged? |
|------|------------------|------------------|-------------------|
| `gated` | `(R/R_max) * update` | yes | yes, on raw ‖update‖ |
| `ungated` | `update` | yes | yes, on raw ‖update‖ |
| `zero` | `0` (S unchanged by NCA) | **yes** | **yes**, on raw ‖update‖ |

### Consumption policy (deliberate)

```text
consumption = consumption_rate * ‖update‖_pre-gate
```

**Not** `gate * ‖update‖`. A starved pattern that *attempts* self-maintenance
and fails still pays for the attempt. This matches the v0 spec formula and is
required if later stages (self-model / `identity_weight`) will reason about
resource trajectories under failed repair. `gate_mode="zero"` is the extreme
test of this: update_net still runs, R is still billed, S only changes via
ambient diffusion.

### Post-death metrics (deliberate)

Two related constants in `metrics.py` (do not reverse-engineer from CSVs):

| Constant | Value | Role |
|----------|------:|------|
| `DEFAULT_ACTIVATION_THRESHOLD` | **0.05** | Only cells with ‖S‖ > 0.05 contribute to `self_mass` / support |
| `DEATH_MASS_EPS` | **1e-6** | `alive=0` and NaN pattern metrics once `self_mass ≤ 1e-6` |

The sharp mass cliff near death (e.g. 0.0501 → 0.0 on the next step in
`check2_dist8.csv`) is the **activation threshold**: when the last above-
threshold cell falls below 0.05, mass jumps to zero. `DEATH_MASS_EPS` is
the near-zero floor used after that, not 0.05.

Once dead, position- and identity-dependent metrics are **undefined** and
written as `nan`:

- `coherence`
- `local_resource_at_pattern_centroid`
- `distance_to_nearest_well`

They are **not** replaced by a grid-center fallback (which previously made
local R climb after death and produced negative residual cosines). World-level
fields (`total_resource`, `mean_update_magnitude`, `total_consumption`) stay
defined. `time_of_death` is the first such timestep.

### Why ambient diffusion / soft wells / R diffusion?

- `SimpleNCA` is a pure residual. With `gate=0`, S would **freeze** forever.
  Ambient S diffusion is the unopposed erosion the gate is meant to fight.
- A one-pixel generation delta cannot sustain a multi-cell pattern under
  per-cell gating. Wells have a narrow Gaussian footprint (`well_sigma=2.5`)
  and R is mildly diffused so distance-to-well is a real gradient.
- These are **world physics**, not a scripted `if R≤0: reset()`.

### Self-maintenance weights

`SimpleNCA` zero-inits its last layer, so a random network never updates.
Before the checks, `train_self_maintenance` teaches the local rule to hold one
base pattern against diffusion (translations + spatial partial gates +
curriculum). At runtime there is still **no controller** reading/writing the
grid.

## Metrics (logged every step → CSV)

| Metric | Meaning | After death |
|--------|---------|-------------|
| `self_mass` | Σ of per-cell ‖S‖ over cells with ‖S‖ > **0.05** | 0 |
| `alive` | 1 if `self_mass` > **1e-6** else 0 | 0 |
| `coherence` | Cosine similarity to S₀ on active support | **nan** |
| `total_resource` | Σ R | still defined |
| `local_resource_at_pattern_centroid` | R at mass-weighted centroid | **nan** |
| `distance_to_nearest_well` | centroid → nearest well | **nan** |
| `mean_update_magnitude` | mean ‖update‖ over grid | still defined |
| `total_consumption` | Σ (consumption_rate · ‖update‖) this step | still defined |

## Acceptance checks (seed=0, horizon=2000)

Latest full run (local `.venv`, CUDA). Artifacts under `results/`; narrative
summary auto-written to `results/observations.txt`.

Plots: `check1_breakeven.png`, `check2_distance.png`,
`check3_gating_d2.png`, `check3_gating_d4.png`. Prior (pre-metric-fix)
outputs are under `results/historic/v0_initial_pre_metric_fix/`.

### Check 1 — break-even near a well

Pattern centered on well `(8,8)`. Sweep of `(gen_rate, consumption_rate)`:

| Setting | mean local R | last-alive coh | Notes |
|---------|-------------:|---------------:|-------|
| Spec default gen=0.05, cons=0.01 | ~0.999 | ~0.97 | Well too generous once field is smooth |
| gen=0.20, cons=0.2–0.5 | 1.00 | ≥0.97 | Still free-climbing to R_max |
| **Selected gen=0.20, cons=1.0** | **~0.670 ± 0.000** | **~0.951** | Narrow band, pattern alive (mass ~90) |
| gen=0.20, cons=3.0 | (dies) | — | tod ≈ 285 |

**Dynamic equilibrium, not a freeze.** At the selected operating point,
`mean_update_magnitude` stays nonzero and essentially flat through the
horizon (ss mean **0.00684**, final **0.00675** — e.g. t=1000 → t=1999 is
0.00686 → 0.00675). Local R ≈ 0.67 is therefore a continuous erosion↔repair
balance, not a static pattern with updates shut off. This is the strongest
single number in the check-1 result; the plot’s third panel shows it
explicitly.

**Why ss local R moved 0.706 → 0.670 (and coh 0.947 → 0.951) vs the
historic snapshot.** Operating rates are unchanged (`gen_rate=0.20`,
`consumption_rate=1.0`). The shift is from the full post-fix re-run:
self-maintenance weights are **retrained** each full pipeline run (CUDA
training is not bit-stable across re-runs even at `seed=0`), and summary
stats now use alive-only / NaN-safe burn-in windows. The pre-gate
consumption formula itself was already what the code charged; this is not
a silent change of the intended operating point. **Latest numbers in this
README / `results/` are the reference** — see
`results/historic/v0_initial_pre_metric_fix/` for the older snapshot.

Plot uses **dual y-axes**: local R @ centroid on the left (scale [0,1]) and
mean R per cell on the right — so the break-even band is actually visible
(earlier versions plotted ΣR ~200–300 on the same axis and flattened local R
to a line at zero).

### Check 2 — distance-dependence

Distances **2 / 4 / 8** from well `(8,8)` along +x (centers `(8,10)`,
`(8,12)`, `(8,16)`):

| Distance | Last-alive coh | Mean alive coh | Final mass | Time of death |
|---------:|---------------:|---------------:|-----------:|---------------|
| 2 | 0.763 | 0.767 | ~99 | survived |
| 4 | 0.584 | 0.593 | ~56 | survived |
| 8 | 0.026 | 0.194 | 0 | **923** |

Farther ⇒ lower mean coherence, and farther ⇒ earlier death (only dist 8
dies in-horizon). Monotonic distance cost holds.

Post-death rows no longer invent a residual cosine or a climbing local R from
a phantom centroid — those fields are `nan` after `time_of_death` (verified
on `check2_dist8.csv`: `alive` flips 1→0 at t=923 with mass 0.0501→0.0;
coherence / local R / distance are `nan` from that row onward — zero
contamination for the rest of the horizon).

`check2_dist2.csv` and `check3_d2_gated.csv` are **bit-identical** (same
placement, seed, gated mode): check 3 reuses that trajectory as its gated
arm rather than re-running an independent trial. Fine for ablation layout;
not extra independent evidence of the distance effect.

### Check 3 — gating ablation (distances 2 and 4)

| Dist | Mode | Last-alive coh | Mean alive coh | Σconsumption | Final mass |
|-----:|------|---------------:|---------------:|-------------:|-----------:|
| 2 | ungated | 0.967 | 0.978 | ~18134 | ~216 |
| 2 | gated | 0.763 | 0.767 | ~13211 | ~99 |
| 2 | zero | 0.085 | 0.161 | ~15795 | ~0.97 |
| 4 | ungated | 0.967 | 0.978 | ~18151 | ~216 |
| 4 | gated | 0.584 | 0.593 | ~8311 | ~56 |
| 4 | zero | 0.085 | 0.161 | ~16059 | ~1.19 |

Ordering **ungated ≥ gated ≥ zero** holds at both distances on mean-alive
coherence. Zero-gate never fully hits `DEATH_MASS_EPS` within 2000 steps, but
mass collapses by ~100× and identity is gone — the intermediate gated case is
real degradation, not a binary alive/dead flip.

**Why zero-gate Σconsumption can exceed gated (not a bug).** Pre-gate billing
means both arms pay for attempted updates; the totals differ by *what state
update_net keeps seeing*:

- **Gated** settles by ~300 steps into a small stable pattern (mass ≈99).
  Per-step consumption then stays low (~6.1–6.8) — a cheap homeostasis.
- **Zero-gate** never stabilizes: ambient diffusion keeps eroding `S`, so
  update_net keeps seeing a never-settling input and produces moderately
  larger updates for nearly the full horizon (per-step consumption ~7.1–8.1
  even late). Integrated cost is higher even though nothing is applied to `S`.

So “cheap-but-stable beats moderately-expensive-but-never-stabilizing over
time.” Gating does not only protect identity; it also lets the system find a
**cheaper steady state** to defend. That still rules out the “failure is free”
fork: zero-gate is billed, just for fruitless attempts on a dissolving field.

**Ungated mass oscillation is expected.** With no resource gate, nothing
bounds growth/shrink of the residual rule: `self_mass` wanders in a persistent
band (~187–273 over the run; ~197–257 after t=500) rather than locking to a
point. That is unconstrained self-maintenance dynamics, not a residual
`update_net` instability to “fix” — the interesting contrast is that
gating *does* pin mass near a low-cost band.

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
reviewed with the accounting and metric policies above.
