# v2 — Action Selection (Counterfactual Self-Maintenance Allocation)

A scarce, independent **attention** resource that the pattern allocates each
decision interval to exactly one candidate region of itself — using an
*extended* self-model (retrained on damage **and** boost interventions) to
choose the candidate predicted to best protect future coherence.

Does **not** modify `v0` step logic, `SimpleNCA`, resource physics, or
`v1` architecture / on-disk training data.

## Setup

```bash
cd EmbodiedNCA
source .venv/bin/activate
python v2/run_v2_checks.py
# reuse artifacts:
python v2/run_v2_checks.py --skip_dataset --skip_train
# smoke test:
python v2/run_v2_checks.py --quick
```

Requires `v0/results/trained_nca.pt`.

## Layout

```
v2/
├── boost_rollout_dataset.py   # damage + boost perturbation rollouts
├── self_model_v2.py           # same SelfModel arch, retrained on combined data
├── action_selection.py        # candidates, validity mask, attention budget
├── run_v2_checks.py           # Check 1 / 2 / 3
├── results/
└── README.md
```

## Step 1 — extended self-model (boost)

v1's self-model saw only damage-type probes (null / noise / zero-out). Boosting
a local gate is a new intervention distribution. `boost_rollout_dataset.py`
adds:

```text
gate_effective = min(1.0, gate_from_R + boost_magnitude)   # patch, duration ~20
```

then lets the boost lapse and records the same future targets as v1. One model
is trained on the **combined** damage + boost set (not a separate model per
type).

**Encoding:** damage keeps v1's non-negative Gaussian bump; boost uses a
*negative* bump (same architecture / three heads, no extra channel). Unsigned
encoding makes boost and damage input-identical — the model cannot learn
distinct outcome maps.

## Step 2 — Check 1 (mandatory gate)

Calibration is reported on the **boost subset only** (not pooled with damage):

- Pearson r / MAE for `coherence_future` and `local_R_future`
- Within distance-bin breakout (same bins as v1)

**Pass bar:** boost within-bin mean r ≥ 0.5 (hard fail below); target ~0.7+
comparable to v1 damage within-bin r. If hard-fail, action selection is **not**
built — that is the v2 finding.

## Step 3 — action selection

Every `K=20` steps:

1. **Candidates** — 6 angular sectors of active-mass support around the
   mass-weighted centroid (fixed rule, no learned proposal).
2. **Mask before scoring** — v1 `validity_mask(S, wells=…)` (support AND not
   equidistant boundary). Drops are logged.
3. **Selection** — non-learned argmax over `self_model_v2` predicted
   `coherence_future` under a boost-at-candidate encoding.
4. **Attention budget** `A` (independent of `R`):

```text
A ← min(A_max, A + regen_rate)     # each step
if A ≥ boost_cost: apply_boost(best); A ← A − boost_cost
```

`regen_rate` / `boost_cost` are swept so A cannot cover every interval —
scarcity is the point.

Baselines for Check 2 under the same budget:

- **random** / **round-robin** — switching policies (force high switch_rate)
- **fixed** — lock a sector forever (no model). Default = lowest-id at t=0;
  **6-way sweep** locks each of sectors 0..5
- **wellward_heuristic** — boost candidate closest in angle to nearest well
  (pure geometry, no model; final zero-learning control)

## Explicit non-goals

- No RL / policy gradients / learned selector
- No confidence/volatility channels
- No multi-agent interaction
- No changes to wells, decay/gen rates, `SimpleNCA`, or base R gating
- No learned candidate proposal

## Acceptance criteria

| Check | What | Gate |
|------:|------|------|
| 1 | Boost-subset within-bin calibration | Hard stop if r < 0.5 |
| 2 | `selected` vs `random` vs `round_robin` (coh / mass, dist 2/4/8) | Report honestly if selected does not win |
| 3 | No selected action on a masked candidate | Bug if any |

## Final verdict (v2 is done)

**Does self-model-driven action selection earn its complexity under this
design, on this geometry? No.**

| Check | Result |
|------:|--------|
| 1 boost within-bin calibration | weak pass (0.54; strong near-well) |
| 2 mean coherence | selected does **not** beat any baseline |
| 2 mechanism | sticky argmax → low trajectory variance (real, logged) |
| 2b 6-way fixed sweep | selected ≡ best fixed sector (no model) |
| 2c wellward_heuristic | ≡ selected at dist=4; differs at dist=2 (near-well) |
| 3 mask compliance | pass |

### Controls, in order

1. **Mean:** selected ≉ better than random / RR / fixed / wellward.
2. **Sticky mechanism:** mode_frac≈0.96–0.98, switch_rate≈0.02 vs random 0.85.
3. **6-way fixed:** selected locks structural best (d2→s4, d4→s2); fixed lock
   of that sector reproduces variance. Original fixed(s0) was worst-in-sweep.
4. **wellward_heuristic** (3-line geometry: boost sector closest in angle to
   nearest well — no model):

| Dist | selected mode / std | wellward mode / std | Match? |
|-----:|--------------------:|--------------------:|:------:|
| 4 | 2 / 0.00095 | 2 / 0.00084 | **yes** (clean operating point) |
| 2 | **4** / 0.00031 | **2** / 0.00078 | no — centroid≈on well; wellward ill-defined; selected still ≡ fixed(s4) |
| 8 | 0 / 0.036 | 2 / 0.027 | dying pattern; not diagnostic |

At the clean distance, a hand-coded rule matches the full pipeline. At
near-well, pure wellward is slightly worse than selected, but selected is
still matchable by zero-learning fixed(best). Neither distance shows the
self-model exceeding a zero-learning ceiling.

### Naming discipline

| Phrase | OK? |
|--------|:---:|
| sticky argmax → smoother than random | yes |
| selected ≡ best fixed / ≡ wellward at dist=4 | yes |
| "self-model earns its complexity" | **no** |
| "confirms homeostasis" | **no** (v0 did; this is allocation-stability on top) |
| "reduce surprise" | **no** (objective is value-argmax) |

Full write-up: `results/observations.txt`.

## After this — v3 scope (if any)

**Do not** extend this design on one map (more candidates, longer horizons,
smarter selector). Ceiling here is match-a-hand-coded-rule.

Interesting v3 requires situations a 3-line rule does not trivially solve:

- multiple wells at competing distances
- wells that turn on/off
- nearer ≠ better (depleted close well vs rich far well)

That means **retraining on varied geometries**, not more machinery on
`(8,8)/(24,24)`.
