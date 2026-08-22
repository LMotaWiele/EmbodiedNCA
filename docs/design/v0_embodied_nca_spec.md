# Implementation spec: v0 Resource-Gated Embedded-Pattern NCA

## Context for the agent

You have access to the existing TransNCA repo (`nca_module.py`, `latentca_model.py`,
`data_generator.py`, `experiment_utils.py`, `train_eval.py`, `ablation_study.py`,
`perturbation_test.py`). That repo implements a *token-prediction* system: a
transformer controller reads/writes a persistent NCA grid and a memory bank to
answer entity-tracking queries. **This is a different, standalone experiment.**
Do not modify the existing transformer/entity-tracking pipeline. Reuse only the
low-level building block: `SimpleNCA` from `nca_module.py` (the `perception` conv
+ `update_net` conv + residual-clamp update rule). Everything else here is new.

## What we're building, in one sentence

A single self-organizing pattern living inside an NCA grid, with no external
controller reading or writing it, whose continued existence depends on locally
available "resource" that decays over time and is only replenished near a small
number of fixed wells — and whose own self-maintenance capacity is mechanically
gated by that resource, so that resource scarcity causes real, physically-modeled
degradation rather than a scripted death event.

## Explicit non-goals for v0 (do not build these yet)

- No transformer, no controller, no separate "agent" network reading/writing the
  grid from outside. The pattern's *own* local update rule is the only dynamics.
- No action selection / forward-model planning / rollout of candidate actions.
- No `confidence`, `volatility`, or `identity_weight` channels or mechanisms.
- No multi-agent interaction.
- No dynamic/learned well placement — wells are fixed, hardcoded coordinates.

These are all real next steps, but v0 exists to answer three narrow, falsifiable
questions (see "Acceptance criteria" below) before any of that is worth building.
If v0's checks don't pass, the next stages aren't worth doing yet either.

## File layout

Create a new, self-contained module — do not edit existing files:

```
src/embodiednca/v0/
├── nca_module.py       # SimpleNCA (perception + update_net + residual clamp)
├── grid_world.py       # ResourceGrid + gated pattern dynamics
└── metrics.py          # coherence, self-mass, survival-time measurement
scripts/run_sanity_checks.py   # the three acceptance-criteria experiments
docs/design/v0.md              # what this is, how to run it, what the plots mean
results/v0/                    # checkpoints, plots, observations
```

## Core state

Two co-located fields over the same `H x W` spatial grid (start with `H=W=32`):

- `S`: pattern state, shape `[C, H, W]` (reuse `SimpleNCA`'s existing channel
  count / perception+update_net structure unchanged — `C=32`, `hidden_dim=64`
  are reasonable defaults, but don't fight the existing class's defaults).
- `R`: resource field, shape `[1, H, W]`, values in `[0, R_max]` (`R_max=1.0`).

## Resource dynamics (independent of the pattern's own update)

```python
generation_mask = torch.zeros(H, W)
# 2-3 fixed wells, hardcoded coordinates, away from the initial pattern location
generation_mask[well_y, well_x] = 1.0   # for each well

def step_resource(R, generation_mask, decay_rate=0.02, gen_rate=0.05):
    R_next = R * (1 - decay_rate) + generation_mask * gen_rate
    return R_next.clamp(0, 1.0)
```

Run this decay/generation step every timestep, unconditionally — it happens
whether or not the pattern is nearby, whether or not the pattern acts. It is
the "physics of the world," not something the pattern controls.

## Consumption

The pattern spends resource in proportion to how much self-maintenance work it
does at that cell. Subtract consumption *before* generation/decay of that step:

```python
consumption = consumption_rate * update_magnitude_map  # elementwise, per-cell
R = (R - consumption).clamp(min=0)
```

where `update_magnitude_map` is the per-cell L2 norm of whatever the pattern's
`update_net` computed at that location this step (see below). Tune
`consumption_rate` empirically so a pattern sitting directly on a well is close
to break-even, not comfortably positive — see Acceptance criteria, check 1.

## Gated self-maintenance (the core mechanic)

Do not modify `SimpleNCA`. Wrap its call:

```python
def gated_nca_step(nca: SimpleNCA, S, R, R_max=1.0):
    perception_out = nca.perception(S)
    update = nca.update_net(perception_out)          # [C, H, W]
    gate = (R / R_max).clamp(0, 1)                    # [1, H, W], broadcast over C
    S_next = S + gate * update
    S_next = torch.clamp(S_next, -10, 10)
    update_magnitude_map = update.norm(dim=0, keepdim=True)  # for consumption
    return S_next, update_magnitude_map
```

This is the entire mechanism by which scarcity matters: where `R` is near 0,
the pattern's own corrective update is suppressed, so ordinary grid diffusion
(unopposed) is what erodes it — not a special-cased death rule. Do not add an
`if R <= 0: reset()` anywhere. There is no reset in v0. A run ends when it ends;
record what happened, don't script it.

## Step order (single timestep)

1. `S, update_magnitude_map = gated_nca_step(nca, S, R)`
2. `R = (R - consumption_rate * update_magnitude_map).clamp(min=0)`
3. `R = step_resource(R, generation_mask, decay_rate, gen_rate)`
4. Log metrics (below), continue.

## Metrics to log every step (`metrics.py`)

- `self_mass`: sum of `S`'s magnitude (e.g. `S.norm(dim=0).sum()`) restricted to
  cells above some small activation threshold — a proxy for "how much pattern
  still exists."
- `coherence`: cosine similarity (or normalized cross-correlation) between the
  current `S` (in the region where it started) and the pattern's state at
  `t=0`. This is your "is it still recognizably itself" signal.
- `total_resource`: sum of `R` over the whole grid.
- `local_resource_at_pattern_centroid`: `R` sampled at the pattern's current
  center of mass (weight cells by `S` magnitude to get a centroid).
- `distance_to_nearest_well`: Euclidean distance from pattern centroid to the
  nearest well coordinate, every step.

Save these as a single CSV/dataframe per run, one row per timestep.

## Acceptance criteria — run these three checks before anything else

Implement each as a distinct scenario in `run_sanity_checks.py`. All three must
show the described qualitative pattern before we move to v1 (self-model /
forward prediction / identity-weight). If any fails, report the actual plots
and numbers — do not adjust parameters silently to force the expected shape;
tell me what happened and we'll decide together whether it's a tuning problem
or a modeling problem.

**Check 1 — break-even near a well.**
Place the initial pattern centered directly on a well. Run for a fixed horizon
(e.g. 2000 steps). `local_resource_at_pattern_centroid` should hover in a
narrow band, neither climbing freely (well too generous) nor draining to zero
even while parked there (well too weak). Report the trajectory and the
steady-state value. If it's not near break-even, this is a `gen_rate` /
`consumption_rate` tuning problem — sweep both before declaring it wrong.

**Check 2 — distance-dependence.**
Run three seeds with the initial pattern placed at increasing distances from
the nearest well (e.g. 2, 8, 20 grid cells away), everything else identical.
Plot `coherence` and `self_mass` over time for all three on one chart. Expect:
the farther-placed pattern should show visibly faster decay / lower steady-state
coherence than the near-placed one, since it has less access to replenishment.
If distance doesn't visibly matter, either resource diffuses too freely across
the grid (it currently doesn't — it's purely local per-cell dynamics — so check
this isn't accidentally true) or consumption/decay rates are mistuned relative
to grid size.

**Check 3 — gating actually does something, in both directions.**
Run three variants from the same initial pattern and seed:
  (a) full gated dynamics as specified above,
  (b) **ungated** control — same `SimpleNCA`, same resource field computed and
      logged for comparison, but the update is *not* multiplied by `gate`
      (i.e. `S_next = S + update`, resource still tracked/consumed but has no
      causal effect on `S`),
  (c) **always-zero-resource** — force `gate = 0` for the entire run (pattern
      gets no self-maintenance at all, ever).
Expect: (a) should show pattern survival/coherence somewhere between (b) and
(c) — worse than the unconstrained control, better than the no-maintenance
floor — and specifically should show *visible, gradual* degradation under
scarcity rather than either no visible effect (gating isn't wired correctly)
or instant total collapse in the very first few steps (gating or decay rates
are miscalibrated to be catastrophic rather than a genuine gradient).

## Deliverables

1. The four files above, runnable with `python run_sanity_checks.py`.
2. One plot per check (matplotlib, saved to disk, not just printed numbers).
3. A short written summary (in `README.md`) of what was observed for each of
   the three checks, including any parameter sweeps needed to reach a
   reasonable operating point, and explicit call-outs of anything that didn't
   behave as expected rather than adjusted-until-it-did.

## What happens after this

Do not start on self-model prediction, identity-weight, or action-selection
until these three checks are reviewed. That's the next spec, not this one.
