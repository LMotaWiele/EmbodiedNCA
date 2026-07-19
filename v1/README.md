# v1 Self-Model + Identity-Weight

Read-only predictive head on top of verified **v0**. Does **not** modify
`v0/grid_world` step logic, `SimpleNCA`, or resource physics.

## Setup

```bash
cd EmbodiedNCA
source .venv/bin/activate
python v1/run_v1_checks.py
python v1/run_v1_checks.py --skip_dataset --skip_train
```

Requires `v0/results/trained_nca.pt`.

---

## Operational API (v2 must inherit this)

Two different untrustworthy regimes — **do not conflate**:

| Regime | Where | Weight | Risk |
|--------|-------|--------|------|
| **No signal** | Plain bg far from pattern / wells | ≈0 (`bg_p95≈0`) | Low if you rank on support only |
| **False signal** | Inter-well **equidistant boundary** | *High* (can exceed pattern peak) | High — looks identity-critical |

```python
from identity_weight import (
    dense_identity_weight,
    validity_mask,
    support_restricted_weight,
    argmax_identity_weight,
    equidistant_boundary_mask,
)
from grid_world import DEFAULT_WELLS

w = dense_identity_weight(model, S, R, active_only=True, device=device)

# Rank only on pattern support AND off the equidistant locus
# (wells computed at runtime — survives well relocation; no hardcoded cells)
masked, mask = support_restricted_weight(w, S, wells=DEFAULT_WELLS)
pick = argmax_identity_weight(w, S, wells=DEFAULT_WELLS)

# validity_mask = (‖S‖ > τ)  AND  NOT  |d_well_i − d_well_j| < eps  (any pair)
```

- **Mass-only** masks correctly drop no-signal bg for on-pattern ranking.
- They do **not** address false-signal equidistant cells on a full-grid map —
  those were never going to pass `‖S‖>τ`. Pass `wells=` so the equidistant
  boundary is excluded explicitly before any resource-allocation decision.

Never raw `argmax(w)` over the full grid.

---

## Check 1 — calibration (summary)

| Metric | Value |
|--------|------:|
| Pooled / mean within-bin coh r | 0.953 / **0.828** |
| local_R p95 abs err \| true&lt;0.2 | **0.178** |
| local_R r at [10,13) | 0.234 (MAE 0.022 — **variance collapse**, not failure) |
| Survival n_dead | 0 (H too short — untested) |

---

## Check 2 / equidistant false-signal

### Headline (stronger than “dominates corners”)

At **typical operating distances**, the map’s single highest confidence is a
**false signal larger than the real signal**:

| Dist | max locus weight | pattern peak | **locus / pattern** | beats pattern? |
|-----:|-----------------:|-------------:|--------------------:|:--------------:|
| 2 | 0.087 | 0.101 | 0.86× | no (pattern still wins) |
| 4 | **0.103** | 0.044 | **2.4×** | **yes** |
| 8 | **0.150** | 0.036 | **4.2×** | **yes** |

vs corners (weaker claim): locus / corner ≈ **7–12×**; `bg_p95=0.0` means ≥95%
of bg is correctly near zero — failure is **rare and localized**.

### Geometry: one locus, not three special cases

Wells `(8,8)` and `(24,24)`. Hot cells:

| Cell | x+y | Role |
|------|----:|------|
| (24,8) | **32** | equidistant saddle |
| (8,24) | **32** | equidistant saddle (symmetric) |
| (16,16) | **32** | same line (midpoint) |

All three lie on **x + y = 32** — the discrete **perpendicular bisector**
(equidistant locus) between the two wells, not “two saddles plus a mid-layout
curiosity.” Runtime rule:

```text
flag cell if  min over well pairs  |dist(cell, A) − dist(cell, B)|  <  eps
```

Hardcoding `(24,8)` / `(8,24)` would break when wells move; the `|d_A−d_B|`
mask does not.

R on the locus is ~0 (not a high-resource pocket): artifact is **S≈0 +
layout-special coordinates** never seen in single-well training.

Plot: `results/check_saddle_hotspots.png` (auto from `run_v1_checks.py`).

### Untrained cosine

Init-dependent (`seed = run_seed + 17`). Prefer stable **naive∝‖S‖ residual**
(0.63–0.84) for structure claims.

---

## Net assessment

| Claim | Status |
|-------|--------|
| Short-horizon coh skill within regimes | **Yes** (within-bin r 0.73–0.89) |
| Pooled r fine-grained only | **Overstated** |
| local_R scarcity | Fat tail; [10,13) r cliff is variance collapse |
| Survival head | Untested |
| No-signal bg | Correctly ~0 (`bg_p95=0`) |
| False-signal equidistant locus | **Confirmed**; beats pattern peak at dist 4/8 |
| Raw argmax safe? | **No** |
| v2 cell ranking | **Must** use support + equidistant exclusion |

## After this

v1 closed. v2 action / resource allocation that ranks cells must call
`argmax_identity_weight(..., wells=current_wells)` (or equivalent dual mask).
Optional polish: train with inter-well probes so unrestricted maps improve —
not required if the mask is mandatory.
