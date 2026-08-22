# Notes index — read this first

Agents: load **this file only**, then the **one** row matching the task. Do not bulk-read notes/.

| Topic | File | Status | One-liner |
|---|---|---|---|
| How to write notes | [README.md](README.md) | meta | layers, front-matter, access rules |
| Layout / results root | [../../docs/design/layout.md](../../docs/design/layout.md) | sealed | `results/<stage>/`; versioned stage copies are the record |
| v0 spec | [../../docs/design/v0_embodied_nca_spec.md](../../docs/design/v0_embodied_nca_spec.md) | sealed | resource-gated residual NCA |
| Stage write-ups | [../../docs/design/v0.md](../../docs/design/v0.md) … v5.md | sealed | measured numbers; v5 closes the series |

## Glossary

| Name | Means |
|---|---|
| gated | residual NCA update multiplied by `R/R_max` |
| identity_weight | max(0, baseline_coh − probed_coh); rank with `validity_mask` |
| selected | argmax over self-model `coherence_future` under a boost encoding |
| wear_aware_heuristic | boost toward the lower-wear well; skip if both floored |
| Stage A | freeze earlier checkpoints; do not retrain to chase a ranking |

## Active next

- Series is **closed at v5**. Do not start v6 in this repo.
- **Do not:** retrain from scratch to fix a missing checkpoint — fix `embodiednca.paths` consumers instead.
