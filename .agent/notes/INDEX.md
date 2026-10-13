<!-- agent-memory-schema: 1 -->
# Notes index — read this first

Agents: load **this file only**, then the **one** row matching the task. Do not bulk-read notes/.

This file is the **authority for what happens next**. Notes hold conditions; this holds the queue.

## Notes

| Topic | File | Status | One-liner |
|---|---|---|---|
| How to write notes | [README.md](README.md) | meta | layers, front-matter, access rules |

## Active next

| # | Item | Source | Added |
|---|---|---|---|
| 1 | Series is **closed at v5**. Do not start v6 in this repo. | human |  |

## Do not

- retrain from scratch to fix a missing checkpoint — fix `embodiednca.paths` consumers instead.

## Glossary (optional)

| Name | Means |
|---|---|
| gated | residual NCA update multiplied by `R/R_max` |
| identity_weight | max(0, baseline_coh − probed_coh); rank with `validity_mask` |
| selected | argmax over self-model `coherence_future` under a boost encoding |
| wear_aware_heuristic | boost toward the lower-wear well; skip if both floored |
| Stage A | freeze earlier checkpoints; do not retrain to chase a ranking |

## Design map

Sealed documents. These are not notes.

| Topic | File | Status | One-liner |
|---|---|---|---|
| Layout / results root | [../../docs/design/layout.md](../../docs/design/layout.md) | sealed | `results/<stage>/`; versioned stage copies are the record |
| v0 spec | [../../docs/design/v0_embodied_nca_spec.md](../../docs/design/v0_embodied_nca_spec.md) | sealed | resource-gated residual NCA |
| Stage write-ups | [../../docs/design/v0.md](../../docs/design/v0.md) | sealed | measured numbers; v5 closes the series |
