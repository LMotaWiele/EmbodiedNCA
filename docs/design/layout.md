# Why this layout

Short rationale for the tree in `AGENTS.md`. Agents should follow AGENTS.md hard rules; this file is background only.

| Path | Why |
|------|-----|
| `src/embodiednca/` | Single home for production code; scaffolding stays mechanical. |
| `src/embodiednca/v0` … `v5` | One package per stage. Later stages import earlier ones (`from embodiednca.v0.nca_module import SimpleNCA`). |
| `src/embodiednca/paths.py` | One results root: `results/<stage>/`. Scripts run from repo root. |
| `tests/` mirror | Fixed mapping so new-module skills generate both halves without guessing. |
| `scripts/` vs `notebooks/` | Scripts are re-runnable stage checks; notebooks are demos. Splitting avoids a catch-all dump. Scripts are not imported by `src/`. |
| `docs/design/` | Spec (`v0_embodied_nca_spec.md`) and per-stage write-ups (`v0.md` … `v5.md`) — protocol and measured numbers. |
| `results/<stage>/` | Checkpoints, plots, CSVs, `observations.txt`. Historic v0 snapshots live at `results/v0/historic/`. |
| `.agent/notes/` | Hard-won findings; **INDEX.md** is the agent entrypoint, then one linked note. |
| `.agent/skills/` | Shared skills once, not cloned under every harness. |
| `.grok/` vs `.agent/` | Model routing and permissions differ per harness; knowledge should not. |
| No project CRG `.mcp.json` | Graph server is global; project MCP only for extra servers. |

## Versioned copies (do not merge)

Near-duplicate modules across stages are the experimental record, not cleanup debt:

| Family | Why they stay separate |
|--------|------------------------|
| `self_model` / `self_model_v2` / `self_model_v3_5` | Same three heads; training distribution and wear channels change. Frozen checkpoints are stage-specific. |
| `rollout_dataset` / `boost_rollout_dataset` / `death_aware_rollout_dataset` | Probe types and physics (v0 vs boost vs per-well wear + death) differ. |
| `action_selection` / `action_selection_v3_5` | 6-way vs 7-way (skip) argmax; v3.5 adds wear-aware heuristics. |
| `wear_dynamics` vs `wear_perwell` | Global scalar vs per-well tracks; different `step_resource` integration (gen_rate vs generation_mask). |

Do not fold these into one "current" module. A later stage that needs v2 behaviour must still import `embodiednca.v2`.

## Checkpoints

| Stage | Loads | Writes |
|-------|-------|--------|
| v0 | — | `results/v0/trained_nca.pt` |
| v1 | v0 NCA | `results/v1/self_model.pt`, `rollout_dataset.pt` |
| v2 | v0 NCA | `results/v2/self_model_v2.pt`, `boost_rollout_dataset.pt` |
| v3 | v0 NCA, v2 self-model | plots / summary only (no retrain) |
| v3.5 | v0 NCA | `results/v3_5/self_model_v3_5.pt`, `death_aware_dataset.pt` |
| v4 | v0 NCA, v3.5 self-model | plots / summary only |
| v5 | v0 NCA, v3.5 self-model | `results/v5/relocated_eval_dataset.pt` |

`--skip_train` / `--skip_dataset` reuse these files. Missing checkpoints are a path bug, not a reason to retrain.
