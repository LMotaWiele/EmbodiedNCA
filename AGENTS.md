# Project rules

Agent-agnostic layout and conventions. Stacks on `~/.grok/AGENTS.md` (graph + explore spawn policy). Prefer **AGENTS.md** — no `CLAUDE.md` / `.claude/` for Grok.

## Overview

- **Purpose**: Closed experiment series on resource-gated embodied Neural Cellular Automata. Stages v0–v5 (v5 closes the project): self-maintenance → self-model → action → wear → chained identity → novel layouts.
- **Stack**: Python >= 3.12 (`pyproject.toml`, hatchling, `.venv`)
- **Package**: `embodiednca` (`src/embodiednca/`, one subpackage per stage)
- **Build / test**: `source .venv/bin/activate` · `pip install -e '.[dev]'` · `pytest`
- **Results root**: `results/<stage>/` (checkpoints, plots, observations). Do not put artifacts under `src/`.

## Layout

```
EmbodiedNCA/
├── AGENTS.md              # this file
├── README.md              # human: what + how to run
├── pyproject.toml         # hatchling, name=embodiednca, >=3.12
├── .venv/                 # local venv (gitignored)
├── src/embodiednca/       # production code, versioned by stage
│   ├── paths.py           # results/<stage>/ and docs/design helpers
│   ├── v0/                # grid_world, nca_module, metrics
│   ├── v1/                # self_model, identity_weight, rollout_dataset
│   ├── v2/ …
│   ├── v3/
│   ├── v3_5/
│   ├── v4/
│   └── v5/
├── tests/                 # mirrors src/ one-to-one (smoke imports)
├── scripts/               # run_sanity_checks / run_vN_checks — not imported by src/
├── notebooks/             # demos — not production logic
├── docs/design/           # spec, stage write-ups, layout rationale
├── docs/                  # other topics
├── results/v0 … results/v5
├── .agent/skills/         # shared SKILL.md (not under .grok/ only)
├── .agent/notes/          # durable findings — INDEX.md first (see rule 7)
├── .grok/agents/          # harness shadows (explore → deepseek-flash)
├── .grok/config.toml      # optional project overrides only
└── .mcp.json              # optional extra MCP only (not CRG)
```

Rationale for buckets: `docs/design/layout.md` (read on demand).

## Hard rules

1. New modules under `src/embodiednca/…` only — never free-floating at repo root.
2. Tests mirror source: `src/embodiednca/<path>/mod.py` ↔ `tests/<path>/test_mod.py` (one style per repo; mechanical).
3. `scripts/` = agent-runnable one-offs; `notebooks/` = demos. Neither imported by `src/`. Promote shared code into `src/`.
4. Design/ADRs / stage protocol → `docs/design/`. README stays the product page.
5. Before a **new top-level directory**, stop and ask. Prefer the tree above.
6. `.agent/` = shared knowledge/skills. `.grok/` / `.opencode/` / `.mcp.json` = harness wiring — do not merge.
7. **Notes (cheap path):** before re-deriving a diagnosis, opening a design debate, or continuing prior investigation → read `.agent/notes/INDEX.md` only, then the **one** linked file. After a finding → update that note’s front-matter (+ ledger row if used). Do not bulk-read notes/ or dump chat. Shape: `.agent/notes/README.md` (on demand). Prefer `docs/design/` for sealed/spec authority over notes.
8. Small focused edits; match neighbors; no unsolicited docs; verify against source not README alone.
9. No commit/push/PR unless asked. No production logic in scripts/notebooks/notes beyond the existing check runners.
10. CRG MCP is **global** (`~/.grok/config.toml`) — no project `.mcp.json` for the graph. Build once: `code-review-graph build --repo .`
11. **Comments** — two registers; do not mix:
    - **`src/`:** 1–2 lines that say what it is and how it works. No reasoning, no spec essays, no "why we didn't do X", no history. Why lives in `docs/design/`. A constraint an editor would otherwise break may stay, still 1–2 lines.
    - **`scripts/` / `tests/`:** same 1–2 line module header. Check protocol and measured numbers live in `docs/design/`.
12. **Do not merge** near-duplicate modules across stages unless they are proven identical. Versioned copies (`self_model` / `self_model_v2` / `self_model_v3_5`) are the experimental record.
13. Checkpoints load from `results/<stage>/`. After moving artifacts, every consumer must follow `embodiednca.paths.results_dir`. Do not retrain from scratch to "fix" a path.

## New module

Create `src/…` + mirrored `tests/…`. Architecture change → `docs/design/`. Durable finding → `.agent/notes/` (via INDEX + front-matter).
