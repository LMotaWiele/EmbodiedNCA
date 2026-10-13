<!-- agent-memory-schema: 1 -->
# Project rules

Agent-agnostic layout and conventions. Stacks on `~/.pi/agent/AGENTS.md` (global routing + spawn policy). Prefer **AGENTS.md** — no `CLAUDE.md` / `.claude/`.

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
├── README.md              # human: what this is + how to run
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
├── docs/                  # other topics by folder
├── results/v0 … results/v5
├── .agent/skills/         # shared SKILL.md (harness-agnostic)
├── .agent/notes/          # durable findings — INDEX.md first (see rule 7)
├── .agent/explain/        # walkthroughs the harness writes for the human — never written by hand
├── .pi/settings.json      # harness wiring: project overrides only
├── .pi/agents/            # subagent definitions (tier → pinned model)
├── .pi/extensions/        # project-local extensions, if any
└── .mcp.json              # optional extra MCP
```

Rationale for buckets: `docs/design/layout.md` (read on demand).

## Hard rules

1. New modules under `src/embodiednca/…` only — never free-floating at repo root.
2. Tests mirror source: `src/embodiednca/<path>/mod.py` ↔ `tests/<path>/test_mod.py` (one style per repo; mechanical).
3. `scripts/` = agent-runnable one-offs; `notebooks/` = demos. Neither imported by `src/`. Promote shared code into `src/`.
4. Design/ADRs / stage protocol → `docs/design/`. README stays the product page.
5. Before a **new top-level directory**, stop and ask. Prefer the tree above.
6. `.agent/` = shared knowledge/skills, portable across harnesses. `.pi/` / `.mcp.json` = harness wiring — do not merge, do not move knowledge into wiring.
7. **Notes (cheap path):** before re-deriving a diagnosis, opening a design debate, or continuing prior investigation → read `.agent/notes/INDEX.md` only, then the **one** linked file. After a finding → update that note’s front-matter, and if a pre-committed condition fired, append the resulting item to INDEX `## Active next`. Do not bulk-read notes/ or dump chat. Shape: `.agent/notes/README.md` (on demand). Prefer `docs/design/` for sealed/spec authority over notes.
8. **INDEX is the queue.** `.agent/notes/INDEX.md § Active next` is the single authority for what happens next and for standing constraints. Notes hold *conditions* (`If → Then`), never a queue. When a condition fires, the Then writes a row into INDEX. Never read a note to find out what to work on.
9. Small focused edits; match neighbors; no unsolicited docs; verify against source not README alone.
10. Commit your own work, with messages naming the task. Never push, force-push, or rewrite commits you didn't make. No production logic in scripts/notebooks/notes. Existing check runners may keep their current logic.
11. **Explanations are not notes.** `.agent/explain/` is written by the harness only. Do not write there. Record progress on a spec in its run record; for other work, in the commit message. Promote a finding to `.agent/notes/*-gotchas.md` only when the trap will recur across tasks in this repo.
12. **Comments** — two registers; do not mix:
    - **`src/`:** 1–2 lines that say what it is and how it works. No reasoning, no spec essays, no "why we didn't do X", no history. Why lives in `docs/design/`. A constraint an editor would otherwise break may stay, still 1–2 lines.
    - **`scripts/` / `tests/`:** same 1–2 line module header. Check protocol and measured numbers live in `docs/design/`.
13. **Do not merge** near-duplicate modules across stages unless they are proven identical. Versioned copies (`self_model` / `self_model_v2` / `self_model_v3_5`) are the experimental record.
14. Checkpoints load from `results/<stage>/`. After moving artifacts, every consumer must follow `embodiednca.paths.results_dir`. Do not retrain from scratch to "fix" a path.

## New module

Create `src/…` + mirrored `tests/…`. Architecture change → `docs/design/`. Durable finding → `.agent/notes/` (via INDEX + front-matter). Next action → INDEX `## Active next`.
