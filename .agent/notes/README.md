<!-- agent-memory-schema: 1 -->
# Agent notes

Durable project memory. Chat dies; **notes + evidence pointers** keep claims.

## Precedence

1. `docs/design/` — sealed authority. Specs and ADRs beat notes.
2. `INDEX.md § Active next` — the queue. The only place that says what happens next.
3. Notes — claims, evidence, and the **conditions** under which the queue changes.

A note never tells you what to work on. It tells you what is true, and what would have to hold for something to become next.

## Layers (do not mix)

| Layer | Where | Contents |
|---|---|---|
| Design authority | `docs/design/` | Specs, ADRs, sealed decisions — not notes |
| Task scope | `*-scope.md` | Locks **before** seeing results |
| Diagnosis | `*-diagnosis.md` | Claims, confounds, run/artifact IDs |
| Gotchas | `*-gotchas.md` | Cross-cutting traps that recur |
| Ledger (optional) | `burden-ledger.md` / `changelog-notes.md` | Hours, milestones, one-line outcomes |
| Index | `INDEX.md` | **Agent entrypoint + queue** |
| Explanations | `.agent/explain/` | Walkthroughs the harness writes for the human — **never written by hand, not notes, not indexed** |

## Access (token-cheap)

1. Read **`INDEX.md` only** first.
2. Open **one** linked file for the task (prefer front-matter; deeper sections on demand).
3. After a finding: update that note’s **front-matter**. If a pre-committed condition fired, append its Then to `INDEX § Active next`.
4. Never bulk-load all of notes/; never paste full chat logs or full eval reports.

## File roles

| Pattern | When |
|---|---|
| Living diagnosis | One open investigation / GO–NO-GO chain |
| Task scope | Pre-committed acceptance criteria before numbers |
| Gotcha card | Small reusable trap across tasks |
| Atomic finding | Optional `YYYY-MM-DD-slug.md` if cited alone |
| Ledger row | After a milestone or expensive run |

One **living diagnosis per open question**. When a chain closes, set `Status:` and open a new doc (or section) for the next claim.

## Note front-matter (required)

Every living note starts with:

```markdown
# Title
**Status:** open | sealed | superseded by X
**Topic:** …
**Last updated:** YYYY-MM-DD

## Current claim
One short paragraph: what is true *now* and what it is *not*.

## Ruled out / confounds
- …

## Evidence (pointers only)
- run / artifact: `path/…`
- code: `src/…`

## Pre-committed criteria
| If | Then |
|---|---|
| … | … |

## Do not
- …
```

Chronology and deep dives **below** that block. Claims at top; raw logs as pointers only.

The four bold fields and the two tables are parsed mechanically — by the pre-commit drift check, and by the routing layer that evaluates `If` rows against run evidence. Keep the headings verbatim. Free prose goes under them, not instead of them.

## Pre-committed criteria

The `If → Then` table is the contract that survives seeing the results.

- Write it **before** the run. A criterion authored after the numbers is not pre-committed, whatever it says.
- Each `If` must be decidable from evidence alone, without rereading the narrative. One condition per row.
- Each `Then` is an action that will be appended to `INDEX § Active next`, or `seal` / `supersede <note>`. It is not itself a queue entry until it fires.
- `Do not` rows that should bind future sessions in *any* task go to `INDEX § Do not`, not here. This section binds only work on this claim.

## Writing rules

- Prefer **decisions and constraints** over narrative.
- Artifacts = evidence; notes = interpretation + pointers.
- Supersede, don’t delete (`Status: superseded by …`).
- Label names (verdicts, modes) are part of the contract — put them in the INDEX glossary if they matter; do not stretch old labels onto new conditions.
- State experimental conditions explicitly when a claim depends on them (oracle vs free, train vs eval, etc.).
- No production logic in notes.
- No walkthroughs. `.agent/explain/` is written by the harness only; do not write there. Record progress on a spec in its run record; for other work, in the commit message. Only a trap that will recur becomes a gotcha card here.
