# Agent notes

Durable project memory. Chat dies; **notes + evidence pointers** keep claims.

## Layers (do not mix)

| Layer | Where | Contents |
|---|---|---|
| Design authority | `docs/design/` | Specs, ADRs, sealed decisions — not notes |
| Task scope | `*-scope.md` | Locks **before** seeing results |
| Diagnosis | `*-diagnosis.md` | Claims, confounds, run/artifact IDs |
| Gotchas | `*-gotchas.md` | Cross-cutting traps that recur |
| Ledger (optional) | `burden-ledger.md` / `changelog-notes.md` | Hours, milestones, one-line outcomes |
| Index | `INDEX.md` | **Agent entrypoint** |

## Access (token-cheap)

1. Read **`INDEX.md` only** first.
2. Open **one** linked file for the task (prefer front-matter; deeper sections on demand).
3. After a finding: update that note’s **front-matter** (+ ledger row if you keep one); add a gotcha only if it will recur.
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

## Pre-committed next
| If | Then |
|---|---|
| … | … |

## Do not
- …
```

Chronology and deep dives **below** that block. Claims at top; raw logs as pointers only.

## Writing rules

- Prefer **decisions and constraints** over narrative.
- Artifacts = evidence; notes = interpretation + pointers.
- Supersede, don’t delete (`Status: superseded by …`).
- Label names (verdicts, modes) are part of the contract — put a glossary in INDEX if they matter; do not stretch old labels onto new conditions.
- State experimental conditions explicitly when a claim depends on them (oracle vs free, train vs eval, etc.).
- No production logic in notes.
