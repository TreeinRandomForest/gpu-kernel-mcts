# AGENTS.md

## Source of truth

- `spec.md` is the source of truth for Milestone A behavior and architecture.
- Before modifying search logic, read the relevant sections of `spec.md`.
- Do not silently reinterpret or expand the scope of `spec.md`.
- If code and `spec.md` disagree, flag the discrepancy rather than changing semantics arbitrarily.

## Development workflow

- Prefer small, reviewable changes.
- For nontrivial changes, inspect the relevant code first and state the intended change before editing.
- Add or update tests for behavioral changes.
- Run the relevant tests after modifications.
- Do not commit or push unless explicitly asked.

## Critical search invariants

Preserve these unless explicitly instructed otherwise:

- MCTS performs one global program search; it does not commit/re-root after each move.
- Only compile-successful, correctness-passing candidates become MCTS nodes.
- `B_gen` counts all candidate-generation calls, including repair/regeneration calls.
- Progressive widening uses the formula defined in `spec.md`.
- PUCT selects semantic strategies.
- UCB selects among existing generated realizations.
- `Q_mean` is used for default selection; `Q_max` is logged.
- MCTS visits do not imply GPU reevaluation.
- Valid locally slower kernels must not be pruned solely for being slower than their parent.
- Secrets such as `RUNPOD_API_KEY` must never be committed or logged.

## Review patterns

When asked for a "spec audit":
- Compare implementation against `spec.md`.
- Report concrete discrepancies with file/line references.
- Do not edit unless asked.

When asked for an "algorithm trace":
- Trace one MCTS iteration through PUCT, progressive widening, UCB, evaluation, budget accounting, and backup.

When asked for "mutation safety":
- Identify affected invariants/tests before changing search logic.
- Add or update tests before or alongside the implementation.
