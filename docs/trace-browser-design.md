# Browser-based trace explorer

This document describes local analysis tooling for persisted MCTS traces. It does not
change the search behavior defined by `spec.md`.

## Purpose

The explorer should make a search run understandable at both the algorithm and kernel
levels. It should support selecting a SQLite trace and run, exploring the
transposition-preserving MCTS DAG, opening any two kernel nodes, displaying their
relationship through semantic strategies, and comparing source, benchmark, and
profiling information.

Because the search graph admits transpositions, the interface must not imply that each
node has exactly one parent. When two nodes are related, it should show a shortest
directed path. When they lie on different branches, it should show paths from a nearest
common ancestor.

## Local architecture and safety

```text
SQLite traces opened read-only
        |
Python query and path-analysis layer
        |
loopback HTTP JSON API
        |
plain HTML/CSS/JavaScript and interactive SVG DAG
```

The server receives an explicit `--trace-dir`, discovers SQLite files beneath that
directory, and assigns opaque IDs to them. Browser requests cannot supply arbitrary
filesystem paths. It binds to `127.0.0.1` by default and performs no writes to traces.

## Phase 1: read-only data layer

- Discover and validate SQLite trace files and list every contained run.
- Return run status, configuration, budgets, node count, and best performance.
- Load graph nodes, strategy edges, realization edges, and invalid generations.
- Load complete source, benchmark, launch, profile, and provenance data for one node.
- Find a shortest strategy path between related nodes or their nearest common ancestor.
- Produce a unified source diff and structured profile comparison for two nodes.

## Phase 2: browser MVP

- Select a trace and run.
- Explore an interactive, transposition-preserving DAG with reward, visit, depth, and
  invalid-proposal filters.
- Select arbitrary A and B nodes, inspect their kernel source and profiles, and compare
  their source and numeric profile fields.
- Show the sequence of strategies between related nodes or from a common ancestor.
- Clearly distinguish final edge statistics from historical decision-time statistics.

## Phase 3: exact selection tracing (implemented)

Legacy traces retain selected iteration steps and final materialized edge statistics,
but not the full candidate score table at each historical selection. New traces add
immutable decision records containing:

- all PUCT candidates, their priors, visits, `Q_mean`, `Q_max`, exploitation term,
  exploration term, total score, and selected flag;
- progressive-widening inputs, allowed child count, existing child count, and the
  expand-versus-UCB decision; and
- all UCB candidates, descents, `Q_mean`, exploitation term, exploration term, total
  score, and selected flag.

The logging must be observational: it must not alter random-number consumption,
selection, budgets, evaluation, or backup.

## Phase 4: algorithm analysis

- Add iteration playback and a cumulative-best-reward timeline.
- Compare runs with different strategy sets, priors, `c_puct`, or generation budgets.
- Plot strategy validity, repair, visit, and reward distributions.
- Add exact decision-math tables after Phase 3 traces exist.
- Export selected paths, kernels, diffs, and profiles for experiment reports.

## Testing

Keep database discovery, SQL queries, graph traversal, and diff construction independent
of the HTTP handler. Test them against small SQLite fixtures containing branching,
transpositions, invalid generations, missing profiles, and multiple runs. Test API path
validation and response status separately. Browser behavior can initially remain a
small dependency-free layer, with end-to-end browser automation added when UI
complexity warrants it.
