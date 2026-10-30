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

- Add iteration playback and a cumulative-best-reward timeline. **Implemented.**
- Compare runs with different strategy sets, priors, `c_puct`, or generation budgets.
  **Implemented.**
- Plot strategy validity, repair, visit, and reward summaries. **Implemented.**
- Add exact decision-math tables after Phase 3 traces exist.
- Export selected paths, kernels, diffs, profiles, relevant decisions, and run context
  as a versioned ZIP analysis bundle. **Implemented.**

## Artifact reconciliation tab

The **Artifact reconciliation** tab has independent trace, run, and node selectors
for A and B. Select the same evidence layer in both columns to compare verified
CUDA source, hash-checked retained PTX, `ptxas` resources and static SASS category
counts, PTX dataflow views, whole-kernel NCU profiles, or the H100 resource graph
with node observations and constraint checks. PTX is displayed with line numbers so
the dataflow view's `ptx_line` fields can be located in the retained artifact.
The dataflow selector exposes the four prompt views and the complete recorded PTX
graph; the latter can be larger than the prompt's bounded `full` view. The tab
also reports the run's configured prompt view when available.

Raw SASS disassembly was not retained in these traces, so the tab cannot show a
PTX-to-SASS instruction mapping. SASS counts are static sites, PTX paths are local
static dependencies, and NCU counters are whole-kernel observations. The H100 graph
has possible links and architectural limits, but it does not assign measured bytes
or latency to a path. Missing artifacts remain explicit; the browser does not
recompile, reprofile, or write to SQLite.

The **NCU Source summary** layer can also show bounded selected-node reports
from `analysis.json` files beneath `--trace-dir` or an explicit
`--source-report-dir`. It matches an exact source SHA-256 and benchmark ID, then
labels the result as a separate-worker measurement. It shows compiler spill
metadata, SASS local instruction sites, whole-kernel local-memory counters,
and the top correlated source lines. The large `.ncu-rep` and CSV exports stay
outside SQLite; the browser remains read-only. A matching source does not
imply that the separate report shares the trace node's timing conditions.

## Testing

Keep database discovery, SQL queries, graph traversal, and diff construction independent
of the HTTP handler. Test them against small SQLite fixtures containing branching,
transpositions, invalid generations, missing profiles, and multiple runs. Test API path
validation and response status separately. Browser behavior can initially remain a
small dependency-free layer, with end-to-end browser automation added when UI
complexity warrants it.
