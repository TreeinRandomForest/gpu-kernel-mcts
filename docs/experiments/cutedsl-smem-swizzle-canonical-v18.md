# Canonical CuTe shared-memory swizzle validation v18

This run validated schema-v3 `shared_memory_swizzle={heuristic,sw64}` through the
normal `CuTeDSLBackend` compile, correctness, benchmark, cache, artifact, and
diagnostic-v2 profiling paths on one H100 SXM. Both states used tile `(128,256)` and
cluster `(2,1)`.

## Integration finding

The initial v17 canonical run failed its distinct-runtime-fingerprint check. Although
the representations and configuration hashes differed, both states produced the
heuristic IR, fatbin, kernel name, and profile. The canonical backend runner had not
forwarded the new typed field to the standalone execution adapter. That forwarding
bug was fixed and given direct regression coverage before v18 was run. The v17 timing
difference was repeated measurement of one kernel and is not SW64 evidence.

## Canonical results

| State | Median (us) | Mean (us) | Stddev (us) | Correct |
| --- | ---: | ---: | ---: | --- |
| Heuristic SW128 | 187.072 | 188.039 | 2.639 | Yes |
| SW64 | 188.576 | 189.475 | 2.483 | Yes |

SW64 was 1.504 us, or approximately 0.80%, slower by median. Both states passed exact
correctness, retained schema-v3 canonical identity, reused their initial JIT artifact
on evaluator lookup, and produced distinct normalized IR, fatbins, and kernel names.

## Diagnostic-v2 profile

| Metric | Heuristic | SW64 |
| --- | ---: | ---: |
| Registers/thread | 154 | 154 |
| Achieved occupancy (%) | 12.38 | 12.37 |
| Instructions executed | 19,074,560 | 19,920,384 |
| Tensor instructions | 1,048,576 | 1,048,576 |
| SM throughput (%) | 84.76 | 86.03 |
| Tensor-pipe utilization (%) | 90.20 | 90.81 |
| L2 throughput (%) | 55.00 | 64.86 |
| L2 sector hit rate (%) | 77.19 | 86.28 |
| Barrier-stall ratio | 3.91 | 2.85 |
| Long-scoreboard stall ratio | 3.82 | 4.01 |
| LSU shared-bank conflicts | 0 | 0 |

SW64 improved the observed L2 hit rate, L2 throughput, SM throughput, and barrier
stall ratio, but executed about 4.4% more instructions and was slightly slower. The
reported LSU bank-conflict metric being zero does not prove that no WGMMA/TMA shared-
memory conflict or layout cost exists; it covers a narrower counter domain.

## Decision

The field remains a valid canonical structural dimension even though it is locally
slower.

## Focused MCTS smoke

The follow-up H100 search started from the heuristic `(128,256)`, cluster `(2,1)`
state and enabled only `change_shared_memory_swizzle`, with `B_mut=1`, `B_gen=0`,
`K_max=1`, and `max_depth=1`.

| Result | Value |
| --- | ---: |
| Iterations | 1 |
| Nodes | 2 |
| `B_mut` | 1 |
| `B_gen` | 0 |
| Profile calls | 1 |
| SW64 backed-up reward | -0.00926965 |
| Best reward | 0.0 (heuristic root) |

The child compiled, passed correctness, became a valid node, and was backed up. Its
negative reward agrees with the standalone result that SW64 is slightly slower at
this schedule. The trace was written to `cutedsl-swizzle-bmut1-v18.sqlite`; it is a
local generated artifact and is not committed.
