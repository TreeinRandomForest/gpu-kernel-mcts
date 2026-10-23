# Sequential strategy choice: seed-0 trace analysis

This analyzes the two completed `B_gen=50` traces in the
[matched pilot](sequential-strategy-choice-seed0.md). It compares recorded
source, compiler artifacts, timing samples, and lightweight NCU profiles.
No new kernels were run for this analysis.

## What the trajectories show

| Call | Freeform best (µs) | Catalog best (µs) | Relevant change |
| ---: | ---: | ---: | --- |
| 6 | 1,372.832 | **697.616** | Catalog reaches a 128×128 WMMA tile with eight FP32 accumulators per warp, K32 double-buffered staging, and padded shared stride 40. |
| 34 | 708.928 | **697.616** | Freeform places productive CTAs in the first 1,024 physical grid positions; this edit also changes the copy path. |
| 40 | **687.568** | 694.640 | Freeform restores a 128×128 asynchronous pipeline after trying smaller tiles. |
| 43 | **676.944** | 690.976 | Freeform hoists asynchronous-copy address generation. |
| 47 | **647.920** | 690.416 | Freeform restores 128×128 tiling and emits `cp.async.ca`; catalog best was already reached at call 41. |
| 48 | **647.008** | 690.416 | Freeform removes output warp barriers and moves work below the inactive-CTA exit. The 0.912 µs gain is smaller than call 48's 3.952 µs sample standard deviation. |

The catalog got close to 700 µs in six calls, then made little best-so-far
progress. Freeform took longer to get there but ended 43.408 µs (6.71%) below
the catalog's best measured median. This is one pair of sampled searches on
different workers. The 30 samples at each candidate estimate measurement
spread, not search-run variance; the root timings differed by 0.75%.

Catalog call 44 has the same state key and node as call 41. Its candidate
source only removes two blank lines. The median changed from 690.976 to
690.416 µs on remeasurement; call 44 does not represent a new optimization.

## Source changes behind the gap

Both best kernels use the same broad design: a 128×128 output macro tile,
K32 double-buffered shared staging with stride 40, eight FP32 WMMA
accumulators per warp, 256 threads, and 40 KiB of static shared memory.
Their remaining differences are concrete implementation choices:

1. **CTA placement.** The catalog best activates a sparse owner pattern:
   only grid coordinates divisible by eight in both dimensions compute a
   macro tile. Freeform calls 34–50 map the 1,024 productive macro tiles to
   the first 1,024 physical CTA indices and make the rest return. No valid
   catalog candidate in this trace used that compact mapping. Freeform call
   33→34 improved from 733.488 to 708.928 µs, but it also replaced
   asynchronous panel copies with synchronous vector loads. Thus the trace
   does not isolate the placement benefit.
2. **Copy cache policy.** Freeform call 47's inline copies use
   `cp.async.ca.shared.global`; the preceding fast freeform kernels use
   `cp.async.cg` after compilation. Call 45→47 improved from 680.528 to
   647.920 µs, with equal reported dynamic instruction counts (132.088
   million) and the same 128 registers/thread. The change also replaced a
   compiler intrinsic with inline assembly and rearranged some address and
   copy statements, so the 32.608 µs gain cannot be assigned solely to
   `.ca`.
3. **Output path.** Freeform call 47→48 removes explicit output
   `__syncwarp` calls and changes the early exit. Its measured 0.912 µs gain
   is too small relative to the observed sample variation to credit as a
   reliable speedup. The catalog's call 40→41 also batches scratch stores
   and removes output warp barriers under `reduce_synchronization`, improving
   its median by 3.664 µs; call 40's sample standard deviation was 5.055
   µs.

The cleanest cache-policy comparison is **catalog call 38→39**. Its source
diff changes exactly one line, `cp.async.cg` to `cp.async.ca`; its PTX differs
only by that qualifier. Median latency worsened from 698.176±2.302 to
737.904±2.089 µs (sample standard deviations). L1TEX throughput as a
percentage of peak active throughput rose from 38.72% to 58.51%, while L2
throughput as a percentage of peak elapsed throughput fell from 52.90% to
32.84%. These percentages are activity measures, **not bytes moved**. The
catalog model made this edit under `pipeline_tensor_core_data_movement` and
later restored `.cg`. The opposite direction in the freeform family is
consistent with an interaction with tile placement, copy code, or another
kernel detail; it does not establish which interaction matters.

| Final kernel | Median (µs) | Registers/thread | Shared memory | Spills | Achieved occupancy | Executed instructions | L1TEX / L2 throughput (% of respective peak) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Freeform call 48 | 647.008 | 128 | 40 KiB | 0 | 23.98% | 130.998M | 61.72 / 36.05 |
| Catalog call 41/44 | 690.416 at call 44 | 126 | 40 KiB | 0 | 22.64% | 140.698M | 39.09 / 52.80 |

These profile differences describe the resulting kernels. They cannot
partition the 43 µs gap by cause, because several source choices differ and
the runs used separate workers. Static SASS site counts likewise do not
measure dynamic work or memory traffic.

## What this says about the strategy catalog

The freeform best did not discover a wholly different GEMM architecture.
The catalog covered its main semantic ingredients through
`tensor_core_multi_accumulator`, `pipeline_tensor_core_data_movement`,
`specialize_workload_shape`, and `reduce_synchronization`. Its call-6
result shows those prompts can reach the fast WMMA design quickly.

The catalog did not produce the compact physical CTA mapping in this run.
`specialize_workload_shape` says to specialize shapes and launch choices,
while the tensor-core prompts permit grouping logical coordinates. That
wording does not prohibit compact mapping, but it does not explicitly direct
the model to compare active CTA placement within an oversized fixed grid.
This is a possible **strategy precision or application** gap, not proof that
a new semantic category is required. Conversely, the catalog *did* try the
`.ca` cache-policy idea, and its isolated trial lost. Simply adding “use
`.ca`” to the menu is unsupported.

The prompt's `paths` dataflow view records static local PTX dependencies and
site counts; it explicitly lacks dynamic counts, measured bytes, alias edges,
and paths across control-flow joins. It does not express physical CTA
placement. The bounded compiler summary records resource usage and static
SASS category counts rather than a raw PTX listing. Source code exposed the
mapping and cache qualifier where present, while lightweight NCU exposed
aggregate activity. We cannot say from this trace that richer dataflow or
compiler context would have changed the model's choice.

## Next discriminating work

1. Remeasure both exported best sources in alternating order on one pinned
   worker to quantify the cross-worker and temporal offset.
2. On the **same source family**, compare compact versus sparse CTA placement
   and `.cg` versus `.ca` as a 2×2 crossover, keeping all other code and
   launch settings fixed. Repeat timing and profiles; check generated PTX
   and resource use. This tests the suspected interaction directly.
3. Repeat the matched freeform/catalog search over declared seeds, after
   resolving the freeform response-label protocol deviation. Classify source
   edits independently from the model's declared strategy ID.
4. If compact placement helps repeatedly, sharpen an existing applicable
   strategy prompt or add a narrowly scoped action with a versioned catalog,
   then compare it under matched search conditions. Keep the MCTS/sequential
   explanation separate until a matched MCTS experiment exists.

## Follow-up: controlled four-kernel crossover

The [2026-10-23 crossover](sequential-cta-cache-crossover/README.md) completed
step 2 on one pinned H100 using the freeform best as the common source. In
that source family, compact CTA placement saved about 28 µs with either
cache policy, and `.ca` saved about 15 µs with either CTA placement. The
estimated interaction was about 1 µs, smaller than the round-to-round
variation. Thus this controlled test did **not** support the suspected
placement × cache-policy dependence. The catalog's earlier `.ca` regression
must depend on some other kernel difference or measurement condition; this
experiment does not isolate which one.
