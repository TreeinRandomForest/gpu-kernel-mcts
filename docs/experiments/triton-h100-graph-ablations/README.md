# H100 CUDA Graph search ablations

These runs use the packaged Triton BF16 GEMM root, the fixed workload in
`spec.md` Section 50, GLM-5.3 Messages with thinking disabled, and the
published worker image at digest
`sha256:1e02c618f5a99b7f212522e0623131c3c26fb27c76834401b4a6bed54fad7874`.
The reward is the verified `triton_h100_cuda_graph_v1` median. The local
controller does not overlay code on the worker. Each seed's arms share one
Nebius H100 and one calibration; seeds 0 and 2 reverse the arm order.

`run.py` uses `B_gen=10`, one repair allowance, uniform strategy priors,
MCTS `k_max=3` and depth 10. Every arm collects `spill_v1` profiles lazily
and saves full SQLite traces. The prompt has no dataflow or H100 graph view.
The model receives the current source, workload, measured timing, and vendor
reference in every arm.

## Exact variation matrix

All rows use seeds 0, 1, and 2; GLM-5.3 Messages with thinking disabled;
the packaged pure-Triton BF16 GEMM root; an H100 SXM; one repair allowance;
the same CUDA Graph correctness and timing contract; and lazy `spill_v1` NCU
collection saved in SQLite. The prompt always has the current complete
source, workload and ABI, H100 target specification, current measured timing,
and a same-worker cuBLAS reference. No row supplies a dataflow view or the
H100 resource graph. `—` means the view is absent from the model prompt,
even though the artifact may be recorded in SQLite.

| Cohort and arm | Search and action policy | NCU in prompt | Compiler in prompt | Prior conversation turns | `B_gen` |
| --- | --- | --- | --- | ---: | ---: |
| Context `mcts-n0-c0` | MCTS, one freeform pure-Triton action | — | — | 0 | 10 |
| Context `mcts-n0-c1` | Same MCTS action | — | Triton summary below | 0 | 10 |
| Context `mcts-n1-c0` | Same MCTS action | `spill_v1` NCU-only view below | — | 0 | 10 |
| Context `mcts-n1-c1` | Same MCTS action | `spill_v1` NCU-only view | Triton summary | 0 | 10 |
| Behavior `mcts-n0-c0` | MCTS, one freeform pure-Triton action | — | — | 0 | 10 |
| Behavior `mcts-strategy` | MCTS, three-action pure-Triton catalog below | — | — | 0 | 10 |
| Behavior `seq-free-h0` | Sequential, open-ended source rewrite | — | — | 0 | 10 |
| Behavior `seq-strategy-h0` | Sequential, choose one applicable catalog action per step | — | — | 0 | 10 maximum |
| Behavior `seq-free-h2` | Sequential, open-ended source rewrite | — | — | 2 | 10 |
| Depth `seq-free-h0` | Sequential, same freeform policy as behavior cohort | — | — | 0 | 25 |

Context and behavior `mcts-n0-c0` have the same configuration but are
independent sampled runs on different workers. The depth arm is also a fresh
run; its call-10 and call-25 checkpoints are compared **within** that run.
`B_gen` includes generation and repair calls. Catalog sequential exhausted
applicable actions and stopped after 2, 3, and 7 calls, respectively.

**Triton compiler summary (`c1`).** The prompt gets `sm_90`,
`registers_per_thread`, `static_shared_memory_bytes`, Triton compiler
`spills` (`n_spills`), one PTX artifact SHA-256, and static PTX instruction
site totals split into `tensor`, `global_load`, `global_store`, `shared_load`,
`shared_store`, `local_load`, `local_store`, `async_copy`, `barrier`, `control`,
and `other`. It also gets schema/status and a limitation note. The full PTX,
SASS, source-line attribution, executed instruction counts, spill bytes, and
dataflow paths are **not** in this view. A concrete root example has 48
registers per thread, 12,288 static shared-memory bytes, zero compiler
spills, and 196 static PTX sites.

**NCU-only prompt view (`n1`).** The worker collects the `spill_v1` bundle:
occupancy; SM, DRAM, L2, L1/TEX and tensor-pipe throughput/activity;
executed instructions and tensor instructions; DRAM read/write bytes; L2
sector hit rate; local load/store sectors and local-load L1 hit/miss sectors;
eligible warps, issue activity, shared bank conflicts; and long-scoreboard,
short-scoreboard, barrier, math-pipe-throttle, and not-selected stall ratios.
The prompt contains both named summary values and their raw NCU metric
records. `launch__registers_per_thread`, its `registers_per_thread` summary,
and any `compiler_*` summary fields are masked so this factor does not
duplicate the explicit compiler summary. Collection is unchanged in `n0`
arms; only prompt visibility changes. Exact raw metric IDs are defined in
[`profiling.py`](../../../src/kernel_mcts/profiling.py).

**Three-action catalog.** The versioned
[`triton_h100_pure_local_v20.json`](../../../configs/triton_h100_pure_local_v20.json)
offers (1) an advancing-pointer `128×128×64`, eight-warp, three-stage loop
rewrite; (2) widening `BN` from 128 to 256 while preserving that source
family; and (3) specializing `M`, `N`, and `K` as `tl.constexpr`. Applicability
and source-edit guards enforce these as pure pointer-Triton transitions.
Freeform arms can make coupled pure-Triton changes, including descriptor and
persistent-kernel rewrites, without selecting one of those IDs.

**Prompt protocol.** MCTS asks for raw replacement source and includes the
selected action text, a compact parent median/sample/warmup summary, and the
cuBLAS reference. Sequential asks for JSON with `selected_strategy_id`,
`hypothesis`, and full replacement source; its current prompt includes the
full benchmark record and recent acceptance/failure feedback even with zero
conversation turns. `h2` additionally replays the two previous user/assistant
message pairs. These protocol differences prevent a clean causal MCTS versus
sequential comparison. The MCTS compact timing field currently says
`measurement: "CUDA events"`, and the catalog action text also mentions
CUDA-event latency; both labels are stale. The recorded benchmark, reward,
and auditor use `triton_h100_cuda_graph_v1` throughout.

## Context group

The four freeform MCTS arms cross NCU profile visibility (`n0`/`n1`) and
Triton compiler-summary visibility (`c0`/`c1`). The `n1` view masks
compiler-derived profile fields, including registers per thread. The compiler
view contains bounded resource fields and static PTX instruction-site counts.
The search action and all other controls are fixed.

## Behavior group

`mcts-n0-c0` is paired with `mcts-strategy` using the existing
`triton_h100_pure_local_v20` catalog. `seq-free-h0` is paired with
`seq-strategy-h0` using the same catalog. `seq-free-h2` changes only the
Messages history from zero to two prior user/assistant turns. These arms
hide NCU, compiler, dataflow, and H100 graph prompt views. The catalog has
three narrowly specified pointer-Triton actions; results describe this
catalog, not all possible strategy sets. MCTS and sequential prompts and
selection rules differ, so compare policy arms within each search method.

The fixed budget counts generation and repair calls. A short run with three
seeds is a screening experiment; close latency rankings require same-worker
remeasurement of the winners.

## Commands

From the repository root, with Nebius and EnMaaS credentials already set:

```bash
.venv/bin/python docs/experiments/triton-h100-graph-ablations/run.py --group context
.venv/bin/python docs/experiments/triton-h100-graph-ablations/run.py --group behavior
.venv/bin/python docs/experiments/triton-h100-graph-ablations/run.py --group depth --budget 25
.venv/bin/python docs/experiments/triton-h100-graph-search/audit.py \
  docs/experiments/triton-h100-graph-ablations/output/*.sqlite
```

Completed arms are skipped on restart. An incomplete trace causes the runner
to stop for inspection; it is never silently overwritten. Each completed
SQLite trace has a JSON summary and exported best source beside it. The
`output/` directory is Git-ignored.

A transient gateway 502/503/504, transport failure, or 120-second hard
deadline is recorded as one invalid generation; three consecutive such
failures stop the run and release the VM. Other gateway errors stop the run
immediately. The first seed-0 attempt completed three context arms, then its
source-only request stalled and failed at the gateway. Its incomplete trace
is retained. The `retry` label reruns the whole seed-0 quartet on one fresh
worker so the primary paired comparison stays on a shared calibration.

## Results: context visibility

Best verified CUDA Graph median in microseconds at `B_gen=10`:

| Seed | Source only | Compiler only | NCU only | NCU + compiler | Same-worker cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (`retry`) | 222.896 | 199.120 | **182.576** | 183.088 | 172.160 |
| 1 | 214.144 | **184.592** | 210.944 | 213.952 | 173.184 |
| 2 | **184.976** | 185.456 | 185.280 | 211.120 | 174.656 |
| Median | 214.144 | 185.456 | **185.280** | 211.120 | — |
| Valid / 30 calls | 26 | 23 | 26 | 26 | — |
| Gateway failures | 0 | 2 | 0 | 1 | — |

All 12 primary context traces passed the Graph timing auditor. The prompt
audit checked every generation: NCU and compiler fields were present only
in their declared arms, NCU's compiler-derived register metric was masked,
and all arms used `spill_v1` collection. Seed 0's three completed pre-retry
arms remain as exploratory repeats and are excluded from this table.

The best arm changes by seed. Compiler-only and NCU-only have nearly equal
three-seed medians, while the combined arm does not improve on the better
single-context arm in any seed. These are sampled search trees, not repeated
measurements of one source. Seed 2's source-only winner is a persistent
descriptor kernel, whereas most other winners are grouped pointer-load
kernels. The compiler-only arms also lost one call to the gateway in seeds 0
and 2, and the combined arm lost one in seed 2. The data do not establish a
causal prompt-context ranking.

## Results: search behavior

Best verified CUDA Graph median in microseconds, again on one worker per seed:

| Seed | MCTS freeform | MCTS catalog | Seq freeform, 0 turns | Seq catalog, 0 turns | Seq freeform, 2 turns | cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **185.824** | 228.704 | 185.088 | 184.208 | 185.888 | 174.064 |
| 1 | **180.976** | 185.088 | 185.152 | 184.528 | 184.704 | 174.176 |
| 2 | 193.408 | **183.936** | 213.024 | 184.448 | 196.096 | 173.424 |
| Median | 185.824 | 185.088 | 185.152 | 184.448 | 185.888 | — |
| Valid / calls | 26 / 30 | 25 / 30 | 22 / 30 | 7 / 12 | 21 / 30 | — |

All 15 behavior traces passed the timing audit. Freeform MCTS won seeds 0
and 1; catalog MCTS won seed 2. The catalog's seed-1 arm used five calls on
strategy-fidelity or correctness failures, while its other two seeds had
all ten calls valid. There is no stable policy winner in this small sample.

Catalog sequential stopped after 2, 3, and 7 calls, respectively, because
the current pointer-Triton source had no remaining applicable action. It
reached about 184–185 µs quickly in each seed, but never used its full
budget. Thus its latency cannot be read as a full-budget catalog versus
freeform comparison. The 0-turn versus 2-turn sequential difference is
below 1 µs in seeds 0 and 1, and favors two turns by 16.928 µs in seed 2.
The latter is a search-path outcome, not proof that history improves a fixed
kernel. No behavior arm had a gateway failure.

The primary context cohort and behavior cohort used different workers and
were sampled independently. Compare arms within a cohort and seed. The
run-level SQLite files, bounded JSON summaries, and best-source exports are
in the ignored `output/` directory.

## Results: sequential search depth

Freeform sequential search with zero prior conversation turns was run to
`B_gen=25` on three fresh workers. The table compares the best candidate
through call 10 with the best through call 25 **within the same run**. This
tests additional search depth without treating an independently sampled
10-call run as the first ten calls of a 25-call run.

| Seed | Best by call 10, µs (call) | Best by call 25, µs (call) | Gain, µs | Valid / 25 | Same-worker cuBLAS, µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 183.024 (4) | **176.832 (14)** | 6.192 | 13 | 172.432 |
| 1 | **178.048 (10)** | 178.048 (10) | 0 | 16 | 170.624 |
| 2 | **184.064 (8)** | 184.064 (8) | 0 | 22 | 173.200 |

All three depth traces passed the Graph timing audit and had no gateway
failures. Seed 0's call-10 winner used pointer loads; its call-14 winner was
a persistent tensor-descriptor kernel. The other two runs did not improve
after call 10 despite 15 more calls. Thus a longer sequential budget can
find a useful source-family transition, but the gain was not reliable across
these three sampled runs. The 176.832 µs winner was 4.400 µs above its
same-worker cuBLAS calibration.

Nebius reported no remaining VM or boot disk after the final run.
