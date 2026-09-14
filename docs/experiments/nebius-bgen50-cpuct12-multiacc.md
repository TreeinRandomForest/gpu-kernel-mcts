# Nebius profiled MCTS search, B_gen=50 with multi-accumulator strategy

## Purpose

This experiment repeated the B_gen=50, `c_puct=12` search after adding
`tensor_core_multi_accumulator`. The new strategy was the only change affecting MCTS
selection. Final-best profiling was also enabled, but runs only after budget
exhaustion and therefore does not affect the search path.

The goal was to increase tensor-core work per staged operand and determine whether
multiple accumulator fragments per warp could improve on the preceding 2,669.9
microsecond best.

## Configuration

- Date: 2026-09-14
- Run ID: `157f75b8-5404-4238-a1a2-eb7e41a6b343`
- Trace: `nebius-llm-bgen50-cpuct12-multiacc.sqlite` (local, not committed)
- Model: `gpt-5.6-terra`, reasoning effort `medium`
- Generation budget: 50 calls, including repairs
- MCTS: `c_puct=12`, `c_pw=1.0`, `alpha_pw=0.5`, `c_ucb=1.0`,
  `K_max=4`, `max_depth=10`, seed 0
- Strategies: eight uniformly weighted actions, including
  `tensor_core_multi_accumulator`
- Repairs: at most one per proposal
- Priors: uniform; `B_prior=0`
- Worker image: `docker.io/saarora/gpu-kernel-mcts:h100-worker-v9`
- Worker: Nebius NVIDIA H100 80GB HBM3 SXM, compute capability 9.0

The fixed workload was 4096 x 4096 x 4096 BF16 GEMM with FP32 accumulation and BF16
output. A was row-major, B was column-major, and C was row-major. The host launch used
a 16 x 16 thread block for every logical 16 x 16 output tile.

## Outcome

- Root median: 28,458.6 microseconds
- Best median: 1,375.6 microseconds
- Best reward: 3.029539
- Root-relative speedup: approximately 20.69x
- Iterations: 33
- Valid non-root nodes: 28
- Lightweight profiles: 12, including the final best
- LLM usage: 136,902 input tokens and 145,469 output tokens
- Aggregate LLM request latency: 1,770.61 seconds
- Transpositions: none

The new best was approximately 1.94x faster than the preceding 2,669.9 microsecond
generated best. It remained about 7.4x slower than the separately measured
approximately 185 microsecond cuBLAS baseline.

## Winning path

```text
root
28,459 us, reward 0
  |
  +-- tensor_core_multi_accumulator, B_gen 20
      2,603 us, reward 2.3916
        |
        +-- reduce_synchronization, B_gen 33
            2,346 us, reward 2.4958
              |
              +-- tensor_core_multi_accumulator repair, B_gen 49
                  1,376 us, reward 3.0295
```

The best candidate passed the fixed-shape correctness test with zero observed maximum
and mean error.

## Kernel evolution

### First multi-accumulator node

The B_gen 20 root child grouped logical blocks into a 64 x 128 output region. Its
eight warps each owned a 32 x 32 warp-level output tile represented by four independent
16 x 16 FP32 accumulator fragments. Each staged A/B tile therefore fed four WMMA
operations per warp instead of one.

This candidate used shared A and B tiles but synchronized around every 16-element K
stage. It reached 2,603 microseconds and immediately became the strongest root
realization.

### Reduced-synchronization parent

The B_gen 33 realization retained four accumulators per warp but loaded WMMA operands
directly from global memory and removed the per-K shared-memory barriers. It reduced
runtime to 2,346 microseconds. Its final-best child later restored shared staging in a
more effective pipelined form.

### Final repaired candidate

The B_gen 48 initial candidate failed compilation. Its B_gen 49 repair produced the
final best:

- Physical block output tile: 64 x 128.
- Warp output tile: 32 x 32 using four FP32 accumulator fragments.
- K stage: 32 elements, containing two 16-wide WMMA steps.
- A and B: aligned, double-buffered shared-memory tiles.
- Transfers: cooperative 16-byte `cp.async` loads.
- Pipeline: future A/B stage committed while the current stage is consumed.
- Output: each warp stores its four independent tiles without a cross-warp reduction.
- Ownership: only one block in each grouped logical region performs work.

The kernel raises register use, but the additional accumulator reuse and improved
operand delivery more than compensate for the occupancy reduction.

## NCU progression

The final-best profiling feature ran after budget exhaustion with
`trigger="final_best"`, so the winning leaf has complete lightweight metrics.

| Node | Median us | Registers/thread | Occupancy | SM throughput | Tensor pipe | DRAM | L2 | L1 | Long scoreboard | Instructions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Root | 28,459 | 30 | 99.42% | 64.92% | 0.00% | 1.41% | 11.35% | 94.87% | 4.78 | 13.6B |
| First multi-accumulator | 2,603 | 72 | 35.47% | 45.10% | 9.73% | 2.70% | 23.02% | 57.76% | 4.00 | 1.03B |
| Reduced synchronization | 2,346 | 64 | 45.82% | 12.47% | 9.21% | 2.33% | 37.32% | 99.57% | 56.44 | 257M |
| Final best | 1,376 | 72 | 35.34% | 30.13% | 15.92% | 7.52% | 26.08% | 94.52% | 7.97 | 379M |

Compared with its direct parent, the final best nearly doubled tensor-pipe
utilization, increased SM and DRAM throughput, and sharply reduced long-scoreboard
stall pressure. Instruction count increased, demonstrating that minimizing
instructions was not the relevant objective: the added work enabled substantially
better tensor-core and memory-system utilization.

The final benchmark was stable: median 1,375.632 microseconds, mean 1,376.143,
standard deviation 3.475, minimum 1,371.648, and maximum 1,386.208 across 30 samples.

## Strategy behavior

All eight strategies received visits. Aggregate statistics across the DAG were:

| Strategy | Visits | Proposals | Generation calls | Repairs | Valid terminal proposals | Invalid terminal proposals | Maximum Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-accumulator tensor core | 8 | 8 | 14 | 6 | 5 | 3 | 3.0295 |
| Reduce synchronization | 7 | 4 | 6 | 2 | 4 | 0 | 3.0295 |
| Reduce register pressure | 6 | 5 | 7 | 2 | 4 | 1 | 1.8427 |
| Tensor-core output tiling | 5 | 3 | 6 | 3 | 3 | 0 | 2.3907 |
| Vectorized memory access | 4 | 4 | 5 | 1 | 4 | 0 | 2.3907 |
| Pipeline tensor-core movement | 4 | 3 | 4 | 1 | 3 | 0 | 1.8350 |
| Shape specialization | 3 | 2 | 3 | 1 | 2 | 0 | 1.8508 |
| Coalesced global memory | 3 | 4 | 5 | 1 | 3 | 1 | 1.8508 |

At the root, the new strategy received six visits, five proposals, and eight calls.
It ended with `Q_mean=2.4019` and `Q_max=3.0295`, making it both the most visited and
highest-valued root action.

The result validates the strategy hypothesis, but also shows that this action is hard
for the model to implement. Only five of its fourteen calls across the DAG were valid;
nine were compile or correctness failures. Repairs were essential to both the first
valid realizations and the final best.

## Failure and repair behavior

Across all strategies:

- 28 calls produced valid candidates.
- 22 calls produced invalid candidates.
- Invalid attempts comprised 17 compile failures and five correctness failures.
- 17 of the 50 calls were repairs.
- At the proposal/iteration level, 28 proposals ended valid and five remained invalid
  after their allowed repair.
- No infrastructure failure occurred.

The call-level validity rate was 56%, lower than the preceding run's 68%. The new
strategy accounted for a substantial portion of the additional failures. Nevertheless,
its successful realizations produced the largest performance improvement observed so
far, illustrating why invalid realizations should not directly penalize a strategy's
Q statistics.

## Controlled comparison

The search-affecting configuration matched the preceding B_gen=50 experiment except
for adding the eighth strategy. Final-best profiling occurred only after selection and
backup were complete.

| Experiment | Strategies | c_puct | B_gen | Best reward | Speedup | Best median |
|---|---:|---:|---:|---:|---:|---:|
| Previous profiled search | 7 | 12 | 50 | 2.3664 | 10.66x | 2,670 us |
| Multi-accumulator search | 8 | 12 | 50 | 3.0295 | 20.69x | 1,376 us |

This single run does not estimate statistical repeatability across stochastic LLM
searches, but it demonstrates that the added action can generate a materially better
kernel under the same budget and exploration coefficient.

## Remaining gap and follow-up

Tensor-pipe utilization increased to 15.92%, a substantial improvement but still far
from sustained hardware utilization. Potential next investigations include:

1. Increase reuse further with additional accumulator fragments or larger K stages,
   while watching the already high 72-register footprint and 35% occupancy.
2. Inspect whether `cp.async.wait_group 0` and block-wide synchronization leave more
   overlap available through deeper buffering or multiple in-flight groups.
3. Add more precise compile guidance for supported WMMA BF16 types and accumulator
   stores to reduce the new strategy's failure rate.
4. Compare `c_puct=6` against 12 with the same eight-strategy configuration if another
   expensive controlled run is justified.
5. Longer-term, evaluate H100-native WGMMA/TMA and CUTLASS/CuTe implementations to
   close the remaining approximately 7.4x cuBLAS gap.

## Trace contents

The local SQLite trace retains every assembled prompt, fixed API instruction, raw
model response, generated source, compiler and correctness evidence, benchmark,
profile, token count, node/edge snapshot, and environment manifest. The trace and
generated CUDA sources remain local artifacts and are intentionally excluded from
version control.
