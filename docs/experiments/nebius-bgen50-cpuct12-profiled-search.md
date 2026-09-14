# Nebius profiled MCTS search, B_gen=50 and c_puct=12

## Purpose

This experiment increased both the generation budget and PUCT exploration coefficient
after a B_gen=10 run with `c_puct=1.5` concentrated every root visit on one strategy.
The goals were to measure strategy coverage under stronger exploration and determine
whether structural tensor-core transformations could exceed the previous 9.00x
root-relative speedup.

This report records an experiment; it does not redefine Milestone A behavior.

## Configuration

- Date: 2026-09-14
- Run ID: `abdf92af-ae64-46bf-898d-be9d5b6cef38`
- Trace: `nebius-llm-bgen50-cpuct12.sqlite` (local, intentionally not committed)
- Model: `gpt-5.6-terra`, reasoning effort `medium`
- Generation budget: 50 calls, including repairs
- MCTS: `c_puct=12`, `c_pw=1.0`, `alpha_pw=0.5`, `c_ucb=1.0`,
  `K_max=4`, `max_depth=10`, seed 0
- Repairs: at most one per proposal
- Priors: uniform; `B_prior=0`
- Worker image: `docker.io/saarora/gpu-kernel-mcts:h100-worker-v9`
- Worker: Nebius NVIDIA H100 80GB HBM3 SXM, compute capability 9.0
- CUDA toolkit: 12.8.2; NCU: 2025.1.1

The workload was a fixed 4096 x 4096 x 4096 BF16 GEMM with FP32 accumulation and
BF16 output. A was row-major, B was column-major, and C was row-major. The host
launched one 16 x 16 thread block for every logical 16 x 16 output tile.

## Outcome

- Root median: 28,457.8 microseconds
- Best median: 2,669.9 microseconds
- Best reward: 2.366390
- Root-relative speedup: approximately 10.66x
- Iterations: 37
- Valid non-root nodes: 34
- Lightweight profiles: 12
- LLM usage: 124,940 input tokens and 134,116 output tokens
- Aggregate LLM request latency: 1,732.45 seconds
- Transpositions: none

The separately measured cuBLAS baseline was approximately 185 microseconds. The best
generated kernel remained about 14.4x slower. A cuBLAS-equivalent root-normalized
reward would be near 5.03, compared with 2.37 for this run.

Relative to the B_gen=10, `c_puct=1.5` experiment, the best median improved from
3,136.8 to 2,669.9 microseconds, an approximately 18% improvement.

## Winning path

```text
root
28,458 us, reward 0
  |
  +-- tensor_core_output_tiling, B_gen 17
      2,913 us, reward 2.2791
        |
        +-- reduce_synchronization, B_gen 18
            2,744 us, reward 2.3391
              |
              +-- specialize_workload_shape repair, B_gen 36
                  2,670 us, reward 2.3664
```

The best candidate passed correctness with zero observed maximum and mean error for
the fixed-shape test.

## Winning kernel structure

The winning branch implemented the intended structural changes:

- One owner block represents a group of eight logical 16 x 16 output tiles.
- The owner covers a 32 x 64 output region.
- Its eight warps each own an independent 16 x 16 accumulator tile.
- A and B are staged in double-buffered shared memory.
- BF16 WMMA operations accumulate into FP32 fragments.
- The final specialization uses 16-byte `cp.async` transfers, commit groups, and
  waits to overlap loading a future K tile with work on the current tile.
- The fixed 4096 dimensions and strides are compiled directly into the specialized
  path.
- No cross-warp reduction of partial K accumulations is required.

The pipeline still performs only one 16 x 16 x 16 MMA per warp for each K stage. This
provides limited arithmetic with which to hide operand-transfer and synchronization
latency. That observation motivated the later `tensor_core_multi_accumulator`
strategy.

## NCU progression on the winning path

| Node | Median us | Registers/thread | Occupancy | SM throughput | Tensor pipe | Instructions |
|---|---:|---:|---:|---:|---:|---:|
| Root | 28,458 | 30 | 99.41% | 64.93% | 0.00% | 13.6B |
| Tensor-core output tiling | 2,913 | 48 | 60.87% | 49.61% | 7.34% | 911M |
| Reduced synchronization | 2,744 | 32 | 95.61% | 59.42% | 7.90% | 1.54B |
| Final specialized best | 2,670 | not collected | not collected | not collected | not collected | not collected |

The final best was created late in the search and was never subsequently selected
for expansion, so the version of the search used for this experiment did not profile
it. Final-best profiling before worker release was implemented afterward. Future
runs should therefore retain NCU metrics for the winning leaf even if MCTS never
expands it.

The parent profile shows that output tiling substantially increased tensor-pipe use,
but utilization remained below 8%. High occupancy was not sufficient by itself: the
root had nearly full occupancy and no tensor work. The remaining gap likely requires
more MMA work per staged operand, although lightweight metrics alone are not a
complete bottleneck diagnosis.

## Exploration and strategy coverage

All seven strategies in the configuration used for this run received visits. Root
statistics at completion were:

| Root strategy | Visits | Proposals | Generation calls | Q_mean | Q_max |
|---|---:|---:|---:|---:|---:|
| Tensor-core output tiling | 7 | 3 | 5 | 2.0896 | 2.3664 |
| Shape specialization | 6 | 3 | 5 | 1.8028 | 1.8482 |
| Vectorized memory access | 5 | 3 | 5 | 1.2364 | 1.8484 |
| Coalesced global memory | 4 | 2 | 2 | 1.4492 | 1.8408 |
| Pipeline tensor-core data movement | 4 | 3 | 5 | 1.4143 | 1.7472 |
| Reduce register pressure | 4 | 2 | 2 | 1.4433 | 1.8364 |
| Reduce synchronization | 4 | 2 | 2 | 1.4020 | 1.9345 |

This confirms that `c_puct=12` corrected the short-run coverage problem. Tensor-core
output tiling emerged as the strongest root strategy and supplied the winning branch.
The larger coefficient also continued allocating substantial effort to weaker root
actions, so a lower value such as 6 may provide a better exploration/exploitation
balance in a future 50-call comparison.

## Failures, repairs, and infrastructure

Across 50 generation calls:

- 34 calls produced valid candidates.
- 15 calls produced invalid candidates: nine compile failures, four correctness
  failures, one timeout, and one launch failure.
- 13 calls were repairs.
- One evaluation ended in infrastructure failure.
- At the terminal proposal level, 34 iterations were valid, two were invalid after
  repair, and one ended in infrastructure failure.

Invalid and repair calls consumed `B_gen`. Invalid and infrastructure outcomes did
not become nodes and received no reward backup. The infrastructure failure was
recorded separately rather than treated as evidence against its strategy.

The 68% call-level validity rate indicates that free-form CUDA tensor-core generation
still spends significant budget on compiler and correctness recovery. Repairs raised
the terminal-proposal success rate substantially, but consumed more than one quarter
of the total generation budget.

## Comparison with earlier profiled runs

| Experiment | c_puct | B_gen | Root strategies reached | Best reward | Speedup | Best median |
|---|---:|---:|---:|---:|---:|---:|
| B_gen=10 exploitative | 1.5 | 10 | 1 of 7 | 2.1977 | 9.00x | 3,137 us |
| B_gen=10 exploratory | 12 | 10 | 6 of 7 attempted | 1.9023 | 6.70x | about 4,250 us |
| B_gen=50 exploratory | 12 | 50 | 7 of 7 | 2.3664 | 10.66x | 2,670 us |

The short exploratory run paid for breadth without enough remaining budget to exploit
the discoveries. At 50 calls, broad coverage found a better structural branch and
then refined it to the best observed generated kernel.

## Follow-up work

1. Use `tensor_core_multi_accumulator` to increase MMA work and operand reuse per
   pipeline stage while monitoring register pressure and occupancy.
2. Confirm that the new final-best profiling behavior captures metrics for future
   winning leaves.
3. Compare `c_puct=6` and `c_puct=12` only if the expected information gain justifies
   another expensive run.
4. Add a resumable-search design before attempting to compose multiple independent
   B_gen=50 runs into one B_gen=100 search.
5. Longer-term, investigate WGMMA/TMA and CUTLASS/CuTe mechanisms rather than relying
   exclusively on free-form WMMA rewrites to close the cuBLAS gap.

## Trace contents

The local SQLite trace retains every assembled prompt, fixed API instruction, raw
model response, generated source, evaluation result, token count, latency, profile,
node/edge snapshot, and environment manifest. The trace and generated CUDA files are
local experimental artifacts and are intentionally excluded from version control.
