# Nebius Sol versus Terra profiled MCTS comparison

## Purpose

This report compares two profiled MCTS searches using different OpenAI models under
the same recorded search budget and MCTS configuration. It is an experiment report,
not a change to the Milestone A specification.

## Runs

| | Sol | Terra |
|---|---|---|
| Date | 2026-09-16 | 2026-09-14 |
| Run ID | `bdd6f080-e24e-4c84-be31-a5ac09583226` | `157f75b8-5404-4238-a1a2-eb7e41a6b343` |
| Trace | `nebius-sol-bgen50-cpuct12.sqlite` | `nebius-llm-bgen50-cpuct12-multiacc.sqlite` |
| Model | `gpt-5.6-sol` | `gpt-5.6-terra` |

Both traces record the fixed `bf16_gemm_4096_h100` workload, generation budget 50,
seed 0, and identical MCTS parameters: `c_puct=12`, `c_pw=1`, `alpha_pw=0.5`,
`c_ucb=1`, `K_max=4`, `max_depth=10`, and at most one repair. Both used an NVIDIA
H100 80GB HBM3 SXM worker and NCU 2025.1.1.

## Outcome

| Metric | Sol | Terra |
|---|---:|---:|
| Root median | 28,245.9 microseconds | 28,458.6 microseconds |
| Best median | **1,172.8 microseconds** | 1,375.6 microseconds |
| Best reward | **3.1816** | 3.0295 |
| Root-relative speedup | **24.09x** | 20.69x |
| Valid generations | **36/50 (72%)** | 28/50 (56%) |
| Compile failures | 12 | 17 |
| Correctness failures | 2 | 5 |
| Generation that found the best | **37** | 49 |
| Iterations | 36 | 33 |
| Profile calls | 14 | 12 |
| Input tokens | 123,918 | 136,902 |
| Output tokens | 176,972 | 145,469 |
| Aggregate LLM latency | 2,789.2 seconds | 1,770.6 seconds |

Sol found a kernel approximately 14.7% faster than Terra's best, while producing
eight more valid kernels. It used 6.6% more total tokens and 57.5% more aggregate LLM
request time.

## Best speedup by generation budget

| B_gen | Sol | Terra |
|---:|---:|---:|
| 5 | **7.93x** | 4.13x |
| 10 | **7.93x** | 6.23x |
| 20 | 7.93x | **10.93x** |
| 30 | **18.29x** | 10.93x |
| 40 | **24.09x** | 12.13x |
| 50 | **24.09x** | 20.69x |

Sol reached its final best at generation 37. Terra's largest late improvement arrived
at generation 49.

## Winning paths

Sol's shortest root-to-best path was:

```text
root
  -> reduce_synchronization                reward 1.8311, speedup 6.24x
  -> tensor_core_multi_accumulator         reward 3.1816, speedup 24.09x
```

Terra's shortest root-to-best path was:

```text
root
  -> tensor_core_multi_accumulator         reward 2.3916, speedup 10.93x
  -> reduce_synchronization                reward 2.4958, speedup 12.13x
  -> tensor_core_multi_accumulator         reward 3.0295, speedup 20.69x
```

The repeated combination of synchronization reduction and multiple independent
tensor-core accumulators is evidence that these transformations are complementary.
Every valid `tensor_core_multi_accumulator` application improved its parent in both
runs: seven of seven for Sol and five of five for Terra. Sol generated this strategy
validly in 70% of its calls, compared with 35.7% for Terra.

## Best-node NCU comparison

| Metric | Sol best | Terra best |
|---|---:|---:|
| Registers per thread | 72 | 72 |
| Achieved occupancy | 34.97% | 35.34% |
| SM throughput | **58.15%** | 30.13% |
| Tensor-pipe utilization | **19.25%** | 15.92% |
| L2 throughput | **56.09%** | 26.08% |
| Long-scoreboard warps per issue | **0.86** | 7.97 |
| Instructions executed | 630.9 million | **378.6 million** |

The two kernels have essentially identical occupancy and register use. Sol's kernel
is faster despite executing more instructions. Its higher SM, tensor-pipe, and L2
throughput, together with its much lower long-scoreboard value, suggests more useful
parallel work and less waiting on memory dependencies. This is an interpretation of
the recorded counters, not a causal proof.

Sol's winning kernel uses eight warps, four accumulators per warp, a 64 by 128 output
block, double-buffered shared memory, `cp.async` staging, and WMMA tensor-core
operations.

## Remaining cuBLAS gap

The separately measured cuBLAS median recorded in the earlier experiment reports was
approximately 185.4 microseconds. Sol's best remains approximately 6.3x slower than
cuBLAS, although it is approximately 24.1x faster than the root.

## Interpretation limits

This comparison controls the recorded workload, generation budget, seed, MCTS
parameters, hardware class, and profiler version. It is not a definitive model
benchmark:

- LLM sampling remains stochastic even with the same MCTS seed.
- The traces do not record a Git commit or dirty-tree state, so exact code and prompt
  equivalence cannot be established.
- The newer Sol trace contains exact PUCT and UCB candidate tables that the older
  Terra trace predates, preventing symmetric decision-level comparison.
- One run per model is insufficient to estimate variance.

A stronger comparison should repeat each model across multiple seeds while recording
the Git revision and clean/dirty state.
