# CuTe pipeline interaction sweep v8

## Scope

This standalone deterministic grid measures cluster-by-mainloop-pipeline interactions
without changing MCTS state. The CTA tile remains `(128,256)`. Three validated
clusters are crossed with the pinned stage heuristic and explicit A/B mainloop depths
2 and 3, for nine trials under a separate `B_tune`.

All trials ran on one H100 SXM under the fixed `bf16_gemm_4096_h100` correctness and
CUDA-event timing contract. Same-run cuBLAS measured `176.288 us` median.

## Results

All nine candidates compiled, passed exact correctness, produced 30 timing samples,
and consumed one `B_tune` each.

| Cluster | Pipeline | Median latency |
|---|---:|---:|
| `(1,1)` | heuristic | 190.384 us |
| `(1,1)` | 2 | 298.416 us |
| `(1,1)` | 3 | 192.800 us |
| `(1,2)` | heuristic | 189.904 us |
| `(1,2)` | 2 | 307.552 us |
| `(1,2)` | 3 | 193.440 us |
| `(2,1)` | heuristic | **184.896 us** |
| `(2,1)` | 2 | 297.344 us |
| `(2,1)` | 3 | 187.040 us |

The best configuration was `(128,256)`, cluster `(2,1)`, with heuristic staging. It
was `1.0297x` faster than the default configuration and had a latency ratio of
`1.04883` versus cuBLAS. Its `184.896 us` median independently reproduces the prior
`184.880 us` schedule-tuning result.

## Interpretation

Two A/B stages do not hide enough latency for any tested cluster. Three stages improve
substantially but remain slower than the pinned shared-memory heuristic. Cluster
`(2,1)` helps both heuristic and three-stage configurations, demonstrating a real
cluster/pipeline interaction, but it does not change the optimal pipeline choice.

This is calibration evidence, not a reason to re-root global MCTS. The fixed search
root and reward normalization remain unchanged. A future structural dimension should
retain heuristic staging as the current best setting while continuing to log the full
typed representation.
