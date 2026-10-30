# Triton winner BN × warps × scheduling crossover — 2026-10-30

This is a controlled `2 × 2 × 2` scan around the exported GLM-5.3 Triton
winner. It varies only output tile width `BN ∈ {128, 256}`, `num_warps ∈
{4, 8}`, and grouped versus ordinary 2D program scheduling. `BM=128`,
`BK=64`, `GROUP_M=8`, the A/B pointer loop, masks, `tl.dot`, and
`num_stages=3` remain fixed. The grouped/256/8 variant is exactly the saved
winner; grouped/128/8 exactly matches MCTS generation 2 in the local trace.

One Nebius H100 SXM worker evaluated all eight variants in four seeded,
shuffled orders. Every evaluation compiled, passed correctness against the
same-worker cuBLAS reference, and produced 30 CUDA-event timings after 10
warmups. The VM and temporary disk were deleted after completion.

## Result

Each value below is the mean of four 30-sample benchmark medians. The final
column is the standard deviation of those four medians, in microseconds.

| Scheduling | `BN` | Warps | Mean median, µs | Round SD, µs |
| --- | ---: | ---: | ---: | ---: |
| Ungrouped | 128 | 4 | 218.004 | 0.332 |
| Ungrouped | 128 | 8 | 221.872 | 0.713 |
| Ungrouped | 256 | 4 | 1,777.612 | 3.667 |
| Ungrouped | 256 | 8 | **199.816** | 1.058 |
| Grouped | 128 | 4 | 223.712 | 1.147 |
| Grouped | 128 | 8 | 238.480 | 1.272 |
| Grouped | 256 | 4 | 1,773.708 | 3.590 |
| Grouped | 256 | 8 | 200.220 | 0.551 |

For the grouped, 8-warp source family actually traversed by MCTS, increasing
`BN` from 128 to 256 saved **38.260 µs**. The original trace measured
239.616 → 199.712 µs across the corresponding generation-2 and generation-3
nodes. The selected generation-3 action was `change_warp_count`, but its
source changed `BN` only; `num_warps` remained 8. This crossover supports the
tile-width edit as the source of that local improvement, subject to the
ordinary limits of same-worker timing measurements.

The `BN=256, warps=4` combinations were roughly **1.77 ms** under both
scheduling policies, whereas `BN=256, warps=8` was about 200 µs. This is a
large interaction between tile width and warp count, not a useful marginal
average for either parameter. Compiler resources and spill behavior were not
captured for these variants, so the cause of the cliff remains unverified.

At `BN=128, warps=8`, grouping cost 16.608 µs in this scan. At
`BN=256, warps=8`, grouped and ungrouped differed by only 0.404 µs in their
four-round means, smaller than their round-to-round spread. The grouped
scheduler was therefore not required to obtain the roughly 200 µs winner
on this workload, though its effect may change with shape or tile policy.

## Evidence and limits

- Source revision: `d06e2347aa7fee93b7a4c9cc6b9290892daac5ca` on
  `feature/triton-h100`; the experiment runner was uncommitted during the
  run, recorded as `dirty_tree=true`.
- Worker image: `docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
  Device/toolchain: H100 80GB HBM3 SXM, CUDA 12.8.2, Torch 2.7.1+cu128,
  Triton 3.3.1, CUTLASS 4.5.1, NCU 2025.1.1.0.
- Workload: 4096³ BF16 GEMM, row-major A/C, column-major B, seed 0,
  `rtol=atol=0.02`. Startup calibration measured Triton root 822.640 µs,
  cuBLAS 176.480 µs, and CUTLASS 311.856 µs.
- [run.py](run.py) generates the eight exact sources and fixes the round
  order. The eight sibling `.py` files are the measured sources.
  [results.json](results.json) stores the worker manifest, source hashes,
  correctness, and all 960 timing samples.

This scan isolates three factors within the winner's source family. It does
not explain the full root-to-winner improvement because the pointer-loop
rewrite and other source changes were held fixed. It uses one shape, seed,
worker, and toolchain; no model generation or NCU replay occurred during
the scan. The worker manifest has no sampled clock or power-state series.
