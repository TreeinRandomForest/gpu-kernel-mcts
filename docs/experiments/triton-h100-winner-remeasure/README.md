# Triton H100 winner remeasurement — 2026-10-30

The exported GLM-5.3 `B_gen=3` winner was re-evaluated on one fresh Nebius
H100 SXM worker against the packaged Triton root, cuBLAS, and CUTLASS. The
runner used four rotated rounds, each with 10 warmups and 30 CUDA-event timing
samples per implementation. Triton root and winner were compiled and checked
for correctness in every round; both vendor baselines also passed correctness.
The VM and temporary disk were deleted after the run.

## Result

| Implementation | Round medians, µs | Mean of round medians, µs | SD across rounds, µs |
| --- | --- | ---: | ---: |
| Triton root | 820.288, 822.048, 822.112, 821.248 | 821.424 | 0.853 |
| Exported winner | 201.920, 202.400, 200.768, 203.136 | **202.056** | 0.994 |
| cuBLAS | 180.224, 179.360, 180.096, 179.104 | **179.696** | 0.548 |
| CUTLASS | 311.056, 310.944, 311.520, 311.168 | 311.172 | 0.249 |

The winner was 4.065× faster than the same-worker Triton root. Its paired
gap to cuBLAS was 21.696, 23.040, 20.672, and 24.032 µs across the four
rounds: a mean of 22.360 µs, or 12.44% of the cuBLAS mean. The new winner
mean is 2.344 µs above the original search's 199.712 µs median. The result
supports a roughly 200 µs kernel and a persistent gap to cuBLAS on this
worker; it does not estimate variability across GPUs, driver stacks, or
input distributions.

## Reproduction and evidence

- Source revision: `d06e2347aa7fee93b7a4c9cc6b9290892daac5ca` on
  `feature/triton-h100`. The measurement script was uncommitted at execution,
  recorded as `dirty_tree=true` in the worker manifest.
- Winner source: [exported Python file](../triton-h100-glm53-bgen3-best.py).
  Packaged root: `src/kernel_mcts/benchmarks/kernels/bf16_gemm_triton.py`.
- Worker image: `docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
- Device/toolchain: NVIDIA H100 80GB HBM3 SXM, compute capability 9.0;
  CUDA 12.8.2, Torch 2.7.1+cu128, Triton 3.3.1, CUTLASS 4.5.1,
  Nsight Compute 2025.1.1.0.
- Workload: fixed 4096 × 4096 × 4096 BF16 GEMM, row-major A/C and
  column-major B, FP32 accumulation, seed 0, `rtol=atol=0.02`.
- Startup calibration, before the rotated rounds, measured Triton root
  822.448 µs, cuBLAS 180.720 µs, and CUTLASS 312.384 µs.
- [run.py](run.py) contains the exact round order and worker lifecycle.
  [results.json](results.json) retains the environment manifest, all 480
  rotated-round timing samples plus startup calibration samples, per-round
  correctness, source hashes, and run ID. No model generation or NCU replay
  was used during this remeasurement.

The Triton figures come from the worker's Python/Triton harness; vendor
figures come from its pinned C++ baseline executable in the same container.
All are CUDA-event timings after warmup. The worker manifest did not include
GPU clock or power-state samples, so the report makes no clock-normalized
claim. The runner used distinct evaluation IDs to force a new verification
and timing pass rather than worker-cache reuse; Triton's JIT cache may still
reuse compiled code.
