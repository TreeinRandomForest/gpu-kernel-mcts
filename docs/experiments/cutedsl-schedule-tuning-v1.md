# CuTe DSL schedule tuning v1

## Purpose

This experiment exhaustively evaluated the initial typed CuTe DSL schedule space
for the fixed `bf16_gemm_4096_h100` workload. Trials ran sequentially on one H100
SXM worker and used the repository's correctness and CUDA-event timing contract.
The raw `cutedsl-tuning.json` report remains a local generated artifact.

## Environment

- Date: 2026-09-19
- GPU: NVIDIA H100 80GB HBM3 SXM, compute capability 9.0
- Driver: 580.173.02
- CUDA toolkit and NVCC: 12.9
- CUTLASS and CuTe DSL: 4.5.1
- PyTorch: 2.8.0+cu129
- Warmups: 10
- Measurements: 30

## Results

The run attempted 12 configurations. Nine compiled and passed correctness. The
three `(256, 128)` CTA-tile configurations failed before measurement because the
pinned implementation restricts tile M to 64 or 128. They correctly consumed
`B_tune` in this run because the constraint was discovered during evaluation;
the static schedule space now excludes them.

| Configuration | Median latency | Relative result |
|---|---:|---:|
| cuBLAS | 177.056 us | 1.000x latency |
| Best CuTe: tile `(128, 256)`, cluster `(2, 1)` | 184.880 us | 1.044x cuBLAS latency |
| Default CuTe: tile `(128, 256)`, cluster `(1, 1)` | 193.024 us | 1.090x cuBLAS latency |

The best schedule was 1.044x faster than the default CuTe schedule, a 4.4%
latency improvement. It remained 4.4% slower than cuBLAS. Its mean latency was
186.516 us with a 3.546 us population standard deviation; the observed range was
183.232-196.544 us. Correctness matched the BF16 cuBLAS reference with zero
observed maximum and mean error.

## Interpretation

A small typed schedule space recovered meaningful performance without source
generation and reduced the earlier default CuTe-to-cuBLAS gap by roughly half.
Cluster shape mattered for the best CTA tile: changing only the cluster from
`(1, 1)` to `(2, 1)` produced the winning configuration. The next tuning-space
extension should inspect the pinned implementation's actual constraints before
exposing pipeline, TMA-layout, WGMMA-arrangement, swizzle, or epilogue controls.
