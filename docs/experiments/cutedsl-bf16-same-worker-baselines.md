# CuTe DSL BF16 same-worker baseline comparison

## Purpose

This experiment evaluates the pinned Hopper CuTe DSL dense-GEMM example against
cuBLAS and the repository's fixed CUTLASS baseline on one H100 worker. All three
implementations use the fixed `bf16_gemm_4096_h100` workload and are measured in one
container so the result does not mix hardware or toolchain environments.

An earlier adapter invoked NVIDIA's aggregate benchmark helper separately for every
sample and reported a suspicious CuTe median of 280.624 microseconds. The corrected
adapter creates one workspace, performs ten warmups, synchronizes once, and then
records thirty individual launches with CUDA events. This report uses only the
corrected measurement.

## Configuration

- Date: 2026-09-18
- Raw report: `cutedsl-comparison-v5.json` (local, not committed)
- Execution order: cuBLAS, fixed CUTLASS, CuTe DSL
- Worker: NVIDIA H100 80GB HBM3 SXM, compute capability 9.0
- GPU UUID: `GPU-46275b3b-11c3-b9f8-c36b-f6af69164b3d`
- Driver: 580.173.02
- CUDA toolkit: 12.9.1; NVCC 12.9
- CUTLASS and CuTe DSL: 4.5.1
- PyTorch: 2.8.0+cu129
- Worker image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v5`
- CuTe CTA tile: 128 x 256
- CuTe cluster: 1 x 1

The workload is a 4096 x 4096 x 4096 GEMM with BF16 inputs and output and FP32
accumulation. A is row-major, B is column-major, and C is row-major. Inputs reproduce
the repository's seed-zero `std::mt19937` BF16 stream. Correctness uses the same
`cublasGemmEx` FP32-compute reference and `atol=rtol=0.02`. Each implementation uses
ten warmups and thirty CUDA-event measurements.

## Results

All implementations passed correctness. CuTe DSL matched the BF16 cuBLAS reference
exactly in the reported output, with zero observed maximum and mean error.

| Implementation | Median us | Mean us | Stddev us | Min us | Max us | Latency / cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| cuBLAS | 178.096 | 178.133 | 1.106 | 176.416 | 181.760 | 1.000 |
| CuTe DSL | 192.544 | 193.694 | 4.539 | 190.144 | 215.584 | 1.081 |
| Fixed CUTLASS | 310.896 | 310.910 | 0.997 | 308.928 | 314.304 | 1.746 |

The fixed CuTe DSL kernel was 8.1% slower than cuBLAS by median latency and 1.615x
faster than the fixed CUTLASS configuration. The CUTLASS number represents one older
SM80-style fixed configuration compiled for H100, not the performance potential of an
autotuned Hopper SM90 CUTLASS kernel.

## Timing-adapter finding

The continuous timing sequence removed the previous 280-microsecond plateau. The
first corrected CuTe sample was 215.584 microseconds, the second was 200.096, and the
remaining samples were predominantly around 190-193 microseconds. This residual
startup transient raises the mean and standard deviation but has little effect on the
median. Future experiments may capture operating state around each implementation or
increase warmups, but should retain all measured samples rather than silently discard
early observations.

## Interpretation

The experiment establishes that a fixed, official Hopper WGMMA/TMA CuTe DSL kernel
can approach cuBLAS on the current workload without kernel search. It does not yet
justify MCTS integration. The next useful CuTe experiment is separately budgeted,
typed schedule autotuning over a small legal configuration space, followed by a
comparison with a strong Hopper-specific CUTLASS reference.

The report manifest captured the GPU identity and primary library/toolchain versions,
but its image digest, project git commit, backend implementation hash, and dirty-tree
state were unset. Future automated reports should populate those provenance fields.
