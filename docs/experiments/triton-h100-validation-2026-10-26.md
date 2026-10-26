# Triton H100 integration validation — 2026-10-26

The Section 50 backend completed a deterministic one-generation smoke and a
three-generation MCTS search on Nebius H100 SXM. All four generated candidates
across the two runs compiled, passed correctness, and finished timing. The
Nebius VMs and temporary disks were deleted after each completed run.

## Reproducibility

- Source revision: `24ae39a6c33c130e880ca6757d8f155f60b9f649` on
  `feature/triton-h100`.
- Worker image: `docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
- Device: NVIDIA H100 80GB HBM3 SXM, compute capability 9.0.
- Toolchain: CUDA 12.8.2, Torch 2.7.1+cu128, Triton 3.3.1,
  Nsight Compute 2025.1.1.0, CUTLASS 4.5.1.
- Workload: fixed 4096 × 4096 × 4096 BF16 GEMM, row-major A/C and
  column-major B; 10 warmups and 30 CUDA-event samples per node.
- MCTS: `gpt-5.6-sol`, medium reasoning, uniform priors, seed 0,
  `B_gen=3`, one repair allowed, `lightweight_v1` NCU profile policy, and
  `configs/triton_h100_strategies.json`. Run ID:
  `aede3994-82e9-4283-a8c8-42b22f98b866`.
- Smoke run ID: `8e426bdf-711d-4a82-938a-bc395da8d5e7`.

The controller recorded `dirty_tree=true` because an unrelated untracked
`.aws` file was present. That file was never staged, committed, or sent to
the worker. The subsequent ignore rule prevents it from making later run
provenance appear dirty.

## Results

| Run | Candidate | Selected action | Median µs | Reward |
| --- | --- | --- | ---: | ---: |
| Smoke | Triton root | — | 822.992 | 0 |
| Smoke | 64 × 64 × 32 tile | `deterministic_smoke` | 559.376 | 0.386125 |
| MCTS | Triton root | — | 822.448 | 0 |
| MCTS | generation 1 | `improve_b_layout_access` | 234.112 | 1.256486 |
| MCTS | generation 2 | `improve_b_layout_access` | **224.048** | **1.300425** |
| MCTS | generation 3 | `change_warp_count` | 295.920 | 1.022196 |

The best MCTS result is 3.67× faster than its same-run root. Its 30 timing
samples ranged from 217.024 to 233.824 µs, with a 3.991 µs population
standard deviation. These are one-run observations, not an ablation result.
The winning source is in [triton-h100-bgen3-best.py](triton-h100-bgen3-best.py).

MCTS generation 1 and 2 both expanded the root; generation 3 expanded the
generation-2 child. The winning source treats the column-major B buffer as
a physical row-major `[N,K]` view, loads it with a Triton block pointer, and
transposes the tile for `tl.dot`. It also changes the root's
`(BM, BN, BK)=(32,64,32)` and four warps to `(128,128,64)`, eight warps, and
three pipeline stages. The selected action therefore names the main prompt
theme but does not isolate the effect of B access from those coupled changes.

Both smoke nodes and the MCTS root and best node received lazy
`lightweight_v1` NCU profiles. PTX was captured and stored for every valid
node: two artifacts in the smoke trace and four in the MCTS trace. The local
trace files are `triton-h100-smoke-24ae39a.sqlite` and
`triton-h100-mcts-bgen3-24ae39a.sqlite`; SQLite traces remain ignored by Git.

## Gaps found during validation

The first published image lacked Python development headers. Triton's first
JIT attempt failed while compiling its CUDA helper against `Python.h`; image
revision `24ae39a` added the headers. Nebius startup now surfaces a failed
worker health stage promptly and redacts Docker startup diagnostics. Initial
Docker attempts also used a Podman-local digest that did not resolve on Docker
Hub; the successful runs used the verified registry digest above.

The worker ran cuBLAS/CUTLASS calibration during startup, but the controller
did not persist the calibration payload in the trace. Its latencies are
therefore unavailable for this report; the root and candidate comparisons
above use measurements retained per node. Persisting calibration is the
remaining Section 50 evidence gap before treating this integration track as
fully documented.
