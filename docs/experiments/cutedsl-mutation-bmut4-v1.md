# CuTe DSL mutation search with B_mut=4

## Purpose

This run extends the one-proposal hardware smoke test to four deterministic CuTe
mutations. It checks PUCT allocation across the initial tile and cluster strategies,
retention of valid slower schedules, and transposition reuse under the remote worker
and normal SQLite trace path.

## Configuration

- Date: 2026-09-20
- Run ID: `75fba105-406f-4741-8e3d-f90bb18bcc0e`
- Workload: `bf16_gemm_4096_h100`
- GPU: NVIDIA H100 80GB HBM3, SXM, compute capability 9.0
- Provider: Nebius
- Worker image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-search-v1`
- CUDA toolkit: 12.9.1
- CUTLASS/CuTe DSL: 4.5.1
- NCU: 2025.2.1
- `B_gen`: 0/0
- `B_mut`: 4/4
- Seed: 0
- `c_puct`: 1.5
- Profiles: root and final best using `lightweight_v1`

## Proposals and results

All four proposals passed static validation, JIT compilation, correctness, and
benchmark evaluation.

| Iteration | Strategy | Resulting schedule | Median latency | Reward | State result |
|---:|---|---|---:|---:|---|
| 1 | `change_cluster_shape` | tile `(128,256)`, cluster `(1,2)` | 188.544 us | 0.0177472 | new node |
| 2 | `change_cta_tile` | tile `(64,128)`, cluster `(1,1)` | 313.008 us | -0.4891502 | new node |
| 3 | `change_cluster_shape` | tile `(128,256)`, cluster `(2,1)` | 187.488 us | 0.0233637 | new best |
| 4 | `change_cluster_shape` | tile `(128,256)`, cluster `(1,1)` | 191.920 us | 0 | cached root |

The fourth transition was the inverse of iteration 3: it changed cluster `(2,1)`
back to `(1,1)`. The transposition table reused the existing root node, so four valid
proposals produced four unique nodes including the root. The explicit proposal still
consumed `B_mut`, while the cached state avoided another GPU evaluation.

The best reward corresponds to `exp(0.0233637) = 1.02364x`, approximately a 2.4%
speedup over this run's measured root. The best `(128,256)` tile and `(2,1)` cluster
matches the best schedule found by the earlier exhaustive nine-schedule standalone
tuning experiment. The exact timings differ between runs, so this agreement supports
the schedule choice rather than a cross-run latency comparison.

The `(64,128)` tile was valid but substantially slower. Keeping and backing up that
measured state confirms that locally slower valid kernels are not pruned.

## Profile comparison

The root and best node both used 154 registers per thread and approximately 12.4%
achieved occupancy. The best node's recorded tensor-pipe utilization was 90.18%
versus 91.89% for the root, and SM throughput was 85.06% versus 86.78%. These metrics
do not individually explain the latency improvement; the result reinforces that NCU
metrics are diagnostic prompt context while unprofiled CUDA-event latency defines
reward.

## Provenance limitations

The trace recorded Git commit `caa36bdc0cda3a8a97edce04ee6508f27ab50ac9`, but the
controller reported a dirty tree and the environment manifest did not include an
immutable image digest. This remains behavioral validation rather than a fully
reproducible performance experiment. The SQLite trace and generated best program are
local artifacts and are not committed.
