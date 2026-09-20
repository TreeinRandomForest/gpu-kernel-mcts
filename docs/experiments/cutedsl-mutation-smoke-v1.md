# CuTe DSL mutation-search smoke validation

## Purpose

This guarded run validates the remote Milestone B path rather than establishing a
performance result. It exercises Nebius provisioning, CuTe worker selection, root
JIT/evaluation, NCU profiling, one deterministic typed mutation, valid-only backup,
SQLite persistence, best-program export, and VM cleanup without making an LLM call.

## Configuration

- Date: 2026-09-20
- Run ID: `6621e67e-6354-42a6-95eb-6402c20b3ee7`
- Workload: `bf16_gemm_4096_h100`
- GPU: NVIDIA H100 80GB HBM3, SXM, compute capability 9.0
- Provider: Nebius
- Worker image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-search-v1`
- CUDA toolkit: 12.9.1
- CUTLASS/CuTe DSL: 4.5.1
- NCU: 2025.2.1
- Generator: deterministic CuTe mutation
- `B_gen`: 0/0
- `B_mut`: 1/1
- Seed: 0
- Profiles: 2 using `lightweight_v1`

## Result

PUCT selected `change_cluster_shape`. The deterministic mutation changed cluster
shape from `(1, 1)` to `(1, 2)` while retaining the `(128, 256)` CTA tile. Compilation,
correctness, and benchmark evaluation all passed, so the candidate became the second
MCTS node and its measured reward was backed up.

| State | Median latency | Reward |
|---|---:|---:|
| Root: cluster `(1, 1)` | 192.144 us | 0 |
| Child: cluster `(1, 2)` | 189.152 us | 0.0156942 |

The reward equals `log(192.144 / 189.152)`. This is approximately a `1.0158x`
speedup, or 1.6%, in this single validation run. It is not evidence of a stable
performance difference without repeated controlled measurements.

The search completed one iteration with two valid nodes, one deterministic proposal,
no LLM generations, and two profiles. The local trace was
`cutedsl-mutation-smoke.sqlite`; generated traces and best-program outputs remain
local artifacts and are not committed.

## Provenance limitations

The controller recorded Git commit `c21689b762d3d8e28c9bbfa5503a3764a2e6b170`, but
the image was built from a dirty tree and the manifest did not capture a registry
image digest. The run therefore validates behavior and wiring but is not a fully
reproducible benchmark artifact. A later performance experiment should use a clean
commit, immutable image digest, multiple repetitions, and same-worker baseline
measurements.
