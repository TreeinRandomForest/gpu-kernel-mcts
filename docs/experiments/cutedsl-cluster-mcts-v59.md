# Independent CuTe cluster mutation MCTS smoke v59

Date: 2026-09-29

## Purpose

Verify that the validated cluster `(1,1) <-> (2,1)` transition is represented as
one deterministic realization beneath the semantic `change_cluster_shape` edge,
with explicit mutation accounting and distinct cached state identity.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cluster-v59`
- workload: `bf16_gemm_4096_h100`
- root: CTA `(64,256,64)`, cluster `(1,1)`, three-stage SW128
- strategy: `change_cluster_shape`
- `B_mut=1`, `B_gen=0`
- `k_max=1`, `max_depth=1`
- profile set: `lightweight_v1`

The existing Nebius H100 VM was reused through the normal authenticated HTTP worker
protocol. No additional VM was provisioned.

## Result

Run `b69f42de-5410-4e1b-b131-619be3bb08a0` completed successfully:

| State | Median (us) | Reward |
|---|---:|---:|
| root cluster `(1,1)` | 675.311983 | 0.000000 |
| B-multicast cluster `(2,1)` | 658.064008 | 0.025873 |

Trace invariants:

- iterations: 1
- `B_mut`: 1
- `B_gen`: 0
- nodes: 2
- profiles: 2
- valid-only backups: 1
- typed mutation proposal records: 1
- distinct state keys: 2
- distinct configuration hashes: 2
- distinct binary hashes: 2
- reused nodes: 0

The clustered child became the global best. The trace is
`cutedsl-cluster-bmut1-v59.sqlite`, with best rendered source
`cutedsl-cluster-bmut1-v59-best.py`; both are local artifacts and are not intended
for source control.
