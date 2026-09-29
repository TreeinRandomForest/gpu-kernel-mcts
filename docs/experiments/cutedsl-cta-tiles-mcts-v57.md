# Independent CuTe three-realization CTA-tile MCTS smoke v57

Date: 2026-09-29

## Purpose

Verify on H100 that progressive widening creates all three validated independent
CTA-tile realizations beneath one semantic `change_cta_tile` edge, without
conflating their states, evaluations, artifacts, or mutation accounting.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cta-tiles-v57`
- workload: `bf16_gemm_4096_h100`
- root: `(64,256,64)`, three-stage SW128
- strategy: `change_cta_tile`
- `B_mut=3`, `B_gen=0`
- `k_max=3`, `max_depth=1`
- profile set: `lightweight_v1`

The existing Nebius H100 VM was reused. A normal authenticated worker service ran
in the v57 container, and the normal `run_mcts_search` orchestration connected over
its HTTP worker protocol. No additional VM was provisioned.

## Result

Run `d9118b04-ae23-4b7a-bf92-973b76b6a536` completed successfully:

| State | Median (us) | Reward |
|---|---:|---:|
| root `(64,256,64)` | 675.119996 | 0.000000 |
| cooperative-M `(128,256,64)` | 449.712008 | 0.406283 |
| narrow-N `(64,128,64)` | 515.680015 | 0.269404 |
| combined-M/N `(128,128,64)` | 620.544016 | 0.084294 |

Trace invariants:

- iterations: 5
- `B_mut`: 3
- `B_gen`: 0
- nodes: 4
- profiles: 2
- valid-only backups: 5
- typed mutation proposal records: 3
- valid proposals: 3
- distinct state keys: 4
- distinct configuration hashes: 4
- distinct binary hashes: 4
- reused nodes: 0

The first three iterations widened the semantic edge with the three deterministic
realizations. The remaining two iterations selected existing cached realizations
and backed them up without GPU reevaluation. The cooperative-M realization remained
the global best; the slower but valid combined-M/N state remained in the DAG.

Local artifacts are `cutedsl-cta-tiles-bmut3-v57-valid.sqlite` and
`cutedsl-cta-tiles-bmut3-v57-best.py`; they are not intended for source control. A
separate partial trace from a sandbox-blocked client connection contains no GPU
search work and must not be used as experimental evidence.
