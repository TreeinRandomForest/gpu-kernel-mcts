# Independent CuTe CTA-tile MCTS smoke v55

Date: 2026-09-29

## Purpose

Verify on H100 that progressive widening can create both validated deterministic
realizations beneath one semantic `change_cta_tile` edge without conflating their
states, evaluations, artifacts, or mutation accounting.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cta-n-v55`
- git commit: `fa1e5f6`
- workload: `bf16_gemm_4096_h100`
- root: `(64,256,64)`, three-stage SW128
- strategy: `change_cta_tile`
- `B_mut=2`, `B_gen=0`
- `k_max=2`, `max_depth=1`
- profile set: `lightweight_v1`

The existing Nebius H100 VM was reused. One container ran the normal authenticated
worker service, and a second invoked the normal `run_mcts_search` orchestration over
its HTTP protocol. No new VM was provisioned.

## Result

Run `278e7159-f47e-4137-bdbc-f92748cd353a` completed successfully:

| State | Median (us) | Reward |
|---|---:|---:|
| root `(64,256,64)` | 675.711989 | 0.000000 |
| cooperative-M `(128,256,64)` | 449.999988 | 0.406519 |
| narrow-N `(64,128,64)` | 514.815986 | 0.271957 |

Trace invariants:

- iterations: 2
- `B_mut`: 2
- `B_gen`: 0
- nodes: 3
- profiles: 2
- valid-only backups: 2
- typed mutation proposal records: 2
- distinct state keys: 3
- distinct binary hashes: 3
- reused nodes: 0

The cooperative-M realization remained the global best. The result also reproduces
the standalone narrow-N improvement within normal measurement variation.

Local artifacts are `cutedsl-cta-tiles-bmut2-v55.sqlite` and
`cutedsl-cta-tiles-bmut2-v55-best.py`; they are not intended for source control.
