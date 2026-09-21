# CuTe pipeline mutation validation

## Scope

This experiment validates the corrected typed `change_pipeline_stages` mechanism on
one Nebius H100 SXM. The backend subprocess now forwards the complete typed
representation rather than reducing it to CTA tile and cluster schedule.

The fixed workload was `bf16_gemm_4096_h100`. The mutation-only search used seed 0,
`B_gen=0`, `B_mut=6`, `c_puct=1.5`, `c_pw=1.0`, `alpha_pw=0.5`, and `k_max=4`.
Run ID: `8304b288-b7bd-4117-90b5-73688968c96f`.

## Standalone feasibility

All candidates compiled, passed exact correctness, produced 30 timing samples, and
had distinct runtime-artifact and normalized-IR hashes.

| A/B stages | Median latency |
|---:|---:|
| 2 | 299.136 us |
| 3 | 191.792 us |
| 4 | 190.624 us |

Stage 2 is clearly under-pipelined. Stages 3 and 4 differ by about 0.6%, which is too
small to interpret confidently from one run. Stage 4 remains excluded from the root
mutation neighborhood because it is the heuristic root's effective stage count.

## MCTS trace

The search completed six valid mutation attempts with no LLM calls, three profile
calls, five unique nodes, and no infrastructure failures.

| B_mut | Strategy | Candidate | Median | Reward | Result |
|---:|---|---|---:|---:|---|
| 1 | `change_cluster_shape` | cluster `(1,2)` | 188.816 us | 0.016556 | new node |
| 2 | `change_pipeline_stages` | stage 2 | 297.392 us | -0.437722 | new node |
| 3 | `change_cta_tile` | tile `(64,128)` | 316.832 us | -0.501043 | new node |
| 4 | `change_cluster_shape` | cluster `(2,1)` | 185.344 us | 0.035115 | new best |
| 5 | `change_cluster_shape` | inverse to cluster `(1,1)` | 191.968 us | 0 | cached root |
| 6 | `change_cluster_shape` | inverse to cluster `(1,1)` | 191.968 us | 0 | cached root |

The stage-2 representation, binary fingerprint, and MCTS state were distinct from the
heuristic root. This establishes that the typed pipeline field reaches JIT execution
and state identity. Stage 3 was validated standalone but was not selected within the
six-attempt MCTS budget: after the slow stage-2 realization, PUCT did not revisit that
strategy soon enough to widen it to a second realization. The run therefore validates
pipeline mutation integration, not exhaustive pipeline exploration.
