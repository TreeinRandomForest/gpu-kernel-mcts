# CuTe WGMMA in-flight validation v11

## Purpose

This standalone H100 experiment tested whether changing the pinned Hopper GEMM's
local `k_pipe_mmas` value from 1 to 2 creates a distinct effective kernel. The
SHA-pinned source transformer changed only that assignment. The experiment crossed
the two values with the pinned pipeline heuristic and explicit mainloop stage 3.
It was a pre-search capability test and consumed no MCTS budget.

## Results

| Pipeline | In-flight groups | Correctness | Median latency | Normalized IR | Fatbin |
|---|---:|---:|---:|---|---|
| heuristic | 1 | exact pass | `193.504 us` | `0f69b81e...` | `1c41a324...` |
| heuristic | 2 | exact pass | `192.176 us` | `0f69b81e...` | `1c41a324...` |
| stage 3 | 1 | exact pass | `192.512 us` | `8578985a...` | `5cc04e20...` |
| stage 3 | 2 | exact pass | `191.936 us` | `8578985a...` | `5cc04e20...` |

All runs reported zero maximum and mean correctness error. Pipeline stage 3 produced
a different effective kernel from the heuristic, as expected. Within either pipeline
setting, however, changing the in-flight value changed the typed configuration and
generated source but not the normalized compiler IR or embedded fatbin.

## Interpretation

The approximately 0.7% heuristic timing difference and 0.3% stage-3 difference are
measurement noise because both pairs executed identical compiled kernels. For this
fixed loop, toolchain, and workload, `k_pipe_mmas=2` does not expose an independent
effective degree of freedom. It may be canonicalized away or may not alter the wait
behavior with the number of WGMMA operations actually outstanding.

## Decision

Do not add `wgmma_inflight_groups` as an MCTS strategy or typed mutation. It has been
removed from canonical `CuteGemmProgram` search state after preserving this evidence:
semantically identical effective kernels must transpose rather than occupy separate
nodes. The guarded transformer remains available only through the standalone
`wgmma-inflight-diagnostic` mode; its output cannot define search identity.
