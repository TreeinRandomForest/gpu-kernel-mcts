# CuTe TMA load-policy validation v12

## Purpose

This standalone H100 experiment tested whether the pinned clustered GEMM could expose
TMA multicast as one bounded copy-policy control. Both points used CTA tile
`(128,256)` and cluster `(2,1)`. In the pinned implementation, B is multicast across
the two CTAs while A remains a single-CTA load. This was a pre-search diagnostic and
consumed no MCTS or tuning budget.

## Results

| TMA load policy | Outcome | Median latency |
|---|---|---:|
| `auto_multicast` | exact correctness pass | `187.216 us` |
| `non_multicast` | illegal memory access at CUDA synchronization | not measured |

The valid pinned run reported zero maximum and mean error across the repository
correctness contract. Its 30 samples had mean `187.901 us`, standard deviation
`2.428 us`, minimum `185.152 us`, and maximum `197.824 us`. The normalized compiler
IR SHA-256 began `671224e5`, and the embedded 19,778-byte fatbin SHA-256 began
`4be702f4`.

The non-multicast override changed only `_make_tma_atoms_and_tensors`, forcing its
multicast dimension from two to one. The kernel launched, but the next CUDA
synchronization reported an illegal memory access. No result JSON was written because
the candidate never completed correctness or timing.

## Interpretation

The TMA atom is coupled to clustered TMA partitioning, CTA coordinates, multicast
masks, and ownership of the shared input tile. Replacing the atom alone creates an
internally inconsistent kernel. This is a deterministic invalid candidate, not an
infrastructure failure and not evidence that all non-multicast implementations are
impossible.

## Decision

Reject `tma_load_policy` as a bounded canonical field and do not add an MCTS strategy
or mutation. A future non-multicast experiment would need one coordinated structural
rewrite that changes atom construction, partitioning, masks, and per-CTA data
ownership together, followed by the same JIT, correctness, artifact, and timing
validation.
