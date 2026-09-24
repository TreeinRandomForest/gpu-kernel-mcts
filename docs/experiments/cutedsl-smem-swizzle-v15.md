# CuTe shared-memory swizzle diagnostic v15

This diagnostic tested the first structural layout transformation around NVIDIA's
pinned Hopper dense-GEMM CuTe implementation. At tile `(128,256)` and cluster
`(2,1)`, NVIDIA's helper heuristically chooses SW128 shared-memory layout atoms for
the BF16/K-major A and B operands. The candidate preserves operand majorness while
forcing SW64; the epilogue already uses SW64 and remains unchanged.

The diagnostic was SHA-guarded against the pinned NVIDIA example and ran both
variants on one H100 SXM under the repository's fixed BF16 GEMM correctness and
measurement contract.

## Results

| Policy | Median (us) | Mean (us) | Stddev (us) | Max/mean error |
| --- | ---: | ---: | ---: | ---: |
| Heuristic SW128 | 188.016 | 188.933 | 2.604 | 0.0 / 0.0 |
| Forced SW64 | 188.528 | 189.535 | 2.658 | 0.0 / 0.0 |

SW64 was 0.512 us, or approximately 0.27%, slower by median. That difference is
small relative to sample dispersion and does not establish a performance winner.

## Artifact evidence

| Policy | Normalized IR SHA-256 | Fatbin SHA-256 | Fatbin bytes |
| --- | --- | --- | ---: |
| Heuristic SW128 | `671224e5094d182a...` | `4be702f4144c0f8f...` | 19,778 |
| Forced SW64 | `cf589d7031e6f911...` | `c8ecd3666e1f55ec...` | 20,801 |

The generated kernel names also encoded different source/destination layout
identities. Therefore the candidate is not a canonical alias: it produces a distinct
compiled kernel while preserving exact correctness.

## Decision

The control was promoted to schema-v3 typed state with values `heuristic` and `sw64`
and a deterministic mutation. The next hardware gate validates the canonical
backend/cache path and profiles both variants. Promotion is based on proven legality
and distinct structure, not a claim that SW64 is faster. This follows the search
invariant that valid locally slower kernels are retained rather than pruned solely
for their immediate reward.
