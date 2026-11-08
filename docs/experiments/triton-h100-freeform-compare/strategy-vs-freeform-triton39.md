# Combined strategy catalog versus freeform Triton MCTS on H100

Three GLM-5.3 seed pairs used the packaged 4096³ BF16 GEMM root and one
Nebius H100 worker per pair. Each pair ran combined v20 catalog MCTS and
pure-Triton freeform MCTS on the **same worker**. The arm order was combined
first for seeds 0 and 2, freeform first for seed 1. Worker manifest IDs and
cuBLAS calibrations match within each pair; all workers and disks were
released. The image was pinned to
`docker.io/saarora/gpu-kernel-mcts@sha256:22eb914d5041a988653d9d19a298d6bdb64f178f175563743d0af4591cd75ee5`
and each manifest recorded Triton 3.9.0.

Both arms used GLM-5.3 Messages with thinking disabled and zero conversation
history, the same source-generation adapter, root, worker package overlay,
`B_gen=25`, `k_max=3`, depth 10, and one repair allowance. Parent NCU was
hidden from prompts, while lazy `lightweight_v1` profiles were collected.
Compiler summaries, dataflow views, and the H100 hardware graph were absent
from prompts. The intended difference was action policy: a combined
Triton/Gluon strategy menu versus one pure-Triton freeform action. The
action-specific instructions and eligible source stages therefore differed.
The search seed does not force deterministic GLM output.

| Seed | cuBLAS, µs | Combined best, µs (call) | Freeform best, µs (call) | Combined minus freeform, µs |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 179.360 | 246.480 (7) | **201.856 (23)** | 44.624 |
| 1 | 178.320 | 244.528 (3) | **225.280 (12)** | 19.248 |
| 2 | 177.664 | 245.952 (20) | **231.312 (20)** | 14.640 |

## Best-so-far by generation call

Failed generations and duplicate realizations still count against `B_gen`.

| Seed | Arm | Call 5, µs | Call 10, µs | Call 25, µs | Valid / 25 | Reused states | Nodes incl. root |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Combined | 257.056 | 246.480 | 246.480 | 25 | 7 | 19 |
| 0 | Freeform | 202.768 | 202.768 | 201.856 | 21 | 0 | 22 |
| 1 | Combined | 244.528 | 244.528 | 244.528 | 25 | 8 | 18 |
| 1 | Freeform | 227.712 | 227.712 | 225.280 | 22 | 0 | 23 |
| 2 | Combined | 254.816 | 254.192 | 245.952 | 22 | 5 | 18 |
| 2 | Freeform | 232.320 | 232.320 | 231.312 | 20 | 1 | 20 |

The combined catalog selected descriptor staging 8/8/8 times, Gluon MMA-v2
port 5/5/8 times, pointer rewrite three times in every seed, and BN widening
9/9/6 times. It never selected `triton_specialize_gemm_dims`, a serial
TMA/WGMMA action, or wait-one pipelining. Seed 2 spent three calls on Gluon
MMA-v2 compilation failures. The other two catalog runs had no invalid
proposals. Seven, eight, and five catalog calls reused states, mostly exact
BN-widening realizations. The freeform runs had 4, 3, and 5 invalid calls,
mostly compilation failures, yet each reached a lower best latency.

All three catalog best sources use the 128×256×64, eight-warp, three-stage
pure-Triton pointer loop with runtime M/N/K parameters and a two-dimensional
CTA grid. The dimension-specialization action is statically applicable to
each exported best source, but no specialization entry appears in their
recorded PUCT candidates; those widened best nodes did not acquire strategy
edges in these traces. This is consistent with a **search-allocation** limit
in the `k_max=3` configuration; no specialization generation was attempted.

The three freeform best sources also use 128×256×64 and eight warps, but add
grouped one-dimensional CTA ordering and coordinated loop/mask choices.
Seeds 0 and 1 use four stages; seed 2 uses three. Seed 1 marks M/N/K as
`tl.constexpr`, while seeds 0 and 2 leave them as runtime parameters. These
source differences do not isolate which edit caused the latency gap. A prior
separate-worker combined run reached 198.944 µs, so the catalog can find a
roughly 200 µs kernel under this configuration.

Freeform won all three measured pairs. The sample is too small for a general
policy ranking, and this comparison does not separate the effects of action
menu, prompt wording, and allowed Gluon exploration. A useful next test is a
matched `k_max=1` catalog control, followed by same-worker crossovers of
dimension specialization, grouped ordering, and stage count on representative
best sources. Search-policy changes to discourage repeated realizations need
a separate spec and invariant review.

## Traces and sources

For each seed `S` in `0,1,2`, the SQLite traces are
`output/combined-glm53-sS-b25-paired39v1.sqlite` and
`output/mcts-glm53-sS-b25-paired39v1.sqlite`. Their exported best sources
have matching `-best.py` suffixes. Each trace stores the exact prompts,
model outputs, generation status, source, per-node measurements, compiler
artifacts, worker calibration, and environment manifest.
