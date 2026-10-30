# H100 `spill_v1` profile, SQLite, and prompt validation — 2026-10-30

This model-free, one-generation MCTS check used the saved grouped `BN=256`
four-warp source as root and the eight-warp source as its fixed proposal. The
ordinary search controller evaluated both on one H100 SXM, lazily profiled the
root before generation and the final best after search, persisted the results
to SQLite, and recorded the exact generated prompt. Both kernels passed
correctness. The VM and disk were deleted.

| Measurement | Four warps (parent) | Eight warps (child) |
| --- | ---: | ---: |
| Triton compiled spills | 256 | 0 |
| Local-load sectors | 134,389,760 | 0 |
| Local-store sectors | 134,791,168 | 0 |
| Local-load L1 hit sectors | 60 | 0 |
| Local-load L1 miss sectors | 134,389,700 | 0 |
| Overall L2 sector hit rate | 97.28% | 66.70% |
| Total DRAM read bytes | 206,765,056 | 186,670,336 |
| Total DRAM write bytes | 796,534,272 | 28,660,480 |
| Tensor-pipe active | 10.45% | 91.51% |

The local-load hit and miss counts sum to the local-load sector total for the
four-warp kernel. Nearly all of its local-load sectors missed L1. The overall
L2 hit rate and total DRAM bytes mix local and ordinary GEMM traffic; they do
not quantify spill-specific L2 or HBM bytes. NCU replay counters are separate
from CUDA-event reward timing.

The [local SQLite trace](output/spill-v1-validation.sqlite) contains one valid
generation, two nodes, and two `spill_v1` profiles. Its generation prompt has
`parent_profile.metric_set = spill_v1`, `compiler_spills = 256`, and the same
local-load sector and L1 hit/miss values as the root's stored profile. The
[bounded summary](output/summary.json) records those values; the SQLite file
is Git-ignored. The trace browser's **Artifact reconciliation → NCU Source
summary** layer matches both nodes to the separately collected Source reports
by source hash and workload, labeling them as separate-worker evidence.

The run used the pinned Triton worker image
`docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`
with the current repository's Python package overlaid in a separate process.
This validates the updated backend and search path on H100 without publishing
a new registry image. [provenance.json](output/provenance.json) records this
distinction. [run.py](run.py) and [run_inside.py](run_inside.py) reproduce the
fixed validation flow; neither calls an LLM.
