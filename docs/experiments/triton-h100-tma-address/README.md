# Isolated TMA store-mask and grouped-CTA arithmetic experiment

The [matched four-action/five-action MCTS pilot](../triton-h100-reuse-strategy/README.md)
produced four best kernels at 198–201 µs. Auditing their retained source and
PTX confirmed that all four use pointer-based `tl.load` and `cp.async`, not
descriptor TMA loads. All use a `128×256×64` tile, eight warps, and grouped
CTA order; their compiler records report 213–215 registers/thread and no
spills. The trace audit checks source and PTX hashes and is recorded in
[`best_audit.json`](../triton-h100-reuse-strategy/best_audit.json). SASS was
unavailable in those ordinary MCTS node records.

We then used the unchanged [TMA kernel](../triton-h100-tma-load/tma_kernel.py)
as a controlled baseline. The aligned 4096³ workload has complete output
tiles, so its C-store mask is redundant. It also has 32 M tiles, divisible
by `GROUP_M=8`, so the grouped scheduler's tail-group calculation always
returns eight. [`variants.py`](variants.py) constructs two separate source
edits and their combination; the constant-group variant adds a launcher guard
requiring `M % 1024 == 0`. These are fixed-shape specializations, not a
general replacement for arbitrary dimensions.

One pinned H100 SXM ran all variants and cuBLAS on identical seeded tensors.
Every variant matched cuBLAS exactly. Three randomized-order rounds used 30
CUDA-event samples each; NCU full Source/SASS was captured separately with
kernel replay and `--cache-control all`.

| Variant | Round medians, µs | Paired median vs baseline | Executed warp instructions |
| --- | --- | ---: | ---: |
| Baseline | 197.328, 204.000, 206.784 | — | 27.754M |
| Unmasked C store | 199.984, 203.936, 202.656 | −0.064 µs | 27.615M |
| Constant group size | 197.776, 202.704, 203.328 | −1.296 µs | 27.681M |
| Both edits | 197.952, 204.464, 208.496 | +0.624 µs | 27.550M |
| cuBLAS reference | 188.976, 188.400, 193.424 | — | — |

All four TMA variants use 154 registers/thread and execute 1,048,576 HGMMA
warp instructions. The edits reduced total instructions slightly, but their
latency changes are small and inconsistent across rounds. Combining the
edits was slower in every round despite having the fewest instructions.
Neither edit warrants a new MCTS strategy or a larger matched search on this
evidence. This also reinforces that aggregate instruction count is not a
latency decomposition.

[`results.json`](results.json) records raw event samples, hashes, correctness,
worker provenance, and report hashes. [`analysis.json`](analysis.json)
contains paired deltas and SASS opcode counts. Large NCU reports and exports
are under ignored `output/`. [`run.py`](run.py) provisions and releases the
worker; [`analyze.py`](analyze.py) validates and summarizes its results.
The instance and disk were both confirmed absent after the run.
