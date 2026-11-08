# Isolating CTA order, tile width, and persistence in wait-one Gluon

## Design

This is a fixed-shape H100 SXM experiment for contiguous BF16
`A[4096,4096]`, `B[4096,4096]`, and `C = A @ B.T`. The validated three-slot
wait-one Gluon kernel and cuBLAS are anchors. A new Gluon source keeps the
three-slot wait-one pipeline, ordinary A/B TMA loads, two-CTA cluster, BK=64,
eight warps, and TMA output store. Its 2×2×2 factors are:

| Factor | Values |
| --- | --- |
| Cluster order | M coordinate fastest; N coordinate fastest |
| Cluster output tile | 256×256; 256×128 |
| Assignment | One tile per cluster; 66 persistent clusters looping over tiles |

The 66-cluster setting launches 132 physical CTAs, matching cuBLAS's CTA
count; the regular 256-wide and 128-wide settings launch 512 and 1024 CTAs.
The narrower tile necessarily changes the HGMMA shape and number of output
tiles, so its effect is not solely a numerical tile-width parameter.

All eight variants and both anchors compiled and passed exact comparison to
`torch.mm` for the fixed seed-0 input. Outputs were checked again after
each timing round. Ten CUDA Graph rounds cyclically rotate the ten arms
through every measurement position. Each arm has 30 event-timed replays per
round. Separate NCU full-set captures use the same seeded inputs and
`--cache-control all`; NCU replay duration is not used as latency. Profiled
Gluon cubin hashes match the timing cubins.

## Results

| Arm | Median µs | CTAs / waves | DRAM read MB | Tensor pipe active | Warp instructions M |
| --- | ---: | ---: | ---: | ---: | ---: |
| cuBLAS | **176.416** | 132 / 1.00 | 150.902 | 90.15% | 16.571 |
| Validated wait-one original | 186.944 | 512 / 3.88 | 174.094 | 84.40% | 24.524 |
| 256, M-fast, regular | 185.736 | 512 / 3.88 | 174.097 | 85.99% | 28.197 |
| **256, M-fast, persistent** | **183.728** | 132 / 1.00 | 174.086 | 86.70% | 28.595 |
| 256, N-fast, regular | 185.328 | 512 / 3.88 | 174.107 | 85.88% | 28.197 |
| 256, N-fast, persistent | 186.152 | 132 / 1.00 | 174.093 | 87.09% | 28.595 |
| 128, M-fast, regular | 289.592 | 1024 / 7.76 | 309.338 | 59.76% | 54.002 |
| 128, M-fast, persistent | 286.448 | 132 / 1.00 | 308.558 | 60.59% | 54.727 |
| 128, N-fast, regular | 283.568 | 1024 / 7.76 | 315.538 | 60.71% | 54.002 |
| 128, N-fast, persistent | 288.488 | 132 / 1.00 | 312.411 | 59.81% | 54.727 |

Within-round paired contrasts give a clearer result for the small changes:

- At 256-wide, M-fast order, persistence was **2.128 µs faster** at the
  median of ten paired differences, and faster in nine of ten rounds. DRAM
  reads changed by only 0.011 MB, so this gain is not explained by reduced
  measured DRAM traffic. The best arm remains 7.312 µs behind same-worker
  cuBLAS by the median-of-round-medians comparison.
- At 256-wide without persistence, N-fast order was 0.496 µs faster at the
  paired median; with persistence it was 2.480 µs slower. Order interacts
  with persistence but did not close the cuBLAS gap.
- At fixed order and assignment, switching to the 128-wide tile added
  97.912–103.768 µs. DRAM reads rose from about 174 MB to 309–316 MB and
  warp instructions rose from about 28 M to 54 M. The 128-wide variants
  emit `HGMMA.64x128x16.F32.BF16`, the same *shape* and 2.097 M dynamic
  HGMMA warp instructions as captured cuBLAS, but do not reproduce its
  traffic, total instruction count, or performance.
- Making the 128-wide variants persistent reduced CTA waves to one but did
  not fix their traffic or latency. Matching cuBLAS's CTA count alone is
  insufficient.

The generalized source allocates its output shared-memory tile alongside
the pipeline buffers, unlike the original source. Its 256-wide variants
therefore use 214.040 KB shared memory per CTA versus 148.504 KB for the
original; all have 154 registers per thread and zero spills. The factor
contrasts compare variants within the generalized source. The original
anchor shows that the refactor itself also changes compilation, so the
best-versus-original difference is not solely a persistence effect.

These are one-worker, one-shape observations. PC samples and aggregate NCU
counters do not identify a unique instruction or schedule that causes the
remaining gap. The narrower tile's high DRAM traffic is a measured property
of this implementation, not evidence that every 256×128 GEMM must be slow.

## Artifacts and next question

- `kernel.py`: generalized Gluon source.
- `paired.py`, `profile_one.py`, `run.py`, `analyze.py`: exactness, balanced
  timing, matched NCU capture, and analysis.
- `results.json`, `output/paired.json`, `analysis.json`: provenance, raw
  rounds, metrics, and paired factor contrasts. PTX, cubins, SASS, and NCU
  CSVs are in `output/`; large `.ncu-rep` files stay local and Git-ignored.

The next isolated target is cuBLAS's operand reuse and lower instruction
count at a 128-wide HGMMA shape. A source rewrite that shares or reuses A
across neighboring 128-wide output tiles would test that hypothesis; merely
changing tile width or CTA count did not. Validate any resulting source on
more shapes and seeds before adding a search strategy.

The Nebius VM and boot disk were removed after profiling.
