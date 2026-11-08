# Wait-one Gluon versus cuBLAS on one H100

## Question and method

How much of the remaining fixed-shape GEMM gap can be associated with the
launch schedule, data traffic, and emitted instructions? This experiment uses
the validated three-slot wait-one Gluon source and the selected cuBLAS kernel
for physically contiguous BF16 `A[4096,4096]`, `B[4096,4096]`, and
`C = A @ B.T`. Both arms use the same seed-0 tensors, pass exact comparison
with `torch.mm`, and run on one H100 SXM worker. Eight CUDA Graph timing rounds
alternate arm order; each arm has 30 event-timed replays per round. Separate
NCU full-set passes capture one launch per arm. NCU replay duration is not
used as latency.

The timing Gluon cubin SHA-256 matches the profiled Gluon cubin. cuBLAS is
identified by its captured NCU kernel name, not by a cubin hash.

## Measured result

| Measure | Gluon wait-one | cuBLAS |
| --- | ---: | ---: |
| Median of eight round medians | 190.216 µs | 178.304 µs |
| Physical CTAs / waves per SM | 512 / 3.88 | 132 / 1.00 |
| Threads / registers / shared memory per CTA | 256 / 154 / 148.504 KB | 384 / 168 / 214.204 KB |
| DRAM read / write | 174.077 / 28.872 MB | 150.912 / 28.752 MB |
| Tensor pipe active | 84.69% | 90.40% |
| Executed warp instructions | 24.524 M | 16.571 M |
| Dynamic HGMMA warp instructions | 1.049 M, `64x256x16` | 2.097 M, `64x128x16` |
| Dynamic TMA load / multicast load warp instructions | 294,912 / 0 | 32,768 / 32,768 |

The median paired Gluon-minus-cuBLAS difference is **11.920 µs**; Gluon was
slower in all eight rounds (paired differences 11.408–12.464 µs). Both arms
drifted upward during the timing sequence, so the paired difference is more
informative than comparing either arm's first and last round. This is one
worker and one shape, not a population estimate.

The cuBLAS capture selected
`nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT`. Its SASS contains 3D TMA
loads, multicast loads, 3D TMA stores, HGMMA, and loop branches. The Gluon
SASS contains ordinary 2D TMA loads and stores, HGMMA, and a loop
`WARPGROUP.DEPBAR.LE ... 0x1`. The cuBLAS HGMMA shape differs, so the larger
dynamic HGMMA instruction count does not imply twice the matrix work.

## CTA ownership: what is established

The Gluon source explicitly maps each program cluster to a 256x256 output
tile and uses two CTAs per cluster. Its 16x16 program grid therefore
launches 512 CTAs. The selected cuBLAS kernel launches only 132 CTAs, or 66
two-CTA clusters. Its name suggests a 256x128 tile family, but the aggregate
NCU launch and Source counters do not establish an exact output tile map,
per-CTA tile count, or whether a persistent scheduler is active. Its SASS
reads `SR_CTAID.Y` and contains repeated output-store paths; these are
consistent with a more complex mapping but do not uniquely identify it.
The 132-CTA count is direct evidence that cuBLAS does not use the same
one-Gluon-program-cluster-per-256x256-tile launch geometry.

The 23.166 MB lower cuBLAS DRAM read count, fewer total executed warp
instructions, and higher tensor-pipe activity are simultaneous observations,
not independent causal estimates of the 11.920 µs gap. The current evidence
does not isolate the benefit of multicast, the CTA schedule, the tile shape,
or individual loop instructions.

## Next bounded test

Keep the validated Gluon tile, three-slot wait-one schedule, TMA paths, and
cluster size, and change only program-cluster order (for example, group
neighboring M tiles at the same N coordinate before advancing N). Compare
paired CUDA Graph timing and NCU DRAM/L2 traffic against the original on the
same input. This tests a specific reuse hypothesis without attributing all
of cuBLAS's behavior to its CTA count. Do not add a search strategy until
the changed source passes exactness on multiple seeds and relevant shapes.

## Artifacts

- `paired.py`, `profile_one.py`, `run.py`, and `analyze.py` reproduce the
  comparison and summarize the captured counters.
- `results.json`, `output/paired.json`, and `analysis.json` hold provenance,
  exactness, round timings, and interpreted metrics.
- `output/gluon.ptx`, `.sass`, and the two `*.sass.csv` Source views support
  instruction inspection. The large `.ncu-rep` files are retained locally
  and Git-ignored.

The Nebius VM and boot disk were removed after profiling.
