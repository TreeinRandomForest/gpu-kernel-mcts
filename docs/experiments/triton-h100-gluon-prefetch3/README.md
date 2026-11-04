# Three-stage Gluon A/B prefetch ring on H100

## Question

Does preloading future A/B tiles remove the TMA barrier poll that dominated the
two-stage source Gluon kernel? This isolates the schedule: both source kernels
use the same `128 × 256 × 64` per-CTA tile, two-CTA cluster, 8 warps, ordinary
per-CTA B load, `4096³` BF16 GEMM, seed 0, and exact output comparison with
`torch.mm`. The candidate uses three shared-memory slots and three transaction
barriers. It preloads the first three K tiles, refills a free slot two K steps
ahead, and waits for the prior WGMMA before reusing that slot.

## Result

All four arms produced exactly the same BF16 output as cuBLAS. On one H100 SXM,
three interleaved CUDA Graph rounds gave the following median latency within
each round (µs):

| Arm | Round 0 | Round 1 | Round 2 | Median of rounds |
| --- | ---: | ---: | ---: | ---: |
| Three-stage source Gluon | 219.760 | 234.496 | 240.928 | 234.496 |
| Two-stage source Gluon | 295.568 | 294.240 | 305.104 | 295.568 |
| Optimized Triton B | 184.304 | 184.080 | 191.760 | 184.304 |
| cuBLAS | 175.312 | 175.136 | 183.392 | 175.312 |

The three-stage kernel was 1.25–1.34× faster than its paired two-stage control,
but remained slower than optimized Triton B. The rising third-round latencies
for all arms make the paired ratios more useful than one isolated latency.

NCU used separate replay passes. Its durations are **not** the latency samples
above. The Source-page PC samples and counters show:

| NCU observation | Three-stage | Two-stage |
| --- | ---: | ---: |
| Tensor pipe active | 66.77% | 55.89% |
| DRAM bytes read | 174.134 MB | 174.097 MB |
| Executed warp instructions | 22.988 M | 18.760 M |
| Shared memory per CTA | 148.504 KB | 99.336 KB |
| Registers per thread / spills | 154 / 0 | 154 / 0 |
| PC samples at branch after `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 841 | 19,528 |
| PC samples at `WARPGROUP.DEPBAR.LE` | 13,485 | 343 |

The barrier poll moved out of the leading sampled hotspot, while WGMMA
completion became the leading site. This is consistent with better overlap of
TMA and computation. PC sampling does not provide a fraction of elapsed time
at an instruction. The candidate uses about 49 KB more shared memory per CTA
and 4.23 M more executed warp instructions; the latency gain is therefore
specific to the schedule and workload, not a general reduction in work.

The paired timing and NCU runs used identical cubins for each source arm,
verified by SHA-256. Both emitted ordinary `UTMALDG.2D` for B; multicast
sites exist in the runtime-selectable binary but the ordinary branch executed.
The HGMMA dynamic count remained 1,048,576 in each arm.

## Artifacts and reproduction

- `kernel.py`: candidate source; `paired.py`: exactness and interleaved CUDA
  Graph timing; `profile_one.py`: NVTX-selected NCU launch.
- `run.py`: provisions one Nebius H100, runs both checks, saves compiler
  PTX/cubin/SASS and NCU reports, then tears down the worker.
- `results.json`, `output/paired.json`, `analysis.json`: source identities,
  timing, compiler resources, and NCU summaries. `analyze.py` regenerates the
  summary from raw CSV files.
- `compile-failure-attempt1.json` and `compile-failure-attempt2.json` preserve
  two initial compiler failures before the successful source revision.

Run `python docs/experiments/triton-h100-gluon-prefetch3/analyze.py` to
regenerate `analysis.json`. The successful run finished with no Nebius VM or
boot disk remaining.

## Next question

The three-stage source is a stronger schedule control, but its leading
`WARPGROUP.DEPBAR.LE` sample site and larger shared-memory footprint do not
yet explain the remaining gap to optimized Triton B. Inspect WGMMA issue and
wait placement, and compare its emitted loop with Triton B before transferring
this prefetch pattern into the optimized source or a search strategy.
