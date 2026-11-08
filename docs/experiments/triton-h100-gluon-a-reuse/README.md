# Two adjacent N tiles with one A panel

This standalone experiment tests whether one persistent two-CTA Gluon cluster
benefits from sharing an A panel between two neighboring 128-wide output
tiles. It uses the fixed 4096 × 4096 × 4096 BF16 `A @ B.T` case on one
Nebius H100 SXM. The source and outputs are [here](kernel.py),
[paired measurements](output/paired.json), [NCU profiles](output), and
[run provenance](results.json).

All Gluon arms use 66 persistent clusters, two CTAs per cluster, eight warps
per CTA, 256 M rows per cluster, three K pipeline slots, and an M-fast
tile schedule. Each cluster covers the same 256-wide output region:

| Arm | Output organization | A loads per K step | BF16 shared memory / CTA |
| --- | --- | ---: | ---: |
| `original_persistent` | One 256-wide accumulator | 1 | 213,016 B |
| `split_reload` | Two 128-wide accumulators | 2 | 229,400 B |
| `split_reuse` | Two 128-wide accumulators | 1, feeding both MMAs | 180,248 B |

The B tiles are loaded separately in both split arms. No new TMA multicast or
remote shared-memory read is used. All arms were compiled with the pinned
Triton 3.9 image and passed bitwise equality against `torch.mm` before
timing. Ten rotating-order CUDA Graph rounds produced:

| Arm | Median of round medians | Range of round medians |
| --- | ---: | ---: |
| cuBLAS | 179.176 µs | 173.904–183.920 µs |
| Original persistent | 186.664 µs | 181.248–194.496 µs |
| Split, reload A | 258.192 µs | 243.392–263.040 µs |
| Split, reuse A | 250.232 µs | 232.272–257.648 µs |

The per-round reload-minus-reuse difference was positive in all ten rounds:
median **7.320 µs** (range 4.688–20.944 µs). The reuse arm was still
**62.496 µs slower** than the original at the paired median. These are
single-worker, single-seed, single-shape observations.

NCU `--set full` captured one separate launch per arm. Its replay duration is
not the CUDA Graph timing above. Key counters:

| Arm | DRAM reads | TMA load instructions | Executed instructions | Shared GMMA instructions | Tensor-pipe active, elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original persistent | 174.113 MB | 65,536 | 28.595M | 1.049M | 87.03% |
| Split, reload A | 174.105 MB | 131,072 | 41.940M | 2.097M | 61.14% |
| Split, reuse A | 174.096 MB | 98,304 | 33.736M | 2.097M | 63.56% |
| cuBLAS | 150.869 MB | 65,536 | 16.571M | 2.097M | 89.80% |

The unchanged DRAM reads show that extra A loads in this case are mostly
served below DRAM; the reuse benefit tracks fewer TMA and total instructions
and 49 KB less shared memory per CTA. Splitting the 256-wide MMA into two
128-wide MMAs doubles the counted GMMA instructions versus the original. This
first split kernel also used `wait_group 0`, whereas the original used
`wait_group 1`; the subsequent [paired wait-policy experiment](../triton-h100-gluon-a-reuse-wait1/README.md)
recovered roughly 47 µs by changing only that wait. The count difference
does not by itself prove that GMMA issue is the sole cause of the remaining
latency gap. The cuBLAS
GMMA count is also not directly comparable without accounting for its SASS
instruction shapes and schedule.

The experiment supports intra-cluster A reuse as a useful local mechanism,
but this two-accumulator realization is not a faster replacement for the
current persistent Gluon kernel, even after wait-policy tuning. The next
source experiment should retain
the efficient 256-wide MMA while targeting instruction overhead or operand
traffic, then repeat any promising result across shapes and seeds before
adding a search strategy.
