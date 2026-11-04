# Let one WGMMA group remain in flight

## Why test this

The three-stage Gluon source kernel reached a paired median of 234.496 µs,
versus 184.304 µs for optimized Triton B in its earlier same-worker run.
Both performed 1,048,576 dynamic `HGMMA.64x256x16.F32.BF16` warp
instructions, launched 512 CTAs with 256 threads each, and used about 148.5
KB shared memory per CTA. In the prior Source reports, their loop waits
differed:

| Kernel | Loop SASS wait | Prior PC samples at wait |
| --- | --- | ---: |
| Three-stage Gluon | `WARPGROUP.DEPBAR.LE gsb0, 0x0` | 13,188 |
| Optimized Triton B | `WARPGROUP.DEPBAR.LE gsb0, 0x1` | 2,090 |

These PC samples came from separate runs and are diagnostic only. Triton B
also uses `LDGSTS` copies, a different CTA mapping, and different register
use; the wait operand alone did not establish the remaining latency cause.
The [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
defines `wgmma.wait_group.sync.aligned 1` as waiting until at most one recent
WGMMA group remains pending.

## Controlled edit

`kernel.py` copies the validated three-slot Gluon source. On each K step it
waits for the current TMA tile, issues the current WGMMA, waits until only
that WGMMA may remain pending, and then refills the previous WGMMA's shared
slot with a future TMA tile. The three-slot A/B ring, per-slot barriers,
tile, grid, B-load policy, inputs, and epilogue match the prior source.
The edit changes both the wait operand and its position relative to WGMMA
issue and the refill; those effects are not separated here.

The first attempt failed Gluon type checking because `warpgroup_mma_wait`
returns a tensor and an assignment inside one branch changed the
loop-carried accumulator type. The successful source retains the async
accumulator and uses the wait for ordering; the failed attempt is preserved
as `compile-failure-attempt1.json`.

## Same-worker result

All four arms passed exact BF16 output comparison with cuBLAS on the fixed
`4096³` workload, seed 0. Each round measured 30 CUDA Graph replays per arm;
the arm order was shuffled. The figures below are within-round medians in µs:

| Arm | Round 0 | Round 1 | Round 2 | Median of rounds |
| --- | ---: | ---: | ---: | ---: |
| Gluon wait-one candidate | 187.696 | 188.272 | 190.432 | 188.272 |
| Gluon wait-zero control | 223.840 | 224.048 | 244.144 | 224.048 |
| Optimized Triton B | 186.720 | 186.672 | 189.104 | 186.720 |
| cuBLAS | 177.936 | 177.792 | 180.448 | 177.936 |

The wait-one schedule was 1.19–1.28× faster than the paired Gluon control.
Its median of round medians was 1.552 µs slower than optimized Triton B.
The third control round was slower than its first two, so the first two
paired rounds support the more conservative approximately 1.19× gain.

Separate NCU replay passes on the same worker confirmed that the timing
cubins matched the profiled cubins by SHA-256. NCU replay duration is not
used as latency here:

| NCU or compiler observation | Wait one | Wait zero |
| --- | ---: | ---: |
| Loop SASS wait operand | `0x1` | `0x0` |
| Tensor pipe active | 84.65% | 66.85% |
| DRAM bytes read | 174.099 MB | 174.100 MB |
| Executed warp instructions | 24.524 M | 22.988 M |
| Shared memory per CTA | 148.504 KB | 148.504 KB |
| Registers per thread / spills | 154 / 0 | 154 / 0 |
| PC samples at loop WGMMA wait sites | 4,738 | 13,456 |
| Dynamic HGMMA warp instructions | 1,048,576 | 1,048,576 |

The near-equal traffic and resources, unchanged HGMMA count, verified wait
operand, and paired latency improvement support a scheduling explanation.
The source and emitted binaries contain other changed instructions, and PC
samples do not quantify elapsed time at one instruction. This experiment
does not prove that the wait operand alone accounts for the gain.

## Artifacts

- `kernel.py`: wait-one source. The control is
  `../triton-h100-gluon-prefetch3/kernel.py`.
- `run.py`: same-worker correctness, paired CUDA Graph timing, compiler
  artifact capture, NCU Source reports, and worker teardown.
- `results.json` and `output/paired.json`: source and binary identities,
  exactness, per-round latency, and NCU provenance.
- `output/candidate.sass` and `output/control.sass`: emitted SASS; raw
  NCU CSV files are in `output/` and `analyze.py` regenerates `analysis.json`.
  The large `.ncu-rep` files remain local and are Git-ignored.

No Nebius VM or boot disk remained after the run.

## Next validation before a search action

This is a fixed-shape H100 result. Exercise more data seeds and relevant
shapes, verify output across repeated graph replays, and check resource and
correctness behavior at those shapes before making this a general Triton
strategy. The optimized Triton B implementation is still distinct from
this Gluon source; the near-equal latency does not make the two kernels
interchangeable.
