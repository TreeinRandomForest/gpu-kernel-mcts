# Independent CuTe software-prefetch mainloop (v63)

## Question

Can the independent three-stage mainloop overlap asynchronous TMA movement with
WGMMA consumption without changing its CTA tile, memory layout, or agent ownership?

## Schedule

The serial root issues and waits for TMA separately for every K tile. The candidate:

1. fills all three shared-memory stages;
2. waits for and consumes the current stage with WGMMA;
3. waits for that WGMMA group to finish reading the stage; and
4. refills the released stage with tile `k + 3` while intervening stages remain full.

Warp zero still issues TMA and participates in WGMMA. This is cooperative software
pipelining, not dedicated producer/consumer warp specialization.

## H100 result

Both schedules passed the fixed BF16 GEMM repository contract exactly with zero
maximum and mean error. Image `cutedsl-prefetch-v63` used 10 warmups and 30
CUDA-event samples on the same H100 SXM VM.

| Schedule | Median | Mean | Range |
|---|---:|---:|---:|
| serial | 675.408 us | 674.902 us | 668.128--681.536 us |
| prefetch | 274.256 us | 274.909 us | 271.520--282.048 us |

Prefetch was 2.46269x faster, with `ln(675.408 / 274.256) = 0.901255`.
The normalized MLIR and fatbin fingerprints were distinct.

## Promotion boundary

Schema v4 includes the schedule in canonical identity. The two schedules form one
paired `change_mainloop_schedule` transition under `B_mut`. Prefetch combinations
with alternate CTA tiles, cluster multicast, stage depths, or SW64 remain excluded
until validated independently.

## Typed promotion and MCTS smoke (v64)

The schema-v4 typed CLI path independently passed exact correctness at 274.608 us
median. A bounded H100 search then selected only `change_mainloop_schedule` with
`B_mut=1`, `B_gen=0`, `k_max=1`, and `max_depth=1`. It created one distinct
prefetch node and performed one valid-only backup:

| State | Median | Reward |
|---|---:|---:|
| serial root | 670.960 us | 0 |
| prefetch child | 274.944 us | 0.892142 |

Run ID: `b48ebfae-3e5f-4449-8550-6deadb5f9721`. The trace and best source are local
generated artifacts, `cutedsl-prefetch-bmut1-v64.sqlite` and
`cutedsl-prefetch-bmut1-v64-best.py`.
