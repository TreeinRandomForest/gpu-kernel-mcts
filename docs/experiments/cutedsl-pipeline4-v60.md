# Independent CuTe four-stage mainloop (v60)

## Purpose

Validate a four-stage realization of `change_pipeline_stages` for the independent
`(64,256,64)`, cluster `(1,1)`, SW128 root under the repository BF16 GEMM contract.

## Typed change

- mainloop and barrier-ring depth: `3 -> 4`;
- mainloop shared memory: `122,880 -> 163,840` bytes;
- epilogue storage offset: `122,880 -> 163,840` bytes; and
- combined shared memory: `163,840 -> 204,800` bytes.

The workload, CTA tile, cluster shape, WGMMA decomposition, swizzle, and epilogue
organization were unchanged.

## H100 result

Both variants passed exact repository-contract correctness with zero maximum and
mean error. They were measured sequentially from image
`cutedsl-pipeline4-v60` using 10 warmups and 30 CUDA-event samples.

| State | Median | Mean | Range |
|---|---:|---:|---:|
| three-stage root | 669.760 us | 669.958 us | 660.992--676.704 us |
| four-stage candidate | 675.440 us | 675.677 us | 667.296--684.224 us |

The candidate/root latency ratio was `1.00848`; its MCTS reward is
`ln(669.760 / 675.440) = -0.008445`. The valid slower state is retained rather
than pruned.

## Promotion boundary

The stage-three root may realize either stage two or stage four under the same
semantic strategy. Each alternate returns only to stage three. Direct stage-two to
stage-four transitions and combinations with SW64, cluster `(2,1)`, or alternate
CTA tiles remain outside the validated typed space.

## Bounded MCTS promotion smoke (v61)

A same-worker `B_mut=2`, `B_gen=0`, `k_max=2`, `max_depth=1` run widened the
single `change_pipeline_stages` action twice. It created distinct valid stage-two
and stage-four nodes, performed two valid-only backups, and made no generation
calls:

| State | Median | Reward |
|---|---:|---:|
| stage-three root | 669.664 us | 0 |
| stage-two realization | 670.048 us | -0.000573 |
| stage-four realization | 675.808 us | -0.009133 |

Run ID: `abf1e361-b948-4836-8822-4a2e7acd7471`. The local trace is
`cutedsl-pipeline-bmut2-v61.sqlite`; it is a generated artifact and is not intended
for source control.
