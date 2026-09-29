# CuTe DSL warp specialization v67

Date: 2026-09-29

This paired H100 SXM experiment compared the independently lowered
`(64,256,64)`, cluster `(1,1)`, three-stage SW128 prefetch kernel with the same
kernel using separate 128-thread TMA-producer and WGMMA-consumer warp groups. Both
runs used the same `cutedsl-specialized-v67` image and repository workload contract.

| State | Correct | Median (us) | Mean (us) |
| --- | ---: | ---: | ---: |
| Cooperative prefetch | yes, exact | 273.776 | 274.013 |
| Warp specialized | yes, exact | 264.912 | 265.741 |

Warp specialization was 1.03346x faster, corresponding to
`log(273.775995 / 264.912009) = 0.032913`. The result validates the bounded paired
transition; it does not validate combinations with CTA geometry, cluster,
pipeline-depth, or shared-memory-swizzle controls.

## Typed MCTS smoke (v68)

The schema-v5 typed renderer passed exact H100 correctness independently at
266.848 us median. A subsequent one-iteration search used only
`change_producer_consumer_specialization`, with `B_mut=1`, `B_gen=0`, `k_max=1`,
and `max_depth=1`. The cooperative prefetch root measured 273.648 us and the
specialized child measured 266.304 us, producing reward `0.0272041`. The run
created two distinct nodes, performed one valid-only backup, and made two profile
calls without GPU reevaluation on MCTS visits.

Run ID: `12f5cd4a-c960-4a2d-a66d-f4dd29ee502c`. The local trace is
`cutedsl-specialized-bmut1-v68.sqlite` and is intentionally ignored by Git.
