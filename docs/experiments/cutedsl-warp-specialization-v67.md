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
