# CuTe DSL wide-specialized cluster interaction v70

Date: 2026-09-29

This evidence-only H100 experiment tested B multicast across a `(2,1)` cluster while
holding the final `(128,256,64)` CTA, three mainloop stages, SW128, prefetch, and
dedicated producer/consumer warp groups fixed.

The cluster `(1,1)` control passed exact repository-contract correctness and measured
185.264 us median. The cluster `(2,1)` candidate compiled and returned from kernel
launch, but device synchronization did not complete. The diagnostic container was
terminated after bounded inspection confirmed the stall at `synchronize started`.

This is a synchronization failure, not a measured slow candidate. The specialized
full/empty stage-ring protocol is not currently cluster-safe when B multicast is
enabled. Clustered specialization remains outside typed MCTS state, and no reward or
backup may be assigned to this failed diagnostic.

## Pipeline-depth follow-up

A separate legal interaction reduced the wide specialized mainloop from three stages
to two while preserving cluster `(1,1)`. It passed exact correctness and produced a
distinct source artifact, but measured 210.272 us versus the same-image three-stage
control's 185.264 us. This is a 13.50% latency regression (`reward=-0.126620`). The
valid slower state is retained as experimental evidence but is not admitted as a
specialized mutation from this result.

## Shared-memory swizzle follow-up

Image v71 added a reproducible `--independent-swizzle-bytes` diagnostic control and
compared SW128 with SW64 on the wide specialized kernel. Both variants passed exact
correctness and produced distinct artifacts. SW128 measured 185.424 us median;
SW64 measured 224.192 us, a 20.91% latency regression (`reward=-0.189858`). SW64 is
therefore not admitted into the specialized MCTS neighborhood.
