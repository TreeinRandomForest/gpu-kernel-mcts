# CuTe DSL persistent scheduler ownership v72

Date: 2026-09-29

This evidence-only H100 diagnostic isolates CUTLASS's
`StaticPersistentTileScheduler` from GEMM, TMA, and WGMMA. Its purpose is to prove
bounded and exactly-once work assignment before composing persistent scheduling
with the independent kernel's producer/consumer barrier protocol.

The logical `(128,256)` CTA grid for the fixed 4096x4096 output contains 32x16, or
512, tiles. The diagnostic launched at most one persistent CTA per H100 SM. The H100
reported 132 SMs, so the scheduler launched 132 persistent CTAs and assigned:

- 512 total tile coordinates;
- 512 unique tile coordinates;
- zero duplicate, missing, or unexpected coordinates;
- four tiles to 116 CTAs; and
- three tiles to the remaining 16 CTAs.

The generated CuTe source and compiled fatbin have stable SHA-256 fingerprints in
the JSON report. The host verifies the complete expected coordinate set rather than
accepting only a count.

This result validates scheduler ownership and bounded termination only. It does not
show that the wide specialized GEMM can safely reuse its shared-memory stage ring
across multiple output tiles. The next evidence-only gate must make the producer and
consumer warp groups instantiate identical schedules, reset their per-work pipeline
counts and accumulators, advance in lockstep, and preserve full/empty barrier phases
between work items. Until that passes exact H100 correctness, persistent scheduling
is neither typed state nor an MCTS mutation.

## Barrier-ring follow-up (v73)

The next evidence-only diagnostic added the final kernel's two-agent structure but
still excluded TMA and WGMMA data movement. A producer thread and consumer thread
independently instantiated the same scheduler and exchanged 64 logical K tiles per
output work item through a three-stage full/empty barrier ring.

The barrier stage and phase derive from a global K-tile ordinal carried across work
items. This matters because 64 is not divisible by three: after the first work item,
the next item begins at stage 1/phase 1 rather than incorrectly restarting at stage
0/phase 0.

On H100, all 132 persistent CTAs terminated within the bounded run. Producer and
consumer each completed all 512 output items, their per-CTA completion counts were
identical, and the distribution remained 116 CTAs with four items plus 16 CTAs with
three. The result validates scheduler lockstep and barrier-phase carry. It still
does not validate TMA payload movement, accumulator reset, WGMMA, or the epilogue;
those remain required before performance measurement or typed-state promotion.

## TMA payload follow-up (v74)

The third diagnostic retained the validated persistent scheduler and full/empty
barrier carry, then added real TMA loads for the first A and B K tile belonging to
each scheduled `(m,n)` output tile. The consumer sampled both shared-memory tiles,
and the host checked each sample against the exact expected global-memory address.

The H100 run again covered all 512 coordinates exactly once. Every A and B sample
matched, with no missing, duplicate, unexpected, or payload-mismatched records.
This establishes persistent scheduler ownership, barrier carry, and TMA coordinate
selection independently of tensor-core computation. The remaining correctness gate
is the full 64-K-tile WGMMA loop with accumulator reset and complete epilogue stores.

## WGMMA issue/reset follow-up (v75-v76)

The next gate added the wide CTA's two WGMMA consumer warp groups while retaining
the exact TMA sample checks. Each scheduled work item creates a fresh FP32
accumulator fragment, explicitly fills it with zero, starts WGMMA with accumulation
disabled, waits for completion, synchronizes both consumer groups, and only then
releases the shared-memory stage.

The first v75 attempt failed during DSL lowering because a dynamic thread-index
branch was incorrectly passed to `const_expr`. It never launched on the GPU and is
recorded as a renderer failure, not as a candidate or performance measurement.

After correcting that ownership branch, v76 completed all 512 scheduled tiles on
H100 without deadlock. Coordinate coverage and sampled A/B payloads remained exact,
and the compiled artifact is distinct from the TMA-only diagnostic. This proves
safe WGMMA issue, accumulator reset, consumer synchronization, and stage release for
one K tile per output work item. Because no accumulator data is stored yet, it is
not a numerical correctness result. The next gate is the persistent epilogue,
followed by extension from one K tile to all 64 K tiles.

## One-K epilogue follow-up (v77)

Image v77 added the complete wide-CTA epilogue to the persistent one-K diagnostic.
The two consumer warp groups staged eight disjoint 64x64 accumulator tiles in shared
memory. One elected CTA warp issued the eight TMA stores, waited for their completion,
and synchronized both consumer groups before any subsequent work item could reuse
the epilogue buffers.

The H100 run stored all 512 output tiles and matched the BF16 result of the one-K
FP32 reference exactly: maximum error and mean error were both zero. Scheduler
coverage and A/B TMA payload samples also remained exact. This validates the entire
persistent path for one K tile. It is not yet the repository workload because the
real GEMM accumulates 64 K tiles per output; that full-K carry is the final
correctness gate before latency measurement or typed-state promotion.

## Full-K follow-up (v78 and v81)

Image v78 extended each persistent output work item from one K tile to all 64 K
tiles. Accumulators are reset once per output tile, while the full/empty stage and
phase are derived from a global K-tile ordinal so the three-stage ring remains
continuous across scheduler work items. The synthetic full-size diagnostic covered
all 512 output tiles exactly once and matched its full-K reference with zero maximum
and mean error. That established internal full-K correctness, but did not yet use the
repository's canonical inputs or cuBLAS reference.

Image v81 closed that comparability gap. It used the repository's deterministic
4096x4096x4096 BF16 inputs, seed 0, cuBLAS FP32-accumulation reference, ten warmups,
and thirty CUDA-event measurements. All ownership and sampled TMA checks remained
exact, and the complete output matched the reference with zero maximum and mean
error. Median latency was 239.312 us (mean 240.363 us, minimum 235.552 us, maximum
261.056 us).

The persistent result is 30.0% slower than the 184.112 us static specialized CuTe
kernel and 35.9% slower than the same-VM 176.096 us cuBLAS result. Persistent
scheduling is therefore a correct, distinct implementation but negative performance
evidence for this workload. It remains outside typed MCTS state because the current
Milestone B design space does not include this scheduler; admitting it requires an
explicit spec amendment rather than a silent semantic expansion.
