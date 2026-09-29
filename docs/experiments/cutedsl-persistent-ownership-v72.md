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
