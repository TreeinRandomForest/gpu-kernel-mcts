# CuTe DSL register repartition v69

Date: 2026-09-29

This evidence-only H100 SXM experiment asked whether the validated dedicated
producer/consumer kernel benefits from complete CUTLASS-style register repartition.
The control already reduced the TMA producer to 40 maximum registers per thread.
The candidate additionally increased the WGMMA consumer to 232.

| Variant | Correct | Median (us) | Mean (us) |
| --- | ---: | ---: | ---: |
| Producer decrease only | yes, exact | 266.016 | 266.439 |
| Producer decrease + consumer increase | yes, exact | 265.840 | 266.768 |

The candidate's median ratio was 1.000662x (`reward=0.000662`), while its mean was
slightly worse. Source and runtime artifacts were distinct, but the timing difference
is within noise. The consumer allocation therefore remains diagnostic-only and is
not admitted as typed state or an MCTS mutation.
