# CuTe DSL wide specialized CTA v69

Date: 2026-09-29

This H100 SXM interaction experiment combined two independently validated controls:
the dedicated TMA-producer/WGMMA-consumer schedule and the `(128,256,64)` CTA with
two WGMMA consumer groups. The comparison used the same v69 image and repository
BF16 GEMM contract.

| State | Correct | Median (us) | Mean (us) |
| --- | ---: | ---: | ---: |
| Specialized `(64,256,64)` | yes, exact | 266.016 | 266.439 |
| Specialized `(128,256,64)` | yes, exact | 185.568 | 186.491 |
| cuBLAS, subsequent same-VM comparison | yes, exact | 176.096 | 176.620 |

The wide specialized state was 1.43352x faster than the narrow specialized control
(`reward=0.360135`) and 5.38% slower than cuBLAS. Its normalized IR, runtime fatbin,
configuration, and source fingerprints were distinct.

Schema v5 already expresses this composition. The admitted search edge is the paired
specialized `(64,256)` to specialized `(128,256)` realization of `change_cta_tile`.
It preserves prefetch, specialization, cluster `(1,1)`, three mainloop stages, and
SW128. No other specialized cross-control combination is validated by this result.

## Typed MCTS smoke (v70)

A two-iteration H100 search started from cooperative prefetch and selected the two
admitted transitions in sequence. It used `B_mut=2`, `B_gen=0`, `k_max=1`, and
`max_depth=2`:

| Node | Median (us) | Root-normalized reward |
| --- | ---: | ---: |
| Cooperative prefetch `(64,256)` | 274.160 | 0 |
| Specialized `(64,256)` | 266.032 | 0.030095 |
| Specialized `(128,256)` | 184.112 | 0.398168 |

The run created three distinct schema-v5 nodes, made three profile calls, and
performed two valid-only backups without any LLM generation. Run ID:
`68071bbf-cf28-45c9-ac16-d68476ddb71a`. The local ignored trace is
`cutedsl-wide-specialized-bmut2-v70.sqlite`.

## Optimization path and original project baseline

The project originally began with a naive CUDA C++ BF16 GEMM root that measured
28,245.9 us in the profiled Sol run. That source is useful historical context, but
it is not the parent of the typed CuTe states: changing from CUDA C++ to the
independently lowered CuTe DSL representation is an implementation change, not an
MCTS mutation.

The comparable typed CuTe optimization path begins with the independent serial
kernel:

| Step | Typed transition | Median (us) | Effect |
| --- | --- | ---: | ---: |
| Historical project root | Naive CUDA C++; separate run | 28,245.9 | Context only |
| 1 | Independent CuTe serial `(64,256,64)` | 675.408 | CuTe path root |
| 2 | `change_mainloop_schedule`: software prefetch | 274.256 | 2.46269x vs step 1 |
| 3 | `change_producer_consumer_specialization` | 266.032 | 1.03091x vs step 2 |
| 4 | `change_cta_tile`: specialized `(128,256,64)` | 184.112 | 1.44495x vs step 3 |

Across the typed CuTe path, steps 1 through 4 improve latency by approximately
3.67x. The table combines the v63 serial/prefetch paired experiment with the v70
MCTS measurements, so it is a documented optimization sequence rather than a claim
that every number came from one uninterrupted run. The final 184.112 us kernel was
approximately 4.55% slower than the separately measured 176.096 us cuBLAS result.
