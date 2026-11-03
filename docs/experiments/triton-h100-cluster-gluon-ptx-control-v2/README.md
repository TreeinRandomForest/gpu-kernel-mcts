# Corrected per-CTA ordinary B TMA control

The first [PTX control](../triton-h100-cluster-gluon-ptx-control/README.md)
likely hung because removing the cluster-owner predicate left `elect.sync` active
once per warp. The corrected patch retains a first-warp guard and elects one
lane in that warp **per CTA**. This gives each CTA one ordinary B TMA request
against its local 49,152-byte A+B transaction barrier. The original and
patched PTX both assemble with CUDA 12.8 `ptxas` after changing their PTX
header from 9.4 to the supported 8.7. The edit is pinned to the original
Gluon PTX and changes the B issuer predicate and B TMA instruction, leaving
the rest of the PTX identical. NVIDIA's [PTX ISA description of `elect.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
explains that election selects one participating lane; the first-warp
predicate is needed to select only one warp in each CTA.

All four arms computed the same seed-0 BF16 4096³ GEMM and matched
`torch.mm` **exactly** in each of three rounds. Each arm used ten warmups and
30 CUDA-event samples per round, with randomized order on one H100 SXM.

| Arm | Median of round medians (µs) | Relevant SASS wait sites |
| --- | ---: | ---: |
| Compiler-generated ordinary B | 431.488 | 4 |
| Compiler-generated multicast B | 324.016 | 2 |
| Reassembled, unchanged multicast PTX | 321.952 | 2 |
| **Patched per-CTA ordinary B PTX** | **312.832** | **2** |

Median paired differences, treatment minus control, had the same sign in all
three rounds: patched ordinary versus generated ordinary **−120.768 µs**;
patched ordinary versus reassembled multicast **−9.120 µs**; and reassembled
versus generated multicast **−2.064 µs**. The reassembled control shows a
small toolkit-version and PTX-header difference in this run. The ordinary B load
itself does not explain the original 100+ µs gap: a correct per-CTA ordinary
load in the multicast arm's compiler context is faster than the multicast
version here. The generated ordinary source has other PTX/codegen differences,
including additional `ptxas` waits; this test cannot assign all 120.768 µs
specifically to those waits.

These absolute times are lower than the earlier source-launch Gluon study
because this harness prebuilds tensor descriptors and launches compiled
kernels directly inside the timing loop. Only comparisons within this harness
should be treated as paired. This kernel still trails the separately optimized
Triton B (~198 µs) and cuBLAS (~179 µs) results from the
[matched study](../triton-h100-cublas-sass/README.md). The patched cubin is a
diagnostic artifact, not an MCTS action or a portable source implementation.
The next engineering step is to express the legal per-CTA TMA and one-wait
schedule in maintained Gluon/Triton source, then compare it with the optimized
Triton B kernel under a matched launch and workload.

[`analysis.json`](analysis.json) records all round medians, paired contrasts,
source and cubin hashes. [`results.json`](results.json) records the worker
environment and artifact bundle. [`paired.py`](paired.py), [`run.py`](run.py),
and [`analyze.py`](analyze.py) reproduce the experiment; the pinned edit is
[`patch_ptx_control_v2.py`](../triton-h100-cluster-gluon-schedule/patch_ptx_control_v2.py).
Raw samples and cubins are in ignored local `output/`. Nebius instance and
disk listings were empty after the run.
