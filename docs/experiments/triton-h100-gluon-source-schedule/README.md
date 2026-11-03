# Gluon source control for ordinary B TMA scheduling

This experiment expresses the validated [PTX per-CTA B-load
control](../triton-h100-cluster-gluon-ptx-control-v2/README.md) in maintained
Gluon source. The kernel places ordinary and multicast `tma.async_load` calls
behind a runtime-uniform `B_MODE` branch and marks that argument
`do_not_specialize`. Both B modes therefore share one compiled WGMMA mainloop.
The static Gluon arms from the earlier experiment, optimized Triton B, and
cuBLAS are measured in the same harness.

One Nebius H100 SXM ran seed-0 BF16 4096³ GEMM with three randomized-order
rounds, ten warmups and 30 CUDA Graph replay samples per arm per round. All
six arms matched the cuBLAS reference **exactly**. The two source modes have
the same cubin hash. Its PTX contains a runtime branch to either a per-CTA
ordinary B TMA or a one-owner multicast B TMA; its SASS has two
`WARPGROUP.DEPBAR.LE` sites. The static ordinary Gluon cubin has four such
sites. The dual-path cubin has both TMA instruction sites; only the selected
branch executes on a launch.

| Arm | Median of round medians (µs) | SASS wait sites |
| --- | ---: | ---: |
| Static Gluon, ordinary B | 423.312 | 4 |
| Static Gluon, multicast B | 314.080 | 2 |
| **Source Gluon, ordinary B** | **296.176** | **2** |
| Source Gluon, multicast B | 305.632 | 2 |
| Optimized Triton B | 185.872 | — |
| cuBLAS | 177.776 | — |

Median paired differences (treatment minus control) were −127.216 µs for
source ordinary versus static ordinary, −8.688 µs for source ordinary versus
source multicast, +110.304 µs for source ordinary versus optimized Triton B,
and +8.656 µs for Triton B versus cuBLAS. The source ordinary improvement
and its remaining gap to Triton B had the same sign in all three rounds.
Source multicast versus static multicast was −9.072 µs median but changed
sign in one round.

The result confirms that ordinary per-CTA B TMA can be expressed in Gluon
source with the one-loop-wait compiler schedule and exact output. The runtime
branch is a code-generation aid: this is a source-level experiment pinned to
Triton 3.9.0, not yet a general search action. The 127 µs improvement cannot
be assigned entirely to the wait instructions because the whole compiled
control flow changes. The Gluon result still trails optimized Triton B by
about 110 µs in this same run. The 8.656 µs Triton B/cuBLAS gap here differs
from the earlier ~19 µs event-timed study because this run uses CUDA Graph
replay and Triton 3.9.0; only paired results within each study should be
compared directly.

The next step is a matched compiler/profile comparison of this source Gluon
arm with optimized Triton B. That should identify which remaining work is in
TMA setup, matrix-multiply scheduling, tile scheduling, or epilogue before
transferring a mechanism into the optimized kernel.

[`analysis.json`](analysis.json) records all round medians, paired differences,
compiler resources, artifact hashes, and run identity. [`results.json`](results.json)
records the worker environment and raw artifact archive. [`kernel.py`](kernel.py),
[`paired.py`](paired.py), [`run.py`](run.py), and [`analyze.py`](analyze.py)
reproduce the experiment. Raw samples, PTX, cubins, and SASS are in ignored
local `output/`. Nebius instance and disk listings were empty afterward.
