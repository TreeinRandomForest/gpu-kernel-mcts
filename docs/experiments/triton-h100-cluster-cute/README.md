# Paired CuTe two-CTA B-multicast experiment

This standalone experiment asks whether sharing a B tile across two M-adjacent
CTAs helps the fixed 4096³ BF16 H100 GEMM. It uses the pinned CUTLASS 4.5.1
Hopper CuTe DSL example through the repository's comparable evaluation contract.
The CTA output tile is `(128,256)` in both arms. The only schedule change is
cluster shape `(1,1)` versus `(2,1)`. The CuTe source maps the two cluster CTAs
to adjacent M tiles, computes a B multicast mask, and passes that mask to the
TMA copy. A remains CTA-private.

The runner used one Nebius H100 SXM worker and pinned image
`docker.io/saarora/gpu-kernel-mcts:cutedsl-wide-swizzle-v71`. It randomized
arm order in each of three rounds. Each evaluation used the same seed-0 input,
cuBLAS reference, 10 warmups, and 30 CUDA-event timing samples. All six
evaluations passed correctness with zero maximum and mean error. The source
hash matched across rounds and arms; each arm's generated cubin was stable
across rounds and different from the other arm.

| Round | Single CTA median (µs) | Two-CTA multicast median (µs) | Saving (µs) |
| ---: | ---: | ---: | ---: |
| 0 | 193.664 | 187.568 | 6.096 |
| 1 | 194.912 | 188.016 | 6.896 |
| 2 | 195.008 | 187.888 | 7.120 |

Median paired saving: **6.896 µs (3.54%)**. The recovered single-CTA cubin
contains four `UTMALDG.3D` sites and no multicast sites. The clustered cubin
contains four `UTMALDG.3D` sites, two of them
`UTMALDG.3D.MULTICAST`. These are static instruction sites, not executed
counts or proof of a particular DRAM-traffic reduction. The measured timing
change includes all effects of cluster launch, mapping, synchronization, and
multicast. Three rounds on one worker establish a bounded improvement for
this implementation, not a general speedup estimate.

This is a **CuTe DSL result**. It does not establish performance for the
current Triton kernel: direct `num_ctas=2` conversion of its descriptor TMA
loads failed LLIR lowering in the separate
[`triton-h100-cluster-tma`](../triton-h100-cluster-tma/README.md) probe. An
explicit Triton source that assigns separate output tiles and B multicast
ownership still needs to compile and pass the same measurement contract.

Run `python docs/experiments/triton-h100-cluster-cute/analyze.py` to reproduce
[`analysis.json`](analysis.json) from the retained output. The large raw timing,
MLIR, fatbinary, cubin, and SASS artifacts are under ignored `output/`.
[`results.json`](results.json) records the worker manifest and run identity;
[`binary_audit.json`](binary_audit.json) records the recovered binary hashes.
The first artifact capture used a decoder that added bytes to escaped MLIR
backslashes, so its original `.fatbin` files are invalid. The intact saved
MLIR was decoded with [`recover_binaries.py`](recover_binaries.py), checked
against the CUDA fatbinary length, and used for the cubins and SASS analysis.
The capture code now applies the corrected decoder and disassembles the
embedded cubin directly.

Nebius instance and disk listings were empty after the run.
