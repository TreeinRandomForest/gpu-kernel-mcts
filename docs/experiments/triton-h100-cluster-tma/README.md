# Pinned Triton two-CTA TMA feasibility probe

This probe asks whether the existing descriptor-based Triton 3.3.1 H100 GEMM
can be launched as a two-CTA cluster and emit TMA multicast instructions.
It uses the unchanged source from
[`triton-h100-tma-load`](../triton-h100-tma-load/tma_kernel.py), pinned worker
image, 4096³ BF16 inputs, and cuBLAS correctness reference. Each configuration
ran in a fresh remote Python process, so a failed compilation did not affect
later probes. The runner saved a hash-checked PTX artifact for any successful
configuration.

| Configuration | Logical output tile | CTAs/cluster | Outcome |
| --- | ---: | ---: | --- |
| Single-CTA control | 128×256 | 1 | Exact correctness; TMA PTX, no multicast |
| Two CTAs, same tile | 128×256 | 2 | Triton LLIR `PassManager::run failed` |
| Two CTAs, double M | 256×256 | 2 | Triton LLIR `PassManager::run failed` |
| Two CTAs, double N | 128×512 | 2 | Descriptor creation rejected the tile (`invalid argument`) |

The two configurations with legal descriptors failed before PTX generation;
therefore this run did **not** benchmark clustered TMA or demonstrate
multicast. The 128×512 case is not evidence about compiler multi-CTA support,
because it failed earlier in descriptor validation. The successful single-CTA
control emitted six 2D TMA load sites and no `.multicast::cluster` qualifier.
The LLIR error does not establish that all Triton 3.3.1 cluster kernels are
unsupported; it establishes only that these direct `num_ctas=2` conversions
of our descriptor-load GEMM did not compile. The compiler's detailed pass
diagnostic was not captured by this probe.

[`probe_results.json`](probe_results.json) contains each outcome, source/PTX
hashes, worker manifest, and error tails. The successful PTX is under ignored
`output/`. No latency result is claimed. Nebius instance and disk listings
were empty after the probe.

For an already validated cluster mechanism, the repository's separate CuTe
DSL schedule experiment measured the same `(128,256)` CTA tile at 193.024 µs
with cluster `(1,1)` and 184.880 µs with cluster `(2,1)`; see
[`cutedsl-schedule-tuning-v1.md`](../cutedsl-schedule-tuning-v1.md). That is
evidence for the CuTe implementation, not a controlled comparison with the
current Triton kernel. A future Triton comparison needs an implementation
that explicitly maps two neighboring output tiles and multicast ownership,
followed by PTX verification and same-worker paired timing.
