# Gluon cluster, B multicast, and double buffering

This follow-up isolates three source choices in a pure-Triton 3.9.0 Gluon
H100 GEMM: one versus two CTAs, ordinary versus multicast B TMA loads in the
two-CTA cluster, and a serial versus double-buffered mainloop. Every CTA owns
the same 128×256 C tile and reduces through 64-wide K slices. The pipelined
arms allocate two A/B shared-memory buffers and issue the next TMA loads while
the previous WGMMA is in flight. This is a simple overlap pattern, not a fully
warp-specialized producer/consumer pipeline. The six variants use one source
with compile-time choices; none enters MCTS.

One Nebius H100 SXM and image
`docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1` ran all arms.
Each used the same seed-0 BF16 tensors, `torch.mm`/cuBLAS reference, 10
warmups, and 30 CUDA-event samples in each of three randomized-order rounds.
All 18 evaluations matched the reference **exactly**. The median of the three
round medians is shown below; paired contrasts are computed within each round.

| Arm | Serial (µs) | Double buffered (µs) |
| --- | ---: | ---: |
| One CTA, ordinary B | 494.864 | **353.424** |
| Two CTAs, ordinary B | 508.288 | 506.832 |
| Two CTAs, multicast B | 531.280 | 390.256 |

Selected median paired differences (treatment minus control):

| Comparison | Difference (µs) |
| --- | ---: |
| Two CTAs ordinary B vs one CTA, serial | +13.632 |
| Multicast vs ordinary B, serial two-CTA | +22.992 |
| Double buffering vs serial, one CTA | −141.440 |
| Double buffering vs serial, ordinary-B two-CTA | −1.760 |
| Multicast vs ordinary B, double-buffered two-CTA | **−115.440** |
| Double-buffered two-CTA multicast vs double-buffered one CTA | +36.304 |

These differences have the same sign in all three rounds. The important
interaction is that multicast is slower in the serial cluster but substantially
faster in the double-buffered cluster. Even with multicast, the double-buffered
cluster is slower than the matched single-CTA kernel. The controlled source
setting changes compiler output as well as hardware instructions; these event
times do not independently quantify multicast traffic, cluster synchronization,
or overlap. In particular, the ordinary-B clustered pipeline's lack of speedup
is an observed result, not yet an explained stall mechanism.

PTX and SASS both contain one `UTMALDG.2D.MULTICAST` site in each multicast
arm and none in ordinary-B arms. Each cubin is stable across rounds and
distinct across arms. All variants report 154 registers/thread and zero
spills. Shared memory rises from 65,536 bytes to 98,312 bytes per CTA in the
double-buffered arms. Static instruction sites and reported resources are not
executed instruction or DRAM-traffic measurements.

These kernels remain substantially slower than the separately optimized
Triton B kernel in the [matched cuBLAS study](../triton-h100-cublas-sass/README.md),
so this experiment does not resolve that kernel's approximately 19 µs gap to
cuBLAS. The next useful diagnostic is an NCU comparison of the three
double-buffered arms, followed by a staged implementation derived from the
optimized Triton source or a matched cuBLAS reference in the same run.

[`kernel.py`](kernel.py) is the source, [`paired.py`](paired.py) is the harness,
and [`analysis.json`](analysis.json) contains all round medians, paired
contrasts, compiler resources, and artifact hashes. Run `python
docs/experiments/triton-h100-cluster-gluon-v2/analyze.py` to regenerate it
from ignored `output/` (samples, PTX, cubins, and SASS). [`results.json`](results.json)
records the worker manifest and run identity. A first attempt used a local
Podman digest that Docker Hub could not resolve; its pre-benchmark record is
[`startup_failed.json`](startup_failed.json). The successful retry used the
tested image tag. Nebius instance and disk listings were empty afterward.
