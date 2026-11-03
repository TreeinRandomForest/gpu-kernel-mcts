# Pure-Triton H100 two-CTA B multicast

This standalone probe uses Triton's Gluon API to implement the cluster change
in source. It is independent of CuTe and does not enter MCTS. The kernel follows
Triton's [Hopper WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html)
and [multi-CTA TMA multicast](https://triton-lang.org/main/getting-started/tutorials/gluon/multicta.html)
patterns. Both arms compute one 128×256 C tile per CTA with 64-wide K slices.
The clustered arm assigns adjacent M tiles to two CTAs, shards A and C along
M, and broadcasts the shared B tile with `tma.async_load(..., multicast=True)`.
Its control uses the same Gluon source, tile and warp count with one CTA and
no multicast. This is a simple one-stage mainloop; it is not a port of the
existing optimized Triton kernel.

The experiment ran on one Nebius H100 SXM using Triton 3.9.0 in image
`docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1`.
The local Podman repository digest could not be resolved by Docker Hub in a
follow-up VM startup, so it must not be treated as a portable pull reference.
Each arm used the same seed-0 BF16 input tensors, `torch.mm`/cuBLAS reference,
10 warmups, and 30 CUDA-event samples per round. Arm order changed across
three rounds. Both arms passed the repository's `rtol=atol=0.02` elementwise
tolerance in all six evaluations.

| Round | Single CTA median (µs) | Two-CTA multicast median (µs) | Cluster penalty (µs) |
| ---: | ---: | ---: | ---: |
| 0 | 493.712 | 535.200 | 41.488 |
| 1 | 496.928 | 533.680 | 36.752 |
| 2 | 498.064 | 534.000 | 35.936 |

Median paired penalty: **36.752 µs (7.40%)**. PTX and SASS both confirm one
cluster multicast TMA load site in the two-CTA arm and none in the control.
Both binaries report 154 registers per thread, zero spills, and 64 KiB shared
memory per CTA. The run establishes that this explicit Triton source compiles,
multicasts, and is correct; multicast did not improve latency in this simple
kernel. The paired timing includes all cluster launch, synchronization, layout,
and multicast effects, so it does not isolate the cost or benefit of multicast.

The earlier [CuTe experiment](../triton-h100-cluster-cute/README.md) showed a
small improvement with its separately optimized pipeline. Those absolute
times are not directly comparable: this probe uses a different kernel and
input generator. A useful next diagnostic is a third arm with two CTAs and
identical tile mapping but ordinary B loads. That would separate the cluster
cost from multicast's incremental effect before attempting a staged Gluon
mainloop.

[`kernel.py`](kernel.py) is the source rewrite, [`paired.py`](paired.py) is the
correctness and timing harness, and [`run.py`](run.py) provisions and cleans up
the worker. [`analysis.json`](analysis.json) is reproduced by
`python docs/experiments/triton-h100-cluster-gluon/analyze.py`. Raw samples,
PTX, cubins, and disassembly are retained under ignored `output/`.
[`results.json`](results.json) records run identity and environment. Nebius
instance and disk listings were empty after the run.
