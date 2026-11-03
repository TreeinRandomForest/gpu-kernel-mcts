# Reordering ordinary B TMA requests

The [NCU comparison](../triton-h100-cluster-gluon-ncu/README.md) found four
executed warp-group wait sites in the ordinary-B two-CTA pipeline versus one
loop wait in the multicast pipeline. PTX has the same two explicit
`wgmma.wait_group` sites in both arms, so the extra SASS waits are introduced
by `ptxas`. This experiment changes the order of ordinary A/B TMA load
requests while keeping the same tiles, cluster, descriptors, and computation.

One Nebius H100 SXM ran three randomized-order rounds with 30 CUDA-event
samples per arm per round. Every one of the nine evaluations passed exact
correctness against `torch.mm`.

| Arm | Median of round medians (µs) | SASS `WARPGROUP.DEPBAR.LE` sites |
| --- | ---: | ---: |
| Ordinary B, A request first | 505.232 | 4 |
| Ordinary B, B request first | 480.640 | 4 |
| Multicast B, A request first | 391.376 | 2 (one loop, one epilogue) |

Moving B first saved **24.688 µs** in the median paired comparison, with the
same sign in all rounds. It did **not** remove the extra wait sites. Multicast
remained **88.592 µs** faster than ordinary B-first. All arms reported 154
registers/thread, zero spills, and 98,312 bytes of shared memory per CTA.
This establishes request order as a contributing source-level factor but does
not isolate the inserted waits from the B transfer mechanism. A separate
CTA-local B descriptor test follows.

[`analysis.json`](analysis.json) contains round medians, paired deltas,
resource counts, and cubin hashes. [`results.json`](results.json) records the
worker manifest and artifact hash. [`kernel.py`](kernel.py),
[`paired.py`](paired.py), [`run.py`](run.py), and [`analyze.py`](analyze.py) are
the source and reproducible harness. The raw samples, PTX, cubins, and SASS
are in ignored `output/`. A sandboxed first launch could not reach Nebius DNS
and provisioned nothing; its pre-run record is [`startup_failed.json`](startup_failed.json).
