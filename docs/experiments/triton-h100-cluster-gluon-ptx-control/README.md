# PTX control for the B load and WGMMA wait schedule

The [NCU study](../triton-h100-cluster-gluon-ncu/README.md) found four
warp-group waits in the ordinary-B clustered kernel but only two in the
multicast-B kernel. The source-level request-order test retained the four
waits. This control starts from the multicast kernel's pinned PTX and replaces
only its B TMA instruction line: it changes the elected issuer from the
multicast owner to one issuer per CTA, replaces `shared::cluster` multicast
TMA with ordinary `shared::cta` TMA, and drops the multicast mask argument.
The original and patched PTX were both assembled with the same CUDA 12.8
`ptxas`; their PTX headers were reduced from 9.4 to 8.7 for this older
assembler. Both assembled successfully.

| SASS observation | Unchanged multicast PTX | Patched ordinary-TMA PTX |
| --- | ---: | ---: |
| `WARPGROUP.DEPBAR.LE` sites | 2 | 2 |
| `UTMALDG.2D.MULTICAST` sites | 1 | 0 |
| Ordinary `UTMALDG.2D` sites | 1 | 2 |
| `HGMMA` sites | 4 | 4 |

This static result shows that the ordinary TMA opcode **does not, by itself,
force `ptxas` to emit four waits**. It does not show that the patched program
works. In the H100 runtime check, the patched cubin was selected first and
did not return for over one minute; the process was terminated and the worker
cleaned up. The validation harness recorded no correctness or latency result
for it. A prior attempt stopped before launch because the direct low-level
launcher was passed a two-dimensional grid; that harness bug was fixed before
the runtime attempt. The patched instruction also changes cluster addressing
and transaction completion semantics, so static assembly is not proof that
the replacement is a valid per-CTA load.

The conclusion for this failed patch is limited: the extra waits are a
compiler schedule outcome of the complete ordinary-B PTX, rather than a
hardware requirement of an ordinary TMA instruction. The
[corrected control](../triton-h100-cluster-gluon-ptx-control-v2/README.md)
restores the first-warp guard and passes exact correctness and paired timing.
This first patched cubin must not be used as an optimization candidate.

[`analysis.json`](analysis.json) records the PTX/cubin hashes and static SASS
counts. [`runtime_hung.json`](runtime_hung.json) and
[`grid_failed.json`](grid_failed.json) record the two VM attempts;
[`paired_grid_failed.json`](paired_grid_failed.json) captures the first
low-level launch error. The pinned edit is
[`patch_ptx_control.py`](../triton-h100-cluster-gluon-schedule/patch_ptx_control.py),
and [`paired.py`](paired.py), [`run.py`](run.py), and [`analyze.py`](analyze.py)
are the runtime and analysis harnesses. PTX, cubins, and SASS are in ignored
local `output/` under the schedule experiment. Nebius instance and disk
listings were empty after termination.
