# Does persistence explain the DRAM-read gap?

The preceding [CTA-order experiment](../triton-h100-dram-reuse/README.md)
reduced the TMA kernel's DRAM reads by up to 15.7 MB, but left roughly 20 MB
to cuBLAS. This test changes the 512-CTA TMA kernel into a 132-CTA persistent
version. Every CTA loops over several 128×256 output tiles; each tile still
uses the same BF16 TMA operand descriptors, `BK=64`, eight warps, three
stages, `tl.dot`, and pointer-based output store. We compare three tile orders
(`GROUP_M=1,8,32`) with the nonpersistent `GROUP_M=8` kernel and cuBLAS on
identical seeded 4096³ inputs. All five outputs matched exactly.

Nsight Compute used application replay, explicit `--cache-control all`, and
the same L2/DRAM read metrics for every arm. One H100 SXM worker and the pinned
Triton 3.3.1 image produced the following whole-kernel counters:

| Kernel | CTAs | Registers/thread | Shared memory/block | DRAM read |
| --- | ---: | ---: | ---: | ---: |
| Nonpersistent, group 8 | 512 | 154 | 148.504 KB | **186.659 MB** |
| Persistent, group 1 | 132 | 232 | 216.088 KB | **170.935 MB** |
| Persistent, group 8 | 132 | 230 | 216.088 KB | **186.662 MB** |
| Persistent, group 32 | 132 | 230 | 216.088 KB | **174.107 MB** |
| cuBLAS | 132 | 168 | 214.204 KB | **150.860 MB** |

Persistence by itself did not change measured DRAM reads at group 8 (a
0.002 MB difference). The persistent group-1 and group-32 traffic values
closely match their nonpersistent counterparts from the previous experiment.
Thus reducing this TMA kernel's grid to 132 CTAs does not explain cuBLAS's
remaining DRAM advantage. Persistent compilation raised register and shared
memory requirements, so this is not a pure CTA-count change at the compiler
level; the unchanged DRAM traffic across all three schedules is the stronger
observation.

The timing rounds drifted with cuBLAS too: its median went from 184.576 to
192.912 µs. Persistent group 8 was close to nonpersistent group 8, and the
other persistent schedules were slower; these three rounds are insufficient
for a stable latency ranking. NCU read lookup misses also varied between this
and the previous worker while DRAM bytes stayed stable, reinforcing that those
miss counters should not be converted directly into DRAM bytes. Exact times,
all read metrics, source hashes, and environment manifest are in
[`results.json`](results.json) and [`analysis.json`](analysis.json).

[`persistent_kernel.py`](persistent_kernel.py) contains the outer tile loop;
[`harness.py`](harness.py) checks outputs and timings; [`run.py`](run.py)
profiles each arm and releases the worker. Its VM and disk were independently
listed as deleted. The remaining traffic gap calls for a separate test of tile
orientation or clustered sharing.
