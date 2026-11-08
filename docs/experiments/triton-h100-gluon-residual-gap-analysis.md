# Where the split A-reuse kernel spends extra work

This analysis uses the [same-worker wait-1 comparison](triton-h100-gluon-a-reuse-wait1/README.md)
and its [NCU source/SASS CSV](triton-h100-gluon-a-reuse-wait1/output).
The fixed workload is 4096³ BF16 `A @ B.T`, with 66 persistent two-CTA
clusters. Each cluster computes the same 256-wide output region.

## Reading one K step

The original Gluon kernel loads one A tile and one 256-wide B tile, issues
one `m64n256k16` WGMMA group (with four K subinstructions), and eventually
stores one 256-wide C tile. The split kernel loads **the same one A tile**
and two adjacent 128-wide B tiles, issues two `m64n128k16` groups, and
stores two adjacent C tiles. It reuses A from shared memory; it does not
multicast A or read another CTA's shared memory explicitly. Both kernels
cover the same output elements and use a per-step `wait_group 1`.

| Same-worker measure | Original 256-wide | Split A reuse, wait 1 |
| --- | ---: | ---: |
| CUDA Graph latency, median of ten round medians | 188.048 µs | 204.128 µs |
| Dynamic warp instructions, all opcodes | 28.595M | 33.736M |
| `HGMMA` dynamic instructions | 1.049M | 2.097M |
| `UTMALDG` dynamic instructions | 0.524M | 0.786M |
| `UTMASTG` dynamic instructions | 0.004M | 0.008M |
| `WARPGROUP` dynamic instructions | 0.528M | 0.791M |
| `BAR` dynamic instructions | 1.323M | 1.593M |
| `R2UR` dynamic instructions | 2.116M | 3.181M |
| `UMOV` dynamic instructions | 3.950M | 4.744M |
| NCU DRAM reads | 174.127 MB | 174.087 MB |
| NCU tensor pipe active, elapsed | 87.11% | 83.96% |

The dynamic opcode counts come from the NCU SASS source CSV's *Instructions
Executed* column, grouped by opcode. They count instruction execution, not
cycles. The extra B load, second MMA, and second store each require setup
and synchronization, reflected in the extra address, uniform-register,
barrier, and warpgroup instructions. The 5.142M total-instruction increase
cannot be converted directly into microseconds: operations overlap, and the
MMA instruction shapes differ. DRAM reads remain almost unchanged, so this
test does not support extra DRAM traffic as the cause of the split gap.

The original kernel already realizes A reuse across the full 256-wide N
region. Splitting it adds instruction work to express that same reuse at
128-wide granularity. The split-wait-0 kernel was much slower because it
serialized WGMMA; [wait 1 recovered about 47 µs](triton-h100-gluon-a-reuse-wait1/README.md).
Allowing wait 2 gave no material additional gain.

## One source-level follow-up

The generalized persistent kernel allocates its C shared-memory scratch
alongside the A/B pipeline buffers. A separate, nonpersistent Gluon source
allocated C after the K loop and used less shared memory, so we moved just
that statement in the persistent kernel. The [paired experiment](triton-h100-gluon-c-scratch-lifetime/README.md)
passed exactness but produced the same 213,016 B shared allocation and the
same machine instructions. Its +0.080 µs paired median difference does not
establish a performance effect. Source-level lifetime alone did not make
the compiler overlay these buffers in the persistent loop.

The remaining approximately 7 µs original-Gluon versus cuBLAS gap is a
different comparison. cuBLAS reads about 151 MB from DRAM on this workload
and has a different `m64n128`/multicast/3D-TMA schedule. These aggregate
measurements do not assign that gap to one cause. The next useful source
experiment should target a specific instruction or traffic mechanism while
retaining the original 256-wide output tile, then use paired timing and
compiler/NCU verification before adding any search strategy.
