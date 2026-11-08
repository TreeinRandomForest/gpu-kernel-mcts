# Source lifetime of the output scratch tile

This one-change H100 test asks whether allocating the C shared-memory tile
after the K loop lets the compiler reuse the A/B pipeline storage. The
[candidate source](kernel.py) moves one allocation statement from the top of
the [validated 256-wide persistent Gluon kernel](../triton-h100-gluon-factorial/kernel.py)
to immediately after the final WGMMA wait and barrier invalidation. Tile
shape, TMA loads, WGMMA, persistence, and launch geometry are unchanged.

Both arms compiled and passed bitwise equality against `torch.mm` on the
4096³ BF16 `A @ B.T` workload. Ten rotating-order CUDA Graph rounds on one
H100 SXM gave:

| Arm | Median of round medians | Shared memory / CTA | Registers / thread |
| --- | ---: | ---: | ---: |
| Original 256-wide persistent | 189.272 µs | 213,016 B | 154 |
| C scratch allocated after K | 189.696 µs | 213,016 B | 154 |
| cuBLAS | 181.232 µs | — | — |

The paired candidate-minus-original difference was **+0.080 µs median**,
with six slower and four faster rounds. The round differences ranged from
−6.672 to +15.376 µs while the worker's absolute timings drifted upward.
This does not demonstrate a speedup or a stable slowdown.

Separate NCU captures measured identical executed instruction counts
(28,594,528), TMA load counts (65,536), and GMMA counts (1,048,576). DRAM
reads were 174.077 versus 174.078 MB. PTX instructions match after removing
source-location and debug metadata. Both compiled variants use the same
213,016 B shared-memory allocation. Moving the source statement alone did
not shorten the physical lifetime of the output scratch in this persistent
loop. A likely explanation is that the compiler reserves separate shared
regions across iterations; this experiment does not isolate the allocator's
reason.

See [paired observations](output/paired.json), [NCU and compiler artifacts](output),
and [run provenance](results.json). The profile cubin hashes match their
timed variants. The worker was released after profiling. This is one shape,
input seed, and H100 worker.
