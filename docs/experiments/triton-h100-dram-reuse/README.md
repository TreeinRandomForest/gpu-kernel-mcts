# Locating the cuBLAS DRAM-read advantage

This follow-up compares fast B's [TMA-load variant](../triton-h100-tma-load/README.md)
with `GROUP_M=1`, `8`, and `32`, plus PyTorch/cuBLAS, on one H100 and the same
seeded 4096³ BF16 tensors. Only `GROUP_M` changes among Triton variants: it
changes the order in which the fixed 512 CTAs visit output tiles. The kernel
still uses 128×256 output tiles, 64-wide K slices, eight warps, three stages,
TMA A/B loads, and the same output store. These variants test whether CTA
ordering changes cross-CTA cache reuse. The harness checks every output against
cuBLAS before timing or profiling.

The profiler collects read sectors requested from L2, read lookup hits and
misses, and actual DRAM read sectors. Each kernel is profiled separately with
Nsight Compute application replay under both `--cache-control all` and
`--cache-control none`, after ten target-specific warmups. `all` asks NCU to
flush caches between replay passes; `none` preserves the warmed cache state.
Neither regime is assumed to reproduce the CUDA-event timing loop's cache
state. The same profiler settings and input initialization are used for every
arm. [NVIDIA's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
documents replay and cache-control behavior.

## Evidence already present in earlier full NCU exports

These values come from the preceding matched-input
[cuBLAS SASS](../triton-h100-cublas-sass/README.md) and
[TMA-load](../triton-h100-tma-load/README.md) runs. They used the same pinned
container and NCU's default cache control, but were collected in different
profiling invocations and are therefore a starting clue, not this experiment's
controlled result.

| Read counter | Fast B | TMA B | cuBLAS |
| --- | ---: | ---: | ---: |
| L2 read sectors requested by TEX source unit | 27,647,804 | 27,409,296 | 27,337,928 |
| L2 TEX read lookup hits | 18,819,800 | 19,400,600 | 19,762,744 |
| L2 TEX read lookup misses | 8,169,316 | 7,974,620 | 6,633,976 |
| DRAM read sectors | 5,833,672 | 5,833,576 | 4,714,304 |
| DRAM read bytes | 186.678 MB | 186.674 MB | 150.858 MB |

The L2 request totals are close, while cuBLAS has about 1.34 million fewer
L2 read lookup misses than TMA B and 1.12 million fewer DRAM read sectors.
This places the measured traffic difference at the L2-to-DRAM side of the
hierarchy. It does **not** yet identify the scheduling or sharing mechanism.
L2 lookup misses are not interchangeable with DRAM sectors: the latter counts
actual memory traffic and can be affected by other requests and cache behavior.

## Controlled same-worker result

All three group variants matched cuBLAS's BF16 output exactly. The NCU cold
and warm traffic measurements were stable to within about 0.07 MB of DRAM read
traffic per arm. Values below are the explicit `--cache-control all` profiles:

| Kernel | L2 TEX read sectors requested | L2 TEX read lookup misses | DRAM read |
| --- | ---: | ---: | ---: |
| TMA, `GROUP_M=1` | 38,171,176 | 8,251,648 | **170.973 MB** |
| TMA, `GROUP_M=8` | 26,565,628 | 8,342,488 | **186.686 MB** |
| TMA, `GROUP_M=32` | 26,843,764 | 8,008,824 | **174.081 MB** |
| cuBLAS | 26,235,264 | 6,849,080 | **150.821 MB** |

Changing only CTA tile order reduced TMA DRAM reads by **15.713 MB** with
`GROUP_M=1`, or **12.605 MB** with `GROUP_M=32`, relative to `GROUP_M=8`.
Thus CTA ordering explains part of the original roughly 36 MB discrepancy.
Even the lowest-traffic order still reads **20.151 MB** more than cuBLAS.

`GROUP_M=1` also produces far more L2 read requests, mostly hits, despite
reading fewer bytes from DRAM. The aggregate L2 lookup-miss counter changes
much less than DRAM traffic for that arm. We therefore cannot derive DRAM
bytes by multiplying L2 lookup misses by sector size; the counter classes are
not a one-to-one physical path attribution. The cuBLAS arm still has fewer L2
read misses than every TMA arm.

The `--cache-control none` profiles reproduced nearly the same DRAM ranking:
171.0, 186.7, 174.1, and 150.8 MB for group 1, 8, 32, and cuBLAS. The CUDA
event medians drifted upward across all kernels during the timing rounds,
including cuBLAS (188.5 → 197.9 µs), so these rounds are not strong evidence
for a latency ranking among the group variants. Exact per-sample times and
worker provenance are in [`results.json`](results.json); all traffic counters
are in [`analysis.json`](analysis.json). The worker VM and disk were both
listed as deleted after the run.

The [persistent follow-up](../triton-h100-dram-persistent/README.md) reduced
the grid to 132 CTAs without reducing DRAM reads at a fixed tile order. The
remaining gap could involve tile orientation, cuBLAS's two-CTA clusters, or a
combination of data placement and loading policy.

[`harness.py`](harness.py) implements the matched-input variants,
[`run.py`](run.py) provisions one worker and gathers the eight profiles, and
[`analyze.py`](analyze.py) checks completeness and reports the traffic counters.
