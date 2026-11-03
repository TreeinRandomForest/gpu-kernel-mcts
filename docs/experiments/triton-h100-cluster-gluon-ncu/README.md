# NCU comparison of double-buffered Gluon GEMM arms

This follow-up profiles the three double-buffered arms from the
[six-arm experiment](../triton-h100-cluster-gluon-v2/README.md) on one Nebius
H100 SXM. Each uses the identical source, seed-0 BF16 inputs, tile, and grid.
The profiling harness checks exact output against `torch.mm`, warms up ten
times, and runs one NVTX-selected launch under NCU 2025.1.1 `--set full` with
cache control. All three passed exact correctness. Each report has one `gemm`
kernel row. The source hash agrees with the earlier six-arm experiment.

The preceding randomized CUDA-event study is the latency comparison. Its
median of round medians was **353.424 µs** (single CTA), **506.832 µs**
(two CTAs, ordinary B loads), and **390.256 µs** (two CTAs, B multicast).
Multicast saved 115.440 µs in the paired two-CTA comparison; the multicast
cluster still trailed single CTA by 36.304 µs. NCU duration below is a
profiler replay observation, not a replacement for those event timings.

| NCU metric | Single CTA | Two CTAs, ordinary B | Two CTAs, B multicast |
| --- | ---: | ---: | ---: |
| Duration (µs) | 308.224 | 477.664 | 345.632 |
| DRAM read (MB) | 174.098 | 174.076 | 174.086 |
| DRAM write (MB) | 28.662 | 28.674 | 28.648 |
| L2 read sectors from TEX | 26.828M | 25.191M | 32.124M |
| Tensor pipe active (% of elapsed cycles) | 57.27 | 38.68 | 51.99 |
| Barrier stall samples | 2,394 | **17,299** | 4,119 |
| Registers/thread | 154 | 154 | 154 |
| Shared memory/CTA (KB, NCU) | 99.336 | 99.336 | 99.336 |

The ordinary-B clustered arm has **four `WARPGROUP.DEPBAR.LE` sites in the
loop**, totaling 1,048,576 executed instructions and 14,946 samples at those
sites. The single-CTA and multicast arms each have one loop wait plus one
epilogue wait, totaling 266,240 executions; their wait sites have 299 and 251
samples, respectively. All three execute 1,048,576 `HGMMA` instructions and
524,288 TMA loads. The multicast arm has a dynamically executed
`UTMALDG.2D.MULTICAST` B load; the other arms have ordinary `UTMALDG.2D`
loads. NCU reports the same launch waves per SM (3.88), register count,
shared-memory allocation, and nearly identical DRAM bytes for all arms.

This points to **warp-group waits and reduced tensor-pipe utilization** as
the immediate reason the ordinary-B cluster is slow. The controlled source
flag changes generated code as well as TMA behavior, so the experiment cannot
attribute the gain to multicast transfer alone. DRAM traffic does not explain
the difference; L2 read sectors are actually higher with multicast. The next
causal test should make the ordinary-B cluster use the same one-wait WGMMA
schedule while leaving its B loads ordinary, then repeat correctness and
paired event timing. If that closes the gap, the optimization target is the
compiler schedule; if it does not, inspect TMA and cluster synchronization.

This Gluon kernel remains much slower than the separately optimized Triton
kernel (about 198 µs) and cuBLAS (about 179 µs) in the
[matched reference study](../triton-h100-cublas-sass/README.md). Directly
porting this cluster mainloop into the optimized kernel is not supported by
these measurements.

[`analysis.json`](analysis.json) contains exact counters, units, SASS counts,
and hashes. Run `python docs/experiments/triton-h100-cluster-gluon-ncu/analyze.py`
to regenerate it from ignored `output/` (`.ncu-rep`, raw CSV, source-view SASS
CSV). [`results.json`](results.json) records the run identity, environment,
correctness, and artifact hashes. The profiling code is [`profile_one.py`](profile_one.py)
and [`run.py`](run.py). The Nebius instance and disk listings were empty after
the run.
