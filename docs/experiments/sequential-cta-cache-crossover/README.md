# CTA placement × asynchronous-copy cache policy

This experiment starts from
[`sequential-sol-choice-freeform-bgen50-seed0-best.cu`](../../../sequential-sol-choice-freeform-bgen50-seed0-best.cu)
and changes two factors in a 2×2 crossover:

| Variant | Productive CTA placement | `cp.async` cache qualifier |
| --- | --- | --- |
| `compact_ca` | first 1,024 physical CTA indices | `.ca` |
| `compact_cg` | first 1,024 physical CTA indices | `.cg` |
| `sparse_ca` | grid coordinates divisible by eight | `.ca` |
| `sparse_cg` | grid coordinates divisible by eight | `.cg` |

The fixed launch has a 256×256 CTA grid. Each mapping assigns every 128×128
output macro tile exactly once, yielding 1,024 productive CTAs. In the source
diff, cache policy changes one inline PTX qualifier; placement changes only
the owner test and macro-tile coordinate expressions. All other source text
is identical. The Python runner regenerates these source files from the
recorded best source and rejects an unexpected base edit.

The runner compiles, correctness-tests, and benchmarks each variant four
times on **one** pinned H100 SXM worker. The four rounds put each variant in
each measurement position once. Each evaluation uses a fresh ID, so the
worker remeasures instead of returning its cached result. After timing, the
runner collects one `lightweight_v1` NCU profile per variant. The report
contains all 30 benchmark samples per evaluation, operating-state telemetry,
compiler resources, PTX artifacts, profiles, worker manifest, and source
hashes. A `finally` path releases the Nebius worker and deletes its disk.

Run from the repository root after Nebius CLI authentication:

```bash
.venv/bin/python docs/experiments/sequential-cta-cache-crossover/run.py
```

The output is `results.json` plus four `.ptx` files. If a run fails, the
partial report remains for diagnosis and the script refuses to overwrite it.

## Results, 2026-10-23

The pinned worker reported one H100 SXM, CUDA 12.8.2, and NCU 2025.1.1.
All 16 evaluations compiled and passed the production BF16 GEMM correctness
check with zero maximum error. Each measurement contains 30 CUDA-event timing
samples; the table shows their median in microseconds. Pre/post telemetry
reported SM and memory clocks of 1980 and 2619 MHz for every evaluation,
P0 state, and no active clock event reason. The Nebius instance and disk lists
were empty after runner cleanup.

| Variant | Round 1 | Round 2 | Round 3 | Round 4 | Mean of round medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| `compact_ca` | 651.408 | 639.040 | 645.296 | 644.000 | **644.936** |
| `compact_cg` | 657.984 | 666.752 | 657.872 | 658.768 | 660.344 |
| `sparse_ca` | 672.192 | 667.984 | 675.328 | 679.376 | 673.720 |
| `sparse_cg` | 688.016 | 685.216 | 688.176 | 691.216 | 688.156 |

Within this one source family, compact placement saves **28.784 µs with
`.ca`** and **27.812 µs with `.cg`**. The `.ca` qualifier saves **15.408 µs
with compact placement** and **14.436 µs with sparse placement**. The
difference of those two cache-policy effects is 0.972 µs; the per-round
interaction estimates range from -10.480 to +9.248 µs. Thus the measured
benefits are largely additive here. This experiment does not establish a
strong placement × cache-policy interaction.

The compact-`.ca` to sparse-`.cg` difference is 43.220 µs in this controlled
source family, numerically close to the 43.408 µs gap between the earlier
freeform and catalog bests. That numerical match does not prove a full
decomposition of the earlier gap: the two historical kernels also differed
in copy implementation, output path, and code generation, and were measured
on separate workers.

All four variants used 128 registers/thread, 40 KiB static shared memory,
and zero compiler-reported spill or stack bytes. Within each placement pair,
the raw PTX differs at exactly eight `cp.async.ca` versus `cp.async.cg`
instruction sites and nowhere else. One lightweight profile per variant
reported:

| Variant | L1TEX throughput (% peak active) | L2 throughput (% peak elapsed) | Achieved occupancy | Executed instructions |
| --- | ---: | ---: | ---: | ---: |
| `compact_ca` | 61.86 | 36.68 | 24.00% | 130.998M |
| `compact_cg` | 38.72 | 54.98 | 24.02% | 130.998M |
| `sparse_ca` | 62.73 | 36.23 | 22.57% | 130.490M |
| `sparse_cg` | 38.89 | 54.22 | 22.50% | 130.490M |

The throughput percentages are activity relative to separate peak
denominators, not measured bytes or cache hit rates. The profiles support a
change in cache-level activity, but do not by themselves explain the latency
gain. The earlier catalog call-38→39 isolated `.cg`→`.ca` edit *lost* about
40 µs; that kernel had a different implementation. Cache policy should
therefore remain a measured, kernel-specific choice rather than a fixed
strategy instruction.

For this freeform kernel, an MCTS realization need not cross a slower
intermediate state to reach compact-`.ca`: compact-`.cg` and sparse-`.ca`
both beat sparse-`.cg`. Jointly proposing the two decisions might still save
generation calls. That search-budget claim needs a matched search experiment;
the four-kernel timing crossover does not test model proposal probabilities.
