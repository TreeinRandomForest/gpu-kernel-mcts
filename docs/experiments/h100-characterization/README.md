# Standalone H100 characterization, phase 1

The maintained inventory that separates this experiment's path observations
from graph-link properties is
[H100 hardware model: current state](../../h100-hardware-model.md).

This experiment runs outside MCTS and makes no model API calls. It starts a
versioned, evidence-labeled H100 measurement set with **global-copy path**
timing across payload sizes. The existing nominal H100 graph keeps physical
link latency and sustainable bandwidth unknown until a probe isolates them.

The Triton probe copies contiguous, aligned `int32` buffers. CUDA events time
64 launches captured in a CUDA Graph, divided by launch count, across 15
rounds. One policy repeats a single source/destination pair; the other rotates
through 4–16 pairs with a larger working set. Sizes are 4 KiB, 64 KiB, 1 MiB,
8 MiB, 32 MiB, 128 MiB, and 256 MiB; rotation starts at 8 MiB. The output
keeps every raw timing sample, working-set size, access pattern, requested
bytes, effective end-to-end throughput, GPU identity, tool versions, and clock
snapshots. A minimal kernel gives a graph-replay launch floor. Output values
are **end-to-end path observations**, not per-link HBM or L2 specifications.

Separate NCU passes inspect an 8 MiB repeated pair and a 128 MiB rotating
case using DRAM read/write bytes and L2 sector hit rate. They use
`--cache-control none` to retain the probe's priming sequence. NCU replay can
still change cache behavior, and its timings are not used for the throughput
curve. See the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
for the cache-control caveat.

Run from the repo root after Nebius CLI authentication:

```bash
systemd-inhibit --what=sleep:idle --mode=block --who=kernel-mcts \
  --why='Keep H100 characterization controller active' \
  .venv/bin/python docs/experiments/h100-characterization/run.py
.venv/bin/python docs/experiments/h100-characterization/analyze.py
```

The runner provisions one pinned H100 SXM worker, uploads only the standalone
probe, downloads results, and deletes the VM and disk. The NCU CSV exports are
local and Git-ignored. The timing data, provenance, and validated analysis
are bounded JSON artifacts. No MCTS search or prompt consumes these results.

## First H100 SXM result

The run used an NVIDIA H100 80GB HBM3 (compute capability 9.0), Torch 2.7.1,
and Triton 3.3.1. Each reported value is the median of 15 CUDA-event rounds;
each round times a replay of 64 captured launches. The probe verifies that
each output matches its input. The minimal kernel's graph-replay median was
**1.017 µs/launch** (p10–p90: 1.013–1.029 µs). It is a measured floor for
this capture/replay method, not a standalone host-launch latency.

| Payload per copy | Working set | Median copy time | p10–p90 | Effective requested read+write throughput |
| --- | --- | ---: | ---: | ---: |
| 4 KiB | 8 KiB, repeated pair | 1.211 µs | 1.206–1.221 µs | 6.8 GB/s |
| 8 MiB | 16 MiB, repeated pair | 3.806 µs | 3.799–3.830 µs | 4,408 GB/s |
| 8 MiB | 256 MiB, rotating pairs | 7.333 µs | 7.308–7.366 µs | 2,288 GB/s |
| 128 MiB | 256 MiB, repeated pair | 90.453 µs | 90.397–90.470 µs | 2,968 GB/s |
| 128 MiB | 1 GiB, rotating pairs | 90.010 µs | 89.975–90.040 µs | 2,982 GB/s |
| 256 MiB | 512 MiB, repeated pair | 178.473 µs | 178.453–178.519 µs | 3,008 GB/s |
| 256 MiB | 2 GiB, rotating pairs | 178.510 µs | 178.489–178.564 µs | 3,007 GB/s |

The [full timing artifact](output/timings.json) contains all 11 cases and
their raw samples. Throughput uses **2 × payload bytes / median time** and
does not measure bytes actually transferred on a physical link. The repeated
8 MiB case is much faster than the rotating case, so working-set behavior
matters. This alone does not prove that the former is L2-only. The separate
NCU replay reported 6.0 MB DRAM reads, 0.18 MB DRAM writes, and 70.59% L2
sector hits for the repeated 8 MiB case. For the rotating 128 MiB case it
reported 134.2 MB DRAM reads, 113.8 MB DRAM writes, and 50.20% L2 sector
hits. These counters establish some traffic at both levels under NCU replay;
they do not isolate the latency or bandwidth of HBM→L2, L2→L1, or L1→register.

The GPU's sampled SM clock was 345 MHz before the sweep and 1,980 MHz after
it; memory clock was 2,619 MHz in both snapshots. These are boundary
snapshots, not per-case clock traces. The curve is a first observation on one
worker, with no cross-worker uncertainty estimate yet. The
[analysis](output/analysis.json) therefore leaves `physical_link_calibrations`
empty. The [provenance](output/provenance.json) records the worker image and
environment manifest. No search prompt consumes these artifacts.

## Read-path probes: L2-sourced versus HBM-sourced

A second standalone probe uses a read-only Triton kernel. Each block reads
1,024 aligned `int32` values, reduces them to one checksum, and writes an
8-byte result. The `.cg` load modifier requests L1 bypass; `.ca` allows L1
caching. The analyzer checks the generated PTX for the requested `ld.global.cg`
or `ld.global.ca` instruction. See [Triton's load documentation](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
for the cache modifier semantics. Like the copy sweep, timing uses 15 rounds of
64 CUDA Graph launches and records raw samples. The effective throughput below
counts **requested input bytes only**; the reduction, output write, and launch
remain in the measured time.

| Input | Load policy | Median read/reduce time | Effective requested read throughput |
| --- | --- | ---: | ---: |
| 1 MiB | `.cg` | 1.650 µs | 636 GB/s |
| 8 MiB | `.ca` | 2.472 µs | 3,394 GB/s |
| 8 MiB | `.cg` | 2.685 µs | 3,125 GB/s |
| 32 MiB | `.cg` | 6.491 µs | 5,170 GB/s |
| 64 MiB | `.cg` | 22.983 µs | 2,920 GB/s |
| 256 MiB | `.cg` | 86.972 µs | 3,086 GB/s |

The [read-path timing artifact](output/read_paths_timings.json) has all seven
cases, raw samples, and PTX load lines. NCU checks were run separately. With
ordinary kernel replay, the 8 MiB `.cg` profile reported **6.17 MB of DRAM
reads**. With application replay, which reruns the priming sequence, it
reported **zero DRAM read bytes** and 87.51% L2 sector hits; the 1 MiB `.cg`
case also reported zero DRAM reads and 99.35% L2 sector hits. The 256 MiB
`.cg` case reported **268.4 MB of DRAM reads**, approximately its full input,
under both replay modes. NCU's L2 hit percentage covers the profiled kernel's
L2 requests, including activity beyond input loads, so it is supporting
evidence rather than a direct fraction of input served by L2.

The [read-path analysis](output/read_paths_analysis.json) labels the 1 MiB and
8 MiB application-replay profiles as **L2-resident read evidence** and the
256 MiB profile as **HBM-sourced read evidence**. The kernel-replay 8 MiB
profile remains mixed. This difference demonstrates why the replay mode must
travel with each counter. NVIDIA [recommends application replay with cache
control disabled](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
when prior GPU work primes caches. Timing and profiling were taken on separate
workers with the same pinned image, so the label is a cache-regime check, not
an exact timing/counter pairing. Neither curve isolates a physical-link
bandwidth: reduction, tiny output traffic, and launch still contribute. The
[first profile provenance](output/read_paths_provenance.json) and
[application-replay provenance](output/read_paths_app_provenance.json) record
their distinct workers; raw NCU CSVs remain local and Git-ignored.

## In-kernel reuse probe

To make cache reuse occur inside the measured kernel, each block repeatedly
reads a cyclically shifted 1,024-element chunk using `.cg` loads and adds the
values into an output checksum. The input contains ones, so correctness is
checked against `1,024 × repeat_count`. Changing the repeat count changes both
read traffic and loop/address arithmetic; this is still a path measurement.

| Input and repeats | Median time | Effective requested read throughput |
| --- | ---: | ---: |
| 8 MiB ×1 | 2.621 µs | 3,201 GB/s |
| 8 MiB ×4 | 3.741 µs | 8,968 GB/s |
| 8 MiB ×16 | 12.928 µs | 10,382 GB/s |
| 8 MiB ×64 | 57.272 µs | 9,374 GB/s |
| 256 MiB ×1 | 86.143 µs | 3,116 GB/s |
| 256 MiB ×4 | 116.255 µs | 9,236 GB/s |

The 8 MiB ×64 NCU application replay reported **zero DRAM read bytes** for
512 MiB of requested reads. It also reported an impossible **101.51% L2 hit
rate**, which the [reuse analysis](output/reuse_analysis.json) flags as an
inconsistent counter across replay passes. The zero-DRAM result and prior
8 MiB application replay support cache-resident reads; the hit percentage
cannot be interpreted literally. The 256 MiB ×4 run reported roughly one
payload of DRAM reads, only 25% of its requested bytes. Its cyclic block
shift permits reuse across blocks, so it is **not** four independent HBM
passes. The 256 MiB ×1 case remains the cleaner HBM-sourced control.

The [reuse timings](output/reuse_timings.json) include raw rounds and the
verified `.cg` PTX load lines. The [provenance](output/reuse_provenance.json)
records the pinned worker. These measurements suggest a useful L2-heavy read
path, but none is assigned to a single hardware-graph edge. A stronger
bandwidth estimate needs a control for per-repeat address and arithmetic cost
and an HBM control with independent buffers. The profiler's out-of-range hit
rate also warrants a single-metric replay check.

The read-path and reuse runs used, respectively:

```bash
.venv/bin/python docs/experiments/h100-characterization/run_read_paths.py
.venv/bin/python docs/experiments/h100-characterization/run_read_paths.py --profiles-only
.venv/bin/python docs/experiments/h100-characterization/analyze_read_paths.py
.venv/bin/python docs/experiments/h100-characterization/run_reuse.py
.venv/bin/python docs/experiments/h100-characterization/analyze_reuse.py
```

Runners refuse to overwrite their existing JSON artifacts. Reproduction
therefore requires a fresh output directory or moving the existing artifacts
aside while preserving them.

## Matched read controls and explicit shared loads

A third run kept the L2 and HBM controls on **one worker**. The 8 MiB cyclic
case revisits the same input within a kernel. The HBM control allocates four
**disjoint** 256 MiB regions and reads each once, eliminating the cross-block
reuse seen in the earlier 256 MiB ×4 experiment. Both compile to `.cg` global
loads and check their output sums.

| Read case | Working set | Median | Effective requested read throughput | Single-metric NCU DRAM reads |
| --- | ---: | ---: | ---: | ---: |
| 8 MiB ×64 cyclic | 8 MiB | 57.535 µs | 9,331 GB/s | 0 B |
| 256 MiB ×4 disjoint | 1 GiB | 334.049 µs | 3,214 GB/s | 1,073,763,840 B |

The disjoint control requested 1,073,741,824 input bytes, so its measured
DRAM read count is approximately one input read. A separate single-metric NCU
pass reported 97.58% L2 sector hits for the 8 MiB case. These support
**cache-resident** and **HBM-sourced** read-path labels, respectively. Their
throughputs still include loop arithmetic, reduction, output, and launch;
they are not physical HBM→L2 or L2→register bandwidths.

For an on-chip path, a CUDA C++ kernel initializes one shared-memory value per
thread, synchronizes, then repeatedly loads it through a `volatile` shared
pointer and accumulates it. A second kernel performs register-only additions
and writes the same expected output. Both use 528 blocks of 256 threads and
check every output. The shared kernel's SASS has `STS` and `LDS` instructions
and no global load; the register control has no `LDS`. At 256 repeats, NCU
counted **1,081,344 shared-load wavefronts**, exactly
`528 blocks × 8 warps × 256 loads`; the register control counted zero.

| Repeats | Shared-load kernel | Register control |
| ---: | ---: | ---: |
| 1 | 1.227 µs | 1.176 µs |
| 64 | 2.191 µs | 1.495 µs |
| 256 | 5.306 µs | 2.416 µs |
| 1,024 | 17.773 µs | 6.026 µs |

At 1,024 repeats, the shared kernel requests 553,648,128 shared-read bytes,
or **31.2 TB/s of effective requested shared traffic** over its whole kernel
time. This is an explicit shared→register load-path observation. Initialization,
barrier, address/instruction scheduling, accumulation, output, and launch are
still included; subtracting the register control does not isolate a physical
shared-memory link latency or bandwidth.

The [matched-path analysis](output/paths_v3_analysis.json),
[read timings](output/clean_paths_timings.json),
[on-chip timings](output/onchip_timings.json), and
[provenance](output/paths_v3_provenance.json) record raw samples, PTX/SASS
checks, one-metric NCU passes, the pinned image, and hashes of the uploaded
source files. The raw SASS dump and NCU CSVs stay local and Git-ignored. Run
with `run_paths_v3.py` followed by `analyze_paths_v3.py` in the same manner as
the earlier commands above. No hardware-graph edge has a calibrated value yet.

The next section adds serialized latency chains to these throughput paths.

## Dependent-read latency paths

A fourth run measures **serialized load chains**. Each global-memory pointer
identifies the next pointer, spaced 128 bytes apart in a fixed-seed random
ring. One thread traverses the complete ring; `.ca` requests all-level
caching and `.cg` bypasses L1. A separate 4 KiB shared-memory chain and a
register arithmetic chain provide on-chip comparisons. After three warmups,
15 launches record `%clock64` cycles and `%globaltimer` nanoseconds **inside
the kernel**, divided by dependent steps. The PTX source requests the cache
operators; the SASS shows `LDG.E.STRONG.SM` for `.ca`,
`LDG.E.STRONG.GPU` for `.cg`, and `LDS` in the shared chain. NVIDIA documents
[`%clock64` and `%globaltimer`](https://docs.nvidia.com/cuda/parallel-thread-execution/)
as a cycle counter and a target-specific nanosecond timer, respectively.

| Chain | Working set | Median cycles/hop | Median ns/hop |
| --- | ---: | ---: | ---: |
| `.ca` global, L1-eligible | 4 KiB | 40.61 | 20.36 |
| `.cg` global, L1 bypass | 4 KiB | 283.56 | 142.14 |
| `.cg` global | 1 MiB | 284.70 | 142.71 |
| `.cg` global | 8 MiB | 284.56 | 142.64 |
| `.cg` global, DRAM participating | 128 MiB | 691.62 | 346.68 |
| Shared-memory chain | 4 KiB | 28.60 | 14.34 |
| Register arithmetic control | — | 4.47 | 2.24 |

The 4 KiB `.ca` profile reported 99.6% L1 sector hits and zero DRAM reads.
The 1 MiB and 8 MiB `.cg` profiles reported zero DRAM reads. The 128 MiB
`.cg` profile reported 70.6 MB of DRAM reads and 49.04% L2 sector hits; it is
an **HBM-participating random-read path**, not an HBM-only load latency. NCU
counted 8,192 shared-load wavefronts in the shared chain, matching its step
count. A separate 1 MiB L2 hit-rate pass reported an impossible 107.14%; the
[analysis](output/latency_analysis.json) flags and excludes that percentage.

These numbers include load issue, address translation, dependent instruction
and loop costs. The random 128 MiB case can incur TLB effects. The register
control measures arithmetic dependencies; subtracting it would not isolate a
physical memory-link latency. SM clock snapshots were 345 MHz before the
sweep and 1,980 MHz afterward, so the in-kernel nanosecond timer is preferable
to converting cycles with a nominal clock. The
[raw timing samples](output/latency_timings.json) and
[provenance](output/latency_provenance.json) retain the conditions. Run the
probe with `run_latency.py`, then `analyze_latency.py`; existing JSON artifacts
must be preserved before a new run. Physical graph-link latencies remain
unknown.

## Access-pattern and grid-size sensitivity

A fifth standalone run varies two generic read orders: contiguous and a
bijection that places adjacent logical lanes **32 words apart** in physical
memory. Every input element is read exactly once. It also varies the launch
grid (132, 528, or 2,112 Triton programs) while holding the input at 8 MiB or
256 MiB. The `.cg` read/reduce kernel checks the total output and records 15
CUDA Graph timing rounds per case. Three complete sweeps use different
randomized case orders on the same worker, with GPU clock/power snapshots at
each sweep boundary.

| Input | Programs | Access | Median of sweep medians | Across-sweep range |
| --- | ---: | --- | ---: | ---: |
| 8 MiB | 528 | Contiguous | 2.639 µs | 2.632–2.641 µs |
| 8 MiB | 528 | Stride 32 words | 17.594 µs | 17.343–17.604 µs |
| 256 MiB | 132 | Contiguous | 229.769 µs | 229.753–229.909 µs |
| 256 MiB | 528 | Contiguous | 93.802 µs | 93.490–93.886 µs |
| 256 MiB | 2,112 | Contiguous | 86.895 µs | 86.869–87.363 µs |
| 256 MiB | 528 | Stride 32 words | 1,448.490 µs | 1,440.513–1,448.753 µs |

At 528 programs, the strided case was 6.7× slower for 8 MiB and 15.4×
slower for 256 MiB. Separate one-metric NCU application replays reported zero
DRAM reads for both 8 MiB cases. For 256 MiB, contiguous reads caused 268.45
MB of DRAM traffic, versus 4,179.30 MB for the strided read: **15.6× traffic
amplification**. This makes access order a necessary condition on any useful
bandwidth curve. With contiguous 256 MiB reads, changing the grid from 132 to
2,112 programs reduced median time by about 2.6×; grid size changes work per
program and scheduling, so this is not a pure occupancy measurement.

The first sweep started with a 345 MHz SM-clock snapshot; subsequent sweep
boundaries were at 1,980 MHz. The within- and across-sweep timing ranges are
small for these cases, but the boundary snapshots are not per-case clock
traces. The [sensitivity analysis](output/sensitivity_analysis.json) and
[raw timings](output/sensitivity_timings.json) retain every case and sample;
the [provenance](output/sensitivity_provenance.json) records the worker and
source hashes. Run with `run_sensitivity.py`, then `analyze_sensitivity.py`.
These are workload-path curves, not physical-link rates.

Next characterization work should measure tensor/scalar throughput and
synchronization. A selected-kernel overlay can relate PTX/SASS and symbolic
dataflow to these paths; its core schema should work for attention as well as
GEMM.
