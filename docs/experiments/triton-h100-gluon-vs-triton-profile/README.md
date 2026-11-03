# Matched H100 profile: source Gluon versus optimized Triton B

This experiment follows the [source Gluon scheduling
result](../triton-h100-gluon-source-schedule/README.md). One Nebius H100 SXM
profiled the exact seed-0 BF16 4096³ GEMM through three implementations:
the source Gluon **ordinary B** branch, optimized Triton B, and cuBLAS.
Each process recreated the same tensors, checked exact output against
`torch.mm`, warmed up ten times, and selected one kernel launch with NVTX for
NCU 2025.1.1 `--set full --cache-control all`. The Gluon cubin hash matches
the preceding matched timing run. All three profiles passed exact correctness.

The preceding randomized CUDA Graph study measured 296.176 µs for source
Gluon ordinary, 185.872 µs for Triton B, and 177.776 µs for cuBLAS. The
NCU durations below are profiling replays, not replacement timing samples.

| Observation | Source Gluon ordinary | Optimized Triton B | cuBLAS |
| --- | ---: | ---: | ---: |
| NCU duration (µs) | 322.240 | 217.952 | 214.112 |
| Grid (CTAs) | 512 | 512 | 132 |
| Launch waves/SM | 3.88 | 3.88 | 1.00 |
| Registers/thread | 154 | 206 | 168 |
| Shared memory/CTA (KB, NCU) | 99.336 | 148.480 | 214.204 |
| HGMMA warp instructions | 1.049M, `64×256×16` | 1.049M, `64×256×16` | 2.097M, `64×128×16` |
| Tensor pipe active (% elapsed cycles) | **55.64** | **86.79** | 90.28 |
| Executed warp instructions | 18.760M | 28.455M | 16.571M |
| DRAM reads (MB) | 174.092 | 186.670 | 150.849 |
| DRAM writes (MB) | 28.718 | 28.878 | 28.703 |
| Long-scoreboard stall samples | 19,141 | 3,749 | 2,851 |

The source Gluon arm has an ordinary `UTMALDG.2D` B load in the executed
branch. Its cubin also contains the unused multicast branch, but NCU's SASS
Source page records **zero executed** `UTMALDG.2D.MULTICAST` instructions.
Triton B instead uses `LDGSTS` async copies; cuBLAS uses 3D TMA loads and
stores, including multicast. Gluon and Triton B have the same CTA count,
waves, and whole-kernel HGMMA count and shape. Gluon reads fewer DRAM bytes
and executes fewer warp instructions, yet is much slower. Raw traffic volume,
launched CTA count, and matrix instruction count do not explain its
approximately 110 µs CUDA Graph gap to Triton B.

The most conspicuous Gluon SASS sample is the branch immediately after
`SYNCS.PHASECHK.TRANS64.TRYWAIT`, the poll of its TMA transaction barrier:
**18,697 of 27,864 Source-page PC samples** (67.1%) land at that branch.
This is a sample concentration, not a measured fraction of kernel time. The
Gluon arm also has about 5.1× as many NCU long-scoreboard samples as Triton B
and much lower tensor-pipe activity. NVIDIA describes long-scoreboard samples
as waits on L1TEX dependencies and notes that PC samples appear at the
consumer rather than necessarily at the producer
([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).
Together with the barrier polling loop, the evidence points to insufficient
overlap between operand arrival and tensor work in this Gluon mainloop. These
counters do not separately quantify TMA latency, barrier costs, or epilogue
time, and the cuBLAS instruction shape and launch geometry differ.

The next controlled test should prefetch additional K tiles with a three-stage
A/B shared-memory ring and per-slot transaction barriers, keeping tile shape,
CTA grid, and B policy fixed. Exact correctness, emitted SASS, paired CUDA
Graph timing, and this barrier-poll hotspot should all be checked before
turning the mechanism into a search action. A third A/B slot is expected to
bring shared memory near Triton B's allocation, but the compiler's actual
resource and occupancy effects must be measured.

[`analysis.json`](analysis.json) records exact counters, units, executed SASS
families, PC hotspots, and artifact hashes. Run [`analyze.py`](analyze.py)
to regenerate it from ignored `output/` (NCU reports, raw CSV, Source-page
SASS CSV). [`profile_one.py`](profile_one.py) is the selected-launch harness;
[`run.py`](run.py) records the worker manifest and cleans up resources.
[`results.json`](results.json) includes the run ID, source identities,
correctness, and report hashes. Nebius instance and disk listings were empty
afterward.
