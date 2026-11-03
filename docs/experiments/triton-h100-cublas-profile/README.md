# Same-worker cuBLAS and fast Triton B comparison

This experiment compares the corrected seed-0 [fast B source](../triton-h100-bd-crossover/trace_B.py)
with the pinned `cublasGemmEx` baseline on one Nebius H100 SXM. Both compute
the fixed 4096³ BF16 GEMM with row-major A/C, column-major B, FP32
accumulation, and BF16 output. Three randomized paired rounds each took 30
CUDA-event timing samples after warmup. Separate NCU application replays
collected the **same `diagnostic_v2` metric set** for each implementation.
The exact B source hash was checked against its corrected SQLite trace.

## Timing

| Implementation | Round 0 | Round 1 | Round 2 | Median of round medians |
| --- | ---: | ---: | ---: | ---: |
| Triton B | 201.184 µs | 200.416 µs | 199.760 µs | **200.416 µs** |
| cuBLAS | 175.856 µs | 176.336 µs | 175.584 µs | **175.856 µs** |
| Paired difference | 25.328 µs | 24.080 µs | 24.176 µs | **24.176 µs** |

Both implementations completed their correctness checks. The Triton harness
checks its output against PyTorch/cuBLAS; the vendor executable compares its
cuBLAS output against another cuBLAS invocation. These are CUDA-event timings,
not NCU replay durations. Raw samples and run provenance are in
[`results.json`](results.json); [`analysis.json`](analysis.json) validates the
round medians and computes the paired differences.

## Matched diagnostic counters

The profiles each describe one selected kernel launch. The cuBLAS kernel name
was `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT`. NCU captured it as the
first GPU kernel in the pinned vendor executable, with grid `(2, 66, 1)`,
block `(384, 1, 1)`, cluster size 2, and one reported wave per SM. The
Triton source launches 512 programs of 256 threads each.

| NCU or compiler observation | Triton B | cuBLAS |
| --- | ---: | ---: |
| Registers/thread | 213 | 168 |
| Shared memory/block | 147,456 B | 214,212 B |
| Achieved occupancy | 12.48% | 14.65% |
| Tensor-pipe active metric | 91.51% | 95.24% |
| SM throughput metric | 86.97% | 90.50% |
| Executed instructions | 39,968,768 | 16,575,186 |
| Tensor instructions executed | 1,048,576 | 2,097,152 |
| DRAM read bytes | 186,665,216 | 150,959,616 |
| DRAM write bytes | 28,641,792 | 28,617,728 |
| L2 sector hit rate | 64.47% | 70.99% |
| Local load/store sectors | 0 / 0 | 0 / 0 |

The cuBLAS launch uses more threads and shared memory per block, fewer
registers per thread, and a one-wave clustered grid. It recorded about 2.4×
fewer total executed instructions and 35.7 MB fewer DRAM read bytes while
maintaining slightly higher tensor-pipe activity. The instruction total is a
count of **warp-level machine-instruction executions**, independent of source
language. Triton launches `512 × 8 = 4,096` warps; cuBLAS launches
`132 × 12 = 1,584`. Dividing the instruction totals by launched warps gives
9,758 for Triton and about 10,464 for cuBLAS. Thus the 2.41× total-instruction
ratio broadly tracks the 2.59× launched-warp ratio. These quotients are
descriptive, not equal-work-per-warp efficiency measures: the kernels divide
tiles and pipeline roles differently. The tensor-instruction counts cannot
be read as relative FLOPs because the implementations can use different
instruction shapes. Likewise, cuBLAS had *more* barrier stalls and shared-bank
conflicts in these aggregate reports despite being faster. No individual
counter establishes the cause.

The two benchmark harnesses differ: Triton uses Python/Torch and Gaussian
random inputs; the cuBLAS executable uses C++ and uniform random inputs.
They use the same shape, layouts, dtype, worker, and CUDA-event timing method,
but not identical input values or setup paths. NCU uses separate application
replays, so cache state and replay duration must not be mistaken for the
timing rounds. The raw cuBLAS NCU output is [`cublas.ncu.csv`](cublas.ncu.csv);
the Triton profile and both parsed metric dictionaries are in `results.json`.
The later [matched-input SASS comparison](../triton-h100-cublas-sass/README.md)
uses one Python harness and selects the same cuBLAS kernel.

## Reproduction and provenance

The worker image is pinned to digest
`d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
The runner overlays the current `kernel_mcts` package for `diagnostic_v2`
support, and records its bundle hash, Git revision/dirty state, worker
manifest, exact source hash, timing samples, and raw NCU metrics. Run
[`run.py`](run.py) from the repository root only when no `results.json` exists;
then run [`analyze.py`](analyze.py). `run.py --recover` only parses an already
saved NCU capture and does not provision a GPU.

An initial unoverlaid attempt completed six timings but stopped before NCU
because the pinned image's worker service rejected `diagnostic_v2`. Its
incomplete record is excluded. The completed retry captured both profiles;
NCU emitted a wide CSV format, so the parser was corrected and the saved
capture was recovered offline. Both attempts' guarded VM and disk were
confirmed deleted. No GPU was reprovisioned for parsing.
