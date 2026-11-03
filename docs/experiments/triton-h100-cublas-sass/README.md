# Matched-input cuBLAS and Triton B SASS comparison

This follow-up to the [same-worker diagnostic comparison](../triton-h100-cublas-profile/README.md)
runs the corrected seed-0 fast B Triton source and PyTorch/cuBLAS on the
**same seeded 4096³ BF16 tensors in one Python harness**. Both calls use CUDA
events, the same warmups, and 30 timing samples per round. Their BF16 outputs
matched exactly (`maximum_error=0`). NCU separately captured one named NVTX
launch per implementation with the full Source/SASS set. The selected cuBLAS
kernel is the same `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` seen in the
earlier C++ `cublasGemmEx` comparison.

## Timing

| Implementation | Round 0 | Round 1 | Round 2 | Median round |
| --- | ---: | ---: | ---: | ---: |
| Triton B | 198.368 µs | 198.208 µs | 197.872 µs | **198.208 µs** |
| cuBLAS | 179.408 µs | 179.376 µs | 187.616 µs | **179.408 µs** |
| Paired gap | 18.960 µs | 18.832 µs | 10.256 µs | **18.832 µs** |

All three rounds happened to execute Triton before cuBLAS despite seeded
shuffling. cuBLAS's third round was 8.2 µs slower than its first two; no clock
or power trace was collected, so the cause is unknown. These event timings
are separate from NCU replay durations. Exact samples and run provenance are
in [`results.json`](results.json); the validated medians are in
[`analysis.json`](analysis.json). The earlier C++ harness measured a larger
24.176 µs paired gap, so the two timing conditions should remain distinct.

## NCU launch and memory evidence

| Observation | Triton B | cuBLAS |
| --- | ---: | ---: |
| Grid | 512 CTAs | 132 CTAs |
| Threads/CTA | 256 | 384 |
| Cluster size | None reported | 2 CTAs |
| Registers/thread | 213 | 168 |
| NCU shared memory/block | 148.480 KB | 214.204 KB |
| Tensor-pipe active metric | 91.73% | 95.22% |
| Raw executed warp instructions | 39,968,768 | 16,570,725 |
| DRAM read bytes | 186.688 MB | 150.858 MB |
| DRAM write bytes | 28.673 MB | 28.676 MB |

These NCU reports replayed separate processes, but both generated the same
input tensor values. cuBLAS still read about 35.8 MB fewer bytes from DRAM.
Cache and replay effects remain possible; this is measured traffic, not
per-edge attribution or a proof of why cuBLAS is faster. The 2.41× total
instruction ratio also reflects 4,096 launched Triton warps versus 1,584
cuBLAS warps. It is not a per-tile code-quality ratio.

## Executed SASS instruction families

NCU's Source export counts **warp-level executions at SASS sites**. Selected
opcode totals from [`analysis.json`](analysis.json):

| Opcode family | Triton B executions | cuBLAS executions | What it identifies |
| --- | ---: | ---: | --- |
| `HGMMA` | 1,048,576 | 2,097,152 | Tensor-core instruction issue; shapes/roles can differ |
| `LDGSTS` | 3,244,032 | 0 | Triton's global-to-shared async-copy path |
| `UTMALDG` / `UTMASTG` | 0 / 0 | 65,536 / 32,768 | cuBLAS TMA-family load/store operations |
| `ISETP` | 5,423,104 | 595,748 | Integer predicates |
| `IMAD` | 3,969,024 | 45,184 | Integer multiply-add family, including move-like variants |
| `SEL` | 3,407,872 | 0 | Selection |
| `LEA` | 3,260,416 | 8,192 | Address calculation |
| `UIADD3` | 2,625,536 | 6,694,878 | Uniform integer addition |

The data-movement instruction *types* differ sharply: B issues many
`LDGSTS` operations, while cuBLAS issues `UTMALDG`/`UTMASTG` and uses a
two-CTA cluster. Instruction counts across these families do **not** measure
equal bytes or equal mathematical work. B also executes many integer
predicate, address, and selection operations. This gives us concrete source
and compiler targets to test: redundant masks/addressing, TMA loading, and
persistent tile scheduling. It does not rank their individual latency costs.

The Triton Source-page instruction total equals its raw NCU counter.
cuBLAS's Source-page sum is 16,816,366, about 1.5% above its raw counter
of 16,570,725. NCU can collect Source and raw counters on separate replay
passes; the exact reason for this discrepancy is unresolved. We retain both
values and use the **raw counter for total-instruction comparisons**.
The SASS table uses within-Source-export opcode counts. No cuBLAS PTX was
available in the NCU report, so this comparison does not infer PTX-level
def-use paths for cuBLAS.

## Artifacts and scope

[`paired_harness.py`](paired_harness.py) pins the exact B source SHA-256 and
uses one set of Torch-generated tensors for both kernels. [`run.py`](run.py)
provisions one pinned H100 worker, checks correctness, times both calls,
captures two NCU reports, and deletes the VM/disk. [`analyze.py`](analyze.py)
validates the saved timing samples and summarizes the NCU exports. The large
raw reports and SASS/PTX CSVs are retained locally under ignored `output/`;
`results.json` records report hashes and sizes. The cuBLAS NCU PTX export
explicitly says PTX is unavailable. The worker manifest records H100 SXM,
NCU 2025.1.1, image digest, source/harness hashes, and Git state. The first
VM attempt reset SSH during file transfer before any benchmark; its empty
record is excluded. The completed retry's VM and disk were independently
listed as deleted.
