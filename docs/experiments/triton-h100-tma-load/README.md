# Isolated TMA operand-load experiment

We replaced fast B's pointer-based A/B loads with Triton 3.3.1 tensor
descriptor loads. The kernel kept its `BM=128`, `BN=256`, `BK=64`, eight warps,
three stages, grouped CTA schedule, `tl.dot`, and pointer-based C store. This
tests single-CTA TMA loading without changing persistence or clusters. B is
logically `[K,N]` column-major, so its descriptor views the same storage as a
physical `[N,K]` row-major tensor; the loaded tile is transposed before dot.
The implementation uses the pinned release's
[`triton.tools.experimental_descriptor` API](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/tools/experimental_descriptor.py)
and `_experimental_descriptor_load`, as in its
[persistent matmul tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/09-persistent-matmul.py).

One Nebius H100 SXM worker used the same pinned image as the preceding
[cuBLAS SASS comparison](../triton-h100-cublas-sass/README.md). A single Python
harness used the same seeded 4096³ BF16 A/B tensors for fast B, the TMA variant,
and PyTorch/cuBLAS. Each implementation had ten warmups and 30 CUDA-event
samples per round, for three paired rounds. All three outputs matched **exactly**.
The event timings are separate from NCU's replay measurements.

| Implementation | Round 0 | Round 1 | Round 2 | Median round |
| --- | ---: | ---: | ---: | ---: |
| Fast B | 199.424 µs | 199.072 µs | 202.496 µs | 199.424 µs |
| TMA loads | 197.072 µs | 196.992 µs | 196.016 µs | 196.992 µs |
| cuBLAS | 181.328 µs | 182.000 µs | 181.536 µs | 181.536 µs |
| TMA minus B | −2.352 µs | −2.080 µs | −6.480 µs | **−2.352 µs paired median** |

The third baseline round is slower than the first two. The paired differences
support a modest speedup in this run, but no clock/power trace or independent
replicate was collected. TMA alone leaves roughly 15 µs to cuBLAS under these
conditions; the cuBLAS kernel also differs in CTA clustering, scheduling, and
other code generation.

## Compiled and measured changes

NCU Source exports confirm `cp.async.bulk.tensor.2d` in TMA PTX and executed
`UTMALDG` in SASS. The baseline executes `LDGSTS`; neither compiled kernel
uses a two-CTA cluster. Selected values from [`analysis.json`](analysis.json):

| NCU observation | Fast B | TMA loads |
| --- | ---: | ---: |
| Executed `LDGSTS` | 3,244,032 | 0 |
| Executed `UTMALDG` | 0 | 540,672 |
| Executed `HGMMA` | 1,048,576 | 1,048,576 |
| Registers/thread | 213 | 154 |
| Shared memory/block | 148.480 KB | 148.504 KB |
| Executed warp instructions, raw counter | 39,968,768 | 27,750,400 |
| DRAM read | 186.678 MB | 186.674 MB |
| DRAM write | 28.696 MB | 28.673 MB |
| Tensor-pipe active | 91.53% | 91.66% |

These are warp-level instruction executions, not equal-sized transfers. The
instruction and register reductions are consistent with less per-thread
addressing work, while DRAM traffic barely changes. They do not isolate the
latency contribution of any one instruction family.

[`tma_kernel.py`](tma_kernel.py) contains the source transformation.
[`paired_harness.py`](paired_harness.py) performs correctness and timing;
[`run.py`](run.py) provisions, captures two full NCU Source reports, and
releases the worker. [`results.json`](results.json) records all event samples,
source hashes, image and worker manifest, and NCU report hashes. Raw reports
and CSV exports remain under ignored `output/`. The worker's instance and disk
were both listed as deleted after the run. The first local attempt stopped
before provisioning because the sandbox blocked DNS; its record is retained
under ignored `output/`.
