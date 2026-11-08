# Matched TMA tile orientation latency

This diagnostic reuses the unchanged TMA kernel and same-input benchmark harness
from `triton-h100-dram-tile`. It compares `128×256` and `256×128` output tiles
with `GROUP_M=8`, `BK=64`, eight warps, and three stages on one H100 SXM.
Three randomized-order rounds each have 30 CUDA-event observations per arm.
Full NCU reports and Source/SASS exports were captured with kernel replay and
`--cache-control all`; raw files are under ignored `output/`.

| Measure | 128×256 | 256×128 |
| --- | ---: | ---: |
| CUDA-event round medians, µs | 202.432, 204.480, 209.248 | 219.456, 219.424, 222.032 |
| Registers/thread | 154 | 174 |
| Shared memory/block, KiB | 148.504 | 148.504 |
| Warp instructions executed | 27.750M | 32.010M |
| HGMMA warp instructions executed | 1.049M | 2.097M |
| Tensor pipe active, % of peak active cycles | 91.72 | 89.94 |
| DRAM read, MB (NCU) | 186.682 | 152.846 |
| Local load/store sectors | 0/0 | 0/0 |

The `256×128` tile is slower in every paired round by 17.024, 14.944, and
12.784 µs, respectively (median 14.944 µs). Its lower measured DRAM reads
therefore do not translate into lower latency here. More registers and executed
instructions, including twice the HGMMA warp instructions, are plausible
contributors. These are associations: NCU counters may come from different
replay passes, and this experiment does not isolate one causal bottleneck.

Machine-readable results and report hashes are in [analysis.json](analysis.json)
and [results.json](results.json). Run `python
docs/experiments/triton-h100-latency-tile/run.py` and then `python
docs/experiments/triton-h100-latency-tile/analyze.py` on a new checkout to
repeat; the runner refuses to overwrite existing results.
