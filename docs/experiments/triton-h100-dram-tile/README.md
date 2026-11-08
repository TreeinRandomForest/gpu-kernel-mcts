# Resolving the Triton/cuBLAS DRAM-read gap

Earlier [CTA-order](../triton-h100-dram-reuse/README.md) and
[persistent-CTA](../triton-h100-dram-persistent/README.md) experiments showed
that CTA ordering changes DRAM traffic by 12–16 MB, while reducing the grid
from 512 to 132 CTAs does not change it at a fixed order. This experiment
tests a same-area output tile orientation. The fixed Triton 3.3.1 TMA kernel
uses `BM×BN=128×256` or `256×128`, `BK=64`, eight warps, three stages, the
same A/B tensor-descriptor loading mechanism, and the same BF16 output contract. `GROUP_M`
controls CTA tile order. The output area and 512-program grid are the same
for both tile shapes.

One H100 SXM worker ran all variants and PyTorch/cuBLAS on identical seeded
4096³ BF16 tensors. Every Triton output matched cuBLAS **exactly**. Nsight
Compute used application replay, explicit `--cache-control all`, and identical
L2/DRAM read metrics for each arm.

| Kernel | DRAM read | L2 TEX read sectors | L2 TEX read lookup misses | Registers/thread |
| --- | ---: | ---: | ---: | ---: |
| 128×256, group 1 | 170.937 MB | 38,367,444 | 7,824,596 | 154 |
| 128×256, group 8 | **186.668 MB** | 27,507,144 | 7,550,856 | 154 |
| 256×128, group 1 | 174.082 MB | 36,533,656 | 8,853,336 | 174 |
| 256×128, group 8 | **152.702 MB** | 35,947,212 | 6,679,504 | 174 |
| 256×128, group 16 | 170.967 MB | 37,123,324 | 8,407,960 | 174 |
| cuBLAS | **150.844 MB** | 26,667,752 | 6,488,876 | 168 |

At the same `GROUP_M=8`, swapping the tile from 128×256 to 256×128 removed
**33.966 MB of the 35.824 MB** Triton/cuBLAS DRAM-read gap (**94.8%**).
The remaining difference is **1.858 MB**. This unclustered 512-CTA Triton
variant nearly matches cuBLAS's DRAM traffic, so cuBLAS's persistent grid and
two-CTA cluster are not required to reach similar DRAM-read volume on this
workload. Tile orientation together with CTA order accounts for almost the
entire measured discrepancy. The L2 lookup-miss count also approaches cuBLAS,
which is consistent with improved cache reuse, but these aggregate counters
cannot assign the saved bytes specifically to A or B.

Lower DRAM traffic did **not** mean lower latency. Across three paired timing
rounds, the 256×128/group-8 kernel was **11.216–14.144 µs slower** than the
128×256/group-8 kernel (median paired slowdown 12.976 µs). The former also
issued more L2 read sectors and used 20 more registers/thread. These
differences are candidate explanations, not isolated causes. cuBLAS timing
drifted from 182.000 to 190.752 µs during the run, so cross-run comparisons
of a few microseconds should be treated cautiously. The same-worker paired
direction for the two Triton tiles was consistent in all rounds.

The result separates the **traffic discrepancy** from the **latency gap**.
Further optimization should not assume that matching cuBLAS's DRAM bytes is
sufficient. Source, timing samples, profiler metrics, report hashes, and the
worker manifest are saved in [`results.json`](results.json) and
[`analysis.json`](analysis.json). Raw NCU exports remain in ignored `output/`.

## MCTS follow-up

The Triton [strategy catalog](../../../configs/triton_h100_strategies.json)
now includes `improve_cross_cta_reuse`. It asks the generator to inspect the
source's output-tile dimensions, CTA mapping, workload strides, and measured
resource/traffic evidence, then propose a coordinated tile or traversal
change. It explicitly keeps CUDA-event latency as the objective; lower DRAM
bytes are only one diagnostic. The opt-in `reuse_v1` profile extends
`spill_v1` with L2 TEX read request/hit/miss sectors and an aggregate L2
read-sector hit rate. The default `lightweight_v1` profile and MCTS search
semantics are unchanged. The Triton MCTS prompt currently has source and
workload layouts but no Triton dataflow graph, so this action reasons from
source and aggregate counters rather than tensor-attributed graph edges.

[`harness.py`](harness.py) contains the five Triton tile/order variants and
the cuBLAS call; [`run.py`](run.py) provisions and releases the worker. The
VM and disk were both listed as deleted after the run.
