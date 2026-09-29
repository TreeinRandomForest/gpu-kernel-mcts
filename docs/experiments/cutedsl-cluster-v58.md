# Independent CuTe cluster multicast validation v58

Date: 2026-09-29

## Hypothesis

Grouping two M-adjacent CTAs into cluster `(2,1)` allows their common B tile to be
loaded once through the typed multicast path while A remains CTA-private. Cluster
geometry, B-copy ownership, multicast masks, barriers, and launch metadata must
change atomically.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cluster-v58`
- workload: `bf16_gemm_4096_h100`
- CTA tile: `(64,256,64)`
- cluster: `(2,1)`
- A TMA copy: CTA-private
- B TMA copy: multicast across cluster M
- mainloop: three-stage SW128
- correctness: canonical seed-0 inputs and cuBLAS FP32-accumulation reference
- timing: 10 warmups and 30 CUDA-event measurements

## Result

The repository contract passed exactly:

- maximum error: `0.0`
- mean error: `0.0`
- nonzero outputs: `16,777,216`
- median: `658.959985 us`
- mean: `658.353066 us`
- range: `642.656028--669.632018 us`

Identity and artifact evidence:

- configuration: `0c588c21ecc8536a5e410297eace7e38cbaa41613bcd6a590e3b99f1b2373138`
- source: `78574769cd3c028991bfb75f4291bb7202647be66218ed91cf81a0dea9c86a42`
- normalized MLIR: `fd63488dbbd6e16011dfbb7cbad37d7760738616a87c1ac56b6be1ba0f50610e`
- fatbin: `32d339ac6efc962c9d368ef3125d0e7c3e11986893a7686a207f8c5bbd8d7528`

The same-image sequential cluster `(1,1)` root measured `673.776001 us` median.
The clustered state was therefore 1.02248x faster, reduced latency by 2.20%, and
would receive reward `ln(673.776001 / 658.959985) = 0.022235`.

## Search decision

The paired cluster `(1,1) <-> (2,1)` transition is admitted as a deterministic
realization of `change_cluster_shape` under one `B_mut`. The clustered state returns
only to the root. Combinations with alternate CTA tiles, SW64, the two-stage
mainloop, cluster-N multicast, or another epilogue organization remain excluded.

The raw reports are `cutedsl-cluster-v58.json` and
`cutedsl-cluster-root-v58.json`. They are local artifacts and are not intended for
source control.
