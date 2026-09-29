# Independent CuTe CTA-N/WGMMA-N validation v54

Date: 2026-09-29

## Hypothesis

Reducing the independent GEMM's CTA N extent from 256 to 128, together with its
WGMMA N instruction and every coupled B-tile, epilogue, storage, and launch field,
should produce a distinct correct implementation and a useful deterministic
`change_cta_tile` realization.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cta-n-v54`
- workload: `bf16_gemm_4096_h100`
- CTA tile: `(64,128,64)`
- WGMMA instruction: `64x128x16`, one consumer warp group
- mainloop: three-stage SW128 TMA pipeline
- epilogue: two disjoint 64x64 BF16 TMA-store buffers plus one guard tile
- launch grid: 64x32 CTAs
- correctness: canonical seed-0 inputs and cuBLAS FP32-accumulation reference
- timing: 10 warmups and 30 CUDA-event measurements

## Result

The complete repository contract passed exactly:

- maximum error: `0.0`
- mean error: `0.0`
- nonzero outputs: `16,777,216`
- median: `514.719993 us`
- mean: `514.775465 us`
- range: `509.855986--524.320006 us`

Identity and artifact evidence:

- configuration: `426d7c1a9b13cda85439ad31510067cd5d5538bd78127442b484e8859588b3d2`
- source: `6ad558dd818755e8d778decc784dc2689d255870c85bf9d2454237506fad8684`
- normalized MLIR: `45131ff3e1994a735bca9af0de7678fe49135a6642db7353df65564643047630`
- fatbin: `6b7f936a754f00a3c26760e0031980cc68b1f5abf3a6c0b5c4e9adfee00a7e83`

The `(64,256,64)` root was then measured in the same VM and image at 673.440009 us
median. The narrow-N state was therefore 1.30836x faster, reduced latency by 23.57%,
and would receive reward `ln(673.440009 / 514.719993) = 0.268776`. The two commands
were sequential rather than interleaved, so later search evaluation remains the
authoritative reward measurement.

## Search decision

The paired `(64,256,64) <-> (64,128,64)` transition is admitted as a deterministic
realization of `change_cta_tile` under one `B_mut`. It atomically changes the B TMA
tile, WGMMA N instruction, epilogue buffers and barriers, shared-memory allocation,
grid N extent, canonical identity, and rendered source. Compositions with SW64, the
two-stage mainloop, cooperative M decomposition, or cluster changes remain excluded
until separately H100-validated.

The raw reports are `cutedsl-cta-n-v54.json` and `cutedsl-root-v54.json`, local run
artifacts that are not intended for source control.
