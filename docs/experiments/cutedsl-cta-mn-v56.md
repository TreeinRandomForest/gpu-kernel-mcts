# Independent CuTe combined CTA-M/N validation v56

Date: 2026-09-29

## Hypothesis

Combining the validated cooperative M decomposition and narrow WGMMA-N control as
CTA tile `(128,128,64)` should produce a distinct correct implementation. The
transition must atomically coordinate both TMA operand tiles, two consumer warp
groups, the WGMMA instruction, four epilogue buffers, shared-memory storage, and the
launch grid.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-cta-mn-v56`
- workload: `bf16_gemm_4096_h100`
- CTA tile: `(128,128,64)`
- WGMMA instruction: `64x128x16`, two M-axis consumer warp groups
- mainloop: three-stage SW128 TMA pipeline
- epilogue: four disjoint 64x64 BF16 TMA-store buffers plus one guard tile
- launch grid: 32x32 CTAs
- correctness: canonical seed-0 inputs and cuBLAS FP32-accumulation reference
- timing: 10 warmups and 30 CUDA-event measurements

## Result

The repository contract passed exactly:

- maximum error: `0.0`
- mean error: `0.0`
- nonzero outputs: `16,777,216`
- median: `619.024009 us`
- mean: `619.044266 us`
- range: `601.823986--629.952013 us`

Identity and artifact evidence:

- configuration: `9a03718f288efbad4e212c4e28c77db185c58618d9e99315e35215d418ff3e55`
- source: `9adb8c9be52a7814524f1a568f2b8cc45febd4e612c4176703de191cf51ac6eb`
- normalized MLIR: `01502dc2ea78df482dc8647f1645c688566d59bbe072e00477a143663c5332fb`
- fatbin: `de3ad022a3155c1eb01d73d2a2798adc1b9857b0eda7171d21f095f8c9e602d2`

Same-image sequential controls measured:

| State | Median (us) | Relative to root |
|---|---:|---:|
| root `(64,256,64)` | 671.535999 | 1.00000x |
| combined `(128,128,64)` | 619.024009 | 1.08483x |
| cooperative `(128,256,64)` | 453.696012 | 1.48060x |

The combined state reduces root latency by 7.82% and would receive reward
`ln(671.535999 / 619.024009) = 0.081424`. It is 1.3644x slower than the cooperative
state, but valid alternatives are not pruned solely for being slower than another
child.

## Search decision

The paired `(64,256,64) <-> (128,128,64)` transition is admitted as a third
deterministic realization of `change_cta_tile` under one `B_mut`. Alternate CTA
states return only to the root; direct alternate-to-alternate transitions and
combinations with SW64, the two-stage mainloop, cluster changes, or another epilogue
organization remain excluded.

The raw reports are `cutedsl-cta-mn-v56.json`, `cutedsl-root-v56.json`, and
`cutedsl-cooperative-v56.json`. They are local run artifacts and are not intended
for source control.
