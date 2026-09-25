# Cooperative independent CuTe GEMM v52

Date: 2026-09-25

This experiment validates the independently lowered `(128,256,64)` CuTe GEMM on one
Nebius H100 SXM. The typed state uses two M-axis WGMMA consumer warp groups and eight
disjoint 64x64 shared-memory epilogue buffers. Each group stages four local
accumulator tiles, after which one elected CTA warp issues all eight TMA stores.

The first v51 full-workload run compiled and launched safely but failed correctness.
Exactly one eighth of the output was zero, and the bounded diagnostic showed a cyclic
shift in the second output-tile row. The register-to-shared copy used the CTA-wide
thread ID even though each shared stage represents one warp-group-local 64x64 tile.
Using `thread_idx % 128` for that local copy slice corrected the ownership mapping.

## Results

The corrected v52 bounded one-K diagnostic reported:

- correctness: pass;
- maximum error: `0.03125`;
- cosine similarity: `1.0`;
- tile permutation: `[0,1,2,3,4,5,6,7]`; and
- nonzero outputs: `32768 / 32768`.

The complete repository-contract run used the fixed BF16 4096x4096x4096 workload,
cuBLAS FP32-accumulation reference, 10 warmups, and 30 CUDA-event measurements:

- correctness: pass;
- maximum and mean error: `0.0`;
- median: `448.384 us`;
- mean: `447.574 us`;
- minimum: `438.176 us`;
- maximum: `454.496 us`; and
- nonzero outputs: `16777216 / 16777216`.

The canonical configuration hash is
`73135dfb7f2b7b3a307e17aa9036e75a027c284d99658aef255a2ba9d28719ba`.
The full-workload source hash is
`a59d4dc9ace07bc537a6c2239dc63652fb6e34aee0aff3a2679d0fccc4d5b25e`;
its normalized MLIR hash is
`d3d8328295a456eb5e058219bec1c9392f4da2b632f6c850a63ea342059916b2`,
and its fatbin hash is
`945c7213ba3c31203d0ea1f07b8ab0e6581c6c7f984353a3ea8bdf7d6940f769`.

The raw local reports are `cutedsl-cooperative-one-k-v52.json` and
`cutedsl-cooperative-v52.json`; they are not intended for source control. The shape
is now exposed through the paired one-group/two-group atomic `change_cta_tile`
mutation admitted by `spec.md` section 47.4. Cooperative SW64 and two-stage
combinations remain excluded from mutation enumeration.
