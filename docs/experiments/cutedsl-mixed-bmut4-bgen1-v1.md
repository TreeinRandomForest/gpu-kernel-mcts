# CuTe DSL mixed search with B_mut=4 and B_gen=1

## Purpose

This guarded run validates mutation-first routing on H100: deterministic typed
mutations must consume `B_mut` before an LLM fallback consumes `B_gen`. It also tests
the boundary between stochastic model output and deterministic CuTe rendering.

## Configuration

- Date: 2026-09-20
- Run ID: `8228440c-9dad-4f4f-881f-e5d5e0cf9276`
- Workload: `bf16_gemm_4096_h100`
- GPU: NVIDIA H100 80GB HBM3, SXM, compute capability 9.0
- Provider: Nebius
- Worker image: `docker.io/saarora/gpu-kernel-mcts:cutedsl-search-v1`
- Model: `gpt-5.6-sol`, reasoning effort `medium`
- `B_mut`: 4/4
- `B_gen`: 1/1
- LLM tokens: 1,963 input and 661 output
- LLM latency: 10.443 seconds
- Profiles: 3 using `lightweight_v1`

## Result

The first four proposals matched the mutation-only experiment: three new valid nodes
and one transposition back to the cached root. The best schedule was again tile
`(128,256)` with cluster `(2,1)`:

- root median: 192.912 us;
- best median: 186.640 us; and
- best reward: 0.0330525, or approximately `1.0336x` root speedup.

After `B_mut` was exhausted, iteration 5 selected `change_cluster_shape` and made one
LLM call. The model chose tile `(128,256)` with cluster `(2,1)`, which was already the
best canonical state. However, it returned a complete reformatted Python launcher and
an incorrect literal `CONFIGURATION_HASH`. `CuTeDSLBackend` correctly rejected that
source because it was not byte-for-byte output from the deterministic renderer. The
proposal was recorded as `INVALID/COMPILE_FAILURE`, created no node, and did not alter
backup statistics.

This was an interface failure rather than a poor schedule decision. The raw model
output contained the correct typed representation but crossed the trust boundary as
model-authored executable source.

## Resulting fix

The mixed CuTe generator now:

1. asks the model for only one typed representation JSON object;
2. also accepts the prior full-source form for compatibility by extracting only its
   literal `KERNEL_MCTS_REPRESENTATION`;
3. statically validates the typed representation;
4. discards all model-authored executable code and embedded hashes; and
5. uses `PinnedCuteGemmRenderer` to create the canonical program.

With this adapter, the observed proposal canonicalizes to the existing `(128,256)`,
`(2,1)` node and should be recorded as a transposition without GPU reevaluation.
Generation traces also fall back to the selected budget mechanism when generator
metadata omits a mechanism label. The following repeat run validates these changes
remotely.

## Successful v2 validation

The corrected controller was rerun on Nebius H100 SXM on 2026-09-20:

- Run ID: `aade2041-6114-4653-8e2d-6b8738954dbd`
- Controller commit: `ecb2ba64c92cfbaf72df5b3ae150d8e8bdddc3a8`
- `B_mut`: 4/4
- `B_gen`: 1/1
- LLM tokens: 1,764 input and 343 output
- LLM latency: 7.061 seconds
- Final nodes: 4
- Profiles: 3

The first four mutations again produced three new nodes and one cached-root
transposition. The fifth iteration selected `change_cluster_shape` after `B_mut` was
exhausted. The model returned a JSON representation changing its `(1,2)` parent to
the existing `(2,1)` best schedule. Static validation passed, the deterministic
renderer produced the canonical source, compilation and correctness passed, and the
transposition table reported `reused_node=1` with the exact existing configuration
hash `2a461f56da0f3f1d59193e00a3d3702e5b1293470e1b7435667132641bebc785`.

The worker reused the evaluation already cached for that canonical program, so the
LLM realization did not cause another GPU evaluation. Its generation record retained
`proposal_mechanism=llm_typed_representation`, `typed_output_format=json`, complete
transformation evidence, and separate `B_gen=1`, `B_mut=4` accounting.

The measured root median was 191.840 us and the best `(128,256)`, `(2,1)` median was
186.016 us. Reward was 0.030829, approximately `1.0313x` root speedup. These timings
are run-local validation evidence; the important v2 outcome is that typed generation,
canonical rendering, transposition detection, and cache reuse all worked remotely.

## Provenance limitations

Both traces recorded dirty controller trees, and neither worker manifest captured an
immutable image digest. The v1 trace used commit
`caa36bdc0cda3a8a97edce04ee6508f27ab50ac9`; v2 used
`ecb2ba64c92cfbaf72df5b3ae150d8e8bdddc3a8`. These runs remain behavioral validation
rather than reproducible performance benchmarks.
