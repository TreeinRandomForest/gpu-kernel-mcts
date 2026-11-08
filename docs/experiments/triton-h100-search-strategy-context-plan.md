# Triton/Gluon search: strategies and parent context

The consolidated target factor space and current support status are in
`triton-h100-ablation-matrix.md`. This document retains the implementation
sequence and earlier experiment history.

Status: implementation in progress. The Gluon adapter passed an H100 backend
smoke. The stage-aware v2 and smaller staged v3 catalogs are implemented;
the generated descriptor-Triton bridge passed H100 validation. The later v18
catalog reached a correct three-slot wait-one Gluon kernel from the packaged
Triton root in three declared H100 seeds; see the adapter README. Other
context/search work remains.
Pause further gap-closing GPU experiments while building and validating this path.

## Why this work is needed

The current H100 Triton catalog has five broad actions. It does not name the
source-level transitions behind the best standalone results: descriptor TMA,
two-CTA Gluon/WGMMA, a three-slot prefetch ring, wait-one/refill ordering, or
persistent cluster assignment. The evidence is implementation-specific:
wait-one improved a three-slot Gluon source substantially; multicast B and
four-slot wait-two did not improve that source; 256-wide persistent assignment
saved about 2 µs in a fixed-shape factorial, while 128-wide Gluon doubled
DRAM traffic. Do not turn one fixed-shape winner into a universal tile rule.

Current implementation boundaries also matter:

- `triton_h100` accepts a Python `launch(A,B,C,M,N,K)` returning a named
  Triton `CompiledKernel`. Its original NCU selection required the name
  `bf16_gemm_triton`. The local adapter now keeps the standalone Gluon
  kernel's `gemm` device body, interprets the search ABI's column-major
  `B[K,N]` as a zero-copy physical `B[N,K]` view, and profiles by the
  compiled name plus NVTX range. The H100 adapter smoke passed; see
  `triton-h100-gluon-search-adapter/README.md`.
- The CLI only permits sequential search for `cuda_cpp`. The sequential
  prompt and compiler-context selector also assume CUDA C++.
- MCTS can show Triton compiler resources and static PTX counts and lazily
  collected NCU metrics. It does not yet include the bounded H100 graph in
  generation prompts. Triton dataflow prompt views are explicitly rejected.
- Section 48 of `spec.md` defines an optional CUDA-only freeform action.
  A Triton-only freeform or hybrid action needs an explicit spec amendment.

## 1. Make the best kernel reachable under the search contract

The contract-compatible source fixture for the validated wait-one Gluon
implementation now preserves `launch(...)`, B's logical shape and strides,
the returned compiled kernel, and recorded NCU selection identity. Verify
source, output, compiled resources, and latency through the *search harness*
against the packaged Triton root and cuBLAS on one worker. CUDA-event search
timing and the prior CUDA Graph timings are different contracts; report both
without comparing their absolute latencies as if interchangeable. Add local
tests for launcher/selection metadata and a bounded H100 smoke. If NCU's
hard-coded name would miss a valid returned kernel, select the launch by a
recorded compiled-kernel identity or NVTX range and test that change.

## 2. Version a precise, stage-aware strategy catalog

Keep the current five-action catalog as the baseline. Build a separate v2
catalog with action prompts and machine-readable applicability for the GEMM
operation, H100 capabilities, and parent implementation stage:

| Action theme | When offered | Required reasoning and guard |
| --- | --- | --- |
| Tile and warp co-design | Ordinary Triton/Gluon | Account for the observed `BN=256`, four-warp register/spill cliff; preserve output coverage and measure latency. |
| Descriptor TMA staging | Pointer-load Triton | Convert A/B physical views safely; check descriptor legality, resource use, and exactness. |
| Cross-CTA tile traversal | Tiled GEMM | State which operand may be revisited; use measured aggregate L2/DRAM only as evidence, not an A/B attribution. |
| Clustered Gluon WGMMA/TMA | Triton GEMM ready for a structural rewrite | Use a legal two-CTA ownership map and ordinary B loads initially; keep accumulator and output ownership explicit. |
| Three-slot wait-one pipeline | Clustered Gluon with barriers/WGMMA | Match TMA barrier phase and shared-slot lifetime; allow one WGMMA group in flight; verify emitted wait and no spills. |
| Persistent cluster assignment | Valid clustered Gluon | Derive worker count from hardware/tiles, cover each output once, preserve pipeline teardown, and check latency and resources. |

The large clustered rewrite may need a versioned source scaffold or a typed
template underneath its semantic action; a prompt alone must first prove it
can generate a valid candidate. Applicability filtering must exclude actions
that cannot operate on the current parent, instead of spending `B_gen` on
impossible requests. Preserve the old catalog and uniform-prior MCTS arm for
controlled comparisons. Add parser/filter tests and prompt-snapshot tests.

The opt-in catalog is `configs/triton_h100_strategies_v2.json`. It declares
the parent stages accepted by each action. `strategy_applicability.py` uses
Python AST syntax and the kernel returned by `launch` to identify pointer Triton, descriptor Triton, clustered
Gluon, other Gluon, or unknown parents. Unknown parents receive no
stage-specific v2 action. MCTS filters before asking for priors or an
expansion profile; an empty eligible set is a dead end. Local tests cover
stage classification, filtering, malformed metadata, prompt guards, and one
fixture realization routed through MCTS. The separately validated Gluon
adapter is the fixture's source. One GLM-5.3 probe of the initial structural
prompt returned an ordinary Triton pointer-load kernel; a second probe with
the validated source supplied as a scaffold returned an AST-identical copy.
The scaffold is an explicit diagnostic override and is absent from the
production v2 catalog. Thus an independently generated structural rewrite
is still an open gate before claiming that the prompt alone reaches it.
The three v2 Gluon actions now require the generated `launch` to return a
kernel classified as `clustered_gluon`. A pointer-load Triton response is
logged as `STRATEGY_MISMATCH` after spending its generation call and before
GPU evaluation; repair receives the expected and observed stages. This
prevents a source-stage mismatch from being admitted under a Gluon action.
The gate does not certify the wait-one or persistent schedule itself.

The direct v2 structural action failed a bounded scaffold-free GLM-5.3
request plus one repair: the first source was pointer Triton and the repair
was not a clustered Gluon kernel. The opt-in staged catalog at
`configs/triton_h100_strategies_v3_staged.json` adds smaller parent/output
stage transitions. Its descriptor/TMA action produced a valid H100 kernel on
the repair call. The next single-CTA Gluon TMA/WGMMA action returned pointer
Triton then no extractable source at the output-token cap. A still smaller
Gluon MMA-v2 language/layout action also failed within two calls. Neither
Gluon stage was sent to GPU evaluation. A diagnostic single-CTA Gluon fixture,
made by changing the launcher and cluster layouts of the previously validated
two-CTA source, did compile, pass correctness, and measure 249.632 µs on H100.
It is not a model-generated intermediate. A bounded two-call attempt to add
the two-CTA cluster from this valid parent kept `num_ctas=1` on both calls;
the candidate-stage guard rejected both before GPU evaluation. The remaining
gate is a model-generated Gluon transition. Further requests should follow a
revised template or an explicitly labeled partial-scaffold experiment.
The subsequent five-round same-worker crossover found the single-CTA fixture
faster than the two-CTA reference by 20.624 µs median paired. An opt-in typed
one-CTA tile action then produced two valid but slower variants (N=128 and
BK=32); a measured-result MCTS replay charged two `B_mut` proposals and kept
both valid nodes. This does not close the scaffold-free model-generation gate
or justify a live MCTS performance claim. The exact samples and constraints
are in the Gluon search-adapter experiment README.
An alternate-root Gluon traversal arm subsequently demonstrated that GLM-5.3
can write a bounded local source edit: two distinct grouped mappings became
valid MCTS children under `B_gen=4`. The root remained faster. The completed
SQLite trace records the explicit root source, local package overlay, parent
NCU profile, compiler summary, prompts, failures, and timings. This resolves
the local-action reachability check while leaving the pointer-Triton-to-Gluon
bridge and a performance-positive strategy open.

## 3. Build one bounded parent-context record for both algorithms

Use the same versioned context builder for sequential and MCTS generation:

1. **Always:** parent source, ABI/layout and correctness contract, measured
   parent CUDA-event median with sample count, calibrated same-worker cuBLAS
   reference if valid, and the chosen action or explicit freeform instruction.
2. **Optional measured evidence:** lazy parent NCU profile with metric-set ID,
   units, replay/cache policy, tensor-pipe and occupancy signals, local-memory
   sectors, L2 read requests/hits/misses, and DRAM bytes. These are whole-
   kernel observations, not per-tensor or per-edge traffic. Keep the profile
   collection policy independent of prompt visibility.
3. **Optional static evidence:** compiler resources, PTX hash and bounded
   instruction classes, then a separately versioned SASS summary of TMA vs
   `LDGSTS`, HGMMA shape, WGMMA wait operands/sites, and local-load/store
   sites. Static sites must not be described as dynamic execution counts.
4. **Optional architecture:** the existing bounded H100 resource graph with
   architectural limits and explicit uncalibrated-link labels; do not invent
   live bandwidth or latency. Parent resource use may be overlaid only where
   compiled or measured evidence exists.
5. **Deferred until validated:** Triton/Gluon PTX dataflow views. Audit tensor
   descriptors, cluster ownership, shared-memory buffers, branches, and
   output stores against source/SASS on several kernels before offering
   counts, paths, subgraphs, or full adjacency in prompts.

Persist the exact rendered context and prompt hash in SQLite. Bounded size,
provenance, and availability/limitations fields are part of the schema.
Test missing profiles and artifacts explicitly; absence should not be filled
with guessed counts or traffic.

## 4. Expose sequential and freeform MCTS modes

Extend the sequential CLI and prompt builder to `triton_h100` using the same
parent-context record. Keep freeform and strategy-menu choices as separate
arms, with configurable history turns and the existing repair choice lock.

For an opt-in strategy-free Triton MCTS arm, represent freeform generation as
one reserved action. PUCT over one action is trivial; the existing exact
progressive-widening formula decides when to generate another realization,
and UCB selects among existing realizations when widening is saturated.
This preserves the global tree, validity rule, `B_gen` accounting, `Q_mean`
selection, and cached evaluation. A hybrid catalog-plus-freeform arm is a
separate later comparison. Amend `spec.md` before implementing either Triton
MCTS variant; do not silently reuse the CUDA-only `freeform_cuda` ID.

## 5. Validation and ablations

First run GPU-independent tests for strategy applicability, prompt provenance,
context masking, freeform action accounting, progressive widening, and UCB.
Then perform a small H100 end-to-end smoke for the Gluon adapter, NCU kernel
selection, sequential Triton, and one-action MCTS. Only after those gates,
compare equal `B_gen`, root, worker image, model, seed, context, metric set,
and timing contract. Separate these factors in reports:

- old versus v2 strategy catalog;
- sequential freeform versus sequential menu;
- strategy-only versus one-action freeform versus hybrid MCTS;
- measured profile, compiler/SASS, hardware graph, and validated dataflow
  visibility, one context factor at a time.

Report best verified latency versus generation calls, valid/repair rates,
action allocation, depth and unique states, prompt tokens, and GPU/API cost.
Do not infer a winning strategy or context from one short sampled tree.
