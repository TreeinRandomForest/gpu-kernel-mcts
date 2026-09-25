# CuTe DSL Search Model

## Relationship to cuBLAS

cuBLAS is a fixed external performance baseline for this project. NVIDIA supplies
many optimized implementations and internal dispatch logic that select a kernel for
the workload, dtype, and hardware. The implementation and selection space are mostly
opaque to this repository. cuBLAS is measured under the same workload contract, but
it is not an MCTS node and MCTS does not modify it.

CuTe DSL search instead constructs explicit, auditable kernels. Each MCTS node is one
complete effective CuTe configuration under a fixed workload, GPU, and toolchain. Its
state may eventually include:

- CTA and cluster decomposition;
- WGMMA and warp-group arrangement;
- pipeline stages and producer/consumer specialization;
- TMA copy layout;
- shared-memory layout and swizzle; and
- epilogue ownership and staging.

Only fields that the typed schema, deterministic renderer, and static validator
actually support belong to the active search space. The current search supports CTA
tile, cluster shape, mainloop pipeline depth, and epilogue pipeline depth.

## Expert template versus structural search

The current renderer uses NVIDIA's pinned Hopper dense-GEMM CuTe implementation as
its executable foundation. The typed fields configure that expert implementation;
they do not replace its WGMMA/TMA algorithm. Consequently, the current search is a
structured way to tune and combine exposed choices around an already optimized
kernel. On this small space, MCTS should be expected to recover roughly the same
answer as exhaustive grid search rather than outperform it.

The next stage still begins from the pinned NVIDIA implementation, but introduces
one bounded structural transformation at a time—for example, a supported alternative
shared-memory layout or swizzle. Each transformation is first diagnostic-only and is
promoted to typed state plus an MCTS strategy only after H100 JIT, correctness,
artifact-identity, and performance validation. This is analogous to adding a CUDA
strategy, except the realization is a typed CuTe transformation rather than an
unrestricted source rewrite.

A simpler independently controlled CuTe GEMM representation remains a separate
Milestone B objective. Until that exists, results must be described as search over
an expert template, not synthesis of a new GEMM implementation.

The first such structural diagnostic overrides NVIDIA's shared-memory layout-atom
heuristic from SW128 to SW64 for the BF16 A/B mainloop while preserving operand
majorness and the already-SW64 epilogue. It remains outside canonical state until an
H100 run proves JIT legality, correctness, and distinct compiled artifacts.
The v15 H100 diagnostic supplied that evidence: both variants were exactly correct
and produced distinct normalized IR, fatbins, and kernel layout identities. SW64 was
0.27% slower by median in this run, which is within noise; promotion is justified by
legality and distinct structure rather than a performance claim.

## One expansion

Conceptually, one new realization is produced as follows:

```text
PUCT selects a semantic StrategyEdge
        |
        v
progressive widening permits another realization
        |
        v
proposal mechanism produces a typed candidate
   |                                  |
   +-- deterministic mutation         +-- stochastic LLM proposal
        |                                  |
        v                                  v
     B_mut                              B_gen
        |                                  |
        +---------------+------------------+
                        v
              static validation
                        v
          deterministic source rendering
                        v
          JIT -> correctness -> benchmark
                        v
                valid child node
```

PUCT continues to select semantic strategies. Progressive widening limits the number
of distinct concrete outcomes beneath a strategy. When an existing realization is
revisited, UCB selects among those existing children. Only valid measured children
become nodes and participate in backup.

### Choosing the proposal mechanism

The core search can configure one proposal mechanism directly or wrap a mutation and
LLM generator in the mixed-mechanism router. Direct CuTe mutation generation uses
`B_mut`, while direct LLM generation uses `B_gen`.

The initial mixed-mechanism router uses a simple mutation-first rule after PUCT has
selected the semantic strategy and progressive widening has admitted a new child:

1. use an untried supported typed mutation when one exists and `B_mut` remains;
2. otherwise use LLM generation when it is supported and `B_gen` remains; and
3. declare that parent-strategy proposal space exhausted when neither mechanism can
   produce another realization.

This is deliberately not another PUCT decision. PUCT chooses the semantic strategy;
the router chooses how to realize it. The budgets remain separate, and exhausted
mechanisms become ineligible without changing PUCT, progressive-widening, or UCB
formulas. A later explicit ablation may compare this rule with a bandit or learned
mechanism policy using observed validity, reward, and cost.

Each proposal trace records the eligible mechanism sequence and the selected
mechanism's budget kind. The CLI constructs mutation-only and mixed CuTe searches,
providers initialize a CuTe worker backend, and both paths have completed guarded
remote H100 validation.

## Deterministic mutations

A deterministic mutation applies a known typed transformation with explicit values:

```text
change_cta_tile: (128, 256) -> (128, 128)
```

Given the parent, strategy, and parameters, it always produces the same canonical
candidate. It does not call an LLM. The `B_mut` budget counts these mutation proposals
separately from model calls. Different mutation paths that reach
the same canonical configuration transpose to the same cached state.

The current standalone mutation mechanisms are:

- `change_cta_tile`;
- `change_cluster_shape`;
- `change_pipeline_stages`;
- `change_epilogue_stages`; and
- `change_shared_memory_swizzle`.

The pipeline mutation changes only A/B mainloop staging and offers explicit depths
`2` and `3`. The pinned heuristic remains the root behavior, and epilogue staging
remains pinned. Explicit depth `4` is deliberately excluded from this initial
neighborhood because it is equivalent to the heuristic for the reference
`(128,256)` tile and would create duplicate effective states.

The epilogue mutation changes only output-pipeline depth and currently offers values
2 and 3 only for tile `(128,256)`, cluster `(2,1)`, the schedule validated on H100.
Pinned depth 4 is represented by `None`; explicit 4 is excluded as an identical-state
alias. Mutation generation filters statically invalid cross-schedule combinations.
The canonical backend CLI accepts a complete tile/cluster tuple for guarded hardware
validation, and a GPU-independent MCTS integration test verifies that an epilogue
realization creates a schema-v3 node under `B_mut` without consuming `B_gen`.
A bounded aggregate validation mode runs stages 2 and 3 through that canonical path
and checks identity, correctness, cache reuse, and distinct runtime fingerprints
before the field is exercised in a larger search.

The swizzle mutation changes NVIDIA's shared-memory layout-atom selection from its
heuristic to SW64 while preserving operand majorness. SW64 is initially legal only
for tile `(128,256)`, cluster `(2,1)`, where standalone H100 validation proved exact
correctness and distinct IR/fatbin identity. Schema v3 makes `heuristic` versus
`sw64` explicit canonical state; the mutation consumes `B_mut` and no `B_gen`.
Canonical v18 validation also confirmed cache reuse and diagnostic-v2 profiling for
both states. A focused search smoke uses a heuristic root at the validated schedule,
only `change_shared_memory_swizzle`, `B_mut=1`, and `B_gen=0`.

```bash
python -m kernel_mcts.search_cli \
  --provider nebius \
  --image docker.io/saarora/gpu-kernel-mcts:cutedsl-swizzle-v18 \
  --trace cutedsl-swizzle-bmut1-v18.sqlite \
  --backend cute_dsl \
  --generator cute-mutation \
  --generation-budget 0 \
  --mutation-budget 1 \
  --cute-root-tile-m 128 \
  --cute-root-tile-n 256 \
  --cute-root-cluster-m 2 \
  --cute-root-cluster-n 1 \
  --cute-strategy change_shared_memory_swizzle \
  --k-max 1 \
  --max-depth 1 \
  --best-output cutedsl-swizzle-bmut1-v18-best.py \
  --nebius-project-id PROJECT_ID \
  --nebius-subnet-id SUBNET_ID \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --confirm-create-and-terminate
```

For guarded experiments, the search CLI accepts all four `--cute-root-*` schedule
arguments together. They are legal only for `--backend=cute_dsl`, are validated and
rendered through the normal typed path, and the complete root representation is
recorded in run provenance. This permits an epilogue-focused run to begin at the
validated `(128,256)`, cluster `(2,1)` state without changing the default CuTe root.
`--cute-strategy` may be repeated to select an explicit subset of the registered
CuTe semantic actions for a guarded ablation. It changes only the configured action
set; selection and budget semantics within that set are unchanged.

The first epilogue-only mutation smoke uses two deterministic proposals so both
validated realizations can be represented beneath one strategy edge:

```bash
python -m kernel_mcts.search_cli \
  --provider nebius \
  --image docker.io/saarora/gpu-kernel-mcts:cutedsl-epilogue-v14 \
  --trace cutedsl-epilogue-bmut2-v14.sqlite \
  --backend cute_dsl \
  --generator cute-mutation \
  --generation-budget 0 \
  --mutation-budget 2 \
  --cute-root-tile-m 128 \
  --cute-root-tile-n 256 \
  --cute-root-cluster-m 2 \
  --cute-root-cluster-n 1 \
  --cute-strategy change_epilogue_stages \
  --k-max 2 \
  --max-depth 1 \
  --best-output cutedsl-epilogue-bmut2-v14-best.py \
  --nebius-project-id PROJECT_ID \
  --nebius-subnet-id SUBNET_ID \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --confirm-create-and-terminate
```

Run `d884b5c3-bc5c-4648-a3a6-d1c68d16b5a9` completed in two iterations with
`B_mut=2`, `B_gen=0`, three nodes, and two valid measured-leaf backups. Its detailed
trace analysis is recorded in
[CuTe epilogue mutation smoke v14](experiments/cutedsl-epilogue-bmut2-v14.md).

## Independent typed root

`--cute-root-kind independent` selects the independently lowered
`(64,256,64)` single-warp-group root on both the client and worker. Worker
calibration therefore measures the same program used as the search root. The
initial independent neighborhood contains two H100-validated transitions:
atomically changing both A and B shared-memory swizzles from 128 bytes to 64 bytes,
or changing the mainloop from three stages to two while coherently rebuilding its
barrier ring and storage offsets. Each consumes `B_mut`; no independent LLM proposal
path is enabled yet. The router excludes the unvalidated two-stage-plus-SW64
combination.

```bash
python -m kernel_mcts.search_cli \
  --provider nebius \
  --image IMAGE_WITH_THIS_CHANGE \
  --trace cutedsl-independent-swizzle.sqlite \
  --backend cute_dsl \
  --cute-root-kind independent \
  --generator cute-mutation \
  --generation-budget 0 \
  --mutation-budget 1 \
  --cute-strategy change_shared_memory_swizzle \
  --k-max 1 \
  --max-depth 1 \
  --best-output cutedsl-independent-swizzle-best.py \
  --nebius-project-id PROJECT_ID \
  --nebius-subnet-id SUBNET_ID \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --confirm-create-and-terminate
```

The SW64 candidate passed the canonical repository contract on H100 but was slower
than SW128 (718.832 us versus 670.592 us median). It remains a valid MCTS node; the
search does not prune it merely for being locally slower.

The two-stage candidate also passed exact correctness with distinct MLIR and fatbin
fingerprints. It measured 671.248 us, effectively tied with the three-stage root.
To smoke-test that transition, replace the command's strategy with
`change_pipeline_stages`; keep `B_mut=1`, `B_gen=0`, `k_max=1`, and `max_depth=1`.

The subsequent four-strategy `B_mut=35` run explored 24 unique nodes, recovered the
same pinned-stage `(128,256)`, cluster `(2,1)` schedule selected by grid tuning, and
showed no meaningful benefit from the additional staging controls. See
[CuTe mutation search B_mut=35 v14](experiments/cutedsl-mutation-bmut35-v14.md).

They are connected to core MCTS through a deterministic mutation generator, covered
by GPU-independent tests, and validated through remote H100 mutation searches.

## Stochastic LLM proposals

An LLM realization consumes `B_gen`, including any repair calls. It may return a
constrained typed representation or structural patch. Fresh calls can produce
different concrete realizations beneath the same semantic strategy.

An LLM does not automatically expand the representable design space. Its output must
still be expressible by the current typed schema, renderer, and validator. For
example, an LLM cannot introduce a new pipeline organization while pipeline structure
is fixed by the renderer.

The roles are therefore:

- deterministic mutations systematically explore known supported choices;
- LLM calls stochastically propose or combine choices within the supported language;
- repository/compiler development adds new schema fields, rendering behavior, and
  proven legality rules, thereby expanding the actual search space.

## Independent TMA-to-shared-memory mainloop

The first step away from the pinned NVIDIA expert template is a GPU-free typed
contract in `cute_independent.py`. It describes the producer side of one fixed
Hopper BF16 GEMM mainloop rather than patching a hidden template choice:

- A transfers a `(128,64)` BF16 tile and B transfers a `(64,256)` tile;
- both global and shared-memory operands are K-major and 16-byte aligned;
- the `(2,1)` CTA cluster loads A independently and multicasts B across the M axis;
- three stages have disjoint A/B storage regions and one arrival-barrier slot each;
- A and B change together between coordinated SW128 and SW64 layouts; and
- canonical JSON and a stable configuration hash identify the complete contract.

Static validation rejects partial layout changes, incompatible multicast ownership,
incomplete tile coverage, insufficient alignment, overlapping stages or operands,
and shared-memory capacity violations. The deterministic output is intentionally a
structural renderer input, not an executable kernel. It cannot become an MCTS node
or consume `B_mut` until it becomes a complete executable kernel and both variants
pass standalone H100 validation.

The next typed layer specifies the consumer and epilogue contract. Two WGMMA
consumer warp groups cover the `(128,256,64)` CTA tile using `64x256x16` operations
and own FP32 register accumulators. A four-stage `(64,64)` epilogue converts those
accumulators to N-major BF16, stages them in a dedicated shared-memory region, and
uses a TMA store. The combined 180,224-byte bound fits the configured H100 per-CTA
limit. Cross-component validation ensures complete WGMMA and epilogue coverage and
prevents storage overlap.

This is a complete *typed design*, not executable CuTe DSL yet. The next lowering
must map every field to CUTLASS 4.5.1 APIs for descriptors, multicast masks,
barriers, WGMMA partitioning, accumulator ownership, and TMA stores. Until that
lowering compiles and passes correctness on H100, the independent representation is
kept outside `CuTeDSLBackend`, transpositions, mutation enumeration, and all budgets.

The first lowering checkpoint maps the typed design to concrete CUTLASS 4.5.1 APIs
for tiled WGMMA construction, cluster and shared-memory layouts, TMA load/store
atoms, and TMA mainloop/epilogue pipeline constructors. The generated module does
not import `HopperWgmmaGemmKernel` or the pinned `dense_gemm.py`. A container-side
binding diagnostic resolved all ten required symbols and records the lowering source
hash plus explicit implemented/missing component lists. It deliberately reports
`executable_kernel=false`: shared storage, coordinates and multicast masks, producer
and consumer loops, the register-to-shared epilogue, TMA store loop, and launch still
require dynamic lowering.

The first standalone dynamic diagnostic lowers only the TMA round trip. It launches
one `(2,1)` cluster, loads distinct A tiles per CTA, multicasts one B tile across the
cluster, and stores the copied values for exact comparison. This intentionally
precedes WGMMA so illegal addresses, multicast masks, and barrier behavior can be
debugged without tensor-core or accumulator state. It remains noncanonical and
cannot consume any search budget.

The binding checkpoint can be repeated inside an image containing the current source:

```bash
python -m kernel_mcts.cute_entrypoint \
  --mode independent-lowering-bindings
```

## Budgets and autotuning

The intended accounting is:

```text
B_gen  = LLM generation and repair calls
B_mut  = deterministic typed mutation proposals
B_tune = standalone mechanical tuning trials
```

Autotuning is not required at every MCTS node. Initially, MCTS should evaluate each
new complete configuration once and reuse its cached result. Standalone or post-search
autotuning remains separately budgeted. Any future leaf-local tuner should be an
explicit ablation rather than hidden work performed at every expansion.
