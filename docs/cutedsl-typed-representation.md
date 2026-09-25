# CuTe typed kernel representation

The project is developing a small Hopper-GEMM-specific language above CuTe DSL.
It is not a replacement for CuTe and is not intended to become a general compiler
IR. Its purpose is to give search a canonical, statically checked kernel design that
can be deterministically lowered into complete CuTe DSL source.

```text
typed Hopper GEMM design
        -> deterministic lowering
CuTe DSL Python kernel
        -> CuTe JIT and NVIDIA toolchain
compiler IR -> PTX -> GPU machine code
```

## Why a separate language?

In raw source, one architectural choice is often distributed across layouts, copy
atoms, barriers, loop structure, and ownership calculations. A text edit can change
only part of that contract and still compile before failing at launch or correctness.
The typed representation makes the complete choice explicit and validates interacting
fields together.

It provides:

- stable canonical serialization and hashing for cache/transposition identity;
- deterministic, auditable mutations rather than implicit source edits;
- rejection of proven shape, ownership, alignment, storage, or dependency errors
  before JIT or GPU evaluation;
- structured LLM output with bounded values; and
- a clean point at which analytical models and graph features can be attached.

Measured GPU performance remains the reward source. Approximate representation-level
costs must not silently prune valid kernels.

## Current schema

`IndependentCuteGemmKernel` contains four coordinated layers.

### TMA/shared-memory mainloop

The initial independent BF16 design uses CTA tile `(64,256,64)`, cluster `(1,1)`,
one consumer warp group, and three shared-memory stages. It records complete A and B
copy contracts: layouts, tile coverage, element size, alignment, SW128/SW64
selection, multicast axis, stage stride, storage offset, barrier count, and producer
ownership. A two-stage mainloop is now renderable as a guarded standalone candidate;
its v49 H100 repository-contract validation passed exact correctness and produced
distinct compiler/runtime fingerprints. It remains outside MCTS until a separate
mutation-promotion change.

### WGMMA consumer

One consumer warp group covers the CTA with `64x256x16` WGMMA operations. A and B
are consumed from shared memory and FP32 accumulators remain in consumer-owned
registers. A cooperative two-group `(128,256,64)` diagnostic is retained separately;
its second output row is not yet correct and is not part of the typed root.

The promoted root has also completed a standalone full-workload checkpoint. It
cycles all 64 K tiles through the typed three-stage shared-memory ring and launches a
`64x16` CTA grid for the `4096x4096x4096` GEMM. The H100 run passed correctness and
measured a 665.792 us median over 30 CUDA-event samples after 10 warmups. Subsequent
backend integration reused the canonical workload inputs and cuBLAS reference and
measured 670.592 us while capturing the compiled artifact and NCU profile.
The two-stage candidate measured 671.248 us under the same 30-sample policy, a
difference within ordinary run-to-run noise.

### Epilogue

The initial epilogue converts register accumulators to N-major BF16, stages `(64,64)`
tiles in a separate four-stage shared-memory region, and uses a TMA global store.
The representation separately records one `(64,64)` physical guard tile required
by the lowered composed layout; it is allocation padding, not a fifth logical
pipeline stage, and participates in identity and shared-memory capacity checks.

### Execution schedule

The dynamic protocol is represented as an agent/buffer/phase DAG:

```text
tma_load_agent -> shared_a/shared_b -> wgmma_agents -> accumulator
                                                        |
                                                        v
epilogue_agents -> shared_c -> tma_store_agent -> global_c
```

Phases record their GPU agent, operation, read/write values, waited-on signals, and
emitted signals. Buffer rings record stage count, memory space, producer, consumer,
and barrier protocol. Validation checks identifier uniqueness, references, stage
consistency, produced signals, and acyclic phase dependencies.

## Legality versus performance

Static legality rejects only proven violations such as incomplete tile coverage,
incompatible multicast ownership, insufficient alignment, overlapping storage,
excess shared memory, missing agents or buffers, and cyclic dependencies. It does
not reject a legal kernel merely because a layout or schedule is predicted to be
slow.

## Lowering status

The static lowering binds the representation to CUTLASS 4.5.1 constructors for
tiled WGMMA, cluster/shared layouts, TMA load/store atoms, and mainloop/epilogue
pipelines. A pinned-container diagnostic resolves those bindings without importing
NVIDIA's expert GEMM kernel.

The first dynamic-lowering diagnostic now generates shared storage, fixed-cluster
coordinates, a B multicast mask, transaction barriers, TMA load/store operations,
and launch configuration. It round-trips two independent A tiles and one multicast B
tile and requires exact BF16 equality. Its generated module imports successfully in
the pinned CUTLASS 4.5.1 container, but it still requires H100 JIT, launch, and
correctness validation.

The diagnostic deliberately omits WGMMA and the production epilogue so TMA address,
multicast, and synchronization failures can be isolated. The representation stays
outside MCTS until those checks pass and subsequent WGMMA/epilogue lowering produces
a complete correct GEMM.

The staged v23 H100 run passed all copy-only checks, including exact two-CTA A/B
round trips with B multicast. The debugging sequence exposed and corrected a
warp-uniform TMA issuance requirement. Detailed evidence is recorded in
[Independent CuTe TMA copy diagnostic v23](experiments/cutedsl-independent-tma-v23.md).

The next standalone diagnostic layers one WGMMA K tile onto the validated single-CTA
TMA path. Two consumer warp groups partition shared A/B, issue `64x256x16` WGMMA
operations into FP32 register accumulators, retile and convert those accumulators into
BF16 shared memory, and use TMA to store the complete output tile. Its stages
separately test WGMMA JIT, instruction issue/completion, and numerical output against
a PyTorch FP32-accumulation reference. This remains a diagnostic lowering and is not
yet a searchable state.

After rebuilding the CuTe image, run:

```bash
python -m kernel_mcts.cute_entrypoint \
  --mode independent-tma-copy \
  --independent-tma-stage compile_only
```

Then advance one stage at a time through `launch_empty`, `single_cta_a_load`,
`single_cta_a`, `single_cta_ab`, `cluster_ab_no_multicast`, and
`cluster_ab_multicast`. The empty launch separates geometry from memory movement;
the load-only stage separates the transaction barrier from TMA-store completion.
Each run prints flushed messages before and after JIT, launch, and synchronization.
The multicast lowering performs a cluster-wide arrive/wait after every CTA
initializes its transaction barrier and before any multicast can target a remote CTA.

The guarded two-stage full-workload validation uses the canonical repository
contract and does not alter MCTS state:

```bash
python -m kernel_mcts.cute_entrypoint \
  --mode independent-tma-copy \
  --independent-tma-stage wgmma_full_workload \
  --pipeline-stages 2 \
  --independent-repository-contract \
  --output /output/cutedsl-independent-pipeline2.json
```

## Generalization

Reusable concepts—values, shapes, layouts, memory spaces, ownership, lifetimes,
buffers, dependencies, and synchronization—can later form a shared substrate.
Kernel-family layers should add their own operations: WGMMA accumulation for GEMM,
reduction trees for reductions, or normalization phases for softmax. This avoids
forcing every kernel into a GEMM schema or prematurely building a general GPU IR.
