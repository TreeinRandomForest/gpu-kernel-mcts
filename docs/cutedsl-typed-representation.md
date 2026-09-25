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

The initial fixed BF16 design uses CTA tile `(128,256,64)`, cluster `(2,1)`, and
three shared-memory stages. It records complete A and B copy contracts: layouts,
tile coverage, element size, alignment, SW128/SW64 selection, multicast axis, stage
stride, storage offset, barrier count, and producer ownership.

### WGMMA consumer

Two consumer warp groups cover the CTA M dimension with `64x256x16` WGMMA
operations. A and B are consumed from shared memory and FP32 accumulators remain in
consumer-owned registers.

### Epilogue

The initial epilogue converts register accumulators to N-major BF16, stages `(64,64)`
tiles in a separate four-stage shared-memory region, and uses a TMA global store.

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

## Generalization

Reusable concepts—values, shapes, layouts, memory spaces, ownership, lifetimes,
buffers, dependencies, and synchronization—can later form a shared substrate.
Kernel-family layers should add their own operations: WGMMA accumulation for GEMM,
reduction trees for reductions, or normalization phases for softmax. This avoids
forcing every kernel into a GEMM schema or prematurely building a general GPU IR.
