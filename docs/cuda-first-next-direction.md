# CUDA-first next research direction

## Status and intent

The current research priority is to pause expansion of the operation-specific CuTe
DSL language and investigate a CUDA-first optimizer enriched with compiler,
profiling, dataflow, and hardware context. Preserve the CuTe implementation and its
results: it is a useful structured-search case study and a working backend, not
discarded code. Do not resume broad CuTe language construction by default in a new
session without an explicit decision to do so.

This is a research direction, not a silent change to the Milestone A or Milestone B
semantics in `spec.md`. Any implementation that changes search behavior, state
identity, budgets, or persistence must first be reconciled with the specification.

## What the CuTe work established

There are two distinct CuTe tracks:

1. The pinned NVIDIA CUTLASS 4.5.1 Hopper dense-GEMM example was adapted to the
   repository correctness and timing contract. A deterministic schedule sweep over
   CTA tile and cluster shape improved the default `193.024 us` schedule to
   `184.880 us`, 4.4% slower than same-run `177.056 us` cuBLAS.
2. The repository built an independent typed Hopper BF16 GEMM representation,
   legality system, and renderer. Its documented structural path was serial
   (`675.408 us`), software prefetch (`274.256 us`), producer/consumer warp
   specialization (`266.032 us`), and a two-consumer-group `(128,256,64)` CTA
   (`184.112 us`). The final state was 4.55% slower than a separately measured
   `176.096 us` cuBLAS result.

The strongest independent path used deterministic typed mutations (`B_mut`) rather
than substantive LLM discovery. Typed LLM generation was smoke-tested, including a
cached transposition, but has not yet demonstrated an advantage over deterministic
mutation search.

The main lesson is a tradeoff. A typed representation provides canonical identity,
legality, deterministic rendering, efficient mutations, and reliable caching. It
also requires substantial operation- and target-specific work before search begins:
schema design, compatibility rules, lowering, GPU debugging, and validation. The
current independent language is specialized to the fixed BF16 GEMM workload and
Hopper mechanisms; it is not yet a general GEMM language, much less a general GPU
kernel language.

## Why prioritize CUDA again

Free-form CUDA generation keeps the candidate program space open inside the search
loop. The compiler supplies lowering and rejects illegal implementations, avoiding a
new hand-built renderer for every operation. The mechanism can accept other CUDA
kernels, although the current workload harness and several configured strategies
remain specialized to the fixed BF16 GEMM and must not be described as universally
general today.

The fastest recorded CUDA path improved the naive root from `28,249.408 us` to a
raw MCTS median of `734.448 us`; an independent 18-trial local autotune reached
`726.384 us`. The CUDA-era cuBLAS reference was `185.376 us`, measured separately.
These results show effective open-ended source generation while leaving a large gap
that richer compiler and hardware context may help close.

## Proposed multi-representation CUDA loop

Keep CUDA source as the generated program and augment each valid node with compact,
auditable views of what the compiler and hardware produced:

```text
CUDA source
    + compiler diagnostics and resource usage
    + AST or LLVM/NVVM summaries
    + PTX summaries
    + SASS instruction-category summaries
    + NCU measurements
    + simplified dataflow/hardware model
        -> LLM proposal
        -> CUDA source
        -> compile, correctness, benchmark, cache
```

NVCC does not naturally expose the CuTe MLIR representation for ordinary CUDA
source. Investigate Clang CUDA AST/LLVM IR or NVVM IR before assuming MLIR is the
right intermediate form. Initial NVIDIA-toolchain evidence can come from retained
PTX, `ptxas` reports, `cuobjdump`, and `nvdisasm`.

Do not place unbounded raw IR, PTX, or SASS dumps into prompts. Extract stable,
structured summaries such as:

- registers, spills, local memory, static/dynamic shared memory;
- load/store widths and address spaces;
- tensor-core and scalar instruction categories;
- barriers, synchronization, and asynchronous-copy mechanisms;
- launch and occupancy constraints;
- parent-to-child compiler deltas, behind an explicit ablation flag.

Compiler artifacts and summaries must be cached with the node evaluation. Ordinary
MCTS visits must not recompile, reprofile, or reevaluate the GPU program.

## Simplified dataflow and hardware model

Build a compact graph that describes where data moves and where computation occurs:

```text
global memory -> shared memory -> registers -> compute -> registers -> global memory
```

Graph nodes should identify memory spaces, compute units, synchronization, and
ownership. Edges can carry bytes, estimated transactions, reuse, dependencies,
bandwidth ceilings, and latency. Overlay target capabilities such as register and
shared-memory capacity, tensor-core throughput, memory bandwidth, SM count, and
occupancy constraints. Reconcile estimates with NCU rather than treating the model
as ground truth.

The first model should be deliberately small and inspectable. Its purpose is to give
the LLM constrained architectural context and to support hypotheses such as “reuse B
across more output work” or “barriers dominate despite available tensor throughput,”
not to simulate the complete GPU.

## Budget roles in this direction

- `B_gen`: all LLM calls proposing CUDA source, including repairs. Only compile- and
  correctness-valid results become nodes.
- `B_mut`: optional deterministic transformations already implemented by the
  repository, potentially including future AST-level rewrites. It does not represent
  compiler-context extraction.
- `B_tune`: concrete local parameter trials around a selected implementation,
  initially after search. Moving tuning to promising internal nodes is a separate,
  explicitly budgeted algorithmic experiment.

## Recommended experiment sequence

1. Define a versioned compiler-context record associated with a complete cached
   evaluation result. Preserve raw artifact provenance separately from prompt-sized
   summaries.
2. Capture PTX and `ptxas` resource evidence for valid CUDA nodes. Add GPU-independent
   parser/serialization tests before remote validation.
3. Add bounded SASS category extraction from the compiled artifact without changing
   reward timing or triggering work on node visits.
4. Add an off-by-default prompt flag for compiler summaries. Persist the exact
   summary and prompt provenance used for each generation.
5. Run equal-budget ablations with the same root, workload, model, seed, strategy
   catalog, worker class, and search parameters:

   - CUDA source only;
   - CUDA plus NCU;
   - CUDA plus compiler summary;
   - CUDA plus NCU and compiler summary.

6. Prototype a small dataflow extractor and Hopper model, first for diagnosis and
   visualization, then as another off-by-default prompt ablation.
7. Only after the summaries prove useful, consider deterministic AST transformations
   or learned strategy priors.

The first implementation should not change PUCT, progressive widening, UCB,
valid-only backup, state identity, or budget semantics. Treat compiler context as
cached evaluation metadata and optional generation context.

## Questions the experiments should answer

- Does compiler context improve valid-generation rate under equal `B_gen`?
- Does it improve best reward or time-to-best beyond NCU alone?
- Which summaries influence generated code, rather than merely increasing tokens?
- Can a small dataflow model correctly identify the winning transformation class?
- Do parent-to-child compiler deltas help, or introduce harmful path dependence?
- Which strategies remain GEMM-specific, and what applicability metadata is needed
  before testing other CUDA workloads?

## New-session handoff

After reading `AGENTS.md` and `spec.md`, inspect this document, `TODO.md`,
`docs/strategy-guide.md`, `docs/research-roadmap.md`, and the current git status.
Begin by auditing the existing CUDA compile/artifact/evaluation path and proposing a
small versioned compiler-context record plus tests. Do not alter search semantics,
resume broad CuTe language expansion, provision a GPU, or make billable API calls
without an explicit request.
