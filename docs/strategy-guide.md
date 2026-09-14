# Search strategy guide

An MCTS action in this project is a semantic optimization strategy, not a specific
code edit. When MCTS selects a strategy, the LLM receives the parent kernel, workload
contract, target hardware, and the parent's lightweight NCU profile. The LLM may
produce several different concrete implementations of the same strategy through
progressive widening.

This guide assumes familiarity with grids, thread blocks, warps, streaming
multiprocessors (SMs), and the GPU memory hierarchy. It does not assume prior
experience with WMMA, tensor-core kernel design, or MCTS.

## Strategy scope and applicability

The configured strategies are not universally applicable to every GPU kernel.
Coalescing, vectorized access, synchronization reduction, register-pressure reduction,
and shape specialization are broad optimization categories, but each is useful only
when the program and workload satisfy its assumptions. For example, vectorization
requires safe alignment, and synchronization can be removed only when memory
dependencies permit it.

The tensor-core output-tiling, pipelining, and multi-accumulator strategies are
matrix-operation strategies. Their current CUDA prompts are specialized further to
the fixed BF16 GEMM workload and launch contract. They should not be offered unchanged
for arbitrary reductions, elementwise kernels, scans, or workloads without compatible
tensor-core operations.

The current configuration has no machine-readable applicability constraints. As the
benchmark suite grows, strategy definitions should declare requirements such as:

```yaml
applicability:
  operations: [gemm]
  dtypes: [bfloat16, float16]
  backends: [cuda_cpp]
  hardware_capabilities: [tensor_cores]
```

The strategy layer should filter the configured catalog against the workload,
backend, hardware manifest, and legal launch constraints before constructing MCTS
actions. This avoids spending generation budget on transformations that cannot apply.
Semantic strategy IDs can remain stable while backend- and workload-specific prompt
variants provide concrete implementation guidance. Applicability filtering is planned
work and is not implemented by the current search.

## Fixed workload and launch constraints

The current benchmark is a 4096 x 4096 x 4096 BF16 GEMM with FP32 accumulation. A
16 x 16 thread block is launched for each logical 16 x 16 output tile. Generated
kernels must preserve the function ABI, tensor layouts, output type, correctness
tolerances, and launch configuration.

The fixed launch is an important constraint. A kernel can still have one physical
block compute a larger output region by grouping logical block coordinates, making
only one block in each group perform the work, and returning from the other blocks.
Such a transformation must ensure that every output element is written exactly once.
Changing the host-side grid or block dimensions is not currently allowed.

## General strategies

### Coalesced global memory

`coalesced_global_memory` asks adjacent threads to access adjacent addresses where
the layouts permit. Coalescing reduces the number of memory transactions needed to
serve a warp. It is most useful when NCU reports weak memory throughput or inefficient
access patterns, but it cannot by itself overcome a scalar arithmetic bottleneck.

### Vectorized memory access

`vectorized_memory_access` asks the generator to use wider aligned loads or stores.
This can reduce instruction count and improve memory-transfer efficiency. The
generator must prove alignment from the fixed workload and avoid changing the order
or ownership of output elements.

### Reduced synchronization

`reduce_synchronization` targets unnecessary block-wide barriers. A barrier stalls
every warp in a block until all participating threads arrive. Removing one is valid
only when no thread can observe incomplete shared-memory writes or reuse shared
storage prematurely.

### Reduced register pressure

`reduce_register_pressure` targets excessive per-thread state. High register use can
limit the number of resident warps, while aggressive reduction can spill values to
local memory or destroy useful instruction-level parallelism. Register count should
therefore be interpreted together with occupancy, stalls, and throughput rather than
minimized in isolation.

### Fixed-shape specialization

`specialize_workload_shape` allows compile-time reasoning about the fixed dimensions.
It may remove boundary checks, simplify indexing, choose exact tile factors, or
unroll known loop extents. The resulting kernel still has to accept the declared
runtime arguments and preserve the workload contract.

## Tensor-core strategies

Tensor cores execute small matrix multiply-accumulate operations cooperatively across
a warp. CUDA's WMMA interface exposes these operations through fragments representing
matrix A, matrix B, and an accumulator. For BF16 GEMM, the accumulator remains FP32
until results are converted to the required BF16 output.

The B_gen=5 Nebius run demonstrated why explicit tensor-core strategies are needed:

- The scalar root reported 0% tensor-pipe utilization and executed about 13.6 billion
  instructions.
- A generated WMMA parent reduced the instruction count to about 461 million, but
  tensor-pipe utilization was only 2.55%.
- The best candidate reached reward 1.8428, approximately a 6.3x speedup over the
  root, but remained far behind the cuBLAS baseline.

The generated design assigned all eight warps in a 256-thread block to the same
16 x 16 output tile. Each warp accumulated a different portion of K, stored a partial
tile in shared memory, and then participated in a reduction. This exposes tensor-core
instructions, but it duplicates accumulator storage, adds shared-memory traffic and
synchronization, and gives the block only one small output tile to produce.

### Independent warp output tiling

`tensor_core_output_tiling` asks each warp to own a distinct output tile and accumulate
the complete K dimension for that tile. No cross-warp reduction is then required.
The eight warps can produce several output tiles concurrently, increasing useful work
per block and making tensor-core execution easier to sustain.

Because the launch grid still names 16 x 16 logical tiles, a larger physical output
tile requires safe grouping of block coordinates. Owner blocks compute the grouped
region; non-owner blocks return without writing. Edge handling must remain correct
even though the current 4096 dimensions divide common tile sizes exactly.

### Pipelined tensor-core data movement

`pipeline_tensor_core_data_movement` combines independent warp-owned output tiles
with staged operand movement. The intended steady state is:

1. Load a future A/B tile from global memory into shared memory.
2. Consume the current shared-memory tile with tensor-core MMA operations.
3. Alternate buffers so data movement overlaps arithmetic where possible.

Loads should be coalesced and, when alignment permits, vectorized or asynchronous.
Shared-memory layouts must avoid bank conflicts. The strategy should not recreate the
cross-warp partial-K reduction that limited the observed WMMA design.

### Multiple accumulator fragments per warp

`tensor_core_multi_accumulator` asks each warp to own several independent accumulator
fragments rather than only one 16 x 16 result. A staged A or B operand can then feed
multiple MMA operations before the block waits for another global-memory transfer.
This increases arithmetic work per pipeline stage and amortizes synchronization,
address calculation, and copy latency.

The strategy was added after the profiled B_gen=50 search produced a correct
double-buffered `cp.async` kernel but measured less than 8% tensor-pipe utilization.
That kernel performed only one 16 x 16 x 16 MMA per warp in each K stage, leaving
little computation with which to hide operand-delivery latency.

More accumulator fragments consume more registers, and larger physical output tiles
may consume more shared memory. The generator must balance this reuse against
occupancy and spilling. Each warp must retain unambiguous output ownership, and any
grouping of the fixed logical block coordinates must avoid missing or duplicate
writes.

## Interpreting the lightweight NCU profile

The profile is evidence, not an automatic diagnosis. Useful relationships include:

- `tensor_pipe_utilization_pct` near zero on BF16 GEMM suggests scalar arithmetic or
  ineffective tensor-core use.
- High occupancy does not imply high performance; the observed WMMA parent had about
  99% achieved occupancy but only 2.55% tensor utilization.
- `registers_per_thread` helps assess occupancy limits and spill risk.
- DRAM, L2, and L1 throughput help locate pressure in the memory hierarchy.
- Long-scoreboard stalls suggest warps are waiting for dependent memory operations.
- Executed instruction count can reveal expensive scalar work, indexing, reductions,
  or synchronization overhead.

The LLM receives both friendly summary fields and the underlying metric names, values,
and units. A strategy prompt should use the metrics relevant to its transformation
without treating any single percentage as a complete performance model.

## Correctness and search behavior

Every generated realization is compiled and checked against the reference before it
can become a search node. A valid kernel may remain in the search even when it is
slower than its parent. Invalid generations and repairs consume generation budget but
are not backed up as measured rewards.

The current production configuration contains eight strategies. `spec.md` describes
an initial set of approximately 22 NVIDIA-derived prompts, so expanding and validating
the strategy catalog remains future work rather than an implied change to MCTS
semantics.
