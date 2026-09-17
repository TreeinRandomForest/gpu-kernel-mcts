# Local kernel autotuning

This document records an experimental extension beyond the current Milestone A search.
It does not override `spec.md`. The initial goal is to determine how much of the gap
between the best generated CUDA kernel and cuBLAS comes from precise configuration
choices rather than a missing kernel architecture.

## Role in the search

The first implementation should tune only the final global-best kernel after MCTS:

```text
MCTS selects a kernel family and implementation
                    |
                    v
extract a typed, bounded tuning configuration
                    |
                    v
generate legal configurations under B_tune
                    |
                    v
compile -> correctness -> benchmark on the warm GPU worker
                    |
                    v
export the fastest valid tuned kernel
```

Post-search tuning avoids multiplying tuning cost across every node and does not
change PUCT, progressive widening, UCB, backup, or the MCTS result. Tuning attempts
use a separate `B_tune`; they do not consume `B_gen` and are not backed up through
the tree. The untuned MCTS best and the tuned best must both be retained and reported.

The controller implements this as an optional phase:

```bash
--autotune \
--tuning-budget 20 \
--tuning-method random \
--tuned-best-output tuned-best.cu
```

It is disabled by default. The worker remains alive until the phase completes.
`random` and deterministic `grid` ordering are currently supported; Ax/BoTorch is a
future candidate-selection method.

Eligible constants must be declared explicitly. Autotuning never guesses which
numeric literals are safe to change:

```cpp
// KERNEL_MCTS_TUNE BLOCK_M=64,128
// KERNEL_MCTS_TUNE PIPELINE_STAGES=1,2,3
#define BLOCK_M 128
#define PIPELINE_STAGES 2
```

An unannotated final kernel produces a persisted `tuning_skipped` result. Each selected
configuration replaces only its corresponding integer `#define`. The current schema
assumes the listed cross-product is safe to attempt; compile and correctness failures
still consume `B_tune` and are retained.

The GPU is required for black-box compile, correctness, and latency measurements.
For tens or hundreds of observations, surrogate fitting and acquisition optimization
can run on the controller CPU; GPU-accelerated or variational GP fitting should not be
necessary initially.

## Parameters for a WMMA GEMM

### CTA output tile

`BLOCK_M` and `BLOCK_N` specify the output region produced by one cooperative thread
block (CTA). Larger tiles can increase reuse of A and B but require more registers,
shared memory, and work per CTA.

Initial choices:

```text
BLOCK_M = {64, 128}
BLOCK_N = {64, 128}
```

### Warp arrangement

`WARPS_M` and `WARPS_N` describe how the block's warps divide its output tile. Their
product determines the number of warps and normally determines the thread-block size:

```text
WARPS_PER_BLOCK = WARPS_M * WARPS_N
THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK
```

Initial choices:

```text
WARPS_M = {1, 2, 4}
WARPS_N = {1, 2, 4}
```

The warp output tile and the number of accumulator fragments follow from the CTA tile
and warp arrangement. More accumulators increase instruction-level parallelism and
reuse, but also increase register pressure.

### K chunk

`K_CHUNK` is the amount of the reduction dimension staged and processed together.
It controls shared-memory footprint, copy granularity, synchronization frequency,
and fragment reuse.

```text
K_CHUNK = {16, 32, 64}
```

It must be divisible by the MMA instruction's K dimension.

### Pipeline stages

`PIPELINE_STAGES` controls how many global-to-shared transfers can be buffered around
the current compute stage. Additional stages may hide memory latency but consume more
shared memory and can increase synchronization or bookkeeping overhead.

```text
PIPELINE_STAGES = {1, 2, 3}
```

### Later parameters

After the basic tuner works, consider exposing:

- asynchronous-copy width and copies per thread;
- shared-memory leading dimensions, padding, and swizzles;
- vector load and store widths;
- accumulator fragment shape and count;
- epilogue ownership and store vectorization;
- `__launch_bounds__` and minimum blocks per SM;
- split-K or cluster shape where the workload and implementation support them.

## Launch geometry

Grid dimensions should normally be derived from the output tile rather than tuned as
independent variables:

```text
grid.x = ceil_div(N, BLOCK_N)
grid.y = ceil_div(M, BLOCK_M)
block.x = 32 * WARPS_M * WARPS_N
```

The present fixed-launch harness causes generated kernels with larger logical CTA
tiles to filter out most launched blocks. A proper parameterized harness should pass
the derived launch configuration directly so inactive blocks are never launched.

## Static legality constraints

Reject configurations before compilation when they violate proven constraints:

- threads per block must not exceed the hardware limit;
- `BLOCK_M` and `BLOCK_N` must divide into the selected warp tiles;
- `K_CHUNK` must be compatible with the MMA K dimension;
- computed shared-memory usage must fit the configured per-block limit;
- required alignments must hold for vector and asynchronous copies;
- pipeline buffers must not overlap live data;
- launch geometry must cover the workload exactly or include correct boundary logic;
- register and occupancy estimates may guide search but should not become hard
  rejection rules unless they prove the launch is impossible.

Every compiled configuration must still pass the normal correctness contract before
it is benchmarked.

## Candidate selection

Begin with random or Sobol sampling over the statically legal configurations. This
provides a simple baseline and validates the parameterization, renderer, evaluator,
and trace records.

Ax can later serve as the candidate-suggestion layer, using Sobol initialization and
BoTorch-based Bayesian optimization. Our SQLite trace remains the source of truth.
Compare adaptive optimization against random/Sobol sampling under the same `B_tune`;
a small, mostly categorical space may not benefit from a Gaussian-process model.

## Trace requirements

Persist every tuning trial, including:

- parent MCTS run and final-best node identity;
- optimizer and optimizer configuration;
- trial number and complete parameter configuration;
- whether the configuration passed static validation;
- compile result and diagnostic;
- correctness result and error statistics;
- benchmark samples and GPU operating-state telemetry;
- measured latency and rank at the time of observation;
- environment-manifest ID, source hash, and binary hash; and
- cumulative `B_tune` and wall-clock cost.

Compilation or correctness failures count against `B_tune` because they consumed a
configuration trial, even though they do not produce a latency observation.

## Initial experiment

1. Parameterize the current best WMMA CUDA kernel using the six initial dimensions.
2. Replace block-index filtering with directly derived grid and block dimensions.
3. Enumerate and statically filter the finite configuration space.
4. Run a small correctness-focused smoke tune.
5. Compare random, Sobol, and Ax/BoTorch selection under equal `B_tune`.
6. Compare the tuned CUDA kernel with the untuned MCTS best, fixed CUTLASS baseline,
   and cuBLAS under the same workload and measurement policy.
7. Use the remaining performance gap and profiles to decide whether the next step is
   additional CUDA parameters or a CuTe DSL WGMMA/TMA kernel family.
