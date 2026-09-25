# Research roadmap

This document records speculative research directions beyond the current Milestone A
implementation. It does not override `spec.md` or commit the project to these designs.
Each direction should be refined into a testable hypothesis before implementation.

## Guiding progression

```text
static kernel features -> analytical cost model -> analytical strategy priors
                       -> AST/IR features       -> learned policy
                                                -> learned value
```

Every experiment should retain uniform-prior MCTS as a baseline, use the same
workloads and correctness checks, and compare methods under the same `B_gen`. Model
predictions must not silently prune valid kernels or leak held-out benchmark results.

## Analytical GPU models

Build lightweight models that help an LLM reason about a kernel before applying a
semantic strategy. Candidate features and estimates include:

- arithmetic intensity and roofline regime;
- global-memory transactions, coalescing, and redundant traffic;
- shared-memory footprint, traffic, and possible bank conflicts;
- register pressure, theoretical occupancy, and launch limits;
- synchronization frequency and estimated exposed latency;
- instruction mix, data reuse, and tensor-core eligibility.

The initial model should combine source-derived features, compiler resource reports,
hardware limits, and cached profiler measurements. Its output should be structured and
auditable, with assumptions and uncertainty recorded alongside each estimate.

Possible uses include enriching generation prompts, ranking relevant strategies, and
rejecting only transformations that violate proven resource or ABI constraints.
Approximate performance predictions should guide exploration rather than become hard
pruning rules, because inaccurate models could remove non-monotonic optimization paths.

Initial research questions:

1. Do analytical summaries improve valid-generation rate?
2. Do they improve best speedup at fixed `B_gen`?
3. Which estimates agree with measured counters, and where do they fail?
4. Does uncertainty-aware guidance outperform deterministic recommendations?

## AST- and IR-based reasoning

Extend analytical reasoning from raw source text to structured program representations.
Useful extracted structure may include:

- loop nests, bounds, and unrolling;
- array subscripts and affine access patterns;
- address spaces and shared-memory lifetimes;
- barriers, warp operations, and dependency structure;
- tensor layouts, reductions, and producer-consumer relationships;
- launch annotations and kernel ABI.

A practical first step is feature extraction from Clang AST, CUDA-aware compiler IR,
or normalized intermediate metadata while continuing to generate complete source.
Direct AST rewriting can follow later. CUDA macros, templates, inline PTX, and library
abstractions make an AST-only representation incomplete, so raw source and compiler
evidence should remain available.

Compare raw-source prompts, raw source plus AST/IR summaries, and constrained
structured transformations. Measure correctness rate, compilation rate, semantic
diversity, speedup, prompt size, and generation cost.

## Typed computation and schedule graphs

Investigate a representation more constrained and hardware-aware than either raw
source or a conventional syntax AST. Separate the immutable mathematical computation
from its GPU schedule:

- the computation graph describes operations such as matrix multiplication,
  reduction, conversion, and epilogue functions; and
- the schedule graph describes tiling, ownership, memory placement, data movement,
  synchronization, and pipelining.

Graph values should represent tensors, tiles, or fragments and carry machine-checkable
properties such as:

- shape, element type, and accumulation type;
- logical and physical layout;
- global, shared, or register memory placement;
- alignment and vectorization requirements;
- thread, warp, warp-group, or CTA ownership; and
- lifetime and pipeline stage.

Edges or operations can represent ordinary copies, TMA transfers, tiled loads,
`ldmatrix`, MMA or WGMMA instructions, reductions, conversions, barriers, and stores.
A verifier should reject incompatible shapes, layouts, ownership, address spaces, and
synchronization before source generation or GPU evaluation. Static resource estimates
could also reject only proven launch or capacity violations while leaving uncertain
performance decisions to search.

Semantic strategies would become typed graph rewrites rather than unrestricted source
rewrites. Examples include enlarging a CTA tile, splitting one accumulator into several
MMA fragments, changing warp ownership, adding a pipeline stage, applying a
shared-memory swizzle, or replacing a global-to-shared copy with TMA. Canonical graph
serialization could improve transposition detection and make transformations easier to
audit.

The validated graph should lower through deterministic renderers, initially to a
known-correct CUDA or CuTe DSL template. CuTe DSL would therefore be an implementation
target rather than necessarily the search representation. This distinction matters:
free-form generation in a higher-level language may exchange CUDA errors for DSL type
or layout errors, whereas a typed graph or restricted configuration schema can prevent
many invalid proposals before compilation.

Start with a small GEMM-specific schedule schema rather than a general GPU compiler IR.
Expose a limited set of parameters and rewrites for CTA tiles, warp layouts, MMA atoms,
pipeline stages, memory-copy mechanisms, shared-memory layouts, and epilogues. Compare
four proposal formats under equal generation budgets:

1. unrestricted CUDA source;
2. CUDA AST transformations;
3. unrestricted CuTe DSL; and
4. typed schedule-graph or configuration transformations rendered deterministically.

Measure compile-valid and correctness-valid proposals per generation call, repair
frequency, unique semantic schedules, compile latency, best kernel latency, and cost per
valid improvement. A successful prototype should demonstrate that the structured
representation improves proposal efficiency without preventing architecture-changing
transformations.

### Overlay with a calibrated GPU resource graph

Overlay the computation/schedule graph with a separate model of GPU resources and
movement paths. Hardware nodes may represent HBM, L2, shared memory, register files,
TMA engines, tensor cores, CUDA cores, SMs, CTAs, warp groups, and clusters. Edges
should describe scope plus uncertain or calibrated attributes such as latency,
sustainable bandwidth, transaction granularity, issue throughput, multicast,
synchronization cost, and contention.

The overlay maps semantic values and operations onto hardware paths:

```text
A/B tile: HBM -> L2 -> TMA -> shared memory
matmul:   shared memory -> WGMMA -> register accumulators
output:   registers -> shared memory -> TMA -> L2 -> HBM
```

Treat this as an uncertainty-aware analytical model rather than a cycle-accurate
simulator. Begin with symbolic bytes, operations, reuse, lifetime, and capacity;
then add roofline bounds, occupancy constraints, and pipeline-overlap estimates.
Calibrate estimates against NCU and timings while retaining measured GPU latency as
the search reward.

Candidate uses include resource validation, bottleneck explanations, strategy
prompts, and optional priors. Predicted performance must not become a hard pruning
rule unless it proves a launch or correctness impossibility. Compare graph-informed
and uninformed search under identical budgets and report sensitivity to incorrect
hardware parameters.

## Analytical strategy priors

Map predicted bottlenecks and transformation opportunities to a normalized prior over
semantic strategies. For example, evidence of strided global loads could increase the
prior for coalescing or layout strategies, while occupancy limits could increase the
prior for register- or shared-memory reductions.

Candidate prior modes are:

1. uniform;
2. analytical only;
3. LLM only;
4. analytical features supplied to an LLM prior;
5. calibrated mixtures of analytical and learned priors.

All prior inputs and outputs should be persisted. Priors must be computed without
using the unobserved performance of candidate children. Evaluation should report
best-found performance versus `B_gen`, validity rate, strategy coverage, calibration,
and sensitivity to inaccurate analytical estimates.

## Learned policy and value functions

Milestone A traces can eventually support learned search guidance resembling parts of
AlphaGo Zero, adapted to stochastic single-player program search.

A learned policy is the nearer-term target. It can predict promising semantic
strategies from the kernel, workload, hardware, static features, and cached profile.
Training targets may use successful strategy outcomes while accounting for exploration
bias, invalid generations, model identity, and multiple stochastic realizations.

A learned value function is more difficult. The relevant value is not simply the
current kernel reward; it should estimate the best valid descendant reachable under a
specified remaining generation budget. Targets are censored by finite searches and
change with the generator, strategy set, hardware, and budget. Initial value studies
should therefore compare several explicit targets, such as observed best descendant,
budget-conditioned improvement, and probability of exceeding a speedup threshold.

Important differences from two-player games include:

- expensive compile, correctness, and GPU measurements;
- stochastic LLM realizations beneath each semantic action;
- a DAG with transpositions rather than a pure tree;
- no opponent or known terminal game outcome;
- changing generators and toolchains that make training data non-stationary.

Start with policy-guided PUCT and retain measured-leaf backup. Introduce learned value
terms only through explicit ablations, with safeguards against training/test leakage
and with `Q_mean`, `Q_max`, and raw measured rewards preserved in traces.

## Closing the vendor-baseline gap

Early BF16 GEMM searches improved the naive root substantially but remained far behind
cuBLAS. A larger generation budget or deeper tree is unlikely to close that gap by
itself when search is refining a scalar or conventional SIMT implementation while the
vendor library uses Hopper-specific tensor-core machinery. Expand the reachable
program families before spending substantially more generation budget.

Add architecture-changing semantic strategies for:

- warp-level and warp-group tiling;
- WGMMA tensor-core operations;
- TMA global-to-shared-memory transfers;
- double- or triple-buffered asynchronous pipelines;
- swizzled shared-memory layouts and bank-conflict avoidance;
- register-fragment tiling and occupancy-aware resource control;
- cooperative, coalesced epilogues; and
- tile size, pipeline stage, and launch-shape tuning.

Several of these changes must be introduced together to yield a valid program. Include
a coarse semantic strategy that replaces a conventional GEMM structure with a correct
Hopper tensor-core template, then let finer strategies optimize its layout, pipeline,
and launch parameters. This provides a bridge that may be unreachable through small
local edits alone.

Evaluate structured implementation layers alongside raw CUDA generation:

- CuTe DSL exposes tensors, layouts, tiling, TMA, and WGMMA as structured concepts and
  could support both LLM transformations and conventional parameter autotuning.
- Constrained CUTLASS templates offer a lower-risk path to exploring expert kernel
  configurations, although this studies library-configuration search more than
  unrestricted kernel generation.
- Inline PTX may be useful for narrowly scoped WGMMA experiments after the higher-level
  paths work.
- Handwritten SASS should not be an initial target because it is fragile, difficult to
  validate, and tightly coupled to a particular GPU generation.

Profile-guided generation should provide occupancy, register pressure, memory
throughput, tensor-core utilization, stall reasons, and achieved FLOP/s. Prompts and
strategy priors can then distinguish incremental tuning opportunities from cases where
the kernel needs a different computational structure.

Suggested experiment order:

1. Inspect and profile the best valid generated kernel against the root, fixed CUTLASS,
   and cuBLAS implementations. Completed for the initial BF16 experiments.
2. Add one Hopper tensor-core macro-strategy and verify that it can produce a valid,
   correct candidate.
3. Add structured CuTe DSL or constrained CUTLASS configuration experiments. The
   first fixed CuTe baseline and nine-point typed schedule space are complete: the
   best schedule measured `184.880 us`, 4.4% behind same-worker cuBLAS. Milestone B
   now targets structural typed CuTe transformations under MCTS.
4. Decompose successful tensor-core kernels into finer layout, pipeline, resource, and
   launch strategies.
5. Add profile-informed prompts and optional priors.
6. Only then compare larger `B_gen` values, initially in the 100–300 range. Increasing
   `max_depth` beyond 10 is lower priority until traces show depth is the limiting
   factor rather than strategy coverage or stochastic realization breadth.

Keep cuBLAS and fixed CUTLASS results outside the MCTS state space as reference
baselines. Compare best-found latency and speedup at equal `B_gen`; do not use baseline
performance to prune otherwise valid nodes.

## Forward-pass graph partitioning and fusion

A longer-term target is optimization across a model forward pass rather than within one
isolated kernel. The search input would be a typed computation/dataflow graph, and a
state would describe both its partition into executable kernels and the schedules of
those kernels.

Graph-level semantic strategies could:

- fuse adjacent operators or GEMM epilogues;
- remove intermediate global-memory materializations;
- recompute cheap values instead of storing and reloading them;
- propagate or change tensor layouts across operator boundaries;
- move fusion boundaries to balance locality against resource pressure;
- form persistent regions; and
- overlap computation, data movement, and communication where dependencies allow.

Kernel-level strategies would then optimize each resulting region using CuTe DSL or
another deterministic backend. They could select tiles, layouts, memory placement,
TMA transfers, warp-group ownership, tensor-core operations, pipelines, and epilogues.
The dataflow representation should track where every value resides, its layout and
lifetime, and its producer-consumer dependencies. Cached profiling evidence could
identify global-memory traffic, exposed synchronization, register or shared-memory
pressure, occupancy limits, and pipeline imbalance.

The goal should generally be an efficient partition into several fused kernels, not a
literal single kernel for an entire forward pass. Excessive fusion can increase live
state, register spilling, shared-memory demand, synchronization, compile complexity,
and load imbalance; different operators may also require incompatible parallel
decompositions. Fusion boundaries are therefore search decisions rather than an
assumption that more fusion is always better.

This direction naturally creates a two-level search:

1. graph search chooses partitions, fusion, recomputation, and inter-operator layouts;
2. schedule search optimizes the implementation of each fused region.

Initial experiments should use a short, common subgraph such as GEMM plus bias,
activation, residual, or normalization. Compare unfused library calls, a manually
fused reference, and searched partitions under identical correctness and end-to-end
latency measurement. Only after those experiments should the graph grow toward a full
transformer block or forward pass.

## External generalization with GPU MODE

After the complete CuTe DSL flow works end to end—typed computation and schedule
representations, typed parameters, structural strategies, deterministic lowering,
profiling, and search—use a bounded selection of
[GPU MODE reference problems](https://github.com/gpu-mode/reference-kernels) as an
external generalization suite. This is a secondary validation milestone, not the main
research objective: attention against FlashAttention, useful operator fusion, and
forward-pass graph optimization remain higher-value targets.

Select roughly four to six problems spanning different computational motifs, such as
elementwise maps, reduction or scan, normalization or softmax, matrix operations, and
an irregular or fused operation. Freeze the search algorithm before inspecting
leaderboard outcomes, then give each problem fixed and separately reported `B_gen`,
`B_mut`, profiling, and wall-clock budgets. Compare the initial implementation,
deterministic mutation search, mixed LLM/mutation search, a strong library baseline,
and the public leaderboard where the hardware and measurement contract are
comparable.

Report more than rank: correctness and compile-valid rates, unique valid schedules,
best latency, search cost, budget-to-improvement curves, and sensitivity across the
published shapes and GPU targets. Treat leaderboard-specific specialization and
fixed-shape tricks as findings rather than allowing them to reshape the general search
architecture. The purpose is to expose overfitting to BF16 GEMM, identify which typed
language features fail to generalize, and provide reproducible external comparison.

As a rough effort allocation, keep GPU MODE evaluation to 20--25% of this research
phase and reserve 75--80% for attention, fusion, dataflow reasoning, and calibrated
architecture models. The maintained
[Popcorn CLI](https://github.com/gpu-mode/popcorn-cli) can be integrated only after a
small manual submission validates the benchmark and authentication workflow.

## Suggested staging

1. Define and validate a static feature schema on existing BF16 GEMM traces.
2. Add an uncertainty-aware analytical summary without changing selection.
3. Supply that summary to generation prompts and measure validity and speedup.
4. Convert analytical scores into an optional strategy-prior provider.
5. Add AST/IR-derived features and compare them with source-only features.
6. Train and evaluate an offline policy model from accumulated traces.
7. Define budget-conditioned value targets and test value-guided search offline.
8. Run controlled online ablations only after offline behavior is understood.

At each stage, document the hypothesis, data split, budget, model and toolchain
versions, failure policy, and stopping criterion before running experiments.
