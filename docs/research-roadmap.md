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
