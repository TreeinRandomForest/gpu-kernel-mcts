# CUDA MCTS implementation for slides

This document is a source for presentation slides and speaker notes. It describes
the implemented CUDA search path. `spec.md` remains authoritative when this document
and the implementation differ.

## Scope and current workload

The current CUDA experiment optimizes one fixed workload:

| Field | Current value |
|---|---|
| Operation | GEMM, `C = A B` |
| Dimensions | `M = N = K = 4096` |
| Inputs | BF16 |
| Accumulation | FP32 |
| Output | BF16 |
| Target | NVIDIA H100 80GB HBM3 SXM, SM90 |
| A layout | Row major, shape `(M,K)` |
| B layout | Column major, shape `(K,N)` |
| C layout | Row major, shape `(M,N)` |
| Entry point | `bf16_gemm_root` |
| Fixed launch | block `(16,16,1)`, grid `(ceil(N/16),ceil(M/16),1)` |
| Correctness tolerance | `rtol = atol = 0.02` |
| Benchmark | 10 warmups, 30 CUDA-event samples, median latency |

The search mechanism can accept complete CUDA source, but the current harness,
fixed ABI, launch contract, and several strategy prompts are specific to this BF16
GEMM experiment. Do not describe the present end-to-end system as validated on every
CUDA kernel.

Suggested slide title: **Current CUDA experiment**

Suggested visual: one GEMM equation with the workload contract listed beside it.

## One global program search

One MCTS run retains the full program graph for the complete budget. It does not
commit to a root move and restart search after every optimization.

One iteration performs:

1. Start at the original kernel.
2. Follow existing edges using PUCT and realization UCB.
3. When progressive widening permits, request one new candidate.
4. Compile and verify the candidate.
5. Benchmark a valid candidate.
6. Back up its measured root-relative reward through the selected path.

There are no simulated rollouts. A newly created valid node receives its value from
an actual GPU measurement.

The run returns the fastest valid kernel found anywhere in the graph. It does not
return the most visited root action.

Suggested slide title: **One MCTS iteration**

Suggested visual:

```text
root
  selection through cached nodes
  semantic strategy selection
  progressive-widening decision
  generation and verification
  measured leaf reward
  backup through selected edges
```

## State and node definition

A state identifies one effective compiled kernel under a fixed evaluation context:

```text
state = (
    binary fingerprint,
    workload contract,
    launch configuration,
    hardware and toolchain context
)
```

The optimization history is absent from the state. Two paths that produce the same
effective state share one canonical node through the transposition table.

Each `SearchNode` contains:

- the complete valid `EvaluationResult`;
- the CUDA source and backend;
- a state key;
- benchmark samples and reward;
- the cached compiled artifact and correctness evidence;
- optional cached NCU profile data;
- one `StrategyEdge` for each available semantic action.

Every node satisfies these invariants:

```text
proposal status = VALID
compile status = SUCCESS
correctness status = PASS
reward and benchmark are present
```

Failed generations remain proposal records. They never become nodes.

Depth is path specific because a canonical node may be reached through paths of
different lengths. `SearchNode` does not store one global depth value.

Suggested slide title: **A node is a verified kernel state**

## The graph has two edge levels

CUDA generation is stochastic. Applying one semantic strategy several times can
produce several different kernels. The implementation therefore separates strategy
selection from realization selection.

```text
SearchNode
  StrategyEdge: reduce_synchronization
    RealizationEdge: generated kernel A
    RealizationEdge: generated kernel B
  StrategyEdge: tensor_core_output_tiling
    RealizationEdge: generated kernel C
```

`StrategyEdge` stores:

- prior probability;
- visit count;
- cumulative backed-up value;
- `Q_mean` and `Q_max`;
- valid and invalid proposal counts;
- generated realization edges.

`RealizationEdge` stores:

- canonical child-node ID;
- descent count;
- cumulative backed-up value;
- realization `Q_mean`.

PUCT chooses a semantic strategy. Progressive widening decides whether to generate a
new realization. UCB chooses among existing realizations.

Suggested slide title: **Semantic strategies have stochastic realizations**

## CUDA strategy catalog

The implemented catalog currently contains eight actions:

| Strategy ID | Intended change |
|---|---|
| `coalesced_global_memory` | Improve adjacent-thread global-memory access |
| `vectorized_memory_access` | Use safe aligned vector loads and stores |
| `reduce_synchronization` | Remove unnecessary barriers without introducing races |
| `reduce_register_pressure` | Reduce per-thread state while retaining useful parallelism |
| `specialize_workload_shape` | Exploit the fixed workload and legal launch choices |
| `tensor_core_output_tiling` | Give warps independent complete output tiles |
| `pipeline_tensor_core_data_movement` | Overlap staged operand movement with MMA work |
| `tensor_core_multi_accumulator` | Increase MMA work and operand reuse per warp |

The first five are broad optimization categories with workload-dependent
applicability. The last three are specialized to the current BF16 GEMM and launch
contract.

The exact prompts live in `configs/strategies.yaml`, outside MCTS logic. `spec.md`
describes an initial catalog of approximately 22 NVIDIA-derived prompts. The current
implementation has eight, so the larger catalog remains planned work rather than
implemented behavior.

Suggested slide title: **Current CUDA strategy catalog**

## Strategy priors

Every node receives a probability distribution over its strategies when its action
edges are first initialized.

The implementation supports:

- uniform priors, with `P(s,a) = 1 / number of strategies`;
- optional LLM priors obtained by a separate provider call.

LLM prior calls use `B_prior`, not `B_gen`. The reported CUDA experiments used
uniform priors, so they consumed no LLM prior calls.

The node receives its cached lightweight profile before priors are requested. This
allows a future LLM prior provider to use measured evidence.

Suggested slide title: **Priors allocate initial attention across strategies**

## PUCT selects a strategy

For node `s` and strategy `a`:

```text
PUCT(s,a) = Q_mean(s,a)
          + c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))
```

Where:

- `Q_mean(s,a)` is the mean leaf reward backed up through the strategy edge;
- `P(s,a)` is the strategy prior;
- `N(s,a)` is the strategy-edge visit count;
- `sum_b N(s,b)` is the total number of action visits at the node.

Default `c_puct` is `1.5`. Experiments exposed it as a CLI parameter and also tested
`c_puct = 12`.

The exploitation term is `Q_mean`. `Q_max` is retained and logged for analysis but
does not affect default selection.

At a node with zero visits, the exploration numerator is exactly zero. All strategies
tie at score zero, and the seeded random generator breaks the tie. This implements
the exact formula rather than replacing zero with one.

Suggested slide title: **PUCT balances measured value and strategy exploration**

## Progressive widening limits LLM branching

One strategy can generate many distinct CUDA kernels. Generating all candidates at
once would create an unbounded branching factor.

For the prospective next visit to a strategy edge:

```text
K_allowed(s,a) = min(
    K_max,
    ceil(c_pw N(s,a)^alpha_pw)
)
```

Implemented defaults:

```text
c_pw = 1.0
alpha_pw = 0.5
K_max = 4
```

The implementation uses `action.visits + 1` as the prospective visit count. The
first selection therefore permits one child.

With the defaults:

| Prospective visits | Maximum distinct valid realizations |
|---:|---:|
| 1 | 1 |
| 2 to 4 | 2 |
| 5 to 9 | 3 |
| 10 and above | 4 |

The count of existing children includes valid canonical realizations. Invalid
proposals consume `B_gen` and remain logged, but they do not occupy a child slot.

Suggested slide title: **Progressive widening controls stochastic branching**

## UCB selects an existing realization

When progressive widening is saturated, MCTS selects an existing realization under
the chosen strategy:

```text
UCB(s,a,s') = Q_mean(s')
            + c_ucb sqrt(log(1 + N(s,a)) / (1 + N(s,a,s')))
```

Where:

- `N(s,a)` is the strategy-edge visit count;
- `N(s,a,s')` is the number of descents through that realization;
- `Q_mean(s')` is the mean backed-up value on the realization edge.

Default `c_ucb` is `1.0`.

The hierarchy is deliberate:

```text
PUCT chooses what semantic change to try.
Progressive widening decides whether to request another implementation.
UCB chooses which existing implementation to revisit.
```

Suggested slide title: **UCB allocates visits among generated kernels**

## Candidate generation and the full prompt

Every generation call starts a fresh LLM session. Grandparent and sibling
conversations are not retained. The search graph owns history.

The OpenAI API receives this fixed instruction:

```text
Return only the complete replacement kernel source. Do not use Markdown fences or
explanatory prose.
```

The input is one JSON document containing:

```json
{
  "task": "Produce a complete replacement GPU kernel preserving the workload contract.",
  "backend": "cuda_cpp",
  "target_hardware": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "form_factor": "SXM",
    "compute_capability": "9.0"
  },
  "workload": {
    "benchmark_id": "bf16_gemm_4096_h100",
    "operation": "gemm",
    "dtype": "bfloat16",
    "shapes": [
      {"dimensions": {"M": 4096, "N": 4096, "K": 4096}, "weight": 1.0}
    ],
    "rtol": 0.02,
    "atol": 0.02,
    "metadata": {
      "entry_point": "bf16_gemm_root",
      "accumulation_dtype": "float32",
      "output_dtype": "bfloat16",
      "tensors": "complete A, B, and C shape and layout contract",
      "abi": "complete parameter and fixed launch contract"
    }
  },
  "strategy": {
    "id": "selected semantic strategy",
    "description": "short description",
    "backend_prompt": "exact CUDA strategy prompt from strategies.yaml"
  },
  "parent_profile": {
    "schema_version": 2,
    "metric_set": "diagnostic_v2",
    "summary": "friendly metric names and values",
    "metrics": "raw NCU metric names, values, and units"
  },
  "parent_kernel": "the complete parent CUDA source",
  "constraints": [
    "Preserve the operation, tensor layouts, dtype, ABI, and entry point.",
    "Return a complete compilable source file, not a patch.",
    "Do not include Markdown fences or explanatory text."
  ],
  "attempt": 0
}
```

The actual prompt includes the full workload metadata, full profile, and complete
parent source rather than the abbreviated strings shown above.

The optional incoming profile-delta ablation adds the selected incoming strategy,
reward change, speedup ratio, and numeric profile-summary deltas. It is off by
default because it introduces path-dependent context into otherwise fresh sessions.

The trace stores the prompt text, API instruction, prompt hash, raw response,
candidate source, token usage, model metadata, and latency.

Suggested slide title: **Each LLM call sees one parent and one strategy**

## Repair prompt

An invalid LLM candidate can receive a bounded repair attempt. The repair prompt
retains the original parent, strategy, workload, hardware, and parent profile, then
adds:

```json
{
  "repair": {
    "previous_candidate": "complete failed candidate",
    "previous_evaluation": {
      "status": "INVALID",
      "invalid_reason": "COMPILE_FAILURE or CORRECTNESS_FAILURE",
      "compile_status": "FAIL or SUCCESS",
      "correctness_status": "NOT_TESTED or FAIL",
      "compilation": "bounded compiler stdout and stderr",
      "correctness": "error evidence when available"
    },
    "instruction": "Repair the failed candidate while preserving semantics."
  }
}
```

Compiler output is bounded before insertion. Default `max_repairs = 2`, so one
logical proposal can consume at most three `B_gen` calls: the initial generation and
two repairs.

Suggested slide title: **Repairs consume the same generation budget**

## Verification sequence

The generated source must pass every gate in order:

1. **Contract check** confirms the CUDA backend and fixed BF16 GEMM workload.
2. **Compilation** uses NVCC with SM90, C++17, `-O3`, relocatable device code, and
   cuBLAS linkage.
3. **Launch and correctness** run the candidate against the repository harness and
   cuBLAS FP32-compute reference using deterministic seed-zero inputs and the declared
   tolerance.
4. **Benchmark** records 30 positive finite CUDA-event timings after 10 warmups.
5. **Reward** uses the fixed root benchmark measured at the beginning of the run.
6. **Binary fingerprint and state key** identify the effective state under the
   workload, launch, hardware, and toolchain context.
7. **Node admission** creates a node or reuses an existing canonical node only when
   compilation and correctness pass and a valid benchmark exists.

Failure classification includes compile failure, launch failure, correctness
failure, timeout, benchmark failure, and infrastructure failure.

Infrastructure failures are distinct from candidate invalidity. The implementation
retries them up to the configured infrastructure retry limit, default one retry.
Infrastructure failures and invalid candidates receive no backup value.

Suggested slide title: **Only verified kernels enter the search graph**

## Performance reward

For one workload shape:

```text
reward(K) = ln(T_root / T_K)
```

Where `T_root` is the original kernel median fixed at the start of the run and `T_K`
is the candidate median.

Interpretation:

| Reward | Meaning |
|---:|---|
| `0` | Equal latency to the root |
| Positive | Faster than the root |
| Negative | Slower than the root |
| `ln(2) = 0.693` | `2x` root-relative speedup |
| `ln(10) = 2.303` | `10x` root-relative speedup |

Linear speedup is recovered as:

```text
speedup = exp(reward)
```

The reward is absolute relative to the fixed root, not incremental relative to the
parent. This makes values comparable across branches.

For a workload distribution, the specification uses the weighted sum of per-shape
log speedups. The current CUDA backend evaluates one fixed shape.

Suggested slide title: **Reward is root-normalized log speedup**

## Lazy profiling

Benchmark timing determines reward. NCU does not run inside the reward timing.

Profiling occurs lazily:

1. A candidate compiles, passes correctness, and receives an unprofiled benchmark.
2. If MCTS later tries to expand that node, it profiles the cached compiled artifact.
3. The profile becomes context for strategy priors and the next LLM proposal.
4. At the end, the global best receives one lightweight profile if it lacks one.

An ordinary visit to an existing node does not trigger profiling. Profiles are
cached on the canonical node. `profile_calls` is reported separately from `B_gen`.

NCU runs one isolated launch with zero warmups and one measurement. It does not
replace the 30-sample unprofiled CUDA-event benchmark.

Suggested slide title: **Timing evaluates nodes, profiling informs expansion**

## NCU metric sets

`lightweight_v1` collects nine metrics:

- registers per thread;
- achieved occupancy;
- SM, DRAM, L2, and L1 throughput;
- tensor-pipe activity;
- long-scoreboard stalls;
- executed instruction count.

`diagnostic_v2` adds:

- DRAM bytes read and written;
- L2 sector hit rate;
- local-memory load and store sectors;
- eligible warps and issue activity;
- shared-memory bank conflicts;
- tensor instruction count;
- barrier, short-scoreboard, math-pipe-throttle, and not-selected stalls.

The profile retains both friendly aliases and exact NCU names, values, and units. The
metric-set schema records the target architecture and compatible NCU versions.

CUDA full-profile collection remains unimplemented. The current system uses the two
bounded metric sets above.

Suggested slide title: **Profiles summarize utilization, traffic, and stalls**

## Backup

After a valid leaf receives measured reward `r`, every selected edge in the traversal
path receives the same root-relative value:

```text
StrategyEdge:
    visits    += 1
    value_sum += r
    Q_mean     = value_sum / visits
    Q_max      = max(Q_max, r)

RealizationEdge:
    descents  += 1
    value_sum += r
    Q_mean     = value_sum / descents
```

There is no discount factor. The implementation backs up the measured leaf reward,
not the parent-to-child improvement.

Invalid proposals and infrastructure failures do not update edge statistics. A
depth limit, cycle, or non-root dead end can terminate at an already valid cached
node and back up that node's measured reward.

`Q_mean` drives selection. `Q_max` records whether an edge has ever led to a strong
descendant and supports later analysis or ablations.

Suggested slide title: **Valid measured reward updates both edge levels**

## Slower nodes remain searchable

CUDA optimization is non-monotonic. A slower intermediate kernel may enable a much
faster descendant.

The implementation therefore does not prune a node because:

```text
child latency > parent latency
child reward < current global-best reward
```

PUCT can reduce visits to poor branches naturally, while `max_depth` and `B_gen`
bound the cost. The default maximum path depth is 10.

Suggested slide title: **The search can cross performance valleys**

## Selecting the final kernel

The root begins as the global best with reward zero. After each iteration, a valid
leaf replaces the global best only when its reward is strictly greater.

Consequences:

- the winner may occur at any depth or under any root strategy;
- the winner need not have the most visits;
- a transposed node can be the winner regardless of which path rediscovered it;
- equal rewards retain the earlier best;
- the final best is profiled before worker release if required.

Optional post-search autotuning starts from this best MCTS kernel. Its output is a
separate tuning result and does not retroactively modify the MCTS graph or its edge
statistics.

Suggested slide title: **The fastest valid node wins globally**

## CUDA budgets and accounting

### `B_gen`

The primary MCTS budget is the total number of candidate-generation LLM calls:

```text
B_gen = initial generations + repairs + regenerations
```

Invalid output still consumes the budget. Descending through cached nodes consumes
no `B_gen`. One MCTS iteration is not necessarily one generation call.

### `B_prior`

LLM strategy-prior calls are counted separately. Uniform priors consume zero.

### `B_tune`

`B_tune` counts concrete local-autotuning configurations evaluated after search. The
CUDA tuner operates only on explicitly annotated integer `#define` choices and can
use grid or seeded random order. Tuning trials do not alter MCTS selection or backup.

### Other accounting

CUDA currently uses `B_mut = 0`. Profile calls, drift probes, GPU evaluations, input
and output tokens, and estimated model cost are reported separately rather than
folded into `B_gen`.

Suggested slide title: **Generation and tuning have separate budgets**

## Evaluation caching and transpositions

An MCTS visit is not a GPU evaluation.

Each valid node caches:

- benchmark samples and reward;
- compiled artifact;
- compilation and correctness evidence;
- source and binary hashes;
- lightweight profile when collected.

When generation produces an existing state key, the transposition table reuses the
canonical node and cached evaluation. The graph still records the additional
incoming parent, strategy, and generation relationship.

New GPU work normally occurs only for a new realization, an explicitly missing
profile, or an enabled measurement-drift probe.

Suggested slide title: **Canonical nodes prevent repeated GPU evaluation**

## Stopping conditions and exceptional paths

The normal CUDA stopping condition is exhaustion of `B_gen`. Search may also stop or
terminate an iteration because of:

- maximum path depth;
- a detected cycle in the DAG;
- a node whose strategies have no available generation mechanism or reachable child;
- exhausted proposal space;
- an unrecoverable infrastructure exception.

The seeded random generator makes zero-visit PUCT tie-breaking reproducible.
Generation remains stochastic because the external model can return different
outputs.

Suggested slide title: **Budgets bound the run, depth bounds trajectories**

## Persistence and auditability

The controller writes an append-oriented SQLite trace containing:

- run configuration and environment manifest;
- root-evaluation attempts;
- every prompt, response, candidate, validation result, and budget index;
- PUCT, progressive-widening, and UCB decision snapshots;
- valid and invalid proposal outcomes;
- node creation and transposition reuse;
- backup records and edge statistics;
- profiles, new-global-best events, and final snapshots;
- optional tuning trials and measurement-drift probes.

The trace browser reconstructs the DAG, kernel source, diffs, profiles, strategy
paths, and selection math without rerunning the search.

Suggested slide title: **The SQLite trace reconstructs every decision**

## Details that complete the MCTS description

The requested topics cover the core algorithm. These additional points prevent
common misunderstandings:

1. **No rollouts.** Hardware measurement supplies the leaf value directly.
2. **No re-rooting.** One run performs one global program search.
3. **Two edge levels.** PUCT selects strategies and UCB selects realizations.
4. **DAG rather than tree.** Transpositions share canonical nodes.
5. **Valid-only nodes and backup.** Failed generations remain logs rather than
   penalized pseudo-nodes.
6. **Cached evaluation.** Visits do not imply compilation, benchmarking, or NCU.
7. **Fixed root normalization.** Reward remains comparable across the graph.
8. **Fresh LLM calls.** Prompt history does not leak across branches by default.
9. **Failure separation.** Candidate invalidity and infrastructure failure have
   different retry and logging semantics.
10. **Search evidence limits.** The present experiments do not yet prove that MCTS
    beats all equal-budget baselines, and the strategy catalog is workload-specific.

Suggested slide title: **Implementation choices that matter**

## Compact algorithm trace

```text
Input:
    verified root, strategies, workload, H100 worker, B_gen

Repeat while generation budget remains:
    node = root
    path = []

    while path depth < max_depth:
        lazily profile node if its actions are being initialized
        initialize strategy priors once
        action = seeded argmax PUCT(node, strategy)
        allowed = progressive_widening(action.visits + 1)

        if valid children < allowed and generation budget remains:
            generate one complete CUDA candidate
            compile and check correctness
            repair if invalid and budget remains

            if no valid candidate:
                end iteration without backup

            benchmark valid candidate
            create or reuse canonical node
            append the selected strategy and realization to path
            back up the candidate's measured root-relative reward
            update the global best
            end iteration

        realization = argmax UCB(existing realizations)
        append strategy and realization to path
        node = canonical child

    if depth or cycle ends traversal at a valid cached node:
        back up that node's measured reward

Profile the final best if needed.
Optionally autotune the final best under B_tune.
Return the best valid kernel found anywhere in the graph.
```

## Primary implementation references

- State and edges: `src/kernel_mcts/search/model.py`
- MCTS control flow and formulas: `src/kernel_mcts/search/mcts.py`
- Proposal, repair, and budget loop: `src/kernel_mcts/proposals.py`
- Prompt construction: `src/kernel_mcts/llm_generation.py`
- OpenAI API instructions: `src/kernel_mcts/openai_client.py`
- Compile, correctness, benchmark, and reward: `src/kernel_mcts/evaluation.py`
- CUDA backend and NCU collection: `src/kernel_mcts/cuda_backend.py`
- Workload contract: `src/kernel_mcts/benchmarks/bf16_gemm.py`
- CUDA strategy prompts: `configs/strategies.yaml`
- Post-search tuning: `src/kernel_mcts/autotuning.py`
- Run orchestration and worker lifecycle: `src/kernel_mcts/orchestration.py`
- Authoritative behavior and architecture: `spec.md`
