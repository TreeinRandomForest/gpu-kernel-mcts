# PTX dataflow prompt-view pilot, Sol seed 0

## Plan

Run five arms at `B_gen=50`: compiler summary only, then the same compiler
summary plus `counts`, `paths`, `subgraph`, or `full` PTX dataflow. Hold the
CUDA BF16 GEMM root, `gpt-5.6-sol` at medium reasoning, seed 0, strategy
catalog, `c_puct=12`, `K_max=4`, maximum depth 10, one repair, and the pinned
H100 worker image fixed. Use `--omit-parent-profile` in every arm, while still
collecting `lightweight_v1` NCU profiles on expansion. Do not use incoming
profile deltas. Each arm gets its own SQLite trace and a newly provisioned
H100 SXM worker under the existing exception-safe provider lifecycle.

The control must be rerun with this code; the historical compiler-only seed 0
trace is descriptive context, not a matched control. Inspect every trace for
SQLite integrity, 50 sequential `B_gen` records, run config, prompt hashes,
the correct view in prompt and generation metadata, extraction status, and
prompt-size-limit incidence. Report valid proposals, best node median latency,
input/output tokens, profile calls, and elapsed wall time. One seed is a pilot,
not a reliable ranking of prompt views.

The strategy catalog hash is
`cc7820e2558eb13b9f9dfa344ee2c2f09b101b56a4648e473c3e0273593dafdc`.
The worker image is
`docker.io/saarora/gpu-kernel-mcts@sha256:f90da644bcabc3a07756ecea933f4e1bc2d7faa5785e1c5d39c851eecedb0429`.

## Read-only evidence check before launch

Three kernels in `prompt-ablation-sol-diagnostic-combined-bgen50-seed0.sqlite`
were inspected on the same recorded H100 worker. Their stored PTX artifacts
were hash-verified and selected by named entry. All compiled graphs explicitly
report `partial` because memory alias paths and control-flow joins are not
proven. The PTX families agree qualitatively with source and SASS, but static
site counts differ across lowering.

| Kernel | Source/PTX evidence | SASS evidence | NCU aggregate observation |
|---|---|---|---|
| Root `6d93418b` | Scalar GEMM with shared staging; PTX has 2 global loads, 2 shared stores, 32 shared loads, no tensor sites | 2 global loads, 2 shared stores, 32 shared loads, no tensor sites | 1.63 GB DRAM read; 1.76 billion bank conflicts; 28,241 µs median |
| Async WMMA `5773eba1` | Source uses `cp.async` and WMMA; PTX has 24 async-copy and 22 tensor sites | 26 global-load and 16 tensor sites | 0.804 GB DRAM read; 7.93 million bank conflicts; 1,158 µs median |
| Async WMMA `3fc840d6` | Source uses `cp.async` and WMMA; PTX has 12 async-copy and 13 tensor sites | 14 global-load and 8 tensor sites | 0.358 GB DRAM read; 87.6 million bank conflicts; 1,140 µs median |

The NCU quantities are whole-kernel measurements, not quantities on individual
PTX edges. In particular, the graph never connects a `cp.async` shared write
to a particular WMMA shared load. The root has a nonzero NCU tensor-pipe
instruction counter despite zero PTX/SASS tensor sites and zero reported
tensor-pipe utilization; this counter is not used to infer tensor paths.
An additional fast kernel (`07deb7c4`, 938 µs) is explicitly `unsupported`
by the PTX block limit, so prompts for such nodes must expose that status
instead of silently inventing a graph. These observations support a pilot
using *static dependency evidence with explicit gaps*, not traffic or
critical-path claims. Runtime path validation remains open.

In a read-only preflight across three existing Sol traces (115 nodes), the
`full` view serialized 107 partial graphs and reported four
`prompt_size_limit` cases and four `unsupported` graphs. Size-limit and
unsupported statuses must be reported as exposure rates in the pilot, not
counted as complete graph prompts.

## Results

The matched compiler-only control completed on 2026-10-15. Its SQLite trace
passed integrity, sequential `B_gen=1..50`, prompt-hash, and compiler-context
checks. The run produced 40 valid candidates, 16 lightweight expansion
profiles, and a best median latency of 865.1 µs. It used 108,283 input tokens
and 192,661 output tokens in 53.7 minutes. One generation was recorded as an
evaluation infrastructure failure after successful compilation; it consumed
one `B_gen` call.

The `counts` arm also completed and passed the same checks: 50 calls, 41 valid
candidates, 17 profiles, and a best median latency of 844.1 µs. All 50 prompts
contained the `counts` view with `partial` status. It used 129,041 input tokens
and 177,048 output tokens in 50.2 minutes. Its instance and disk were deleted
normally. Nebius CLI authentication had failed during control-worker teardown;
that instance and disk were subsequently deleted after browser
reauthentication.

The `paths` arm completed with 50 calls, 42 valid candidates, 17 profiles,
and a best median latency of 831.4 µs. All 50 prompts carried the `paths`
view with `partial` status and passed prompt-hash and metadata checks. It used
169,640 input tokens and 194,779 output tokens in 55.8 minutes. The root's
selected local paths were mostly isolated sites, while some generated parents
had five- or six-site paths. These are local PTX dependencies, not measured
traffic paths.

The `subgraph` arm completed with 50 calls, 40 valid candidates, 14 profiles,
and a best median latency of 780.9 µs. All 50 prompts carried the partial
bounded adjacency view. It used 220,395 input tokens and 192,256 output tokens
in 57.8 minutes. The winning node arose from two successive
`tensor_core_multi_accumulator` realizations at calls 34 and 38. The root's
sampled subgraph contained 10 sites and three edges while explicitly omitting
152 sites and 119 edges; the view's selection policy must be considered when
interpreting the run.

The `full` arm completed with 50 calls, 40 valid candidates, 17 profiles, and
a best median latency of 1,087.1 µs. All 50 prompts carried a partial graph;
none hit the explicit prompt-size limit or the extractor's unsupported status.
It used 880,052 input tokens and 183,488 output tokens in 54.9 minutes. Its
winning node came from `tensor_core_multi_accumulator` at call 39 followed by
`vectorized_memory_access` at call 48. The worker instance and disk were
deleted after the run.

| Prompt view | Valid / 50 | Best median µs | Change vs. control | Input tokens | Output tokens | Profiles | Minutes | Graph exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Compiler only | 40 | 865.1 | — | 108,283 | 192,661 | 16 | 53.7 | No graph |
| Counts | 41 | 844.1 | −2.4% | 129,041 | 177,048 | 17 | 50.2 | 50 partial |
| Paths | 42 | 831.4 | −3.9% | 169,640 | 194,779 | 17 | 55.8 | 50 partial |
| Subgraph | 40 | **780.9** | **−9.7%** | 220,395 | 192,256 | 14 | 57.8 | 50 partial |
| Full adjacency | 40 | 1,087.1 | +25.7% | 880,052 | 183,488 | 17 | 54.9 | 50 partial |

All five SQLite files passed integrity checks, sequential `B_gen=1..50`,
stored-prompt SHA-256 checks, and exact prompt/metadata view agreement. The
control exposed no graph; every graph-arm prompt exposed its requested view.
No prompt contained a parent NCU profile. The stored search configurations
differ only in `dataflow_prompt_view`; the root source hash, git commit, model,
seed, strategy catalog, and pinned worker image match. Each run used a fresh
worker, so small hardware or runtime drift is still possible. The compiler
control's one evaluation infrastructure failure consumed a generation call.
The provider project listed no remaining instances or disks after all arms.

## Interpretation

This seed's bounded subgraph found the fastest candidate, about 9.7% below
the matched compiler-only control. The full adjacency arm ended about 25.7%
above that control and used 8.13 times as many input tokens. It used 4.0 times
the subgraph arm's input tokens. Its average stored prompt was 69,265
characters, versus 15,929 for subgraph and 7,296 for control. These are
observed search outcomes and context costs, not evidence that one view causes
better or worse kernel generation. The five searches sampled different trees
despite the shared seed and settings. One seed cannot rank the views or
separate prompt effects from stochastic trajectory effects.

The fastest subgraph path used two successive
`tensor_core_multi_accumulator` realizations at calls 34 and 38. Inspection of
the generated source shows a shift from a scalar-style root to WMMA with
larger output tiling and multiple accumulator fragments. The trace establishes
the edits and measured latency; it does not establish which graph facts, if
any, caused the model to choose or implement them. The full arm's best was
found late, at call 48, so a smaller budget would have understated its best
result. The best-so-far curves and winner source diffs are more informative
than only the final minima.

The `partial` label matters for every graph arm. PTX gives static local
dependencies, not measured bytes, runtime path frequency, or a proven
critical path. The bounded subgraph also omits many sites and edges by design.
The pilot therefore tests *prompting with current partial evidence*, not the
value of a complete, validated dataflow graph. No candidate used a parent NCU
profile in its generation prompt, although lightweight profiles were collected
on expansion under the normal search pipeline.

## Next experiments and search design

1. Repeat the compiler, subgraph, and full arms across additional matched
   seeds before assigning a view ranking. Include counts and paths if budget
   permits. Compare best-so-far curves, validity, strategy/ancestor paths,
   input-token cost, and exposure status rather than only final best latency.
2. Inspect representative successful and failed edits against their exact
   prompts. Test whether a specific graph fact led to a useful code change;
   do not infer graph use from the view label alone. Improve the bounded
   subgraph selector if it repeatedly hides relevant transfer paths.
3. Reconcile static PTX/SASS edges with measured whole-kernel NCU traffic
   where the measurement supports it. Carry uncertainty through source and
   AST annotations. NCU source/SASS correlation can identify instruction
   regions, but sampled stalls should not be represented as exact time per
   AST node.
4. Prototype a hierarchical source/AST plus dataflow view that can suggest
   local edits while retaining cross-region paths. Test that view against the
   current compiler and subgraph controls with the same search budget.
5. Consider a separately budgeted post-MCTS refinement loop on the best
   verified kernel: inspect source/PTX/SASS and profiles, propose one local
   edit, then compile, check correctness, and measure. Compare its gain per
   generation call with deeper MCTS rather than assuming linear refinement
   is preferable.

The current fastest pilot kernel remains well above the approximately
178 µs cuBLAS baseline previously measured for this BF16 GEMM in
`docs/experiments/cutedsl-bf16-same-worker-baselines.md`. Prompt-view work may
help the CUDA search, but backend and search-space choices remain material.

A [trace-level analysis](ptx-dataflow-view-trace-analysis-seed0.md) compares
winner edits, prompt evidence, strategy allocation, and best-so-far curves.
