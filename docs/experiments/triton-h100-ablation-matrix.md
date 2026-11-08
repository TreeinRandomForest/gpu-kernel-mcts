# Triton H100 search and context ablation matrix

Status: research plan, not a declaration that every arm is implemented. The
fixed starting workload is the packaged Triton BF16 GEMM on one H100 under
the `triton_h100` launch, correctness, and `triton_h100_cuda_graph_v1`
CUDA Graph replay timing contract in `spec.md` Section 50. Earlier runs used
per-call events around the Python launcher and are a separate historical
timing cohort. “Triton + Gluon” means the same root and backend allow
source transitions into Gluon; it is not a second root or a second GPU backend.

## Target axes and current support

| Axis | Planned arms | Current boundary |
| --- | --- | --- |
| Allowed source language | Pure Triton; Triton with Gluon transitions | Both have versioned MCTS strategy catalogs. A packaged-root run reached a correct Gluon wait-one kernel through staged actions. Pure-Triton sequential freeform rejects non-Triton source stages before GPU evaluation. |
| Search algorithm | Global MCTS; sequential refinement | Pure-Triton sequential support passed local tests and two seed-0 H100 smoke runs; see `triton-h100-sequential-pure/README.md`. Triton/Gluon sequential transitions remain a later arm. |
| Action policy | Explicit strategy catalog; freeform | Triton MCTS now supports an opt-in pure-Triton freeform-only arm with reserved `freeform_triton`, ordinary PUCT/progressive widening/UCB, and pre-evaluation stage rejection. H100 runs at `B_gen=6` and 25 reached 219.824 and 201.488 µs respectively; see `triton-h100-freeform-mcts/README.md`. Pure-Triton sequential has guarded menu and stage-restricted freeform modes. A catalog-plus-freeform hybrid is a later, distinct arm. |
| H100 context | None; bounded architecture graph; calibrated links; evidence-backed parent resource/traffic overlay | The graph exists as a diagnostic artifact. Triton MCTS does not yet insert it into prompts. Calibrated and live-overlay levels require explicit evidence and versioning; unknown traffic or link performance must remain unknown. |
| Compiler context | None; resources; bounded PTX instruction classes; richer versioned PTX/SASS or line views | Triton MCTS supports an optional bounded resource/PTX-site summary. PTX artifacts may be stored even when hidden from the prompt. Static sites are not executed instruction counts. Richer SASS/Source views are later arms. |
| NCU context | Hidden; lightweight; diagnostic; spill/reuse diagnostics | The backend exposes `lightweight_v1`, `diagnostic_v2`, `spill_v1`, and `reuse_v1` metric sets. Collection policy and prompt visibility are separate choices; hiding a profile need not prevent collection for trace analysis. Values are whole-kernel measurements, not per-tensor or per-edge traffic. |
| Dataflow context | None; counts; paths; bounded subgraph; full adjacency | CUDA has these prompt views. Triton/Gluon dataflow views are currently rejected by the CLI. They require source/PTX/SASS reconciliation and per-node provenance before prompt use. |

This is a **target factor space**, not a request to run its full Cartesian
product. Some combinations are invalid or depend on prior work: a Gluon-only
action does not belong in a pure-Triton arm; a traffic overlay requires
measured evidence; a dataflow view requires a validated extractor. Record an
arm as unavailable rather than silently substituting a different prompt.

For the freeform-only MCTS arm, use `--triton-freeform-mcts` with
`--backend triton_h100 --algorithm mcts` and an LLM generator, omitting
`--strategies`. The flag rejects a strategy file or mutation budget. Its
prompt and output are persisted with the trace; the run metadata records the
action policy and prompt version/hash. Local tests, a six-call H100 smoke,
and a 25-call depth run are recorded in the freeform-MCTS report.

A three-seed `B_gen=25` GLM comparison is now recorded in
`triton-h100-freeform-compare/README.md`. Seeds 1–2 pair the arms on the same
worker. This is a near-context comparison: the raw-source MCTS and JSON
sequential prompt protocols, timing payload size, and feedback visibility
still differ and are audited there.

## Dataflow validation gate

For each newly accepted Triton or Gluon node, bind a static graph to its
source hash, compiled PTX/SASS identity, kernel entry point, and run/node ID.
The graph should represent operand movement, shared-memory tiles, barriers,
tensor-core operations, and output stores where compiler evidence supports
them. Reconcile several unlike kernels, including pointer-load Triton,
descriptor Triton, and serial/pipelined Gluon, against source and emitted
code. Check tensor orientation, descriptor coordinates, ring-slot lifetimes,
branch/loop paths, and output ownership. Mark missing or ambiguous edges;
never invent dynamic counts, per-tensor DRAM bytes, or path frequencies from
static code. Whole-kernel NCU traffic can test aggregate plausibility but
cannot by itself prove individual graph edges. Only after this audit should
counts, paths, subgraphs, or adjacency enter generation prompts. Persist the
exact bounded view and its limitations in SQLite.

## Comparison sequence

1. Make the four core modes runnable: pure Triton and Triton + Gluon, each
   under MCTS and sequential search, with explicit catalog and freeform
   policies. Preserve the same packaged root and verification pipeline.
2. Compare action policies with a fixed, modest context. Keep strategy-only,
   freeform-only, and hybrid MCTS separate. In sequential search, record the
   selected menu ID or freeform choice and lock it during repair.
3. Vary one context factor at a time: profile visibility/metric set, compiler
   summary, H100 graph level, then validated dataflow view. A larger factorial
   is justified only after these smaller comparisons identify interactions
   worth testing.
4. Use paired seeds and equal `B_gen`, model and inference settings, worker
   image, workload, root, correctness tolerance, timing method, profile
   collection policy, and search settings. Declare sequential history length,
   repair limit, and acceptance rule; they are part of the algorithm arm, not
   hidden prompt differences. Keep compilation, correctness, and timing as
   the admission gates for every candidate.

Report best verified latency versus generation calls, valid and repair rates,
depth/unique states, action allocation, prompt tokens, NCU calls, and GPU/API
cost. Store exact rendered prompts, context versions, masks, artifacts, and
run configuration. Single short runs are feasibility checks, not evidence of
a winning search policy or context level.
