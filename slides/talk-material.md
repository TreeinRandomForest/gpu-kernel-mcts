# GPU kernel optimization with MCTS — talk material

Working outline for a 50-minute talk. Each section is one prospective slide. Times are speaking targets and sum to 50 minutes. The slide text is deliberately short; the notes carry the explanation. Figures and exact comparisons can be inserted as the deck takes shape.

## Narrative

The question is whether a global search over semantic GPU optimization strategies uses a fixed kernel-generation budget more effectively than simpler search. CUDA C++ generation demonstrates the search machinery and the limits of unrestricted source edits. A typed CuTe DSL design space makes structural choices explicit and enables deterministic, validated mutations. The same search architecture could target another accelerator if that target supplies a kernel representation and trustworthy evaluation path; this portability is a research direction, not a measured result here. The current evidence establishes working search and promising kernel improvements; it does not yet establish that MCTS beats all equal-budget baselines.

## 1. Title — 1 minute

**On slide:** GPU kernel optimization with Monte Carlo Tree Search

**Say:** Introduce the project as a search experiment over *programs*, with measured GPU latency as the reward. Preview the two kernel representations: generated CUDA C++ and typed CuTe DSL.

## 2. Research question — 2 minutes

**On slide**

- Given a correct root kernel and a fixed workload, how should we spend a limited generation budget?
- Main comparison: best valid speedup versus candidate-generation calls, `B_gen`.
- Compare MCTS with independent generation, greedy search, best-of-K, and beam search where data are available.

**Say:** The goal is search efficiency, not a claim that tree search is automatically better. Keep vendor kernels as external performance references. The root defines reward zero.

**Visual:** One simple plot schema with `B_gen` on the horizontal axis and best root-relative speedup on the vertical axis. Fill with actual comparable runs later.

**Source:** [spec.md, Sections 1, 3, 18, 26–27](../spec.md).

## 3. Why kernel optimization is hard — 2 minutes

**On slide**

- Correctness is a hard constraint, including layout and numerical tolerance.
- Performance depends on coupled choices: tile ownership, data movement, pipeline, launch, and epilogue.
- A locally slower valid kernel may enable a faster descendant.

**Say:** A source edit can compile but fail correctness. A correct edit can move performance in the wrong direction. Search has to retain valid intermediate states and measure the actual hardware outcome.

**Visual:** A three-state optimization path with a temporary latency regression before an improvement. Label it as an illustration unless drawn from a real trace.

**Source:** [spec.md, Sections 17, 23–24](../spec.md).

## 4. Fixed workload and reward — 3 minutes

**On slide**

- Initial target: `4096 × 4096 × 4096` BF16 GEMM, FP32 accumulation, BF16 output.
- A row-major, B column-major, C row-major.
- Root-normalized reward: `log(T_root / T_candidate)`.
- Correctness precedes timing; median CUDA-event latency supplies the reward.

**Say:** The workload contract is immutable during a search. The benchmark requires `rtol = atol = 0.02`, 10 warmups, and 30 measured launches. A candidate must pass the full contract before it can become a node. Preserve raw timing samples because small differences can be noise.

**Visual:** Compact workload contract next to the evaluation sequence.

**Source:** [spec.md, Sections 5, 17–20, 29.1](../spec.md); [implementation notes](../docs/implementation-notes.md).

## 5. H100 as the target system — 2 minutes

**On slide**

- One H100 SXM worker per run, reused across candidate evaluations.
- CUDA C++ path: conventional CUDA and WMMA candidates.
- CuTe path: Hopper WGMMA, TMA, shared-memory pipelines, and warp-group ownership.

**Say:** Explain only the hardware features needed to interpret later results. The controller owns search and LLM calls; the worker owns compile, correctness, timing, and profiling. Record exact hardware and toolchain provenance, since cloud measurements drift.

**Visual:** Controller-to-worker architecture diagram, then a small Hopper memory/compute path.

**Source:** [spec.md, Sections 6 and 45](../spec.md); [implementation notes](../docs/implementation-notes.md).

## 6. Standard MCTS vocabulary — 3 minutes

**On slide**

- Selection chooses where to search.
- Expansion adds a new state.
- Evaluation measures that state.
- Backup updates statistics along the selected path.

**Say:** State the adaptation to a single-player, stochastic program-search problem. A run maintains one global DAG; it does not commit to a root action and restart after each move. The output is the best valid kernel discovered anywhere.

**Visual:** Tiny tree or DAG with one highlighted selected path.

**Source:** [spec.md, Sections 7–10 and 16](../spec.md).

## 7. Semantic actions and stochastic realizations — 4 minutes

**On slide**

- PUCT chooses a semantic strategy using `Q_mean`, visits, and a strategy prior.
- One strategy may generate several distinct kernels.
- Progressive widening caps distinct realizations: `min(K_max, ceil(c_pw N(s,a)^alpha_pw))`.
- When widening is saturated, UCB chooses an existing realization.

**Say:** A strategy such as output tiling is not a deterministic code edit. This creates two selection levels: strategy, then concrete implementation. Show where uniform and LLM-provided strategy priors fit. `Q_max` is logged for analysis but does not replace `Q_mean` in default selection.

**Visual:** A state with two strategy branches, each with multiple concrete children. Annotate PUCT, widening, and UCB.

**Source:** [spec.md, Sections 9–12 and 23](../spec.md).

## 8. One search iteration — 4 minutes

**On slide**

1. Traverse the current DAG and select a strategy.
2. Widen or descend to a cached realization.
3. Generate or mutate a candidate when widening permits it.
4. Validate, benchmark, add or reuse a node, then back up measured reward.

**Say:** Use one actual trace if possible. Show a failed generation staying in the proposal log, a repaired generation consuming another `B_gen`, or a transposition reusing an existing node. Ordinary visits reuse cached GPU evaluation. A valid slower candidate remains eligible for expansion.

**Visual:** Annotated trace screenshot or a four-step animation from the SQLite trace browser.

**Source:** [spec.md, Sections 11–16, 20–23, 27](../spec.md); [trace browser design](../docs/trace-browser-design.md).

## 9. Identity, budgets, and evidence — 3 minutes

**On slide**

- Only compile-successful, launch-successful, correctness-passing candidates are nodes.
- Canonical identity and binary fingerprints let distinct paths transpose to one state.
- `B_gen`: all LLM candidate and repair calls. `B_mut`: typed mutation proposals. `B_tune`: separate tuning trials.
- Visits, proposals, and GPU evaluations are different counts.

**Say:** These distinctions are necessary to interpret any search-efficiency plot. Show one concrete DAG transposition if the trace is available. Explain why equal `B_gen` is necessary but mixed mutation runs also require their `B_mut` disclosure.

**Visual:** Four counters beneath one example trace; a merged node for transposition.

**Source:** [spec.md, Sections 7, 21–23, 27, 47.3–47.7](../spec.md).

## 10. CUDA strategy catalog — 1 minute

**On slide**

- General examples: coalescing, synchronization, register pressure, shape specialization.
- GEMM examples: tensor-core output tiling, pipelined data movement, multiple accumulators.
- Strategy applicability depends on operation, dtype, backend, and hardware.

**Say:** The present CUDA prompts were developed around the fixed BF16 GEMM. Avoid suggesting that they transfer unchanged to reductions or normalization. Explain how a semantic ID is resolved to a backend-specific prompt.

**Visual:** Two or three real strategy prompts as short excerpts, with their intended kernel change.

**Source:** [strategy guide](../docs/strategy-guide.md); [spec.md, Section 8](../spec.md).

## 11. Context given to the generator — 3 minutes

**On slide**

- Parent source, workload, H100 target, selected strategy, and backend constraints.
- Cached NCU summary when the parent is selected for expansion.
- A fresh LLM request for each generation, with bounded repair feedback if needed.

**Say:** NCU profiles are collected lazily, after a valid node is measured and when it becomes a parent. Mention representative metrics: tensor-pipe utilization, occupancy, registers, memory throughput, stalls, and instruction count. A profile is evidence to guide a proposal, not a correctness proof or a replacement reward. Discuss PTX or MLIR here only as compiler evidence that can explain whether two designs actually differ; put broader compiler-guided search in next steps.

**Visual:** Prompt anatomy paired with one short NCU summary. Avoid a wall of prompt text.

**Source:** [spec.md, Sections 13 and 20](../spec.md); [implementation notes](../docs/implementation-notes.md); [research roadmap](../docs/research-roadmap.md).

## 12. CUDA search result and diagnosis — 4 minutes

**On slide**

- Fastest recorded CUDA MCTS run: `B_gen = 50`, best valid median `734.448 µs` versus `28,249.408 µs` root, about `38.46×` root-relative speedup.
- A separate 18-trial CUDA autotune reached `726.384 µs` under `B_tune = 18`; tuning did not change the MCTS result.
- The generated and tuned CUDA implementations still remained roughly `4×` slower than the separately measured cuBLAS reference near `185 µs`.

**Say:** This `diagnostic_v2`, incoming-profile-delta-off run found its best node at generation 34 with the `tensor_core_output_tiling` strategy. The exported kernel uses WMMA and asynchronous copies. The standalone tuner remeasured an annotated version of that kernel, so its `739.088 µs` input measurement and `726.384 µs` tuned measurement are a within-tuning-run comparison; do not subtract `726.384` directly from the MCTS run's `734.448` as a causal tuning gain. The earlier `B_gen=50` run at `2,669.9 µs` remains a useful search-path example but is not the fastest recorded CUDA search.

**Visual:** Latency ladder: root, fastest MCTS node, separately tuned variant, and cuBLAS reference. Label both budgets and separate measurement runs. Add a short source or trace excerpt if space permits.

**Source:** [fastest CUDA MCTS trace](../nebius-sol-bgen50-cpuct12-variantdiagnostic_v2-deltas-off.sqlite); [CUDA tuning trace](../nebius-sol-tuning-wmma-grid-btune18.sqlite); [README result summary](../README.md).

**Exact slide data:** [CUDA performance numbers](cuda-performance-numbers.md).

## 13. Search-method comparison — 3 minutes

**On slide:** Best valid speedup versus `B_gen`, with validity and API cost alongside it.

**Say:** Populate this slide only from runs using the same root, workload, worker class, generator, and generation budget. If those comparisons are not ready, state the current evidence limit explicitly: the CUDA run demonstrates a working MCTS search and a large root-relative improvement, while the broader superiority question remains open. Do not place differently budgeted or differently provisioned runs on one causal comparison chart without labels.

**Visual:** Reserved plot with curves for independent, greedy, best-of-K, MCTS uniform, and MCTS LLM-prior where measured. Add uncertainty or repeated runs if available.

**Source:** [spec.md, Sections 3 and 26–27](../spec.md).

## 14. Why typed CuTe DSL — 2 minutes

**On slide**

- Hopper performance depends on coordinated WGMMA, TMA, layout, ownership, and pipeline choices.
- A typed state makes these choices explicit and can reject proven illegal combinations before JIT.
- Deterministic rendering makes source, identity, and mutation evidence auditable.

**Say:** Be precise about the current search spaces. One CuTe branch configures NVIDIA's pinned expert template. A second, independently lowered root gives the project more direct structural control. They are separate experimental starting points, not interchangeable performance baselines. A strategy names the intended kind of optimization; the typed representation records the complete resulting implementation. Changing a strategy label alone changes no kernel.

**Visual:** A typed configuration alongside the CuTe source sections it controls.

**Source:** [spec.md, Section 47](../spec.md); [CuTe search model](../docs/cutedsl-search-model.md).

## 15. How a CuTe strategy changes a kernel — 3 minutes

**On slide**

- **Bounded template choice:** `change_pipeline_stages` proposes a supported depth; the renderer updates the pinned implementation's mainloop configuration.
- **Coordinated structural change:** `change_cta_tile` can change the independent kernel from one to two consumer warp groups, rebuilding ownership, storage, barriers, epilogue, and launch together.
- Each proposal becomes a canonical typed state, passes static checks, renders to complete CuTe source, then undergoes JIT, correctness, and timing.

**Say:** Spend about a minute on each example. Pipeline depth is a useful parameter example, but it is not an unconstrained integer knob: only validated values and combinations enter the current search space, and an explicit setting that duplicates the pinned heuristic should not create a second state. The cooperative transition is a code-structure change expressed as one atomic typed mutation, not a single-field edit. PUCT selects the semantic strategy; the proposal mechanism chooses concrete values and produces the complete new state. Progressive widening and realization UCB then work as before. The initial mixed router tries an untried supported mutation first and falls back to an LLM typed proposal; those calls consume `B_mut` and `B_gen` respectively. This routing does not change the PUCT score.

**Visual:** A side-by-side comparison: `pipeline_stages: pinned heuristic → 2` with the affected mainloop shown, and `(64,256,64), 1 consumer group → (128,256,64), 2 groups` with its coupled fields listed. Label the first as tuning an exposed expert-template choice and the second as a structural change in the independent lowering.

**Source:** [spec.md, Sections 47.3–47.7](../spec.md); [CuTe search model](../docs/cutedsl-search-model.md).

## 16. CuTe results and limits — 4 minutes

**On slide**

- Pinned expert-template tuning: best schedule near `184.9 µs` on H100, close to same-worker cuBLAS near `177 µs`.
- Expert-template mutation search, `B_mut = 35`, `B_gen = 0`: 24 unique nodes, 12 transpositions; best `186.816 µs` versus `193.136 µs` root in that run.
- Independent typed root and cooperative mutation smoke: `668.832 µs` to `448.608 µs`, reward `0.399383`, with `B_mut = 1` and `B_gen = 0`.

**Say:** Keep each comparison within its own measurement run. The expert-template search recovers a schedule already found by small grid tuning; it is not evidence that MCTS outperforms exhaustive search on that finite space. The independent result shows a valid structural transition and an auditable search step, while still leaving a gap to the expert template and cuBLAS.

**Visual:** Two clearly separated panels or slides if needed: expert-template space and independent typed space. Show latency in microseconds and budgets on each.

**Source:** [schedule tuning](../docs/experiments/cutedsl-schedule-tuning-v1.md); [B_mut=35 search](../docs/experiments/cutedsl-mutation-bmut35-v14.md); [cooperative v52 and v53 smoke](../docs/experiments/cutedsl-cooperative-v52.md).

## 17. What failed and what the traces taught us — 1 minute

**On slide**

- Free-form CUDA generation spends calls on compile and correctness repair.
- Some apparent CuTe configuration changes compiled to the same effective artifact.
- Partial structural edits can launch yet fail correctness.
- Small timing differences need repeatable, same-worker evidence.

**Say:** Use one concrete example: the proposed WGMMA in-flight setting changed source/configuration but produced identical IR and fatbin, so it was excluded from canonical state. A separate multicast-only TMA override caused an illegal memory access, showing that ownership and partitioning must change together. These are useful findings about the design space, not just unsuccessful trials.

**Source:** [CUDA B_gen=50 search](../docs/experiments/nebius-bgen50-cpuct12-profiled-search.md); [WGMMA in-flight diagnostic](../docs/experiments/cutedsl-wgmma-inflight-v11.md); [TMA load-policy diagnostic](../docs/experiments/cutedsl-tma-load-policy-v12.md).

## 18. Transfer to other accelerators — 3 minutes

**On slide**

- Reuse the global search: semantic strategies, PUCT, widening, realization UCB, budgets, and valid-only backup.
- Supply a target adapter: kernel language or typed representation, compiler/runtime, correctness harness, timing, and artifact identity.
- Translate strategies to the target's real controls: tile mapping, on-chip memory, data movement, synchronization, and compute engines.
- Recalibrate priors and performance context on the new hardware.

**Say:** Present this as an architectural hypothesis, not a portability result. The workload's mathematical contract can be shared, but legal layouts, numerical behavior, timing method, and profiler signals need target-specific validation. A CUDA prompt or CuTe renderer does not transfer directly to a custom accelerator. For a concrete illustration, AWS NKI exposes tile-level and instruction-level interfaces for Trainium, while Tenstorrent Metalium separates reader, compute, and writer kernels connected by circular buffers. Their exposed decisions differ from Hopper's WGMMA/TMA choices, but the search can still select semantic transformations and measure valid realizations. If a target lacks useful counters, start with correctness and timing; add profiler-informed prompts only after the metrics are reliable.

**Visual:** Shared search core feeding two adapters: current H100 backend and a hypothetical accelerator backend. Under each adapter, show representation, compiler/runtime, correctness/timing, and profile/artifact identity. Mark the second path "proposed, unvalidated."

**Source:** [spec.md, Sections 29, 43, 45, 47](../spec.md); [AWS Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/index.html); [TT-Metalium getting started](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html).

## 19. Next experiments — 1 minute

**On slide**

- Complete equal-budget comparisons against simpler search methods.
- Broaden the independently controlled CuTe transformations after standalone validation.
- Test analytical and compiler-derived context as optional prompt features or strategy priors.

**Say:** PTX, MLIR, SASS, and profiler counters can help explain generated code and identify duplicated effective states. Any analytical performance estimate should guide search, not hard-prune a valid kernel. Longer-term work includes learned policy/value models and fusion, but neither is part of the current result claim.

**Source:** [spec.md, Sections 43–47](../spec.md); [research roadmap](../docs/research-roadmap.md).

## 20. Closing — 1 minute

**On slide**

- MCTS gives an inspectable search over semantic strategies and concrete kernel realizations.
- Typed CuTe states make coupled Hopper decisions explicit and searchable.
- The decisive open test is performance per fixed generation budget against strong, fair baselines.
- Portability to another accelerator remains an experiment to run, beginning with a target-specific evaluator and typed design space.

**Say:** Return to the research question. Invite discussion about which representation and feedback signals can make the search space both expressive and efficient.

## Figure and data checklist

- Export one readable trace DAG with a winning CUDA path and one CuTe transposition.
- Build the CUDA latency ladder from the fastest `B_gen=50` run and separate `B_tune=18` trace, with cuBLAS marked as an external reference.
- Plot equal-budget search-method curves only after verifying matching workload and generator conditions.
- Show two CuTe result groups separately: pinned expert-template search and independent typed lowering.
- Use one compact typed-state diff for the atomic cooperative mutation.
- Check every displayed latency against its original trace or experiment report before presenting.

## Likely cuts if the 50 minutes include questions

Shorten Slides 5, 9, 10, and 17 first. Keep the research question, one full iteration, the comparison evidence, the distinction between the two CuTe spaces, and the portability claim's current evidence limit.
