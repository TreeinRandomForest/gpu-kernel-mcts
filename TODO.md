# Implementation status

`spec.md` is the source of truth for Milestones A and B. This file records implementation
and validation status. Checked implementation items have automated coverage; any
completed hardware validation is identified explicitly.

## Bounded Triton H100 integration track

Section 50 of `spec.md` defines this precursor to the MI355X workload.

- [x] Add a packaged Triton root, fixed BF16 GEMM launcher contract, worker
  backend, provider selection, versioned strategies, deterministic smoke
  candidate, and a pinned Torch/Triton image recipe.
- [x] Add local tests for compile/correctness/timing/profile wiring and backend
  selection without changing MCTS selection or generation accounting.
- [ ] Build and pin the Triton H100 worker image digest; verify Torch, Triton,
  CUDA, NCU, and CUTLASS versions on a real H100 worker.
- [ ] Run a root and deterministic one-candidate smoke; inspect cuBLAS/CUTLASS
  calibration, correctness, candidate timing, and trace records.
- [ ] Run a small LLM search with `configs/triton_h100_strategies.json` and
  confirm a lazily collected `lightweight_v1` NCU profile on a selected node.
- [ ] Reuse the validated Triton plumbing for the separate MI355X Section 49
  workload, then add ROCm-specific JIT, profiling, and shape coverage.

## Experimental Triton/ROCm MI355X track

Section 49 of `spec.md` defines the workload and preserved search behavior.
Complete the first seven items before a source-only LLM search. Add the ROCm
profiler before profile-informed search; compiler, hardware, and dataflow views
are separate context ablations rather than prerequisites for a measured root.

- [ ] Pin the vLLM PR revision and a ROCm/PyTorch/Triton worker image. Record the
  exact fused kernel, launch configuration, dispatch wrapper, stock reference,
  tool versions, image digest, and `gfx950` identity.
- [ ] Add an attach-to-existing-GPU SSH worker option for a personally supplied
  endpoint (`ssh://user@host:port` or equivalent), key, optional jump host, and
  tunnel to the existing worker protocol. Validate host identity and hardware;
  keep credentials and private endpoint details out of traces. Add vendor
  provisioning only when a provider with usable MI355X capacity is selected.
- [ ] Build an MI355X ROCm worker image and startup manifest. Verify container
  GPU access, `supports_mx()`, Triton JIT, `amd-smi`, profiler availability,
  and the requested worker environment before evaluation.
- [ ] Add the PR fused operation as a `triton_rocm` root/backend, including its
  launcher and split-K behavior. Keep the stock separate-quantization operation
  as a comparison reference, not a search node. Define candidate source/config
  boundaries and backend-specific state identity.
- [ ] Implement the versioned block32 workload harness: the fifteen primary
  `(M,N,K)` cases, expanded shapes and formats only after reference validation,
  independent QDQ correctness, bit-exact stock comparison, fallback/edge cases,
  dispatch checks, graph replay, and full-operation HIP timing. Persist every
  case's result, aggregate reward, noise, and root/stock baselines.
- [ ] Wire `triton_rocm` and the workload into the worker, controller CLI,
  generation/repair prompts, SQLite trace, and artifact export without changing
  MCTS tree policy or `B_gen` accounting. Run a root-only GPU smoke before search.
- [ ] Define versioned Triton semantic strategies and applicability/legality
  checks for tile, K packing, split-K, warps, stages, waves, fusion, and memory
  movement. Keep the PR's MI355X tuning table as baseline evidence; test
  strategy prompts and invalid-proposal handling before budgeted MCTS runs.
- [ ] Add profiler selection by backend/hardware: retain NCU for NVIDIA and use
  `rocprofv3`/ROCm Compute Profiler for AMD. Probe counter access on the actual
  VM/container, distinguish tracing from hardware counters, version metric sets,
  and keep profiler replay timing out of the reward.
- [ ] Capture bounded Triton compiler evidence (TTIR/TTGIR, LLVM IR, AMDGPU ISA,
  resource usage and code-object identity). Persist raw artifacts separately from
  prompt summaries; add summary levels as explicit context ablations.
- [ ] Add an MI355X/CDNA4 resource model with documented capacities and
  compiler-reported VGPR/SGPR/LDS/occupancy constraints. Calibrate traffic and
  bandwidth edges only against measured ROCm counters; distinguish estimates
  from measurements in traces and prompts.
- [ ] Extract a bounded dataflow graph from Triton source and lowered IR for each
  valid node, including quantization, scale, load/store, dot, split-K, and
  reduction paths. Validate claims against code and available profiling evidence;
  compare graph summary, paths, subgraphs, and adjacency prompt views later.
- [ ] Validate optimized candidates in pinned vLLM tests and a separately
  controlled serving A/B. Hold the PR's `lm_head` guard fixed when attributing
  an effect to the fused Triton operation.

## Implemented

- [x] MCTS search with PUCT strategy selection, progressive widening, and UCB realization selection.
- [x] Exact generation-budget accounting, including repair and regeneration calls.
- [x] Valid-only backup; invalid and infrastructure failures do not update search values.
- [x] Path-dependent depth handling across transpositions.
- [x] Cached complete evaluation results without GPU reevaluation on ordinary visits.
- [x] `B_prior` accounting and seeded zero-visit PUCT tie-breaking.
- [x] Structured SQLite trace records for runs, iterations, proposals, nodes, edges, and final statistics.
- [x] Complete environment manifests with stable IDs and hardware/toolchain validation.
- [x] RunPod provider lifecycle abstraction with one reusable worker per run and exception-safe release.
- [x] Concrete RunPod Pods REST client with H100 SXM provisioning, readiness polling, termination, and a guarded probe CLI.
- [x] Authenticated, idempotent HTTP worker service and controller transport with a CUDA worker image.
- [x] H100 data-center discovery and safe managed network-volume reuse/creation.
- [x] Fixed 4096 x 4096 x 4096 H100 BF16 GEMM workload and packaged CUDA root kernel.
- [x] Backend-neutral compile, correctness, benchmark, reward, and state-identity pipeline.
- [x] CUDA/NVCC backend and cuBLAS correctness/benchmark harness with GPU-independent tests.
- [x] Fixed cuBLAS and pinned CUTLASS diagnostic performance baselines outside the MCTS state space.
- [x] Provider-to-MCTS orchestration with deterministic mock integration coverage and SQLite traces.
- [x] Guarded one-generation H100 smoke-search CLI with a packaged valid candidate.
- [x] Root rewards normalized to exactly zero using the run's measured root benchmark as the fixed reference.
- [x] Remote BF16 GEMM validation completed on an NVIDIA H100 80GB HBM3 SXM worker.
- [x] Provider-neutral LLM completion interface and OpenAI Responses API kernel-generator adapter with fresh-session prompts and usage metadata.
- [x] Guarded controller-only CLI for one configured OpenAI kernel-generation request.
- [x] Guarded H100 candidate compilation, correctness, benchmark, and JSON-report CLI.
- [x] Configuration-driven OpenAI MCTS search CLI with uniform priors and best-kernel export.
- [x] Idempotent root-evaluation infrastructure retries with complete attempt traces and sanitized failure diagnostics.
- [x] Read-only SQLite trace visualization as a transposition-preserving Graphviz DAG.
- [x] Lazy Nsight Compute metric extraction cached per node/environment and supplied to generation prompts.
- [x] Ephemeral RunPod storage mode without network-volume data-center affinity.
- [x] Nebius H100 SXM VM provider with an SSH-tunneled, NCU-capable worker container and exception-safe VM/disk deletion.
- [x] Exact immutable PUCT, progressive-widening, and UCB decision snapshots with
  browser-based decision-math inspection.
- [x] Trace-browser iteration playback, cumulative-best timeline, and per-strategy
  visit, validity, repair, and reward summaries.
- [x] Cross-run trace-browser comparison of reward progress, strategy allocation,
  validity, configuration, hardware, and best-node profiles.
- [x] Downloadable, versioned trace-analysis bundles containing selected kernels,
  source diff, profiles, strategy relationship, relevant decisions, and run context.
- [x] Per-benchmark GPU operating-state telemetry and explicit periodic root/best
  drift probes that preserve fixed reward normalization.
- [x] Optional post-search random/grid autotuning framework for explicitly annotated
  CUDA templates, with separate `B_tune` accounting and SQLite trial records.
- [x] Standalone Nebius autotuning CLI for an existing annotated CUDA kernel, with
  pre-provision validation and exception-safe worker release.
- [x] Comparable fixed Hopper CuTe DSL BF16 GEMM baseline using the repository input,
  correctness, and CUDA-event timing contract.
- [x] Typed CuTe schedule schema with deterministic identity, static validation, and
  exhaustive standalone `B_tune` evaluation on one worker.
- [x] CuTe schedule-tuning validation on H100 SXM: the best `(128, 256)` CTA tile with
  `(2, 1)` cluster measured `184.880 us`, 4.4% slower than same-worker cuBLAS and
  4.4% faster than the default CuTe schedule.
- [x] Milestone B phase-1 typed `CuteGemmProgram` schema, canonical identity,
  structured static legality results, deterministic pinned-template renderer, and
  backward-compatible SQLite/browser representation metadata.
- [x] Milestone B phase-2 `CuTeDSLBackend` evaluation path with deterministic-source
  verification, bounded pinned-template JIT/evaluation, correctness and timing reuse,
  telemetry, typed evaluation metadata, and in-memory artifact caching. The guarded
  H100 backend-mode validation passed on 2026-09-19; a repeat run remains to validate
  the added initial-JIT and enriched-provenance reporting fields.
- [x] Milestone B phase-3 artifact/kernel-identity support: diagnostic cache/module
  inventory, in-memory JIT callable inspection, bounded normalized MLIR persistence,
  runtime-artifact-first fingerprint selection, and a cached single-launch NCU path
  separate from correctness and reward timing. H100 validation captured the embedded
  fatbinary and all nine lightweight NCU metrics with exact generated-kernel filtering.
- [x] Initial Milestone B typed-mutation neighborhood over the already validated CTA
  tile and cluster-shape dimensions. Mutations are deterministic, statically checked,
  path-independent, and emit complete transformation evidence without consuming
  `B_gen` or `B_tune`. This is standalone design-space validation, not MCTS enablement.
- [x] Core MCTS integration for deterministic CuTe proposals with separately reserved
  and persisted `B_mut`, no mutation repair loop, budget-aware widening eligibility,
  finite proposal-space termination, and a GPU-independent end-to-end CuTe MCTS test.
  Remote CuTe orchestration is wired. The guarded Nebius H100 SXM mutation-search
  smoke run passed on 2026-09-20 with one valid mutation, two profiled nodes, and
  `B_gen=0`, `B_mut=1`. A subsequent `B_mut=4` run exercised both strategies and a
  cached-root transposition and recovered the standalone tuner's best `(128, 256)`,
  `(2, 1)` schedule.
- [x] GPU-free pinned-source structural-capability diagnostic that records the exact
  CUTLASS example hash, callable signatures, CLI controls, validation evidence, and
  bounded pipeline/WGMMA/TMA/epilogue/warp/scheduler source evidence. Discovered
  controls remain `evidence_only` until legality and H100 behavior are validated.
- [x] Use the targeted WGMMA/warp-group assignment diagnostic to select and implement
  one bounded template control. `single_warp_group` forces `atom_layout_mnk=(1,1,1)`
  and a 128-thread CTA while leaving tile, pipeline, layouts, and epilogue fixed.
- [x] Validate `single_warp_group` standalone on H100 for JIT, correctness, distinct
  artifact identity, and timing before adding an MCTS strategy or mutation. The v10
  run passed exact correctness and produced distinct artifacts, but regressed from
  `192.976 us` to `3015.728 us`; keep this pathological choice out of MCTS.
- [x] Validate the SHA-pinned `wgmma_inflight_groups=2` control standalone on H100
  against the default value 1, including its interaction with explicit stage 3.
  All four v11 configurations passed exact correctness, but values 1 and 2 produced
  identical normalized IR and fatbin hashes at each pipeline setting. The apparent
  0.3–0.7% timing differences are measurement noise. Reject this as an independent
  search dimension and do not add an MCTS strategy or mutation.
- [x] Remove `wgmma_inflight_groups` from the canonical `CuteGemmProgram` state after
  preserving the v11 diagnostic evidence. Distinct canonical states must not encode
  the same effective compiled kernel. The SHA-guarded source transformer remains
  available only through a standalone diagnostic mode and cannot affect search
  identity, transpositions, or MCTS nodes.
- [x] Add a bounded, noncanonical TMA load-policy diagnostic for the validated
  `(128,256)` tile with cluster `(2,1)`. It compares the pinned automatic multicast
  load with redundant non-multicast loads, is guarded by the exact pinned source
  hash, and cannot affect CuTe search identity or MCTS budgets.
- [x] Validate `tma_load_policy={auto_multicast,non_multicast}` standalone on H100.
  The pinned multicast configuration passed exact correctness at `187.216 us` median.
  The one-field non-multicast override launched but hit an illegal memory access,
  because the TMA atom no longer matched clustered partitioning, CTA coordinates,
  and multicast masks. Reject it as a bounded field and add no MCTS mutation. A
  future non-multicast experiment would require one coordinated structural rewrite.
- [x] Add a SHA-guarded, noncanonical epilogue-stage diagnostic over depths
  `{2,3,4}` for tile `(128,256)`, cluster `(2,1)`. It preserves the pinned A/B
  mainloop depth, restores the original method after evaluation, and serializes
  diagnostic failures instead of losing the requested configuration and error.
- [x] Validate epilogue stages 2, 3, and pinned 4 on one H100. All passed exact
  correctness and produced distinct normalized IR and fatbins. Medians were
  `190.000`, `189.728`, and `188.448 us`; the sub-1% timing differences do not
  establish a performance winner beyond the pinned stage 4 in this run.
- [x] Promote validated epilogue depth to `CuteGemmProgram` schema v2 and add a
  deterministic `change_epilogue_stages` mutation. Explicit values 2 and 3 are
  initially legal only for tile `(128,256)`, cluster `(2,1)`; `None` preserves pinned
  stage 4, and explicit 4 is excluded as an identical-state alias.
- [x] Add canonical backend CLI schedule arguments and schema-v2 epilogue forwarding,
  plus a GPU-independent MCTS test proving one epilogue realization consumes
  `B_mut`, creates a valid node, and consumes no `B_gen`.
- [x] Add one canonical epilogue-validation command that evaluates stages 2 and 3,
  verifies schema/schedule identity, validity, correctness, in-memory artifact-cache
  reuse, and distinct configuration/runtime fingerprints, and retains both complete
  backend reports.
- [x] Run the canonical backend-mode H100 smoke for epilogue stages 2 and 3 and verify
  the report preserves schema v2, configuration identity, cached second evaluation,
  correctness, and the same distinct runtime artifacts observed by the diagnostic.
  Both passed exact correctness and cache reuse; medians were `189.008 us` and
  `189.168 us`, and their IR/fatbin hashes matched the earlier diagnostic.
- [x] Add an auditable all-or-nothing CuTe root-schedule override to the search CLI so
  a guarded mutation run can begin at the validated `(128,256)`, cluster `(2,1)`
  state where epilogue mutations are immediately legal.
- [x] Run and inspect a small mutation-only search from that root. The trace contains
  the root plus distinct stage-2 and stage-3 nodes, two valid typed mutations, exact
  `B_mut=2`/`B_gen=0` accounting, complete canonical representations, and valid-only
  backups. Both candidates were slightly slower than the pinned stage-4 root, so the
  root correctly remained best at reward zero.
- [x] Complete corrected standalone H100 validation of bounded mainloop
  `pipeline_stages={2,3,4}` and repeat `B_mut=6` after forwarding the typed field into
  the backend subprocess. All stages passed exact correctness and produced distinct
  runtime/IR hashes; medians were `299.136`, `191.792`, and `190.624 us`. The search
  created a distinct stage-2 node; stage 3 was not selected within six attempts.
  `None` preserves the heuristic, explicit stage 4 remains omitted from the reference
  mutation neighborhood, and epilogue staging remains pinned at 4.
- [x] Run the nine-point standalone `B_tune` interaction grid for the `(128,256)`
  tile across three validated cluster shapes and mainloop stages `{heuristic,2,3}`.
  Cluster `(2,1)` with heuristic staging won at `184.896 us`, independently
  reproducing the earlier `184.880 us` result; it was 4.88% slower than same-run
  cuBLAS. Stage 2 was consistently under-pipelined, and stage 3 did not beat the
  heuristic for any cluster.
- [x] Add core mixed-mechanism proposal routing beneath each semantic strategy using
  the specified mutation-first policy, falling back to LLM generation only when the
  typed neighborhood is unavailable or `B_mut` is exhausted. The router preserves
  separate `B_mut`/`B_gen` accounting and logs eligible mechanisms plus the selected
  budget kind. The first guarded mixed H100 run routed correctly but exposed that raw
  LLM-authored Python could not satisfy deterministic-render identity. The controller
  now requests, validates, and canonically renders typed JSON. The corrected H100 run
  passed: the LLM selected the existing best schedule, which canonicalized to a valid
  cached transposition with no GPU reevaluation. Learned or bandit routing is a later
  explicit ablation.

## Next

### CUDA-first compiler and hardware context

- [ ] Pause broad operation-specific CuTe language expansion and preserve the current
  backend and measurements as a structured-search case study. Follow
  `docs/cuda-first-next-direction.md` for the active research handoff.
- [x] Define a versioned, cached compiler-context record for valid CUDA evaluations,
  beginning with PTX path/hash provenance and parsed `ptxas` resource evidence.
  GPU-free parser, evaluation, transport, and SQLite tests pass; ordinary MCTS visits
  do not trigger context extraction or GPU reevaluation.
- [x] Validate v1 PTX naming and entry-point `ptxas` parsing on the pinned H100
  worker. The run exposed that worker-local PTX paths expire on teardown.
- [x] Store size-capped, hash-checked PTX in a separate SQLite artifact table and
  attach bounded SASS instruction categories to the valid CUDA node context.
  GPU-free transport, parser, persistence, and orchestration tests pass.
- [x] Validate durable PTX transfer and entry-point SASS parsing on the pinned
  H100 worker: the root and one valid candidate produced two hash-verified PTX
  blobs, two node links, and parsed SASS summaries in the SQLite trace.
- [x] Compare the root on one H100 worker with and without `--keep`, `--keep-dir`,
  and `-Xptxas=-v`: both passed correctness, normalized SASS fingerprints matched,
  and four interleaved timing medians spanned 1.17 microseconds. See
  `docs/experiments/compiler-context-h100-v2.md`.
- [ ] Add launch/resource constraints to the compiler record. Investigate Clang
  CUDA AST/LLVM/NVVM only after PTX/SASS evidence is reliable; ordinary NVCC
  compilation does not expose a CuTe-style MLIR view.
- [x] Build and persist a versioned, diagnostic-only H100 SXM resource graph for
  HBM, L2, L1, shared memory, registers, SMs, and compute units. Record sourced
  capacities and architectural limits, a published HBM peak, explicit unknown
  link bandwidths/latencies, NCU percentage observations, and per-kernel
  compiler/launch constraint comparisons in the trace browser. See
  `docs/h100-resource-graph.md`.
- [ ] Calibrate sustainable bandwidth and latency ranges for graph links with
  same-worker measurements and NCU byte/time evidence. Keep published peaks
  distinct from observed performance and leave unsupported links unknown.
- [x] Persist bounded, exact-source dataflow graphs for the fixed BF16 GEMM root
  and smoke candidate: memory placement, source-level transfer demand, shared
  reuse, scalar computation, synchronization, output ownership, and possible H100
  paths. Show them in the trace browser without changing search decisions. See
  `docs/bf16-dataflow.md`.
- [x] Give each valid CUDA node a cached diagnostic dataflow graph at creation.
  Preserve the exact-source BF16 models and use static SASS instruction categories
  as a disconnected inventory beside logical A/B/GEMM/C nodes for generated CUDA.
  Keep physical routes, dynamic bytes, and value dependencies unknown; support
  read-only derivation for older traces.
- [x] Show independently selectable A/B dataflow diagrams beside the existing
  source comparison. Label source-level quantities, logical relationships, static
  SASS site counts, and unknowns directly in the diagrams.
- [x] Extract bounded value dependencies from retained PTX for generated CUDA
  nodes, with explicit unsupported/partial status for control flow, aliasing, and
  untracked memory. Keep the compiled graph separate from the logical workload view.
- [ ] Reconcile representative PTX paths with SASS and same-worker NCU observations
  before interpreting paths as measured traffic or bottlenecks.
- [ ] Add a parent/child dataflow diff in the trace browser. Compare aligned
  memory/compute stages and evidence provenance before showing quantitative deltas;
  do not compare unknown traffic as zero.
- [ ] Profile both kernels on the same worker for DRAM/L2/shared byte or sector
  counts, then reconcile the source-level dataflow with measured traffic before
  attributing bottlenecks. Assess only proven launch/resource violations as hard
  failures. Keep cross-kernel fusion behind a separately specified workload contract.
- [x] Add an off-by-default CUDA prompt option for bounded parent `ptxas` resources
  and static SASS site counts. Persist the exact prompt, versioned summary, and
  option flags without changing profiling or MCTS budget semantics.
- [x] Run the equal-budget CUDA-only, NCU, compiler-summary, and combined prompt
  pilot at `B_gen=10`; verify all prompt records, hashes, budgets, and worker
  manifests. See `docs/experiments/compiler-prompt-ablation.md`.
- [x] Run matched-seed comparisons of those four arms at `B_gen=50`, then report
  final best latency, validity, token cost, and cross-seed variability. See
  `docs/experiments/terra-sol-prompt-context-bgen50.md`.
- [x] Add off-by-default prompt plumbing for four deterministic views of a recorded
  PTX graph: counts, local paths, bounded adjacency subgraph, and full adjacency
  (with an explicit prompt-size limit). Persist the exact view and prompt per call.
- [x] Run a one-seed Sol `B_gen=50` pilot of compiler-only, counts, paths,
  subgraph, and full PTX graph prompts. Validate exact prompt exposure,
  budgets, traces, and token cost. See
  `docs/experiments/ptx-dataflow-view-seed0.md`.
- [x] Analyze the five pilot traces for winner edits, graph exposure,
  strategy allocation, NCU diagnostics, and best-so-far curves. See
  `docs/experiments/ptx-dataflow-view-trace-analysis-seed0.md`.
- [ ] Run matched-seed, equal-`B_gen` ablations across the four PTX graph views and
  a compiler-summary-only baseline. Record exact prompts, token use, validity,
  best latency, and variation across seeds; keep NCU context and other flags fixed.
- [ ] Evaluate a bounded dataflow/hardware summary after PTX/SASS/NCU reconciliation.

### Sequential CUDA refinement experiment

- [x] Build a standalone, non-MCTS single-session loop with no required strategy
  choice: prompt from the current verified CUDA source and its measured latency,
  NCU profile, compiler context, and bounded dataflow view; ask the model for an
  optimization hypothesis and replacement source. Charge every proposal/repair
  to a fixed generation-call budget. Keep the current kernel after an invalid
  proposal; advance to every valid proposal, including locally slower ones,
  while tracking the best verified kernel separately. Extract the next context
  only after the normal compile, correctness, and timing gates. Persist exact
  prompts, candidate/source lineage, feedback, context versions, token usage,
  profiling calls, and wall/GPU cost. Keep this experiment separate from MCTS
  semantics and reuse the existing correctness and timing contract. The worker
  occupancy wall time is recorded as a cost proxy; provider billing is not yet
  reconciled. See `docs/experiments/sequential-cuda-refinement.md`.
- [ ] Reconcile the worker occupancy proxy with provider billing before reporting
  dollar cost for sequential runs.
- [x] Smoke-test the loop at a small generation budget, then compare a
  `B_gen=50` run with the existing MCTS traces on the same root, model,
  workload, worker image, and measurement contract. Plot best-so-far latency
  against generation calls; also report accepted-chain depth, validity,
  profiling cost, and tokens. Fix a bounded session-memory policy before
  comparisons so growing history does not silently change prompt exposure.
  Completed Sol seed 0 with `B_gen=3` smoke and `B_gen=50` full run; see
  `docs/experiments/sequential-sol-bgen50-seed0.md` and its plot. The full
  run's best was 655.856 µs at call 44, with 47 valid calls and 41 profiles.
- [x] Inspect each sequential proposal's source/hypothesis against the current
  strategy catalog. The per-call table and source spot checks in
  `docs/experiments/sequential-sol-bgen50-seed0.md` identify split-K tensor-core
  bootstrapping and CTA grid remapping as the clearest catalog gaps.
- [ ] Remeasure the 655.856 µs kernel on a fresh matched worker and replay
  call-34 versus call-44 sources in alternating order to test the CTA mapping
  effect. Repeat sequential seeds before drawing a ranking conclusion.
- [x] Record the matched sequential strategy-choice plan: two-turn versus
  full-history context, H100 graph off/on, and freeform versus catalog choice.
  Fix the compiler, PTX `paths`, NCU, root, worker image, and `B_gen=50`
  settings within each pair. See
  `docs/experiments/sequential-strategy-menu-plan.md`.
- [x] Implement an auditable sequential strategy choice in one generation
  call: parse `selected_strategy_id`, freeze/hash the menu and prompt version,
  persist the declared ID and adherence status for every proposal and repair,
  and leave compile/correctness validity and `B_gen` accounting unchanged.
  Add GPU-free generation, CLI, trace, and repair tests.
- [ ] Smoke-test that instrumentation, then run the first matched
  two-turn/no-H100-graph freeform and catalog arms at `B_gen=50`. Repeat over
  declared seeds; compare best-so-far curves, validity, edits, profiles,
  tokens, and GPU cost. The historical 655.856 µs run is not a fresh control.
  Seed-0 smoke and pair completed; see
  `docs/experiments/sequential-strategy-choice-seed0.md`. Repeat seeds and
  classify source edits before drawing a general ranking.
- [ ] Resolve the freeform response-label deviation before repeated strategy
  comparisons: all 50 seed-0 freeform calls returned self-named IDs rather
  than `freeform`. Preserve those raw labels in the trace; if the prompt or
  parser changes, version the protocol and rerun a fresh matched pair.
- [ ] Extend the same paired comparison to two-turn/graph and
  full-history/graph-off and graph-on cells. Keep `lightweight_v1` collection
  fixed for this factorial study; test diagnostic NCU as a later factor.
- [ ] Validate the GLM-5.3 Anthropic-compatible Messages gateway with one
  API-only response and a small H100 CUDA search before running matched MCTS
  or sequential ablations. The adapter and GPU-free tests are documented in
  `docs/messages-gateway.md`. On 2026-10-26, both the adapter and exact curl
  request returned HTTP 401 with the `.bashrc` key after export; gateway
  authorization or the key must be corrected before the GPU smoke.
- [ ] Independently classify sequential source transitions against declared
  strategy IDs. Version and test catalog additions only for recurring useful
  edits that are absent or prohibited by current strategy wording. Seed-0
  trace classification is in
  `docs/experiments/sequential-strategy-choice-trace-analysis-seed0.md`;
  the controlled CTA-placement/cache-policy crossover is in
  `docs/experiments/sequential-cta-cache-crossover/README.md`. Repeated
  matched search seeds and independent edit classification remain open.
- [x] Specify an opt-in hybrid CUDA MCTS action `freeform_cuda` in Section 48
  of `spec.md`, leaving baseline Milestone A semantics unchanged.
- [ ] After the sequential strategy study, implement and compare hybrid MCTS
  versus strategy-only MCTS under matched MCTS budgets, context, priors, and
  worker settings. Do not infer the cause of the MCTS/sequential gap from the
  sequential menu comparison alone.
- [ ] After the matched strategy study, test a compact history policy to
  control the 1.37M input-token cost observed in the two-turn seed-0 run.
- [ ] Ablate one factor at a time with repeated runs: generation-call depth;
  compiler-summary detail; dataflow view and explanatory text; visible NCU
  metric bundle; and an unconstrained prompt versus a fixed explicit strategy
  menu from which the model chooses. Hold the other context and acceptance
  rules fixed. Keep profiling collection and prompt visibility distinct where
  possible, and report both performance and context/profiling cost.

### Near-term CUDA experiment order

1. [x] Finish and verify the twelve matched `gpt-5.6-sol`, `B_gen=50`
   source-only, NCU-only, compiler-only, and combined traces using the pinned
   worker image and `lightweight_v1` metrics. Collate seed-level best latency,
   validity, token cost, and profile calls. See
   `docs/experiments/terra-sol-prompt-context-bgen50.md`.
2. [x] Run matched-seed Sol `diagnostic_v2` NCU-only and NCU-plus-compiler arms
   with the same root, image, declared strategy catalog, and search settings. Compare both
   arms within each seed and against their `lightweight_v1` counterparts. One
   additional combined-only seed 3 is recorded in the same report.
3. [x] Plot best-so-far latency versus `B_gen` and inspect representative winning
   and failed trajectories to identify which compiler/NCU facts correlate with
   useful edits. Treat three-seed differences as exploratory rather than a
   reliable causal ranking. See the trajectory section and figure in
   `docs/experiments/terra-sol-prompt-context-bgen50.md`.
4. [x] Build operation- and dtype-independent bounded PTX value-dependency
   extraction for valid CUDA nodes. Persist hash-checked, versioned within-block
   evidence; derive it read-only for older traces; keep workload-specific labels
   separate and unsupported paths explicit. See
   `docs/ptx-dataflow-extraction-plan.md`.
5. [ ] Reconcile selected compiled graphs with source, SASS, and same-worker NCU
   traffic before assigning quantitative movement claims. The initial
   source/PTX/SASS smoke is recorded in `docs/ptx-dataflow-extraction-plan.md`;
   runtime path validation remains open.
6. [ ] Only after that validation, test an off-by-default compact dataflow/hardware
   prompt summary against the strongest existing prompt arm at equal `B_gen` and
   matched seeds. Use the results to choose targeted PTX/SASS details or refine
   existing strategy prompts before adding new strategy categories.

Keep runtime code fixed during future prompt comparisons so every arm runs
under matched implementation settings.

### Prioritized CuTe search/generalization work

- [x] Enable typed LLM proposals for the independently lowered CuTe root. Prompt
  for a complete versioned typed representation, parse and statically validate it,
  require an admitted parent-to-child structural transition, render it through the
  deterministic independent lowering, charge every initial/repair call to `B_gen`,
  and persist mechanism-specific failures without creating invalid nodes.
- [x] Add separately budgeted post-search autotuning for the best independent CuTe
  result. Preserve the untuned winner, persist every `B_tune` trial and failure,
  report tuned and untuned results separately, and prove that tuning cannot alter
  MCTS visits, Q values, backups, or `B_gen`/`B_mut` accounting.
- [ ] Run bounded H100 validation of the independent mixed generator and post-search
  tuner. Confirm that LLM proposals remain within admitted typed transitions, every
  call and repair is charged to `B_gen`, static tuning failures remain trace records,
  the unchanged baseline is not re-evaluated as a trial, and MCTS counters and
  backups are identical before and after final-result tuning. The v82 smoke validated
  separate `B_mut=1`, `B_gen=1`, and `B_tune=2` accounting plus canonical LLM
  transposition reuse. It exposed and fixed retention of a slower tuned trial;
  rerun the bounded smoke to validate that corrected persisted/output behavior.
- [ ] Run the four required ablations from `spec.md` section 47.9: typed only; typed
  plus final tuning; typed plus LLM; and typed plus LLM plus final tuning. Hold the
  root, strategies, worker class, MCTS settings, measurement contract, and declared
  seeds fixed; report all budgets, validity rates, latency, cost, and canonical-state
  coverage separately.
- [x] Refactor the admitted independent CuTe transitions behind reusable structural
  transformation descriptors with applicability, parameter schema/enumeration,
  coupled-field rebuild, legality, deterministic lowering, and before/after
  evidence. Preserve the current canonical states and mutation graph while doing
  this mechanical refactor. The first slice routes production enumeration through
  versioned descriptors for all six admitted transformation families and records a
  stable transformation ID. A temporary parity implementation proved identical
  proposal order, parameters, and child hashes across ten representative states and
  was then removed; production now has one descriptor-driven enumeration path.
- [ ] Exercise the reusable transformations over a controlled GEMM shape/dtype
  sensitivity suite, including non-divisible shapes with explicit boundary/padding
  behavior. Keep arbitrary operations and architectures as a subsequent milestone;
  do not encode GEMM-specific fields into core MCTS.

- [x] Implement an initial searchable expert-template `CuTeDSLBackend` with canonical
  typed state, deterministic rendering, static legality checks, separate mutation
  accounting, and remote MCTS integration. The validated active fields are CTA tile,
  cluster shape, mainloop pipeline depth, and epilogue pipeline depth. A `B_mut=35`
  run recovered the same pinned-stage `(128,256)`, cluster `(2,1)` schedule as grid
  tuning; the additional staging fields produced no improvement beyond noise.
- [x] Build Milestone B's simpler independently controlled typed Hopper GEMM rather
  than describing current expert-template parameter search as new kernel synthesis.
  Introduce structural WGMMA, TMA, layout/swizzle, warp-specialization, and epilogue
  transformations one bounded diagnostic at a time, promoting only distinct,
  correct H100-validated controls to MCTS state. The independent root now lowers
  without importing NVIDIA's dense-GEMM implementation, and its admitted mutation
  graph reaches the validated 184.112 us wide warp-specialized endpoint.
- [x] Define the first GPU-free independent TMA-to-shared-memory contract for the
  fixed BF16 Hopper target. It coordinates complete A/B CTA tiles, K-major global
  and shared layouts, cluster-aware B multicast, three non-overlapping pipeline
  buffers, arrival-barrier slots, alignment, and the paired SW128/SW64 choice. Its
  canonical renderer output is explicitly structural and cannot enter MCTS yet.
- [x] Combine the independent TMA/shared-memory contract with a minimal WGMMA
  consumer and epilogue to render a complete executable CuTe DSL GEMM. Validate the
  two layout variants standalone on H100 before adding identity, backend, or search
  integration. The experimental `(128,256,64)` cooperative form now has a distinct
  typed state with two consumer warp groups and eight disjoint no-reuse epilogue
  buffers. Each group stages its own four local accumulator tiles; one elected CTA
  warp then issues all eight TMA stores. The v52 H100 validation passed both the
  bounded one-K diagnostic and the complete repository contract with exact tile
  identity and correctness. The full workload measured 448.384 us median over 30
  samples. `spec.md` section 47.4 now admits the paired one-group/two-group
  transition as an atomic `change_cta_tile` mutation under `B_mut`. The mutation
  now rebuilds every coupled typed field atomically, is selectable from the CLI,
  and has GPU-independent generator and end-to-end MCTS coverage proving
  `B_mut=1`, `B_gen=0`. The v53 remote mutation smoke completed with one valid
  proposal and backup, two distinct nodes and binaries, and two profile calls. The
  root measured 668.832 us and the cooperative child 448.608 us median, producing
  reward `0.399383` without an LLM call.
- [x] Define the complete typed structural contract for that kernel: two WGMMA
  consumer warp groups cover the `(128,256,64)` CTA tile with FP32 register
  accumulators, and a four-stage N-major BF16 TMA-store epilogue owns disjoint
  shared storage. Cross-component legality checks reject incomplete WGMMA coverage,
  incompatible ownership, unsafe epilogue tiling, overlap, and excess CTA storage.
- [x] Lower the complete independent contract into actual CUTLASS 4.5.1 CuTe DSL
  calls. The lowering must implement—not merely annotate—TMA descriptors and
  multicast masks, producer/consumer barriers, WGMMA partitioning, accumulator
  ownership, and the output TMA store before it is called executable.
- [x] Complete and validate the static half of that lowering against the pinned
  CUTLASS 4.5.1 container. The generated module directly binds tiled WGMMA, cluster
  and shared layouts, TMA load/store atoms, and both pipeline constructors without
  importing NVIDIA's dense-GEMM kernel. All ten required API bindings resolved. The
  report remains `executable_kernel=false` and enumerates every missing dynamic
  kernel component.
- [x] Extend the independent language with a typed execution schedule. GPU agents,
  staged shared-memory buffer rings, dynamic phases, read/write sets, waited-on
  signals, and emitted signals now form a canonical dependency DAG. Validation
  rejects unknown ownership, inconsistent stage counts, unproduced signals, and
  cycles before lowering.
- [x] Render the first dynamic phase as a standalone TMA copy-only diagnostic. One
  `(2,1)` cluster independently round-trips two A tiles, multicasts one B tile across
  both CTAs, and writes one B result back. The generated kernel includes shared
  storage, cluster coordinates, a multicast mask, transaction barriers, TMA
  load/store atoms, and exact BF16 A/B checks. It is not an MCTS state.
- [x] Build the updated CuTe image and run `independent-tma-copy` on H100. Record JIT,
  launch, exact A/B round-trip correctness, generated-source hash, and any barrier or
  multicast failure before adding WGMMA.
  The first v20 attempt hung before producing a report. The debug path now emits
  flushed JIT/launch/synchronize milestones and a cluster-wide barrier-initialization
  handshake. `compile_only` passed, while the first `single_cta_a` launch hung at
  synchronization, ruling out JIT and multicast. Empty launch passed and A-load-only
  then hung, isolating the issue to TMA load/barrier handling. The generated kernel
  had incorrectly entered `cute.copy` from one thread rather than warp-uniformly;
  load/store TMA issuance now follows the pinned warp-predication plus elected-arrival
  protocol. All v23 stages then passed: empty launch, A load/barrier, A round trip,
  single-CTA A/B, clustered A/B without multicast, and clustered A/B with multicast.
  Final A and B comparisons were exact with zero maximum error.
- [x] Add the first standalone WGMMA lowering stages on top of the validated TMA
  tiles. The generated diagnostic constructs the typed `64x256x16` tiled MMA,
  partitions shared A/B across two consumer warp groups, allocates FP32 register
  accumulators, and emits fence/GEMM/commit/wait. It remains outside MCTS.
- [x] Validate `wgmma_compile_only`, `wgmma_issue_only`, and `wgmma_one_k` on H100
  in that order. The final stage must match a PyTorch FP32-accumulation reference
  within the BF16 workload tolerances before implementing the production epilogue.
  The first compile-only attempt correctly exposed an invalid FP32-register to BF16-
  global direct copy. Compile and WGMMA issue then passed, but
  the first numerical run had large error because an accumulator tensor created from
  shape alone lost the MMA fragment's thread/value-to-C mapping. The diagnostic now
  uses `tiled_mma.make_fragment_C(tCgC)`. The v26 run produced identical error,
  strongly indicating a zero output and implicating the diagnostic global-copy path.
  Explicit scalar mapped stores in v27 proved WGMMA produced nonzero accumulators but
  wrote only 256 C elements—one per thread—because integer indexing did not flatten
  the hierarchical fragment. `autovec_copy` in v28 produced the same 256 nonzero
  elements, confirming that Hopper WGMMA accumulator ownership requires the tiled
  register-to-shared retile used by its epilogue. The diagnostic now lowers the typed
  FP32-register to BF16-shared conversion and complete shared-to-global TMA-store
  epilogue for revalidation. The first combined epilogue run triggered an illegal
  memory access, so `wgmma_r2s_only` now isolates the register-to-shared boundary
  before the shared-to-global TMA store is enabled. That boundary also triggered an
  illegal access; `wgmma_r2s_first_tile` now distinguishes the first tiled store
  from indexing or buffering across later epilogue tiles. The first-tile v31 run
  passed, localizing the error to later-tile traversal. A tentative fragment-count
  change did not fix the v32 run, and a hierarchical slicing attempt failed during
  v33 compilation. NVIDIA's pinned Hopper epilogue confirms that CTA-wide tile count
  and scalar fragment traversal are intentional, so both are restored. The new
  `wgmma_r2s_two_tiles` stage narrows the first failing tile without diverging from
  the reference lowering. Its v34 H100 run passed; `wgmma_r2s_four_tiles` now checks
  the complete four-tile partition expected for each M-oriented consumer warp group.
  The v35 four-tile run failed, so `wgmma_r2s_three_tiles` isolates whether tile three
  or tile four is the first invalid access. The v36 three-tile run passed, proving
  tile four is the boundary. `wgmma_r2s_four_reuse_zero` now distinguishes an invalid
  fourth accumulator tile from an invalid fourth shared-memory stage by reusing
  epilogue buffer zero for all four stores. Its v37 run passed, proving the fourth
  accumulator tile is valid and shared stage index three is the fault. Because the
  typed contract requires four stages, `wgmma_r2s_four_padded` tests whether the
  generated shared allocation is one tile short without changing that contract. Its
  first rendering attempt referenced a single-stage layout before definition; the
  padding then used the equivalent statically known epilogue-tile element count. The
  v39 padded run passed, confirming an undersized generated allocation. The ordinary
  four-tile allocation still failed in v40, proving the physical footprint is five
  tiles although the pipeline retains four logical stages. The typed epilogue now
  records that extra guard-tile padding explicitly, and renderer allocation, state
  identity, and shared-memory legality accounting all consume the same value. The
  full v41 register-to-shared traversal passed. Before re-enabling TMA, the global
  epilogue-tile coordinate stride was aligned with NVIDIA's reference ordering. The
  v42 combined run wrote all 32,768 output elements but failed correctness, so store
  coverage is complete while numerical ordering remains wrong. Compact cosine,
  64x64 tile-permutation, and within-tile transpose diagnostics now distinguish
  global tile order from accumulator-to-shared mapping errors. The v43 result had
  only 0.13 cosine similarity and remained far from correct under every tested tile
  permutation, localizing the error upstream of TMA. Comparison with NVIDIA's Hopper
  mainloop initially suggested direct fragment construction, but pinned CUTLASS 4.5.1
  rejected that during v44 compilation. Inspection of the exact pinned source showed
  that A/B partitioning was correct; the defect was using a warp-group coordinate
  rather than `tidx` for `tiled_mma.get_slice`. A v45 run disproved that hypothesis:
  it produced the identical incorrect output. Further inspection showed the pinned
  implementation does use the original warp-group slice, synchronizes all warp
  groups before its collaborative epilogue, and uses tile stride `(shape[1], 1)`.
  The diagnostic now matches those pinned v4.5.1 details together. In v46 this raised
  cosine similarity from 0.13 to 0.50 and mapped the first four output tiles
  correctly, while the four tiles written after buffer-ring wrap remained corrupt.
  `wgmma_one_k_no_reuse` uses eight logical epilogue stages plus the physical guard
  tile to test whether four-stage shared-buffer reuse is the remaining fault. Its v47
  output was identical to v46, ruling reuse out. `wgmma_one_group` now runs the same
  path with a `64x256x64` CTA tile and one consumer warp group to isolate cooperative
  two-warp-group decomposition from common operand and epilogue logic. That H100
  control passed exactly: cosine similarity 1.0 with zero maximum and mean error.
- [x] Promote the validated `(64,256,64)`, cluster `(1,1)`, single-warp-group design
  to schema v3 of the initial independent typed root. Shared-memory ownership,
  legality checks, deterministic lowering, and execution-agent counts now match the
  correct control. Preserve `(128,256,64)` as the diagnostic-only
  `wgmma_two_group` path until its second output row is correct. The ordinary
  `wgmma_one_k` lowering of the promoted state passed on H100 with cosine similarity
  1.0 and zero maximum and mean error; configuration hash
  `88f1c51b9352902fa905d1b32547b7e71752369aa8b29f2385949828aa4de6c1`.
- [x] Extend the promoted independent root from the validated single K tile
  (`K=64`) to the complete `K=4096` workload with a staged TMA/WGMMA mainloop, then
  validate correctness, timing, profiling, and artifact provenance before MCTS use.
- [x] Validate staged K traversal and full M/N coverage independently on H100. The
  three-stage ring cycles 64 K tiles with per-stage barrier phase tracking; the
  complete `4096x4096x4096` grid launches 64x16 CTAs. All 16,777,216 outputs were
  written, cosine similarity was 1.0, and correctness passed. With 10 warmups and 30
  CUDA-event measurements, median time was 665.792 us (mean 666.053 us, range
  656.736--674.976 us). This is a diagnostic checkpoint because it still uses seeded
  PyTorch inputs rather than the repository's canonical input generator.
- [x] Route the independent full-workload lowering through the repository workload,
  correctness, artifact, benchmark, and profiling contracts. The deterministic
  file-backed renderer now round-trips the typed state through `CuTeDSLBackend`,
  uses the canonical seed-0 input generator and cuBLAS reference, caches the full
  evaluation artifact, fingerprints normalized MLIR/fatbin output, and supports the
  existing NCU metric sets. H100 validation on 2026-09-25 passed correctness with
  30 CUDA-event samples (median 670.592 us) and a lightweight profile reporting
  154 registers/thread, 6.2% achieved occupancy, and 27.06% tensor-pipe utilization.
  Search-root selection and independent typed proposal mechanisms are tracked in
  the next completed item.
- [x] Make the independent CuTe root selectable end to end with matching worker
  calibration. The CLI, RunPod, Nebius, and worker bootstrap now carry an explicit
  `independent` root kind. Admit only the paired SW128/SW64 shared-memory mutation
  under `B_mut`; both operands change atomically and all other independent strategy
  IDs remain ineligible. The SW64 candidate passed the repository contract on H100
  on 2026-09-25 (correctness pass, median 718.832 us versus 670.592 us for SW128).
  A GPU-independent MCTS regression proves the valid candidate becomes a node,
  consumes one mutation and zero generations, and retains the slower valid state.
- [x] Validate a two-stage independent mainloop as the next structural control.
  The typed builder, storage offsets, barrier ring, canonical identity, and generated
  full-workload source now support two stages, but mutation enumeration intentionally
  excludes it from mutations pending a separate promotion change. The v49 H100
  repository-contract run passed exact correctness and measured 671.248 us median
  over 30 samples, versus 670.592 us for the three-stage root. It produced distinct
  normalized MLIR (`ca8b5177...`) and fatbin (`923b9382...`) fingerprints. This
  evidence supports the mutation promotion recorded below.
- [x] Promote the validated independent `3 -> 2` mainloop transition under
  `change_pipeline_stages`. The mutation rebuilds barrier slots, A/B storage,
  epilogue offset, execution-buffer stages, canonical identity, and rendered source
  together. Enumeration prevents composing stage two with SW64 because that pair has
  not been H100-validated. GPU-independent MCTS coverage proves one proposal consumes
  `B_mut=1`, `B_gen=0` and creates a valid measured node. The v50 remote H100 smoke
  completed on 2026-09-25 with one valid mutation, one valid-only backup, two unique
  nodes, two profile calls, and no generation calls. The three-stage root measured
  673.728 us median and the two-stage child measured 668.784 us (`reward=0.007365`),
  a near-noise difference consistent with the standalone validation.
- [x] Validate and promote the independent four-stage mainloop as a second
  `change_pipeline_stages` realization. The typed A/B storage rings, barrier ring,
  epilogue offset, execution buffers, canonical identity, and rendered source all
  change atomically. The v60 H100 repository-contract run passed exact correctness
  and measured 675.440 us median versus 669.760 us for the same-image three-stage
  root (`reward=-0.008445`). It remains searchable because valid slower states are
  not pruned. Stage two and stage four each return only to the stage-three root;
  unvalidated cross-control combinations remain excluded.
- [x] Exercise both pipeline-depth realizations in one bounded H100 search. The v61
  run used `B_mut=2`, `B_gen=0`, `k_max=2`, and `max_depth=1`; it created distinct
  stage-two and stage-four nodes and backed up both valid measurements. The root,
  stage-two, and stage-four medians were 669.664, 670.048, and 675.808 us,
  respectively. No LLM calls or hidden reevaluations were used.
- [x] Add and validate an independently lowered software-prefetch mainloop schedule.
  It prefills the three-stage TMA ring and refills each released stage while later
  full stages are consumed, without introducing a dedicated producer warp. The v63
  H100 repository-contract comparison passed exact correctness and measured 274.256
  us versus 675.408 us for the same-image serial schedule: a 2.46269x speedup and
  `reward=0.901255`. Promote it as schema-v4 typed state and the paired deterministic
  `change_mainloop_schedule` mutation; cross-control compositions remain excluded.
- [x] Validate the typed prefetch state through the CLI and MCTS paths. The v64 typed
  CLI run passed exact correctness at 274.608 us median. A one-iteration H100 search
  consumed `B_mut=1`, `B_gen=0`, created one distinct node, and backed up reward
  `0.892142` from 670.960 us serial and 274.944 us prefetch medians.
- [x] Add a typed dedicated producer/consumer warp-group partition. The v67 H100
  paired comparison passed exact correctness and measured 264.912 us for the
  specialized state versus 273.776 us for cooperative prefetch, a 1.03346x speedup
  (`reward=0.032913`). Schema v5 records distinct ownership and identity; the paired
  `change_producer_consumer_specialization` realization consumes one `B_mut` and is
  exposed only from the validated prefetch state.
- [x] Validate typed warp specialization through MCTS. The v68 H100 smoke used
  `B_mut=1`, `B_gen=0`, `k_max=1`, and `max_depth=1`; it measured 273.648 us for
  cooperative prefetch and 266.304 us for the specialized child, backed up reward
  `0.0272041`, and persisted two distinct schema-v5 nodes and two profiles.
- [x] Test complete producer/consumer register repartition as an evidence-only
  specialization diagnostic. The producer already used a 40-register decrease; v69
  added the matching 232-register consumer increase. Both variants were exactly
  correct and distinct, but medians of 266.016 and 265.840 us differed by only
  0.066%, while the repartitioned mean was slightly slower. Do not promote this
  control to typed MCTS state without stronger evidence.
- [x] Validate the wide-CTA plus warp-specialization interaction. The v69
  `(128,256,64)` specialized kernel passed exact H100 correctness and measured
  185.568 us versus 266.016 us for the same-image `(64,256,64)` specialized control,
  a 1.43352x speedup (`reward=0.360135`). Same-VM cuBLAS measured 176.096 us, leaving
  a 5.38% gap. Admit only this paired `change_cta_tile` transition under `B_mut`;
  keep all other specialized combinations excluded.
- [x] Validate the admitted specialized composition through MCTS. The v70 H100 run
  used `B_mut=2`, `B_gen=0`, `k_max=1`, and `max_depth=2`; cooperative prefetch,
  narrow specialization, and wide specialization measured 274.160, 266.032, and
  184.112 us, respectively. It created three distinct schema-v5 nodes, made three
  profile calls, performed two valid-only backups, and reached reward `0.398168`.
- [x] Test cluster-M multicast on the wide specialized kernel as an evidence-only
  interaction. The cluster `(1,1)` v70 control passed exactly at 185.264 us. Cluster
  `(2,1)` compiled and launched but deadlocked at device synchronization, so the
  existing specialized full/empty barrier protocol is not cluster-safe. Keep this
  composition outside typed MCTS state.
- [x] Test two mainloop stages on the wide specialized kernel. The candidate passed
  exact H100 correctness at 210.272 us versus 185.264 us for the same-image
  three-stage control, a 13.50% regression (`reward=-0.126620`). Retain the result
  as valid negative evidence without admitting the composition to typed MCTS.
- [x] Test SW64 on the wide specialized kernel. The v71 candidate passed exact H100
  correctness and produced a distinct artifact, but measured 224.192 us versus
  185.424 us for same-image SW128: a 20.91% regression (`reward=-0.189858`). Keep
  SW64 available as a reproducible diagnostic, but outside the admitted specialized
  MCTS neighborhood.
- [x] Build an evidence-only persistent CTA tile-scheduler ownership diagnostic for
  the final wide specialized kernel. Image v72 launched 132 persistent CTAs on the
  H100 and assigned the complete 32x16 logical grid exactly once: 512 assignments,
  512 unique tiles, and zero missing, duplicate, or unexpected tiles. Of the 132
  workers, 116 received four tiles and 16 received three. This validates bounded
  scheduling and ownership only; it does not yet validate reused GEMM barriers.
- [x] Validate persistent producer/consumer barrier-ring carry independently of
  GEMM data movement. Image v73 ran 132 persistent CTAs over all 512 output work
  items, with 64 K tiles per item and a three-stage full/empty ring. Producer and
  consumer independently scheduled and completed all 512 items in exact lockstep;
  116 CTAs completed four items and 16 completed three. The stage/phase calculation
  uses a global K-tile ordinal, so it does not incorrectly restart at each output
  tile. TMA, WGMMA, accumulator reset, and epilogue correctness remain the next gate.
- [x] Validate persistent TMA addressing and payload movement independently of
  WGMMA. Image v74 scheduled all 512 `(m,n)` output tiles exactly once, loaded the
  corresponding first A and B K tiles through the carried three-stage barrier ring,
  and matched every sampled shared-memory value to its expected global-memory
  address. There were zero missing, duplicate, unexpected, or payload-mismatched
  records. Full-K WGMMA, per-work accumulator reset, and epilogue stores remain the
  next gate.
- [x] Validate persistent wide-CTA WGMMA issue and per-work accumulator reset before
  adding the epilogue. The first v75 attempt was statically rejected because a
  dynamic thread-ownership branch was incorrectly wrapped in `const_expr`; it never
  launched and produced no candidate result. Corrected image v76 covered all 512
  tiles exactly once, retained exact A/B TMA samples, reset FP32 accumulators for
  every output work item, issued WGMMA from both consumer warp groups, waited for
  completion, and released every stage without deadlock. Numerical output remains
  untested until the persistent epilogue is added.
- [x] Validate the persistent wide-CTA epilogue for one K tile per output work item.
  Image v77 staged the two consumer groups' eight disjoint 64x64 accumulator tiles,
  issued the complete TMA-store epilogue, and synchronized before shared-buffer
  reuse. All 512 output tiles matched the one-K FP32 reference exactly, with zero
  maximum and mean error, while ownership and TMA payload checks remained exact.
  The remaining correctness gate is carrying all 64 K tiles per output item.
- [x] Integrate the validated persistent scheduler into an evidence-only copy of the
  final wide specialized GEMM. The v70 profile is compute-bound (92.22% tensor-pipe,
  87.32% SM, 28.62% DRAM) and the static 512-CTA grid executes in roughly four waves
  on H100. Image v78 carried all 64 K tiles through the three-stage ring and matched
  its synthetic full-K reference exactly. Image v81 then passed the repository
  4096x4096x4096 BF16 contract against the cuBLAS FP32-accumulation reference with
  zero maximum and mean error. Its 239.312 us median was 30.0% slower than the
  184.112 us static specialized kernel and 35.9% slower than the 176.096 us cuBLAS
  result. Retain persistent scheduling as valid negative evidence; do not promote it
  to typed MCTS state without an explicit Milestone B spec amendment.
- [x] Exercise both validated independent mutations in one bounded search. The v50
  `B_mut=2`, `B_gen=0`, `max_depth=1` run created separate SW64 and two-stage children
  directly beneath the SW128 three-stage root, performed two valid-only backups, and
  terminated with three unique nodes. The prohibited stage-two plus SW64 composition
  was never proposed. The trace records the second-step PUCT scores (`0.75` for the
  unvisited pipeline action versus `0.311221` for the visited swizzle action).
- [x] Validate CTA N extent / WGMMA N decomposition as the next independent
  structural control. The `(64,128,64)` variant atomically narrows
  the B tile and WGMMA instruction, uses two 64x64 epilogue tiles, recalculates
  shared storage, and launches a 64x32 CTA grid for the fixed workload. It passed
  the H100 repository contract exactly and measured 514.720 us median over 30
  samples with distinct normalized MLIR and fatbin fingerprints. `spec.md` now
  admits the paired transition, and mutation enumeration exposes it under
  `change_cta_tile` with one `B_mut` charge while excluding unvalidated combinations.
  A sequential same-VM/image root measurement was 673.440 us, a bounded 1.30836x
  speedup (`reward=0.268776`) before the MCTS smoke.
- [x] Run a bounded remote MCTS smoke that widens `change_cta_tile` twice from the
  independent root and confirms that the cooperative-M and narrow-N realizations
  become distinct valid nodes with `B_mut=2`, `B_gen=0`, cached evaluations, and
  valid-only backups. The v55 run completed in two iterations with three distinct
  state keys and binary hashes, two valid-only backups, and no generation calls.
  The 675.712 us root widened to the 450.000 us cooperative-M child
  (`reward=0.406519`) and the 514.816 us narrow-N child (`reward=0.271957`).
- [ ] Research a separate calibrated GPU resource/interconnect graph and map typed
  computation/schedule values onto it. Start with bytes, operations, reuse, storage,
  ownership, and pipeline overlap; later calibrate uncertain latency/bandwidth terms
  from NCU and timings. Use it for explanations, prompts, and optional priors—not as
  an unmeasured reward or performance-based hard-pruning rule.
- [x] Identify and implement one SHA-guarded shared-memory layout/swizzle diagnostic
  in the pinned NVIDIA template. The bounded alternative preserves MN/K majorness
  while forcing SW64 layout atoms; for the BF16 workload this changes A/B from the
  heuristic SW128 choice while the epilogue remains SW64. It is diagnostic-only and
  cannot enter canonical state or consume an MCTS budget.
- [x] Run the same-worker H100 swizzle diagnostic. Heuristic SW128 and forced SW64
  both passed exact correctness and produced distinct normalized IR, fatbins, and
  kernel layout identities. Medians were `188.016 us` and `188.528 us`; the 0.27%
  difference is within run noise and does not establish a performance winner.
- [x] Promote the validated shared-memory control to `CuteGemmProgram` schema v3 and
  a deterministic `change_shared_memory_swizzle` mutation with `heuristic` and
  `sw64` values. SW64 remains limited to the validated `(128,256)`, cluster `(2,1)`
  schedule; rendering, identity, transposition, persistence, and `B_mut` behavior
  have GPU-independent regression coverage.
- [x] Run the combined canonical H100 validation/profile for heuristic and SW64.
  The corrected v18 run passed every schema-v3, cache, correctness, artifact, and
  diagnostic-v2 profile check. SW64 measured `188.576 us` versus `187.072 us` for
  the heuristic and executed 4.4% more instructions despite higher L2 hit/throughput.
  The initial v17 run correctly failed distinct-artifact validation and exposed a
  missing backend-runner forwarding path, which now has regression coverage.
- [x] Run a focused `B_mut=1`, `B_gen=0` H100 smoke from `(128,256)`, cluster `(2,1)`
  with only `change_shared_memory_swizzle`. The run created two valid nodes in one
  iteration, charged exactly `B_mut=1` and `B_gen=0`, and backed up the SW64 reward
  of `-0.00927`; the heuristic root correctly remained best. The schema-v3 trace is
  `cutedsl-swizzle-bmut1-v18.sqlite` (local run artifact, not committed).
- [ ] Add bounded local autotuning for parameterized kernel families, initially for
  CUTLASS SM90 WGMMA/TMA configurations. Keep the algorithm fixed while tuning
  discrete choices such as tile sizes, pipeline stages, warp layout, vector width,
  shared-memory layout, and launch geometry. Start with constrained grid/random
  search, then compare TPE or Bayesian optimization against random search under an
  equal, separately reported `B_tune` budget. Persist every attempted configuration,
  including compile/correctness failures and timings, and expose the best valid tuned
  configuration as the strategy realization without charging mechanical trials to
  `B_gen`.
- [ ] Add an LLM strategy-prior adapter while preserving separate `B_prior` accounting.
- [ ] Add at least three more end-to-end benchmark kernels for the Milestone A minimum suite.
- [ ] Add strategy applicability metadata and filter actions by operation, dtype,
  backend, hardware capabilities, and workload constraints before MCTS selection.
- [ ] Add post-optimization sensitivity analysis for optimized kernels. Evaluate each
  fixed kernel over controlled perturbations of input dimensions, aspect ratios,
  alignment/divisibility, and supported input/output/accumulation dtypes. For every
  case, rerun correctness and compare latency with cuBLAS measured under the identical
  case and environment. Report absolute latency, throughput, the kernel-to-cuBLAS
  latency ratio, and degradation relative to the kernel's optimized workload. Keep
  this diagnostic sweep separately budgeted from `B_gen` and `B_tune`, and retain
  invalid or unsupported cases rather than silently excluding them.
- [ ] After the typed CuTe DSL search flow is complete, run a bounded GPU MODE
  external-generalization study over four to six computational motifs. Freeze the
  search configuration, use equal explicit budgets, and report validity, latency,
  cost, and shape/hardware sensitivity in addition to leaderboard placement. Keep
  this secondary to attention, fusion, and forward-pass optimization; see
  `docs/research-roadmap.md`.

## Validation status

- Unit suite: `282 passed, 1 skipped` at the time this checklist was last updated.
- The opt-in pytest remains skipped in the local unit suite because it requires an
  attached GPU. Equivalent root compilation, correctness, timing, and vendor-baseline
  validation completed through the remote worker lifecycle on 2026-09-11.
- Lightweight Nsight Compute extraction is covered by GPU-independent command, parser,
  protocol, caching, and persistence tests. Numeric metric collection succeeded on a
  Nebius H100 SXM host, in the privileged worker container, and through the automated
  provider smoke lifecycle on 2026-09-14.
- The guarded one-iteration MCTS smoke search completed through the Nebius provider on
  2026-09-14. It also validated root normalization: the unchanged root remained the
  global best with reward zero when the generated candidate was slower.
- The guarded controller-only OpenAI generation check completed with
  `gpt-5.6-terra` on 2026-09-11. The generated candidate subsequently compiled,
  passed correctness, and achieved a `1.328x` speedup over the packaged root on H100.
