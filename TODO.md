# Implementation status

`spec.md` is the source of truth for Milestones A and B. This file records implementation
and validation status. Checked implementation items have automated coverage; any
completed hardware validation is identified explicitly.

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
- [ ] Run the canonical backend-mode H100 smoke for epilogue stages 2 and 3 and verify
  the report preserves schema v2, configuration identity, cached second evaluation,
  correctness, and the same distinct runtime artifacts observed by the diagnostic.
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

- [ ] Implement Milestone B's searchable `CuTeDSLBackend`: begin with a simpler typed
  Hopper GEMM representation, deterministic rendering, and static legality checks;
  then expose structural WGMMA, TMA, pipeline, layout, warp-specialization, and
  epilogue strategies beneath the existing MCTS strategy layer.
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
