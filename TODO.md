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

- [x] Implement an initial searchable expert-template `CuTeDSLBackend` with canonical
  typed state, deterministic rendering, static legality checks, separate mutation
  accounting, and remote MCTS integration. The validated active fields are CTA tile,
  cluster shape, mainloop pipeline depth, and epilogue pipeline depth. A `B_mut=35`
  run recovered the same pinned-stage `(128,256)`, cluster `(2,1)` schedule as grid
  tuning; the additional staging fields produced no improvement beyond noise.
- [ ] Build Milestone B's simpler independently controlled typed Hopper GEMM rather
  than describing current expert-template parameter search as new kernel synthesis.
  Introduce structural WGMMA, TMA, layout/swizzle, warp-specialization, and epilogue
  transformations one bounded diagnostic at a time, promoting only distinct,
  correct H100-validated controls to MCTS state.
- [x] Define the first GPU-free independent TMA-to-shared-memory contract for the
  fixed BF16 Hopper target. It coordinates complete A/B CTA tiles, K-major global
  and shared layouts, cluster-aware B multicast, three non-overlapping pipeline
  buffers, arrival-barrier slots, alignment, and the paired SW128/SW64 choice. Its
  canonical renderer output is explicitly structural and cannot enter MCTS yet.
- [ ] Combine the independent TMA/shared-memory contract with a minimal WGMMA
  consumer and epilogue to render a complete executable CuTe DSL GEMM. Validate the
  two layout variants standalone on H100 before adding identity, backend, or search
  integration.
- [x] Define the complete typed structural contract for that kernel: two WGMMA
  consumer warp groups cover the `(128,256,64)` CTA tile with FP32 register
  accumulators, and a four-stage N-major BF16 TMA-store epilogue owns disjoint
  shared storage. Cross-component legality checks reject incomplete WGMMA coverage,
  incompatible ownership, unsafe epilogue tiling, overlap, and excess CTA storage.
- [ ] Lower the complete independent contract into actual CUTLASS 4.5.1 CuTe DSL
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
- [ ] Validate `wgmma_compile_only`, `wgmma_issue_only`, and `wgmma_one_k` on H100
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
  `B_mut=1`, `B_gen=0` and creates a valid measured node. A remote smoke remains.
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
