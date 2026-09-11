# Implementation status

`spec.md` is the source of truth for Milestone A. This file records implementation
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

## Next

- [ ] Generalize the smoke-search CLI into a configuration-driven LLM search entry point.
- [ ] Add lazy Nsight Compute profiling and cache profiles per node/environment.
- [ ] Add offline CUTLASS SM90 WGMMA/TMA autotuning and persist the selected configuration; keep its tuning cost separate from MCTS budgets.
- [ ] Add an LLM strategy-prior adapter while preserving separate `B_prior` accounting.
- [ ] Add at least three more end-to-end benchmark kernels for the Milestone A minimum suite.
- [ ] Add periodic root/global-best remeasurement and environment-drift handling.

## Validation status

- Unit suite: `138 passed, 1 skipped` at the time this checklist was last updated.
- The opt-in pytest remains skipped in the local unit suite because it requires an
  attached GPU. Equivalent root compilation, correctness, timing, and vendor-baseline
  validation completed through the remote worker lifecycle on 2026-09-11.
- The guarded one-iteration MCTS smoke search also completed remotely. The subsequent
  root-normalization correction is covered by automated tests but has not yet been
  rerun on the H100.
