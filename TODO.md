# Implementation status

`spec.md` is the source of truth for Milestone A. This file records implementation
status only; a checked item means that code and automated tests exist, not that an
optional hardware integration test has run.

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
- [x] Fixed 4096 x 4096 x 4096 H100 BF16 GEMM workload and packaged CUDA root kernel.
- [x] Backend-neutral compile, correctness, benchmark, reward, and state-identity pipeline.
- [x] CUDA/NVCC backend and cuBLAS correctness/benchmark harness with GPU-independent tests.

## Next

- [ ] Run and record the opt-in BF16 GEMM validation on an H100 SXM worker; see [docs/h100-validation.md](docs/h100-validation.md).
- [ ] Implement the concrete RunPod API client, remote worker service, and transport adapters. The current provider uses injected protocols and test doubles.
- [ ] Connect provider acquisition, root calibration, generation, evaluation, persistence, and cleanup in an end-to-end run entry point.
- [ ] Add lazy Nsight Compute profiling and cache profiles per node/environment.
- [ ] Add concrete LLM generator and LLM-prior adapters while preserving `B_gen` and `B_prior` accounting.
- [ ] Add at least three more end-to-end benchmark kernels for the Milestone A minimum suite.
- [ ] Add periodic root/global-best remeasurement and environment-drift handling.

## Validation status

- Unit suite: `83 passed, 1 skipped` at the time this checklist was created.
- Skipped test: real H100 compilation, cuBLAS correctness, and timing.
- No claim of GPU correctness or performance should be made until that test passes on the requested hardware class.
