# Triton H100 CUDA Graph reward contract

`spec.md` Section 50 now defines `triton_h100_cuda_graph_v1` for Triton and
Gluon search. The benchmark captures one call through the candidate's
`launch(A, B, C, M, N, K)` ABI at fixed tensor addresses, verifies replay
correctness and compiled cubin identity, warms the graph, and uses the median
of CUDA-event-timed replays as search latency. Capture or replay failure is a
benchmark failure. Per-call CUDA-event timing around the ordinary Python
launcher remains in `diagnostics.ordinary_launch_*` and is never the reward.
If that optional diagnostic fails, its error type is recorded and the valid
Graph reward remains usable.

The worker calibrates its packaged Triton root, cuBLAS, and CUTLASS with the
same Graph contract. The C++ vendor capture uses a warmed nonblocking CUDA
stream and records timing events on that stream. Search startup rejects a
worker whose calibration has
a missing or different timing contract. The contract, samples, and diagnostic
ordinary-launch samples are serialized in the benchmark record and SQLite
trace; the run configuration also carries the contract name. CUDA C++ and
CuTe DSL timing are unchanged.

The earlier Triton runs used per-call events around the Python launcher.
Their latency values and search reward curves belong to a separate historical
cohort. The [same-worker crossover](../triton-h100-gluon-timing-reconcile/README.md)
explains why the two timing methods can rank Gluon sources differently.
Graph replay measures captured GPU work; a one-off application launch may
also incur host-side descriptor and submission costs.

## H100 validation

The [runner](run.py) overlays the local package onto a temporary pinned-image
H100 worker, restarts it to recalibrate, checks the root and vendor timing
contracts, then compiles, correctness-checks, and Graph-benchmarks the
[one-CTA](../triton-h100-gluon-search-adapter/single_cta_gluon_fixture.py)
and [persistent](../triton-h100-gluon-search-adapter/persistent_factorial_fixture.py)
Gluon sources through the updated search backend. It records source and
overlay hashes, environment manifest, calibration, and per-candidate
benchmarks in `results.json` and `validation.json`. Worker release is handled
by the provider context even if validation fails.

## Result

The validation completed on one H100 SXM worker with Triton 3.9.0 and CUDA
12.8.2. The root, cuBLAS, and CUTLASS calibration records all carried
`triton_h100_cuda_graph_v1`; both Gluon candidates compiled, passed the
backend correctness check, and returned Graph-timed benchmark records. Each
row is the median of 30 event samples after 10 warmups, in microseconds.

| Implementation | Graph replay reward/reference | Ordinary launcher diagnostic |
| --- | ---: | ---: |
| Packaged Triton root | 816.704 | 832.576 |
| cuBLAS | 173.056 | 178.064 |
| CUTLASS | 307.680 | 309.664 |
| One-CTA Gluon | 181.056 | 250.288 |
| Persistent Gluon | 179.760 | 267.568 |

The ordinary diagnostics reproduce the large one-CTA versus persistent host
launch gap. Graph replay reduces that gap to 1.296 µs on this single worker;
this is a contract validation, not a statistical ranking of the two kernels.
The worker and its boot disk were deleted after the run. The pinned published
image still contains the older implementation; the runner validated an
overlaid local package with the bundle hash recorded in `results.json`.
