# Implementation notes

These notes record concrete choices made while implementing `spec.md`. They do not
override the specification.

## Search structure

The search is one global program search. PUCT selects a semantic `StrategyEdge`.
Progressive widening determines when another stochastic realization may be
generated, and UCB selects among existing `RealizationEdge` instances. Only
compile-successful, correctness-passing programs become nodes. Transpositions form
a DAG, so depth is computed from the selected path rather than stored globally on a
canonical node.

Every candidate-generation call consumes `B_gen`, including repair calls. Prior-LLM
calls are tracked separately as `B_prior`. Only measured valid leaves are backed up.
Selection uses `Q_mean`; `Q_max` is retained for trace analysis.

## Evaluation and identity

Tier-zero evaluation is:

```text
normalize source -> compile -> correctness -> benchmark -> fingerprint -> reward
```

The result retains the compiled artifact in memory and serializable compilation,
correctness, timing, launch, worker, and environment evidence. Ordinary MCTS visits
reuse this result.

At run start, orchestration treats the benchmark from the valid root evaluation as
the immutable `T_root`. The root retains all measured evidence but receives reward
exactly `0.0`. Valid candidate rewards are recomputed by the controller from their raw
benchmarks against that same `T_root`; this prevents an earlier worker-startup
calibration sample from introducing a nonzero root reward through measurement noise.

Normalized source hashes identify exact or formatting-only duplicates. The compiled
state key deliberately excludes source text and ephemeral worker identity. It hashes
the binary fingerprint, workload, launch configuration, and stable hardware/toolchain
context so different source programs that produce the same effective binary may
transpose. The environment-manifest ID is still recorded for auditability.

Failures remain distinct:

- compiler rejection, launch failure, correctness failure, timeout, and benchmark failure are invalid candidate realizations;
- worker, transport, tool availability, and malformed protocol responses are infrastructure failures;
- unexpected configuration/programming errors propagate rather than being silently reclassified.

## Initial BF16 GEMM

The initial workload is deliberately narrow:

- shape: `M=N=K=4096`, weight `1.0`;
- inputs: BF16;
- accumulation: FP32;
- output: BF16;
- `A`: row-major `[M, K]`;
- `B`: column-major `[K, N]`;
- `C`: row-major `[M, N]`;
- entry point: `bf16_gemm_root`;
- launch: block `(16, 16, 1)`, grid `(ceil(N/16), ceil(M/16), 1)`.

The root is a simple shared-memory CUDA-core implementation intended as a readable
baseline, not an expert kernel. The harness generates deterministic inputs and uses
cuBLAS with FP32 accumulation as the correctness reference. It benchmarks with CUDA
events after warmup and retains every timing sample plus summary statistics.

Worker calibration also measures fixed cuBLAS and CUTLASS performance references
with the same shape, data layouts, deterministic inputs, tolerances, warmups, and
measurement count. CUTLASS is pinned in the worker image. These references are
reported separately and never enter the MCTS state space or budget accounting.
The current CUTLASS reference is a fixed, untuned Tensor Core configuration, not
an auto-tuned Hopper SM90 WGMMA/TMA performance ceiling.

## Remote execution and secrets

`RunPodProvider` owns lifecycle policy but depends on injected client and transport
interfaces. It acquires one worker per run, captures and validates its manifest once,
reuses it for evaluations, and releases it in a `finally` path. `RunPodRESTClient`
implements the client interface against the official Pods REST API. The authenticated
worker service exposes health, manifest, and tier-zero evaluation endpoints through
RunPod's HTTPS proxy. Each pod receives a random worker-only bearer token. Evaluation
IDs are idempotent: an identical retry returns the cached result, while reuse with a
different payload is rejected. The worker retains compiled artifacts for its run.

The HTTP service starts before GPU calibration and reports authenticated startup
stages through `/health`. If initialization fails, the controller receives only the
stage and exception category, prints the diagnostic, and still terminates the pod;
arbitrary exception text and credentials are not returned.

The H100 SXM provisioning path attaches an existing network volume and pins the pod
to explicitly configured matching data-center IDs. The REST payload omits
`volumeInGb` when `networkVolumeId` is present. Pod termination does not delete the
network volume; its lifecycle and storage charges remain externally managed.

Automatic placement uses read-only `runpodctl datacenter list --output json` data to
find exact GPU-ID matches with positive stock status. It then prefers an adequately
sized volume with the configured managed name in one of those data centers. Creating
a missing persistent volume requires a separate explicit confirmation. Existing,
unrelated, undersized, and user-managed volumes are never silently repurposed or
deleted.

`RUNPOD_API_KEY` is read only from its configured environment variable. CUDA child
processes receive a small allowlisted environment rather than the controller's full
environment. Credentials must not appear in prompts, manifests, traces, subprocess
arguments, or error payloads.

## Run orchestration

The provider-neutral orchestration layer starts the SQLite run, acquires one worker,
records its manifest, evaluates the root, runs one global MCTS, and releases the
worker in a `finally` path. Remote evaluation IDs are stable hashes of the run,
program, and workload, so an infrastructure retry is idempotent rather than a second
logical GPU evaluation. A deterministic mock integration test exercises transient
infrastructure failure, invalid-generation repair, valid-only node creation, budget
accounting, trace materialization, and cleanup without LLM or GPU resources.

The guarded smoke-search CLI adds one remote iteration using a packaged direct BF16
GEMM candidate. It is intentionally a wiring test rather than an optimization claim:
the one-shot generator performs no LLM call, while `B_gen=1` still exercises the same
search budget, trace, and backup paths used by a future API-backed generator.

## Known limitations

- The CUDA harness, root, cuBLAS baseline, and fixed CUTLASS baseline have run
  successfully through the remote lifecycle on an H100 SXM. The opt-in pytest remains
  skipped in ordinary local test runs because no GPU is attached locally.
- CUDA artifact directories are retained because search nodes cache compiled artifacts; run-level cleanup policy is not implemented yet.
- SASS is normalized and hashed when `cuobjdump` is available. The fallback hashes executable bytes and may deduplicate less reliably across builds.
- The CUDA backend currently supports only the fixed BF16 GEMM ABI and launch configuration.
- Profiling methods intentionally raise `NotImplementedError`; lazy profiling is a later phase.
