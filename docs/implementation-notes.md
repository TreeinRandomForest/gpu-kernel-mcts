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

`NebiusProvider` uses the authenticated `nebius` CLI to create one temporary
`gpu-h100-sxm` VM and boot disk per search run. It reaches the VM only through SSH,
starts the same worker image with `--gpus all --cap-add=SYS_ADMIN`, and exposes the
worker to the controller through a loopback-only SSH tunnel. The ephemeral worker
token is uploaded through SSH standard input in a mode-0600 file, never placed in a
process argument, and removed immediately after container launch. Release closes the
tunnel, deletes the VM, and then deletes its boot disk, including on startup,
manifest-validation, and search failures. The SSH private key and Nebius CLI
credentials remain controller-local.

If the Nebius CLI reports a failed instance-create command after the VM was
actually created, the client recovers the VM by its unique generated name and
deletes the VM before deleting its attached boot disk. CLI failures include a
bounded, secret-redacted stderr diagnostic; incomplete cleanup identifies the
resource IDs that may require manual removal.

## Run orchestration

The provider-neutral orchestration layer starts the SQLite run, acquires one worker,
records its manifest, evaluates the root, runs one global MCTS, and releases the
worker in a `finally` path. Remote evaluation IDs are stable hashes of the run,
program, and workload, so an infrastructure retry is idempotent rather than a second
logical GPU evaluation. A deterministic mock integration test exercises transient
infrastructure failure, invalid-generation repair, valid-only node creation, budget
accounting, trace materialization, and cleanup without LLM or GPU resources.

CUDA benchmarks capture best-effort `nvidia-smi` snapshots immediately before and
after timing, including temperature, SM and memory clocks, power draw/limit,
performance state, utilization, and active clock-event reasons. These subprocesses
run outside the CUDA-event timing interval. The serialized benchmark retains both
snapshots for every root, candidate, and explicit drift measurement.

An optional measurement-drift policy remeasures the root and current global best
after a configured number of newly created valid nodes. It emits complete
`measurement_drift_probe` events with latency ratios, telemetry, and a configurable
suspect threshold. Probes do not create search nodes, consume `B_gen`, update cached
benchmarks, redefine `T_root`, or change selection and backup values. Enable it with
`--measurement-drift-interval`; the default zero disables the extra GPU work.

Optional post-search autotuning keeps the worker alive after MCTS and tunes only an
explicitly annotated final-best CUDA template. It has independent `B_tune` accounting
and random/grid candidate ordering. Trials undergo the normal compile, correctness,
benchmark, and telemetry pipeline but never become MCTS nodes or alter backups.
Dedicated `tuning_runs` and `tuning_trials` tables preserve every attempt. Unannotated
kernels are recorded as skipped rather than modified heuristically.

The post-Milestone A CuTe path pins CUTLASS/CuTe DSL 4.5.1 and adapts NVIDIA's Hopper
dense-GEMM example to the same fixed BF16 workload, cuBLAS reference, tolerances,
warmups, and individual CUDA-event measurements used by repository baselines. A
standalone deterministic tuner evaluates a typed CTA-tile/cluster-shape grid under a
separate `B_tune`; trials never create MCTS nodes or alter `B_gen`. On H100 SXM the
best tile `(128, 256)` with cluster `(2, 1)` measured `184.880 us`, versus
`193.024 us` for the default cluster and `177.056 us` for same-worker cuBLAS.

Milestone B phase 1 adds a versioned `CuteGemmProgram` whose canonical JSON and hash
describe kernel structure without duplicating the fixed workload contract. The first
renderer deterministically targets the pinned NVIDIA template and rejects structural
controls that it cannot yet express. SQLite schema version 8 adds nullable typed
representation, configuration-hash, validation, transformation, and proposal-mechanism
fields to the existing node and generation tables. Existing CUDA traces and browser
queries remain valid; CuTe nodes use the same graph, profile, decision, and comparison
APIs, while analysis bundles export rendered CuTe programs with a `.py` suffix.
`CuteGemmProgram` schema v2 adds H100-validated epilogue depths 2 and 3 for tile
`(128,256)`, cluster `(2,1)`. Pinned depth 4 remains encoded as `None`, preventing an
identical effective kernel from receiving two canonical identities.

Milestone B phase 2 implements `CuTeDSLBackend` without enabling CuTe MCTS. It accepts
only source that exactly matches the deterministic renderer for its embedded typed
representation. The first compile invokes the pinned repository-contract adapter in a
bounded subprocess, which triggers CuTe JIT, correctness, and CUDA-event measurement;
the resulting artifact caches all evidence so the evaluator's correctness and benchmark
stages, repeated evaluations, and ordinary visits do not repeat GPU work. Valid results
flow through `BackendKernelEvaluator` and therefore receive the standard reward,
state-key, compilation/correctness evidence, telemetry, worker/environment identity,
and trace serialization.

Milestone B phase 3 inspects the CuTe JIT callable rather than assuming a filesystem
artifact. It records generated kernel names, bounded normalized MLIR and its hash, and
any exposed or MLIR-embedded cubin, fatbin, SASS, or PTX payload hashes. Effective identity
prefers those binary forms, then normalized MLIR, then the phase-2 source/template
fallback; every choice is still combined with launch configuration. The compiler IR
is retained in evaluation metadata for auditability and later structured-analysis
experiments. A separate cached NCU subprocess skips correctness and benchmark timing,
filters the generated kernel, and performs one launch, so profiling cannot change the
CUDA-event reward. The H100 validation confirmed exact generated-kernel filtering,
all nine lightweight metrics, reward/profile separation, and an embedded fatbinary
fingerprint.

The next standalone slice adds deterministic typed mutations for CTA tile and cluster
shape. Every proposal carries static validation and before/after configuration
evidence; no mutation invokes an LLM, JIT, GPU evaluation, or search-budget counter.
The same target reached through tile-then-cluster or cluster-then-tile transitions has
identical canonical identity. Core MCTS now reserves these transitions against a
separate `B_mut`, disables mutation repairs, retains the normal selection/widening/UCB
roles, and persists both budget counters. Existing generation-only searches remain
backward compatible. The search CLI, RunPod/Nebius provider configuration, CuTe image
entrypoint, and worker backend selection now support guarded mutation-only and
mutation-first mixed CuTe searches. H100 search validation and broader structural
controls remain pending. The first mutation-only Nebius H100 SXM run passed on
2026-09-20: one cluster-shape mutation was valid, both nodes were profiled, `B_gen`
remained zero, and `B_mut` was exactly one. A subsequent corrected mixed run consumed
four typed mutations followed by one LLM JSON proposal; deterministic rendering
canonicalized the proposal to the existing best node, and the worker reused its
cached evaluation.

The guarded smoke-search CLI adds one remote iteration using a packaged direct BF16
GEMM candidate. It is intentionally a wiring test rather than an optimization claim:
the one-shot generator performs no LLM call, while `B_gen=1` still exercises the same
search budget, trace, and backup paths used by a future API-backed generator.

## LLM generation

The provider-neutral `LLMClient` returns text, resolved model identity, token usage,
latency, and provider metadata. `OpenAIResponsesClient` implements it with one
independent Responses API request per completion: it supplies neither a conversation
nor `previous_response_id`, and response storage defaults to disabled. The API key is
read only by the local controller from its configured environment variable.

`LLMKernelGenerator` builds a backend-aware JSON prompt from the parent kernel,
workload, hardware, selected strategy, profile summary, and bounded repair evidence.
It returns a complete replacement program and preserves the exact prompt, raw output,
model metadata, usage, latency, and response ID in the generation trace.

The controller-only `kernel_mcts.llm_cli` command provides a guarded intermediate
check before provisioning a GPU. It requires an explicit model, configured strategy,
new output path, and `--confirm-api-call`; makes exactly one request; writes only the
extracted candidate source; and prints response identity, resolved model, token usage,
latency, and output path without printing the prompt, raw response, or API key.
The guarded command completed successfully with `gpt-5.6-terra` on 2026-09-11.
The generated coalescing candidate subsequently compiled and passed correctness on
H100. Its median latency was `21431.792 us` versus `28464.76755 us` for the freshly
measured root, giving reward `0.2837916691` and speedup `1.328x`.

The separate `kernel_mcts.evaluate_cli` command provisions one validated worker,
measures the packaged root once, and evaluates one candidate through the same tier-zero
compile, correctness, and benchmark pipeline used by search. Valid candidate rewards
are normalized against that measured root. Infrastructure failures receive one
idempotent retry, the worker is released in a `finally` path, and a non-overwritten
JSON report retains the environment manifest and complete root/candidate evidence.

The search CLI supports both the original deterministic smoke generator and the
OpenAI generator. OpenAI mode loads semantic strategies from configuration, checks
LLM configuration before provisioning, uses uniform strategy priors, and permits an
arbitrary positive `B_gen`. Every initial or repair generation remains an independent
Responses API call. The SQLite run records the configured model, while each generation
records the resolved model and usage. An optional new `--best-output` path exports the
best valid kernel after the global search completes.

Root evaluation uses the same configurable infrastructure-retry count as candidate
evaluation. Every retry reuses the stable root evaluation ID and emits a complete
`root_evaluation_attempt` trace event. Exhausted retries occur before MCTS starts,
consume no `B_gen`, create no nodes or backups, and record the final root status,
attempt count, and sanitized underlying error type before worker cleanup.

Lightweight Nsight Compute profiling is lazy: a valid node is profiled only when it is
first selected for expansion. The controller requests an authenticated profile of the
worker-cached evaluation ID, so profiling reuses the compiled artifact and does not
repeat compilation, correctness, or timing. The worker caches the versioned numeric
profile, and the node caches it for later visits. Friendly summary fields and raw
metric names, values, and units are persisted and supplied to subsequent generation
prompts. Profile calls are counted separately from `B_gen` and `B_prior`. Full NCU
reports and rule recommendations are not yet collected.

After budget exhaustion, the final global-best node is also lightweight-profiled
before worker release if it has no cached profile. The trace labels this call with the
`final_best` trigger, updates the persisted node profile, and includes it in the run's
profile-call count. A best node already profiled during expansion is not profiled
again.

Search and candidate-evaluation CLIs also support explicit ephemeral storage. This
mode sends neither a network-volume ID nor data-center affinity, uses only the pod's
container disk, and relies on the mandatory termination lifecycle. Durable traces and
exported kernels remain on the controller.

## Known limitations

- The CUDA harness, root, cuBLAS baseline, and fixed CUTLASS baseline have run
  successfully through the remote lifecycle on an H100 SXM. The opt-in pytest remains
  skipped in ordinary local test runs because no GPU is attached locally.
- CUDA artifact directories are retained because search nodes cache compiled artifacts; run-level cleanup policy is not implemented yet.
- SASS is normalized and hashed when `cuobjdump` is available. The fallback hashes executable bytes and may deduplicate less reliably across builds.
- The CUDA backend currently supports only the fixed BF16 GEMM ABI and launch configuration.
- Strategy configuration does not yet express applicability by operation, dtype,
  backend, hardware capability, or workload constraint. All configured strategies are
  currently exposed to MCTS, so the production catalog remains tailored to BF16 GEMM.
- Full profiling remains unimplemented; the lightweight metric set has been validated
  through the automated Nebius H100 SXM worker lifecycle.
