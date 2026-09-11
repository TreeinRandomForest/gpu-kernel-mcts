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
- Full profiling remains unimplemented; the lightweight metric set still requires H100 validation.
