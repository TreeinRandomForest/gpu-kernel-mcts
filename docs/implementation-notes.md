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

## Remote execution and secrets

`RunPodProvider` owns lifecycle policy but depends on injected client and transport
interfaces. It acquires one worker per run, captures and validates its manifest once,
reuses it for evaluations, and releases it in a `finally` path. `RunPodRESTClient`
implements the client interface against the official Pods REST API. The authenticated
worker service exposes health, manifest, and tier-zero evaluation endpoints through
RunPod's HTTPS proxy. Each pod receives a random worker-only bearer token. Evaluation
IDs are idempotent: an identical retry returns the cached result, while reuse with a
different payload is rejected. The worker retains compiled artifacts for its run.

`RUNPOD_API_KEY` is read only from its configured environment variable. CUDA child
processes receive a small allowlisted environment rather than the controller's full
environment. Credentials must not appear in prompts, manifests, traces, subprocess
arguments, or error payloads.

## Known limitations

- The CUDA harness has not yet been compiled or run on an H100 in the recorded development environment.
- CUDA artifact directories are retained because search nodes cache compiled artifacts; run-level cleanup policy is not implemented yet.
- SASS is normalized and hashed when `cuobjdump` is available. The fallback hashes executable bytes and may deduplicate less reliably across builds.
- The CUDA backend currently supports only the fixed BF16 GEMM ABI and launch configuration.
- Profiling methods intentionally raise `NotImplementedError`; lazy profiling is a later phase.
