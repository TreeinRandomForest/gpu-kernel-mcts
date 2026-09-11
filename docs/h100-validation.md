# H100 BF16 GEMM validation

This procedure validates the packaged root kernel and CUDA harness on the hardware
target. Run it on an H100 SXM worker, not on an arbitrary CUDA machine.

## Prerequisites

- NVIDIA H100 SXM with compute capability 9.0;
- a compatible NVIDIA driver;
- CUDA toolkit with `nvcc`, CUDA headers, and the cuBLAS development library;
- `cuobjdump` for preferred SASS fingerprints (optional; executable hashing is the fallback);
- the project virtual environment with test dependencies installed.

Do not print or record `RUNPOD_API_KEY` while collecting validation evidence.

## Confirm the environment

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvcc --version
cuobjdump --version
```

Confirm that the reported GPU model and form factor match the requested H100 SXM
environment manifest before running benchmarks.

## Run the hardware test

Build and push the worker image to a registry accessible by RunPod:

```bash
docker build -f Dockerfile.worker -t REGISTRY/gpu-kernel-mcts:REVISION .
docker push REGISTRY/gpu-kernel-mcts:REVISION
```

The image tag should be immutable for a recorded experiment. After pushing, the
guarded lifecycle probe can provision an H100, wait for the authenticated worker to
finish root calibration, and terminate it:

```bash
.venv/bin/python -m kernel_mcts.runpod_cli \
  --image REGISTRY/gpu-kernel-mcts:REVISION \
  --network-volume-id RUNPOD_NETWORK_VOLUME_ID \
  --data-center-id MATCHING_RUNPOD_DATA_CENTER_ID \
  --confirm-create-and-terminate
```

Alternatively, discover current H100 availability and reuse the managed
`gpu-kernel-mcts` volume where possible:

```bash
.venv/bin/python -m kernel_mcts.runpod_cli \
  --image REGISTRY/gpu-kernel-mcts:REVISION \
  --auto-volume \
  --preferred-data-center-id RUNPOD_DATA_CENTER_ID \
  --confirm-create-and-terminate
```

If no reusable managed volume exists, the command stops before creating storage.
To permit creation of a persistent 50 GB volume, add
`--confirm-create-volume`. Volume creation and storage billing are independent of
pod termination. The optional `--preferred-data-center-id` restricts both managed
volume reuse and creation to an available data center selected by the operator.

### Volume-selection rules

Every RunPod command must use exactly one volume-selection mode:

1. **Manual:** provide both `--network-volume-id` and `--data-center-id`. They
   must refer to the same RunPod data center.
2. **Automatic:** provide `--auto-volume`. Do not combine it with either manual
   identifier.

`--preferred-data-center-id` is valid only with `--auto-volume`; it restricts
automatic selection to that data center, which must currently report the requested
GPU as available.

For `kernel_mcts.runpod_cli`, automatic mode reuses a suitable managed volume when
one exists. If none exists, the command fails before creating a pod unless
`--confirm-create-volume` is also supplied. That separate flag authorizes creation
of persistent, billable storage.

For `kernel_mcts.search_cli`, automatic mode is always reuse-only. The search CLI
does not create network volumes and fails before creating a pod if no suitable
managed volume exists.

This command creates a billable pod. Its `finally` path requests termination, but
you should also confirm deletion in the RunPod console after any interrupted or
failed probe. The network volume must already exist in the specified data center.
Pod termination does not delete the network volume, which may continue to incur
storage charges.

To run the test directly from a checkout already present on an H100 worker:

From the repository root:

```bash
RUN_H100_INTEGRATION=1 .venv/bin/python -m pytest \
  tests/test_cuda_backend.py::test_packaged_root_on_h100 -vv -s
```

The test must:

1. compile the packaged root and harness with `nvcc -arch=sm_90`;
2. execute deterministic correctness comparison against cuBLAS;
3. satisfy the workload's BF16 tolerances;
4. complete warmups and all timing repetitions;
5. return a positive median latency.

The guarded worker probe additionally checks and reports fixed cuBLAS and CUTLASS
BF16 baselines. Both must pass correctness under the same workload contract and
produce the configured number of timing samples.

## Run one MCTS smoke iteration

After lifecycle calibration succeeds, run one deterministic PUCT, progressive-
widening, evaluation, and backup cycle on the same worker architecture:

```bash
.venv/bin/python -m kernel_mcts.search_cli \
  --image REGISTRY/gpu-kernel-mcts:REVISION \
  --trace smoke-search.sqlite \
  --auto-volume \
  --preferred-data-center-id RUNPOD_DATA_CENTER_ID \
  --confirm-create-and-terminate
```

The smoke generator makes no LLM call and is restricted to `B_gen=1`. It emits a
packaged, correctness-first CUDA implementation that is distinct from the root.
The command records the root, generation, evaluation, node/edge statistics, backup,
and final result in the local SQLite file, then terminates the worker in a `finally`
path. It does not create a network volume.

## Evaluate one generated candidate

Before connecting an LLM generator to MCTS, compile and correctness-test one reviewed
candidate through the production worker evaluation path:

```bash
.venv/bin/python -m kernel_mcts.evaluate_cli \
  --image REGISTRY/gpu-kernel-mcts:REVISION \
  --candidate candidate.cu \
  --report candidate-evaluation.json \
  --auto-volume \
  --preferred-data-center-id RUNPOD_DATA_CENTER_ID \
  --confirm-create-and-terminate
```

The command is reuse-only for network volumes and refuses to overwrite the report.
It evaluates the root and candidate on one worker, always terminates the pod, and
returns exit status `0` only when the candidate compiles, passes correctness, and
completes benchmarking. Invalid candidates return status `1` after writing their
compile/correctness evidence. Review compiler diagnostics in the JSON report rather
than weakening the workload contract or numerical tolerances.

## Run an OpenAI-backed MCTS search

After the guarded generation and candidate evaluation checks pass, run a small global
MCTS search with configured semantic strategies and uniform priors:

```bash
.venv/bin/python -m kernel_mcts.search_cli \
  --image REGISTRY/gpu-kernel-mcts:REVISION \
  --trace llm-search.sqlite \
  --generator openai \
  --model MODEL_ID \
  --strategies configs/strategies.yaml \
  --generation-budget 3 \
  --max-repairs 1 \
  --max-infrastructure-retries 1 \
  --best-output best-kernel.cu \
  --auto-volume \
  --preferred-data-center-id RUNPOD_DATA_CENTER_ID \
  --confirm-create-and-terminate
```

This command can incur both OpenAI API and RunPod charges. The local controller makes
fresh model requests and sends only generated kernel/evaluation data to the worker;
neither API key is sent to the other service. Start with a small `B_gen`, inspect the
SQLite trace and exported best kernel, and increase the budget only deliberately.
Root infrastructure failures are retried idempotently according to
`--max-infrastructure-retries`; each attempt is retained in the SQLite event trace and
does not consume generation budget.

## Visualize a search trace

Render the latest run in a trace database as a Graphviz DAG:

```bash
.venv/bin/python -m kernel_mcts.trace_viz \
  --trace llm-search.sqlite \
  --dot search.dot \
  --png search.png
```

PNG output requires the Graphviz `dot` executable on `PATH`; DOT output does not need
an additional Python dependency. Use `--run-id RUN_ID` when the database contains
multiple runs. Large graphs can be restricted with `--max-depth DEPTH` or
`--min-edge-visits VISITS`. Existing outputs are never overwritten unless `--force`
is supplied.

Program nodes show reward, linear speedup, median latency, and aggregate action visits.
Diamond nodes represent semantic strategies selected by PUCT, and their outgoing
realization edges show UCB statistics. The renderer preserves transpositions as shared
program nodes and includes invalid and infrastructure-failure proposals without
embedding generated source, prompts, or profiler payloads.

Then run the complete suite on the same revision:

```bash
.venv/bin/python -m pytest -q
```

## Recorded validation

On 2026-09-11, the remote lifecycle validation completed on an NVIDIA H100 80GB
HBM3 SXM worker with compute capability 9.0. For the fixed 4096 x 4096 x 4096 BF16
GEMM workload, the recorded results were:

- packaged root: correctness passed, maximum error `0.03125`, mean error
  `4.04688344e-06`, median `28468.4951 us` over 30 samples;
- cuBLAS baseline: correctness passed with maximum error `0.0`, median
  `185.376007 us` over 30 samples;
- fixed untuned CUTLASS baseline: correctness passed with maximum error `0.0`,
  median `312.7519835 us` over 30 samples.

A guarded one-generation smoke search also completed, producing two valid search
nodes and a durable local SQLite trace. That run exposed a controller-level reference
issue: the root had been scored against an earlier startup calibration rather than
against itself. Orchestration now fixes the newly measured root benchmark for the run,
assigns the root reward exactly `0.0`, and recomputes every valid candidate's reward
against that same benchmark. This correction has automated coverage; the remote smoke
command should be rerun before recording post-fix search rewards.

The guarded OpenAI generation and candidate-evaluation sequence also completed on
2026-09-11. `gpt-5.6-terra` generated a global-memory-coalescing candidate using
`1155` input tokens and `1437` output tokens. On H100, both root and candidate passed
compilation and correctness with maximum error `0.03125` and mean error
`4.04688344e-06`. The freshly measured root median was `28464.76755 us`; the
candidate median was `21431.792 us`, producing root-normalized reward
`0.2837916691`, or a `1.328x` speedup (`24.7%` lower latency). This remains far below
the cuBLAS performance reference and is evidence of one successful transformation,
not a state-of-the-art result.

## Record the result

Record the following in the run trace or review notes:

- git revision and dirty-tree state;
- environment-manifest ID and worker ID;
- GPU model, UUID, form factor, and driver version;
- CUDA, NVCC, cuBLAS, and cuobjdump versions;
- container image tag and digest;
- correctness maximum and mean error;
- individual timing samples and summary statistics;
- binary fingerprint.

If compilation, correctness, or timing fails, retain the structured result and
classify it according to the evaluator rules. Do not weaken tolerances or silently
change the workload to make the test pass; report any implementation/spec mismatch.
