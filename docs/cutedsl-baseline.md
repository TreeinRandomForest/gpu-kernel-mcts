# CuTe DSL BF16 GEMM Baseline

See [CuTe DSL Search Model](cutedsl-search-model.md) for how typed mutations, LLM
realizations, MCTS selection, budgets, and the cuBLAS baseline fit together.

This is an isolated post-Milestone A baseline. It adapts NVIDIA's pinned
CUTLASS 4.5.1 Hopper dense-GEMM example to the fixed
`bf16_gemm_4096_h100` shape and data types:

```text
M = N = K = 4096
A/B/C = BF16
accumulation = FP32
tile = 128 x 256
cluster = 1 x 1
warmups = 10
iterations = 30
```

Build the separate CuTe image with:

```bash
podman build \
  -f Dockerfile.cutedsl \
  --build-arg KERNEL_MCTS_GIT_COMMIT="$(git rev-parse HEAD)" \
  --build-arg KERNEL_MCTS_DIRTY_TREE="$(test -n "$(git status --porcelain)" && echo 1 || echo 0)" \
  -t docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v4 \
  .

podman push docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v4
```

On an H100 host with container access to the GPU, run:

```bash
mkdir -p cutedsl-output
podman run --rm \
  --device nvidia.com/gpu=all \
  -e KERNEL_MCTS_CONTAINER_IMAGE=docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v4 \
  -v "$PWD/cutedsl-output:/output" \
  docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v4 \
  --output /output/comparison.json
```

Use the equivalent NVIDIA runtime flags if Docker rather than Podman manages
the host GPU.

The default `comparison` mode runs cuBLAS, fixed CUTLASS, and CuTe DSL in that
order inside one container. It records one environment manifest and reports
median latency ratios against the same-run cuBLAS result. The CuTe evaluation
uses a small C++ input generator to reproduce the
existing `std::mt19937` BF16 input stream, calls the same `cublasGemmEx`
reference contract, applies `atol=rtol=0.02`, and records 30 separate CUDA-event
timing samples in one continuous sequence after 10 warmups. The last stdout line
is JSON containing the
fixed configuration, full correctness statistics, timing samples and summary,
and the SHA-256 of the pinned NVIDIA example.

## Interpretation

The comparison aborts rather than reporting ratios if any implementation fails
correctness or the CuTe result does not complete the comparable contract.

## Initial typed schedule space

The standalone CuTe tuning experiment represents backend configuration with a
typed `CuteSchedule`; it does not rewrite source code or create MCTS nodes. The
initial finite space contains three CTA tiles and three cluster shapes:

```text
CTA tiles:      (64, 128), (128, 128), (128, 256)
cluster shapes: (1, 1), (1, 2), (2, 1)
```

The existing `(128, 256)` CTA tile with a `(1, 1)` cluster is the default. A
schedule is rejected before JIT compilation if it is outside the declared space
or does not cover the fixed 4096-by-4096 output tile grid cleanly. Each legal
schedule has deterministic JSON serialization and a stable configuration ID.

This first space deliberately does not expose WGMMA atoms, pipeline stages, TMA
layouts, shared-memory swizzles, or epilogue policy. Those parameters should be
added only after confirming how the pinned NVIDIA example constrains them.
The pinned implementation rejects CTA tile-M values other than 64 and 128, so
the previously considered `(256, 128)` tile is excluded statically and does not
consume `B_tune`.

Run the complete deterministic grid on one worker with:

```bash
sudo docker run --rm --gpus all \
  -e KERNEL_MCTS_CONTAINER_IMAGE=docker.io/USER/gpu-kernel-mcts:REVISION \
  -v "$PWD/cutedsl-output:/output" \
  docker.io/USER/gpu-kernel-mcts:REVISION \
  --mode tune \
  --output /output/cutedsl-tuning.json
```

Every legal schedule that reaches JIT/evaluation consumes one unit of `B_tune`,
including failed attempts. Static rejections are retained but do not consume the
budget. The JSON report contains every complete valid result or failure, the
environment manifest, the best schedule, the default schedule result, and latency
ratios against the same-worker cuBLAS result. These trials remain outside MCTS and
do not consume `B_gen`.

## Backend evaluation smoke test

Milestone B phase 2 evaluates the deterministically rendered reference through the
normal `BackendKernelEvaluator` contract. Rebuild the CuTe image, then run on H100:

```bash
sudo docker run --rm --gpus all \
  --entrypoint python \
  -e KERNEL_MCTS_CONTAINER_IMAGE=docker.io/USER/gpu-kernel-mcts:REVISION \
  -e KERNEL_MCTS_CONTAINER_DIGEST=sha256:IMAGE_MANIFEST_DIGEST \
  -v "$PWD/cutedsl-output:/output" \
  docker.io/USER/gpu-kernel-mcts:REVISION \
  -m kernel_mcts.cute_baseline_cli \
  --mode backend \
  --output /output/cutedsl-backend-evaluation.json
```

The backend verifies that source exactly matches the deterministic rendering of its
embedded typed representation, then runs the pinned JIT/correctness/benchmark harness
once in a bounded subprocess. Correctness and benchmark stages reuse that cached
artifact result; evaluating the same rendered state again does not launch the GPU.
The report is a normal serialized `EvaluationResult` with source hash, state key,
artifact fingerprint, complete timings, telemetry, environment identity, and typed
representation metadata.

Pass the registry manifest digest (the `sha256:...` portion of the pulled image's
RepoDigest) through `KERNEL_MCTS_CONTAINER_DIGEST`. The Git commit and dirty-tree
state are embedded by the documented build arguments. If these values are not
provided, the manifest records them as unavailable rather than inventing provenance.

If runtime JIT diagnostics are unavailable, the backend fingerprint falls back to the
rendered source, canonical representation, pinned example hash, and launch
configuration. Phase 3 prefers an exposed cubin, fatbin, SASS, or PTX hash, followed
by normalized MLIR, and combines that artifact identity with launch configuration.

## Phase-3 artifact diagnostic

Before selecting a cubin/SASS extraction method or an NCU kernel filter, run one
instrumented JIT to observe what the pinned CuTe runtime actually materializes:

```bash
sudo docker run --rm --gpus all \
  --entrypoint python \
  -v "$PWD/cutedsl-output:/output" \
  docker.io/USER/gpu-kernel-mcts:REVISION \
  -m kernel_mcts.cute_baseline_cli \
  --mode diagnostic \
  --output /output/cutedsl-artifact-diagnostic.json
```

The diagnostic snapshots standard CUDA/CuTe cache locations and temporary roots
before and after JIT, hashes created or modified files, ranks cubin/fatbin/SASS/PTX
fingerprint candidates, records newly loaded CUDA-related host modules, and captures
bounded metadata about the actual benchmark callable and workspace. The H100
diagnostic found that the pinned runtime retains its JIT module in memory rather than
materializing a new cache file. Its MLIR contains an embedded CUDA fatbinary, so the
production backend extracts and hashes that payload directly. It also records exposed
kernel names and normalized MLIR hash and text (text is omitted above a bounded size);
normalized MLIR remains the fallback when no embedded binary is available.

Validate the dedicated one-launch NCU path on H100 with:

```bash
sudo docker run --rm --gpus all \
  --entrypoint python \
  -v "$PWD/cutedsl-output:/output" \
  docker.io/USER/gpu-kernel-mcts:REVISION \
  -m kernel_mcts.cute_baseline_cli \
  --mode backend-profile \
  --profile-set lightweight_v1 \
  --output /output/cutedsl-backend-profile.json
```

This mode first performs the ordinary unprofiled correctness and CUDA-event benchmark,
then launches a separate subprocess under NCU. That subprocess skips correctness and
timing and invokes the selected generated kernel exactly once. The resulting profile
is cached by artifact and metric-set identity and does not change the benchmark or
reward. The guarded H100 run confirmed the exact kernel-name filter and captured all
nine lightweight metrics while leaving the root reward at zero.

## Initial typed-mutation neighborhood

Before expanding that neighborhood, inspect the pinned CUTLASS 4.5.1 Python source
without importing it or requiring a GPU:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode structural-capabilities \
  --example /opt/cutlass/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py \
  --output cutedsl-structural-capabilities.json
```

The report records the exact source hash, the `run()` and
`HopperWgmmaGemmKernel.__init__()` signatures, exposed CLI options, validation
assertions and exceptions, and bounded executable-source evidence for pipeline,
WGMMA, TMA, epilogue, warp-specialization, and scheduling controls. Matching method
definitions include their typed parameters and return expressions; docstrings are
excluded from the identifier inventory. Discovery is deliberately
reported as `evidence_only`: a name appearing in the pinned implementation does not
yet establish that it is an independent, legal, or useful MCTS mutation. The report
is the input to selecting and validating the next structural control.

The first GPU-free structural neighborhood exposes only controls already validated by
the pinned renderer: CTA tile and cluster shape. Inspect it locally with:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode design-space \
  --output cutedsl-design-space.json
```

The report contains four one-hop proposals from the reference configuration. Each
records its canonical candidate representation, static-legality result, parent and
child configuration hashes, changed fields, and `typed_mutation` mechanism. These
proposals perform no JIT or GPU work and consume neither `B_gen` nor `B_tune`.
Different mutation orders that reach the same representation produce the same
configuration hash, preserving transpositions. Pipeline, TMA-layout, WGMMA-layout,
shared-memory, and epilogue mutations remain rejected until the renderer can express
and validate them. The neighborhood is connected to core MCTS with separate `B_mut`
accounting. Remote orchestration is wired, while H100 search validation remains
pending.

## Guarded remote mutation search

The CuTe image now dispatches to the authenticated worker service when a provider
supplies `KERNEL_MCTS_WORKER_TOKEN`; without that token it retains the standalone CLI
used by the commands above. The search CLI selects the CuTe backend explicitly and
passes that choice to either RunPod or Nebius.

Run the smallest mutation-only Nebius validation with:

```bash
.venv/bin/python -m kernel_mcts.search_cli \
  --provider nebius \
  --image docker.io/USER/gpu-kernel-mcts:CUTE_REVISION \
  --trace cutedsl-mutation-smoke.sqlite \
  --backend cute_dsl \
  --generator cute-mutation \
  --generation-budget 0 \
  --mutation-budget 1 \
  --nebius-project-id PROJECT_ID \
  --nebius-subnet-id SUBNET_ID \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --best-output cutedsl-mutation-best.py \
  --confirm-create-and-terminate
```

After that guarded run succeeds, enable mutation-first LLM fallback with
`--generator cute-mixed`, a positive `--generation-budget`, a positive
`--mutation-budget`, and the usual `--model`. The initial mixed run uses the two
built-in typed CuTe strategies, so it does not accept a separate strategy file.
The mutation-only command completed successfully on Nebius H100 SXM on 2026-09-20.
See [CuTe mutation smoke validation](experiments/cutedsl-mutation-smoke-v1.md) for the
exact transition, timings, budgets, and provenance limitations. Mixed LLM-fallback
validation remains pending.

A subsequent four-proposal mutation-only search selected both cluster alternatives,
one slower CTA tile, and one inverse mutation that transposed back to the cached root.
It recovered the standalone tuner's `(128, 256)` tile with `(2, 1)` cluster as the
best schedule. See [CuTe mutation search B_mut=4](experiments/cutedsl-mutation-bmut4-v1.md).

The first `B_mut=4`, `B_gen=1` mixed run correctly fell back to the LLM after four
mutations. The model selected the existing best schedule but returned noncanonical
Python, exposing a representation/rendering boundary. The controller now requests a
typed JSON representation and renders trusted canonical source locally. See
[CuTe mixed search B_mut=4, B_gen=1](experiments/cutedsl-mixed-bmut4-bgen1-v1.md).
The corrected rerun passed on H100: the LLM-generated JSON canonicalized to the
existing best node and reused its cached evaluation.

The original feasibility mode remains available for diagnosing adapter failures:

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  docker.io/saarora/gpu-kernel-mcts:cutedsl-baseline-v4 \
  --mode feasibility
```

Its result is marked `comparable_to_repository_baselines=false` because the
unmodified upstream harness:

- generates different deterministic inputs from the CUDA C++ harness;
- checks `atol=0.02` and `rtol=0.001`, rather than the workload's separate
  `atol=rtol=0.02` contract;
- returns one aggregate execution time instead of all 30 CUDA-event samples.

Do not use a feasibility-mode timing to claim a performance comparison with
CUDA C++, CUTLASS C++, or cuBLAS.
