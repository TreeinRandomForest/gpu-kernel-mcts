# CuTe DSL BF16 GEMM Baseline

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

The normal search-worker image remains unchanged. Build the separate image with:

```bash
podman build \
  -f Dockerfile.cutedsl \
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
