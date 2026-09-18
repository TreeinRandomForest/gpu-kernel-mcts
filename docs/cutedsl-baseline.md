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
