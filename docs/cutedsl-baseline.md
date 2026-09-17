# CuTe DSL BF16 GEMM Baseline

This is an isolated post-Milestone A feasibility experiment. It adapts NVIDIA's
pinned CUTLASS 4.5.1 Hopper dense-GEMM example to attempt the fixed
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
  -t docker.io/saarora/gpu-kernel-mcts:cutedsl-feasibility-v1 \
  .

podman push docker.io/saarora/gpu-kernel-mcts:cutedsl-feasibility-v1
```

On an H100 host with container access to the GPU, run:

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  docker.io/saarora/gpu-kernel-mcts:cutedsl-feasibility-v1
```

Use the equivalent NVIDIA runtime flags if Docker rather than Podman manages
the host GPU.

The last stdout line is JSON. A successful result records the fixed CuTe DSL
configuration, correctness success, aggregate execution time, and the SHA-256
of the pinned NVIDIA example.

## Interpretation

The current result is deliberately marked
`comparable_to_repository_baselines=false`. The upstream example:

- generates different deterministic inputs from the CUDA C++ harness;
- checks `atol=0.02` and `rtol=0.001`, rather than the workload's separate
  `atol=rtol=0.02` contract;
- returns one aggregate execution time instead of all 30 CUDA-event samples.

Therefore this first run answers only whether the adapted BF16 Hopper kernel
JIT-compiles, passes its reference check, and executes. Do not use its timing
to claim a performance comparison with CUDA C++, CUTLASS C++, or cuBLAS.

The next implementation step is a shared evaluation adapter that gives the
CuTe DSL kernel the exact repository input, correctness, and per-sample timing
protocol. Only that result may be included in baseline comparisons.
