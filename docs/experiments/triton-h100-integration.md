# Triton H100 integration plan

This is the bounded Section 50 milestone. It exercises Triton source through
the existing H100 worker and MCTS pipeline before the MI355X Section 49 work.
The operation remains the repository's fixed 4096³ BF16 GEMM with row-major A/C,
column-major B, FP32 accumulation, and the existing tolerance. The packaged
Triton root and its same-worker timing define reward zero for this track.

## Implementation sequence

1. Package the Triton root and a deterministic distinct tile-size proposal.
2. JIT each candidate on the worker, save cubin/PTX, run correctness against
   PyTorch/cuBLAS, and time with CUDA events. Preserve source in state identity
   because a launcher/grid change can leave the cubin unchanged.
3. Select `triton_h100` through existing RunPod or Nebius worker provisioning.
   Keep the search tree policy, `B_gen`, node validity, and trace protocol.
4. Profile a selected valid node with NCU `lightweight_v1`; record Torch/Triton
   versions and the image digest. Keep profile replay time out of reward.
5. Build an image, run the deterministic one-generation smoke, then one small
   LLM search using `configs/triton_h100_strategies.json`. Inspect both traces
   before reusing the plumbing for MI355X.

## Validation commands

Build and publish a versioned image using `Dockerfile.triton-h100`. The GPU
commands below require a reachable H100, Nebius project/subnet IDs, and an
image in a registry the VM can pull. They provision a billable VM, so run only
when GPU validation is authorized.

```bash
docker build -f Dockerfile.triton-h100 -t REGISTRY/gpu-kernel-mcts:triton-h100-v1 .
docker push REGISTRY/gpu-kernel-mcts:triton-h100-v1

.venv/bin/python -m kernel_mcts.search_cli \
  --provider nebius --image REGISTRY/gpu-kernel-mcts:triton-h100-v1 \
  --nebius-project-id PROJECT_ID --nebius-subnet-id SUBNET_ID \
  --backend triton_h100 --generator smoke --generation-budget 1 \
  --trace triton-h100-smoke.sqlite --confirm-create-and-terminate

.venv/bin/python -m kernel_mcts.search_cli \
  --provider nebius --image REGISTRY/gpu-kernel-mcts:triton-h100-v1 \
  --nebius-project-id PROJECT_ID --nebius-subnet-id SUBNET_ID \
  --backend triton_h100 --generator openai --model MODEL \
  --strategies configs/triton_h100_strategies.json --generation-budget 3 \
  --trace triton-h100-bgen3.sqlite --confirm-create-and-terminate
```

The image recipe pins Torch 2.7.1 with CUDA 12.8 and CUTLASS 4.5.1; Torch
pins its Triton dependency. Pin the built image by digest for the actual
experiment. The local test suite uses a fake GPU process; it cannot validate
Triton lowering, H100 correctness, benchmark latency, or NCU counter access.
Read the published digest from the registry, for example with
`skopeo inspect --format '{{.Digest}}' docker://REGISTRY/gpu-kernel-mcts:TAG`;
Podman's local image digest can differ from the registry manifest digest.
The image includes Python development headers because Triton compiles a CUDA
helper against `Python.h` during its first JIT invocation.

## Exit evidence

Save the image digest, worker manifest, root/candidate correctness and timing,
cuBLAS/CUTLASS calibration, one valid generated node, an NCU profile, and the
SQLite trace. Do not use the MI355X PR kernel as this H100 root or infer AMD
performance from this track.
