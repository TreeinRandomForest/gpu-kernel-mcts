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

Then run the complete suite on the same revision:

```bash
.venv/bin/python -m pytest -q
```

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
