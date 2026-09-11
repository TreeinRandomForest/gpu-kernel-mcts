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
