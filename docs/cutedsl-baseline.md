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
definitions include their typed parameters, assignments, branch conditions, and
return expressions. Pipeline evidence also records assignments to stage count,
shared-memory-capacity, and occupancy state. Docstrings are excluded from the
identifier inventory. Discovery is deliberately
reported as `evidence_only`: a name appearing in the pinned implementation does not
yet establish that it is an independent, legal, or useful MCTS mutation. The report
is the input to selecting and validating the next structural control.

WGMMA and warp-specialization evidence likewise records targeted assignments for MMA
instruction shapes, K tiling, warp-group counts, thread layouts, and tiled-MMA
construction. This diagnostic precedes any WGMMA representation field: it is used to
identify one bounded control and its tile/layout coupling without treating internal
implementation variables as independently mutable.

The first bounded WGMMA template control compares the pinned automatic decomposition
with `single_warp_group`. For the `(128,256)` root, the pinned example chooses two
128-thread warp groups to reduce register spilling; the alternative forces
`atom_layout_mnk=(1,1,1)` and a 128-thread CTA while leaving tile shape, pipeline,
layouts, and epilogue unchanged. Evaluate it standalone with
`--mode backend --wgmma-configuration single_warp_group`. It is not an MCTS strategy
or mutation until H100 JIT, correctness, artifact identity, and timing are validated.
The guarded v10 evaluation completed that validation: both configurations passed
exact correctness and produced distinct runtime artifacts, but
`single_warp_group` regressed from `192.976 us` to `3015.728 us`. This is useful
negative evidence, so the configuration remains representable for diagnostics but is
not part of the default MCTS mutation neighborhood.

The next pre-search diagnostic tested the pinned mainloop's `k_pipe_mmas` value as
`wgmma_inflight_groups={1,2}`. This controls how many WGMMA groups may remain in
flight before `warpgroup.wait_group`; it does not change the number of CTA warp
groups. Value 1 preserves the pinned source, while value 2 is applied by a
syntax-aware transformation guarded by the exact pinned example SHA-256. The adapter
refuses a changed source hash or an ambiguous assignment instead of silently patching
a different implementation. This diagnostic value is deliberately absent from the
canonical `CuteGemmProgram` representation.

Evaluate the new control outside MCTS with:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode wgmma-inflight-diagnostic \
  --wgmma-inflight-groups 2 \
  --output /output/cutedsl-wgmma-inflight-2.json
```

The guarded v11 validation evaluated values 1 and 2 with both the pinned pipeline
heuristic and explicit stage 3. All four configurations passed exact correctness.
The measured medians were `193.504 us` and `192.176 us` under the heuristic, and
`192.512 us` and `191.936 us` with stage 3. Despite the different generated-source
and configuration hashes, values 1 and 2 produced identical normalized compiler IR
and fatbin hashes within each pipeline setting. The apparent 0.3–0.7% differences are
therefore timing noise rather than evidence of distinct kernels.

This control is rejected as an independent search dimension and must not become an
MCTS strategy or mutation. Keeping it in canonical search state would assign
different configuration identities to the same effective compiled kernel, contrary
to transposition semantics. It has therefore been removed from the typed state and
schema; the guarded transformer remains only in standalone diagnostic tooling. See
[CuTe WGMMA in-flight validation v11](experiments/cutedsl-wgmma-inflight-v11.md).

### TMA load-policy diagnostic

The pinned implementation derives TMA tensor layouts from its shared-memory layouts;
it does not expose an independent bounded TMA-layout enum. Its clustered input loads
do expose one concrete copy-policy decision: when a tile is shared across CTAs, use a
multicast TMA load or let each CTA issue a non-multicast load. The first TMA diagnostic
holds the CTA tile at `(128,256)` and cluster at `(2,1)`. Under that cluster, the
pinned policy multicasts B across two CTAs while A remains a single-CTA load.

`auto_multicast` preserves the official implementation. `non_multicast` wraps only
the pinned `_make_tma_atoms_and_tensors` helper and forces its multicast dimension to
one for both inputs. The alternative may fail JIT or launch if downstream partition
or copy semantics require multicast; such a failure is valid diagnostic evidence,
not a reason to weaken correctness checks. The override is guarded by the exact
pinned example SHA-256 and is absent from canonical `CuteGemmProgram` state.

Run both standalone points on the same H100 worker:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode tma-copy-diagnostic \
  --tma-load-policy auto_multicast \
  --output /output/cutedsl-tma-auto-v12.json

python -m kernel_mcts.cute_baseline_cli \
  --mode tma-copy-diagnostic \
  --tma-load-policy non_multicast \
  --output /output/cutedsl-tma-non-multicast-v12.json
```

Each report uses the repository input, correctness, and timing contract and captures
JIT diagnostics. Compare correctness, normalized compiler IR, embedded fatbin hash,
and timing before deciding whether this is a real searchable dimension. It remains
outside MCTS and consumes no `B_mut`, `B_gen`, or `B_tune` during this validation.

The guarded v12 H100 experiment rejected this as a bounded independent control. The
pinned `auto_multicast` configuration passed exact correctness with zero observed
error and measured `187.216 us` median. Its normalized compiler-IR hash was
`671224e5...` and its embedded fatbin hash was `4be702f4...`. The
`non_multicast` override reached kernel launch but CUDA reported an illegal memory
access at synchronization, so it produced no valid benchmark or artifact report.

Changing only TMA atom construction is insufficient: clustered TMA partitioning,
CTA coordinates, copy masks, and data ownership still encode the multicast policy.
This failure is candidate invalidity rather than an infrastructure outage. Do not add
`tma_load_policy` to canonical state or MCTS. Any future non-multicast experiment must
be a coordinated structural transformation of all coupled components. See
[CuTe TMA load-policy validation v12](experiments/cutedsl-tma-load-policy-v12.md).

### Epilogue-stage diagnostic

The pinned `_compute_stages` heuristic returns both the A/B mainloop depth and a
hardcoded epilogue depth of 4. The next pre-search diagnostic holds the validated
tile `(128,256)` and cluster `(2,1)` fixed, preserves the pinned A/B depth, and tests
epilogue depths 2, 3, and 4. This first bounded diagnostic excludes single buffering
because it would change producer/consumer hazard assumptions in addition to depth;
that exclusion is not a universal legality conclusion for future CuTe kernels.

Run each point in a separate container on the same H100 worker, changing the stage
and output filename:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode epilogue-stage-diagnostic \
  --epilogue-stages 2 \
  --output /output/cutedsl-epilogue-stage2-v13.json
```

The diagnostic mode requires the exact pinned example SHA-256 and marks its output
noncanonical. Successful reports include correctness, timing, normalized IR, and
runtime artifacts. JIT, launch, or evaluation exceptions are serialized as
`diagnostic_failed` reports with the requested control and error rather than being
silently lost. No point consumes `B_mut`, `B_gen`, or `B_tune`.

The guarded v13 H100 run validated all three depths with zero observed correctness
error. Stages 2, 3, and pinned 4 measured `190.000`, `189.728`, and `188.448 us`
median respectively. Every depth produced a distinct normalized compiler-IR hash and
embedded fatbin hash. The differences are below 1% and overlap run-level timing
variation, so the experiment establishes a real structural dimension rather than a
performance improvement.

Epilogue depth is promoted in `CuteGemmProgram` schema v2. Explicit values 2 and 3
are initially legal only for the validated tile `(128,256)`, cluster `(2,1)` and are
available through `change_epilogue_stages`. `None` means the pinned depth 4. Explicit
4 is deliberately excluded because it would give the same effective kernel a second
canonical identity. See
[CuTe epilogue-stage validation v13](experiments/cutedsl-epilogue-stages-v13.md).

Validate a promoted state through the canonical `CuTeDSLBackend` path with all four
schedule components supplied together:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode backend \
  --tile-m 128 \
  --tile-n 256 \
  --cluster-m 2 \
  --cluster-n 1 \
  --epilogue-stages 2 \
  --output /output/cutedsl-epilogue-canonical-stage2-v14.json
```

Unlike the pre-search diagnostic, this command constructs schema-v2 canonical state,
renders it deterministically, parses it in the backend subprocess, and evaluates it
through the normal compile/correctness/benchmark cache. Schedule arguments are
accepted only by backend modes and must be supplied as a complete tuple.

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
configuration hash, preserving transpositions. TMA-layout, WGMMA-layout, and
shared-memory mutations remain rejected until the renderer can express and validate
them. Epilogue mutations are available only after reaching their validated schedule.
The neighborhood is connected to core MCTS with separate `B_mut`
accounting. Remote orchestration is wired, while H100 search validation remains
pending.

The renderer now has a pre-search pipeline feasibility control. `pipeline_stages=None`
preserves the pinned `_compute_stages` heuristic. Explicit values `2`, `3`, and `4`
override only the A/B mainloop stage count while retaining the pinned epilogue stage
calculation. Explicit values 2 and 3 are available as typed mutations after corrected
H100 validation. Explicit 4 is diagnostic-only because it aliases the heuristic for
the reference tile.

Run each feasibility point with the backend evaluation contract, changing only the
stage value and output filename:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode backend \
  --pipeline-stages 2 \
  --output /output/cutedsl-pipeline-stage-2.json
```

The initial H100 feasibility sweep appeared to pass JIT and exact correctness for all
three values, reporting medians of `190.384 us` (2), `189.936 us` (3), and
`190.224 us` (4). The later mutation trace proved that the backend subprocess had
ignored the pipeline field, so these were repeated root measurements rather than
pipeline results. They must not be used as pipeline-performance evidence. Stages 2
and 3 remain in the deterministic mutation neighborhood for corrected validation;
stage 4 is omitted to avoid duplicating the heuristic root's effective configuration.

The first `B_mut=6` guarded search revealed that the backend subprocess initially
forwarded only the tile/cluster schedule and ignored the typed pipeline field. The
pipeline proposals therefore compiled the root binary and transposed to the root;
their timing differences were measurement noise, not pipeline evidence. The adapter
now forwards `pipeline_stages` and applies the bounded override inside the subprocess;
the corrected results below supersede that first run.

The corrected v7 feasibility sweep produced distinct runtime and normalized-IR
hashes for every stage count and passed exact correctness. Median times were
`299.136 us` (2), `191.792 us` (3), and `190.624 us` (4). The corrected
`B_mut=6` search created a distinct stage-2 node at `297.392 us`; stage 3 was not
selected within that search budget. Cluster `(2,1)` remained best at `185.344 us`
(`1.0357x` root speedup). See
[CuTe pipeline mutation validation](experiments/cutedsl-pipeline-bmut6-v7.md).

Run the bounded cluster-by-pipeline interaction sweep outside MCTS with:

```bash
python -m kernel_mcts.cute_baseline_cli \
  --mode pipeline-tune \
  --output /output/cutedsl-pipeline-interactions.json
```

This holds the CTA tile at `(128,256)` and evaluates clusters `(1,1)`, `(1,2)`, and
`(2,1)` crossed with the pinned heuristic and explicit mainloop stages 2 and 3. All
nine legal trials consume one `B_tune` each, including failed JIT, launch, correctness,
or benchmark attempts. The experiment does not alter MCTS state or re-root search.

The guarded v8 sweep completed all nine valid trials. The heuristic pipeline with
cluster `(2,1)` won at `184.896 us`, reproducing the earlier independent `184.880 us`
measurement. It was `1.0297x` faster than the default and 4.88% slower than same-run
cuBLAS (`176.288 us`). Explicit stage 2 was consistently much slower (`297–308 us`),
while stage 3 was closest with cluster `(2,1)` at `187.040 us` but did not beat the
heuristic. See [CuTe pipeline interaction sweep](experiments/cutedsl-pipeline-interactions-v8.md).

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
