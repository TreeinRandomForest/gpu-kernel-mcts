# CuTe DSL backend phase-2 validation

## Purpose

This experiment validated that the typed reference CuTe GEMM can pass through the
normal backend evaluator contract on H100 while reusing one cached JIT/evaluation
artifact.

## Result

- Date: 2026-09-19
- GPU: NVIDIA H100 80GB HBM3 SXM, compute capability 9.0
- Evaluation status: `VALID`
- Compile status: `SUCCESS`
- Correctness status: `PASS`
- Maximum and mean observed error: 0
- Warmups and measurements: 10 and 30
- Median latency: 195.568 us
- Mean latency: 196.090 us
- Population standard deviation: 2.497 us
- Observed range: 193.024-206.208 us
- Reward: 0, because the same cached reference measurement defined the smoke root

The serialized evaluation included the typed representation and schema version,
configuration hash, source hash, artifact fingerprint, state key, launch configuration,
worker/environment identity, complete timing samples, and before/after GPU telemetry.
The evaluator's compilation evidence reported `reused cached CuTe JIT artifact`,
confirming that the second logical evaluation did not launch the GPU again.

## Follow-up from the first report

The first report exposed only the near-zero cached lookup duration and omitted the
CuTe/CUTLASS and PyTorch library versions. Backend mode now records the initial
JIT/evaluation evidence separately from the cached evaluator lookup and enriches the
manifest with driver, CuTe/CUTLASS, and PyTorch versions. Image digest and repository
commit remain explicit build/run inputs; missing values remain null. A repeat smoke
run should verify these reporting additions before phase 3.
