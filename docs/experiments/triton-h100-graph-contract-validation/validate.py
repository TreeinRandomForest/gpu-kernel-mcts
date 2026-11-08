"""Validate the updated Triton backend timing contract on an H100 worker."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram
from kernel_mcts.serialization import serialize_benchmark
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend
from kernel_mcts.triton_harness import GRAPH_TIMING_CONTRACT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"complete": False, "timing_contract": GRAPH_TIMING_CONTRACT,
              "candidates": {}}
    try:
        backend = TritonH100Backend(TritonBackendConfig(
            artifact_root=Path("/tmp/graph-contract-validation-artifacts")))
        for name, path in (("single_cta", args.single),
                           ("persistent", args.persistent)):
            program = KernelProgram(path.read_text(), backend="triton_h100")
            compiled = backend.compile(program, BF16_GEMM_WORKLOAD)
            if not compiled.success or compiled.artifact is None:
                raise RuntimeError(f"{name} compile failed: {compiled.stderr[-800:]}")
            correct = backend.check_correctness(compiled.artifact, BF16_GEMM_WORKLOAD)
            if not correct.success:
                raise RuntimeError(f"{name} correctness failed")
            benchmark = backend.benchmark(compiled.artifact, BF16_GEMM_WORKLOAD)
            report["candidates"][name] = {
                "artifact_id": compiled.artifact.artifact_id,
                "correctness": correct.success,
                "benchmark": serialize_benchmark(benchmark),
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(name, benchmark.median_us,
                  benchmark.diagnostics["ordinary_launch_median_us"], flush=True)
        report["complete"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1200:],
                           "traceback_tail": traceback.format_exc()[-2400:]}
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
