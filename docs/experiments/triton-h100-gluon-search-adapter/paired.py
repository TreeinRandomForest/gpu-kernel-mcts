"""Same-worker CUDA-event crossover of one- and two-CTA Gluon candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import statistics
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.benchmarks.triton_bf16_gemm import load_triton_bf16_gemm_gluon_reference
from kernel_mcts.domain import KernelProgram
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend


ARTIFACT_ROOT = Path("/tmp/gluon-search-adapter-paired-artifacts")
ROUNDS = 5


def cublas_timing() -> dict:
    import torch

    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    timings = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return {"median_us": statistics.median(timings), "timings_us": timings,
            "warmup_count": 10, "stddev_us": statistics.pstdev(timings)}


def exact_source(path: Path) -> bool:
    import torch

    spec = importlib.util.spec_from_file_location("paired_exact_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    module.launch(a, b, c, 4096, 4096, 4096)
    return bool(torch.equal(c, torch.mm(a, b)))


def summarize(observations: list[dict], contender: str = "two_cta") -> dict:
    by_round = {}
    for row in observations:
        by_round.setdefault(row["round"], {})[row["variant"]] = row["benchmark"]["median_us"]
    if len(by_round) != ROUNDS or any(not {"single_cta", contender}.issubset(round_rows)
                                       for round_rows in by_round.values()):
        raise ValueError("incomplete paired observations")
    deltas = [by_round[index][contender] - by_round[index]["single_cta"]
              for index in range(ROUNDS)]
    return {
        "delta_definition": f"{contender}_minus_single_cta_us",
        "paired_delta_us": deltas,
        "median_paired_delta_us": statistics.median(deltas),
        "single_cta_median_of_round_medians_us": statistics.median(
            by_round[index]["single_cta"] for index in range(ROUNDS)),
        "contender_median_of_round_medians_us": statistics.median(
            by_round[index][contender] for index in range(ROUNDS)),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cta", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--include-cublas", action="store_true")
    args = parser.parse_args()
    backend = TritonH100Backend(TritonBackendConfig(artifact_root=ARTIFACT_ROOT))
    programs = {"single_cta": KernelProgram(args.single_cta.read_text(),
                                             backend="triton_h100")}
    paths = {"single_cta": args.single_cta}
    if args.variant:
        for item in args.variant:
            name, separator, path = item.partition("=")
            if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in programs:
                parser.error("--variant must be a unique NAME=PATH")
            programs[name] = KernelProgram(Path(path).read_text(), backend="triton_h100")
            paths[name] = Path(path)
    else:
        programs["two_cta"] = load_triton_bf16_gemm_gluon_reference()
    result = {
        "schema_version": 1, "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "rounds": ROUNDS, "candidates": {}, "observations": [], "complete": False,
    }
    import triton
    result["triton_version"] = triton.__version__
    try:
        artifacts = {}
        for name, program in programs.items():
            row = {"source_sha256": hashlib.sha256(program.source.encode()).hexdigest()}
            result["candidates"][name] = row
            compiled = backend.compile(program, BF16_GEMM_WORKLOAD)
            row["compile_success"] = compiled.success
            if not compiled.success or compiled.artifact is None:
                row["compile_stderr"] = compiled.stderr
                if name == "single_cta" or not args.variant:
                    raise RuntimeError(f"{name} compile failed")
                continue
            artifacts[name] = compiled.artifact
            row["kernel_name"] = compiled.artifact.kernel_name
            row["compiler_context"] = compiled.compiler_context
            correctness = backend.check_correctness(compiled.artifact, BF16_GEMM_WORKLOAD)
            row["correctness_success"] = correctness.success
            row["maximum_error"] = correctness.maximum_error
            if not correctness.success:
                if name == "single_cta" or not args.variant:
                    raise RuntimeError(f"{name} correctness failed")
                del artifacts[name]
                continue
            row["exact_vs_torch_mm"] = exact_source(paths[name])
            if not row["exact_vs_torch_mm"]:
                raise RuntimeError(f"{name} failed exact comparison")
        if len(artifacts) < 2:
            raise RuntimeError("no correct contender for paired timing")
        for round_index in range(ROUNDS):
            order = list(artifacts)
            if args.include_cublas:
                order.append("cublas")
            random.Random(431 + round_index).shuffle(order)
            for position, name in enumerate(order):
                if name == "cublas":
                    benchmark = cublas_timing()
                else:
                    timing = backend.benchmark(artifacts[name], BF16_GEMM_WORKLOAD)
                    benchmark = {"median_us": timing.median_us,
                                 "timings_us": list(timing.timings_us),
                                 "warmup_count": timing.warmup_count,
                                 "stddev_us": timing.stddev_us}
                observation = {
                    "round": round_index, "order": position, "variant": name,
                    "benchmark": benchmark,
                }
                result["observations"].append(observation)
                print(f"round {round_index} {name}: {benchmark['median_us']:.3f} us", flush=True)
        result["summary"] = (
            {name: summarize(result["observations"], contender=name)
             for name in order if name != "single_cta"}
            if args.variant or args.include_cublas else summarize(result["observations"])
        )
        result["complete"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"[-2000:]
        raise
    finally:
        print("RESULT_JSON=" + json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
