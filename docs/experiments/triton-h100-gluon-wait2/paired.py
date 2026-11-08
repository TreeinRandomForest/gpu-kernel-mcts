"""Matched wait-two versus wait-one timing and repeated correctness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import traceback
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ("candidate", "control", "fast-b", "paired", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "schema_version": 1, "observations": [], "orders": {},
              "source_sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                for name in ("candidate", "control", "fast_b", "paired")}}
    save(args.output, report)
    try:
        import torch
        candidate = load(args.candidate, "prefetch3_candidate")
        control = load(args.control, "prefetch3_control")
        fast_b = load(args.fast_b, "prefetch3_fast_b")
        paired = load(args.paired, "prefetch3_paired_helper")
        assert (candidate.CTA_M, candidate.CTA_N, candidate.BLOCK_K, candidate.WARPS) == (
            control.CTA_M, control.CTA_N, control.BLOCK_K, control.WARPS)
        torch.manual_seed(0)
        m = n = k = 4096
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        reference = torch.mm(a, b.T)
        names = ("candidate", "control", "triton_b", "cublas")
        outputs = {name: torch.empty_like(reference) for name in names}
        compiled = {}
        calls = {}
        report["compiler"] = {}
        for name, module in (("candidate", candidate), ("control", control)):
            desc, grid = paired.descriptors(module, a, b, outputs[name])
            compiled[name] = module.gemm[grid](*desc, True, True, 1,
                                                num_warps=module.WARPS, num_ctas=2)
            report["compiler"][name] = paired.write_compiler_artifacts(
                args.output.parent, name, compiled[name])
            calls[name] = lambda compiled=compiled[name], desc=desc, grid=grid: compiled[grid](
                *desc, True, True, 1)
        save(args.output, report)
        calls["triton_b"] = lambda: fast_b.launch(a, b.T, outputs["triton_b"], m, n, k)
        calls["cublas"] = lambda: torch.mm(a, b.T, out=outputs["cublas"])
        for name in names:
            calls[name]()
            torch.cuda.synchronize()
            exact = bool(torch.equal(outputs[name], reference))
            report.setdefault("correctness", {})[name] = {"exact": exact,
                "maximum_error": float((outputs[name].float()-reference.float()).abs().max().item())}
            save(args.output, report)
            if not exact:
                raise ValueError(f"incorrect output: {name}")
        graphs = {name: paired.capture_graph(calls[name], torch) for name in names}
        for round_index in range(3):
            order = list(names)
            random.Random(2409 + round_index).shuffle(order)
            report["orders"][str(round_index)] = order
            for name in order:
                measurement = paired.measure_graph(graphs[name], torch)
                report["observations"].append({"round": round_index, "name": name,
                                                "benchmark": measurement})
                save(args.output, report)
                print(round_index, name, measurement["median_us"], flush=True)
            for name in names:
                exact = bool(torch.equal(outputs[name], reference))
                report.setdefault("post_round_correctness", {}).setdefault(name, []).append(exact)
                save(args.output, report)
                if not exact:
                    raise ValueError(f"incorrect output after round {round_index}: {name}")
        # Keep timing fixed at seed 0 and 4096^3; stress correctness separately.
        for seed, m, n, k in ((1, 4096, 4096, 4096),
                              (2, 4096, 4096, 4096),
                              (0, 2048, 2048, 2048),
                              (0, 4096, 2048, 2048)):
            torch.manual_seed(seed)
            extra_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            extra_b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
            extra_reference = torch.mm(extra_a, extra_b.T)
            case = {"seed": seed, "m": m, "n": n, "k": k, "replays": 5,
                    "exact": {}}
            report.setdefault("stress", []).append(case)
            for name, module in (("candidate", candidate), ("control", control)):
                extra_c = torch.empty_like(extra_reference)
                extra_desc, extra_grid = paired.descriptors(module, extra_a, extra_b, extra_c)
                for _ in range(5):
                    compiled[name][extra_grid](*extra_desc, True, True, 1)
                    torch.cuda.synchronize()
                    if not torch.equal(extra_c, extra_reference):
                        case["exact"][name] = False
                        save(args.output, report)
                        raise ValueError(f"incorrect {name} output for {(seed, m, n, k)}")
                case["exact"][name] = True
                save(args.output, report)
        report["complete"] = True
        save(args.output, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[-1600:],
                           "traceback_tail": traceback.format_exc()[-2500:]}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
