from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Sequence

from .cute_baseline import run_hopper_bf16_comparable
from .cute_program import CuteGemmProgram, validate_cute_gemm_program
from .cute_independent import (
    independent_cute_gemm_from_dict,
    validate_independent_cute_gemm,
)
from .cute_independent_tma import render_independent_tma_copy_diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-json", required=True)
    parser.add_argument("--mode", choices=("evaluate", "profile"), default="evaluate")
    parser.add_argument(
        "--representation-kind",
        choices=("pinned", "independent"),
        default="pinned",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    value = json.loads(arguments.representation_json)
    if not isinstance(value, dict):
        raise ValueError("CuTe representation must be an object")
    if arguments.representation_kind == "independent":
        program = independent_cute_gemm_from_dict(value)
        legality = validate_independent_cute_gemm(program)
        if not legality.valid:
            raise ValueError("independent CuTe representation is not statically legal")
        rendered = render_independent_tma_copy_diagnostic(
            program,
            debug_stage="wgmma_full_workload",
            repository_contract=True,
            profile_single_launch=arguments.mode == "profile",
        )
        with tempfile.TemporaryDirectory(
            prefix="kernel-mcts-independent-runner-"
        ) as directory:
            source_path = Path(directory) / "independent_cute_gemm.py"
            source_path.write_text(rendered.source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                "kernel_mcts_independent_cute_gemm", source_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load rendered independent CuTe program")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run = getattr(module, "run_diagnostic", None)
            if not callable(run):
                raise RuntimeError("independent CuTe renderer omitted run_diagnostic")
            result = run()
        print(json.dumps(result, sort_keys=True))
        return 0

    program = CuteGemmProgram(**value)
    legality = validate_cute_gemm_program(program)
    if not legality.valid:
        raise ValueError("CuTe representation is not statically legal")
    result = run_hopper_bf16_comparable(
        schedule=program.schedule,
        pipeline_stages=program.pipeline_stages,
        epilogue_stages=program.epilogue_stages,
        wgmma_configuration=program.wgmma_configuration,
        smem_swizzle_policy=(
            "forced_sw64"
            if program.shared_memory_swizzle == "sw64"
            else "heuristic"
        ),
        raise_on_correctness_failure=False,
        capture_jit_diagnostics=True,
        profile_single_launch=arguments.mode == "profile",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
