from __future__ import annotations

import argparse
import json
from typing import Sequence

from .cute_baseline import run_hopper_bf16_comparable
from .cute_program import CuteGemmProgram, validate_cute_gemm_program


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-json", required=True)
    parser.add_argument("--mode", choices=("evaluate", "profile"), default="evaluate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    value = json.loads(arguments.representation_json)
    if not isinstance(value, dict):
        raise ValueError("CuTe representation must be an object")
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
