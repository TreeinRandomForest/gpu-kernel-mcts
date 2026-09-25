from __future__ import annotations

import ast
from dataclasses import dataclass

from .cute_independent import (
    IndependentCuteGemmKernel,
    independent_cute_gemm_from_dict,
    validate_independent_cute_gemm,
)
from .cute_independent_tma import render_independent_tma_copy_diagnostic
from .domain import KernelProgram


REPRESENTATION_NAME = "KERNEL_MCTS_INDEPENDENT_CUTE_GEMM"


def independent_cute_gemm_from_source(source: str) -> IndependentCuteGemmKernel:
    tree = ast.parse(source)
    values: list[object] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == REPRESENTATION_NAME
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError(
            f"program must define one literal {REPRESENTATION_NAME}"
        )
    return independent_cute_gemm_from_dict(values[0])


@dataclass(frozen=True, slots=True)
class IndependentCuteGemmRenderer:
    """Render the independently lowered, repository-contract GEMM root."""

    def render(self, kernel: IndependentCuteGemmKernel) -> KernelProgram:
        legality = validate_independent_cute_gemm(kernel)
        if not legality.valid:
            messages = "; ".join(item.message for item in legality.violations)
            raise ValueError(f"illegal independent CuTe GEMM: {messages}")
        lowered = render_independent_tma_copy_diagnostic(
            kernel,
            debug_stage="wgmma_full_workload",
            repository_contract=True,
        )
        header = f"{REPRESENTATION_NAME} = {kernel.as_dict()!r}\n"
        return KernelProgram(header + lowered.source, "cute_dsl")
