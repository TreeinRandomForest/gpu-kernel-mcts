"""Conservative static applicability for versioned Triton/Gluon strategies."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence

from .domain import KernelProgram, Strategy, WorkloadContract
from .triton_action_fidelity import (
    COALESCED_LAYOUT_ACTION, TRITON_CONSTEXPR_ACTION, TRITON_POINTER_LOOP_ACTION,
    TRITON_WIDE_TILE_ACTION, coalesced_layout_actionable,
    triton_constexpr_actionable, triton_pointer_loop_ready,
    triton_wide_tile_actionable,
)


TRITON_PARENT_STAGES = frozenset({
    "pointer_triton", "descriptor_triton", "gluon_mma_v2", "single_cta_gluon",
    "clustered_gluon", "gluon_other", "unknown",
})


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def triton_parent_stage(source: str) -> str:
    """Classify only syntax that establishes a source stage; unknown stays unknown."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "unknown"
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    launch = functions.get("launch")
    if launch is None:
        return "unknown"
    kernel_launches = [
        node.value
        for node in ast.walk(launch)
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Subscript)
            and isinstance(node.value.func.value, ast.Name))
    ]
    launched_names = {call.func.value.id for call in kernel_launches}
    if len(launched_names) != 1:
        return "unknown"
    kernel = functions.get(launched_names.pop())
    if kernel is None:
        return "unknown"
    decorators = {_dotted_name(decorator) for decorator in kernel.decorator_list}
    launch_calls = {_dotted_name(node.func) for node in ast.walk(launch)
                    if isinstance(node, ast.Call)}
    kernel_calls = {_dotted_name(node.func) for node in ast.walk(kernel)
                    if isinstance(node, ast.Call)}
    launch_constants: dict[str, int] = {}
    for statement in launch.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Constant)
                and type(statement.value.value) is int):
            launch_constants[statement.targets[0].id] = statement.value.value

    def cluster_count(value: ast.AST) -> int | None:
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return value.value
        if isinstance(value, ast.Name):
            return launch_constants.get(value.id)
        return None

    clustered = any(
        keyword.arg == "num_ctas" and (cluster_count(keyword.value) or 0) >= 2
        for call in kernel_launches
        for keyword in call.keywords
    )
    if "gluon.jit" in decorators:
        if ("TensorDescriptor.from_tensor" in launch_calls
                and "warpgroup_mma" in kernel_calls
                and "mbarrier.allocate_mbarrier" in kernel_calls):
            return "clustered_gluon" if clustered else "single_cta_gluon"
        if "mma_v2" in kernel_calls or any(
            name.endswith(".mma_v2") for name in kernel_calls
        ):
            return "gluon_mma_v2"
        return "gluon_other"
    if "triton.jit" in decorators:
        constructors = {
            "TensorDescriptor.from_tensor", "TensorDescriptor",
            "create_2d_tma_descriptor", "tl.make_tensor_descriptor",
        }
        descriptor_names = {
            target.id
            for statement in ast.walk(launch)
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            for target in (statement.targets if isinstance(statement, ast.Assign)
                           else [statement.target])
            if isinstance(target, ast.Name)
            and isinstance(statement.value, ast.Call)
            and _dotted_name(statement.value.func) in constructors
        }
        descriptor_parameters = {
            parameter.arg
            for call in kernel_launches
            for parameter, argument in zip(kernel.args.args, call.args)
            if isinstance(argument, ast.Name) and argument.id in descriptor_names
        }
        descriptor_construction = bool(descriptor_names) or bool(
            launch_calls & constructors
        )
        descriptor_load = "tl._experimental_descriptor_load" in kernel_calls or any(
            name.endswith("_desc.load") for name in kernel_calls
        ) or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
            and node.func.value.id in descriptor_parameters
            for node in ast.walk(kernel)
        )
        return "descriptor_triton" if descriptor_construction and descriptor_load else "pointer_triton"
    return "unknown"


def applicable_strategies(
    strategies: Sequence[Strategy], program: KernelProgram,
    workload: WorkloadContract, hardware: Mapping[str, object],
) -> tuple[Strategy, ...]:
    stage = triton_parent_stage(program.source) if program.backend == "triton_h100" else "unknown"
    eligible = []
    for strategy in strategies:
        rule = strategy.applicability
        if rule.get("backend", program.backend) != program.backend:
            continue
        if rule.get("operation", workload.operation) != workload.operation:
            continue
        if "gpu_model" in rule and rule["gpu_model"] != hardware.get("gpu_model"):
            continue
        if "parent_stages" in rule and stage not in rule["parent_stages"]:
            continue
        if strategy.id == COALESCED_LAYOUT_ACTION and not coalesced_layout_actionable(
            program.source
        ):
            continue
        if strategy.id == TRITON_POINTER_LOOP_ACTION and triton_pointer_loop_ready(
            program.source
        ):
            continue
        if strategy.id == TRITON_WIDE_TILE_ACTION and not triton_wide_tile_actionable(
            program.source
        ):
            continue
        if strategy.id == TRITON_CONSTEXPR_ACTION and not triton_constexpr_actionable(
            program.source
        ):
            continue
        eligible.append(strategy)
    return tuple(eligible)
