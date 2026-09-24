from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType


PINNED_CUTE_GEMM_SHA256 = (
    "bb7b76d893219757e2f3701abf1a7c8c819b9966e38aaed30a768596a55aca9f"
)


def validate_pinned_cute_gemm_source(path: str | Path) -> str:
    """Require the exact official source used by structural diagnostics."""

    source = Path(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != PINNED_CUTE_GEMM_SHA256:
        raise ValueError(
            "pinned CuTe GEMM source hash mismatch: "
            f"expected {PINNED_CUTE_GEMM_SHA256}, got {digest}"
        )
    return digest


def transform_wgmma_inflight_groups(
    source: str,
    groups: int,
    *,
    expected_sha256: str = PINNED_CUTE_GEMM_SHA256,
) -> str:
    """Replace the pinned WGMMA in-flight-group constant exactly once."""

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "pinned CuTe GEMM source hash mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    if groups not in (1, 2):
        raise ValueError("WGMMA in-flight groups must be 1 or 2")
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "k_pipe_mmas"
    ]
    if len(assignments) != 1:
        raise ValueError(
            "pinned CuTe GEMM must contain exactly one k_pipe_mmas assignment"
        )
    assignment = assignments[0]
    if not isinstance(assignment.value, ast.Constant) or assignment.value.value != 1:
        raise ValueError("pinned k_pipe_mmas assignment must have value 1")
    if groups == 1:
        return source

    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: assignment.lineno - 1]) + assignment.col_offset
    assert assignment.end_lineno is not None and assignment.end_col_offset is not None
    end = sum(len(line) for line in lines[: assignment.end_lineno - 1]) + assignment.end_col_offset
    transformed = source[:start] + f"k_pipe_mmas = {groups}" + source[end:]
    if transformed.count(f"k_pipe_mmas = {groups}") != 1:
        raise ValueError("WGMMA transformation did not produce exactly one assignment")
    return transformed


def load_pinned_cute_gemm(
    path: str | Path,
    *,
    wgmma_inflight_groups: int = 1,
) -> ModuleType:
    """Load the pinned module, applying a hash-guarded structural transform if needed."""

    example_path = Path(path)
    spec = importlib.util.spec_from_file_location(
        "kernel_mcts_pinned_cute_gemm", example_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned CuTe DSL example")
    module = importlib.util.module_from_spec(spec)
    if wgmma_inflight_groups == 1:
        spec.loader.exec_module(module)
        return module
    source = example_path.read_text(encoding="utf-8")
    transformed = transform_wgmma_inflight_groups(source, wgmma_inflight_groups)
    exec(compile(transformed, str(example_path), "exec"), module.__dict__)
    return module
