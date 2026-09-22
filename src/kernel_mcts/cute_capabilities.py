from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Mapping, Sequence


_REQUIRED_CLASS = "HopperWgmmaGemmKernel"
_REQUIRED_FUNCTION = "run"
_CONTROL_TERMS: Mapping[str, tuple[str, ...]] = {
    "pipeline": ("pipeline", "stage"),
    "wgmma": ("wgmma", "mma", "2cta"),
    "tma": ("tma", "tensormap", "cp_async"),
    "epilogue": ("epilogue", "tma_store"),
    "warp_specialization": ("warp", "producer", "consumer"),
    "scheduler": ("scheduler", "raster", "swizzle", "persistent"),
}
_PIPELINE_STATE_NAMES = {"ab_stage", "epi_stage", "occupancy", "smem_capacity"}
_WGMMA_STATE_NAMES = {
    "k_pipe_mmas",
    "mma_inst_shape_k",
    "mma_inst_shape_mn",
    "mma_inst_tile_k",
    "mma_warp_groups",
    "num_threads_per_warp_group",
    "num_warps",
    "thr_mma",
    "tiled_mma",
    "warp_group_thread_layout",
}


def inspect_cute_structural_capabilities(path: Path) -> Mapping[str, object]:
    """Describe structural controls visible in pinned CuTe source without executing it."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    run = _top_level_function(tree, _REQUIRED_FUNCTION)
    kernel_class = _top_level_class(tree, _REQUIRED_CLASS)
    constructor = _class_method(kernel_class, "__init__")
    if run is None or kernel_class is None or constructor is None:
        missing = []
        if run is None:
            missing.append(_REQUIRED_FUNCTION)
        if kernel_class is None:
            missing.append(_REQUIRED_CLASS)
        elif constructor is None:
            missing.append(f"{_REQUIRED_CLASS}.__init__")
        raise ValueError("pinned CuTe source is missing required symbols: " + ", ".join(missing))

    return {
        "status": "ok",
        "schema_version": 1,
        "diagnostic": "cute_structural_capabilities",
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "line_count": len(source.splitlines()),
        },
        "symbols": {
            _REQUIRED_FUNCTION: _function_description(run),
            f"{_REQUIRED_CLASS}.__init__": _function_description(constructor),
        },
        "cli_options": _cli_options(tree),
        "validation_evidence": _validation_evidence(tree, source),
        "candidate_structural_controls": _candidate_controls(tree, source),
        "selection_decision": "not_made",
        "interpretation": (
            "Evidence only: discovered names and expressions are not yet validated "
            "as legal, independent, or performance-relevant search controls."
        ),
    }


def _top_level_function(
    tree: ast.Module, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def _top_level_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name),
        None,
    )


def _class_method(
    node: ast.ClassDef | None, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if node is None:
        return None
    return next(
        (
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ),
        None,
    )


def _function_description(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Mapping[str, object]:
    return {
        "line": node.lineno,
        "parameters": _parameters(node.args),
        "returns": _unparse(node.returns) if node.returns is not None else None,
    }


def _parameters(arguments: ast.arguments) -> list[Mapping[str, object]]:
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    result = [
        _parameter(
            argument,
            "positional_only" if index < len(arguments.posonlyargs) else "positional_or_keyword",
            defaults[index],
        )
        for index, argument in enumerate(positional)
    ]
    if arguments.vararg is not None:
        result.append(_parameter(arguments.vararg, "var_positional", None))
    result.extend(
        _parameter(argument, "keyword_only", default)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    )
    if arguments.kwarg is not None:
        result.append(_parameter(arguments.kwarg, "var_keyword", None))
    return result


def _parameter(
    argument: ast.arg, kind: str, default: ast.expr | None
) -> Mapping[str, object]:
    return {
        "name": argument.arg,
        "kind": kind,
        "annotation": _unparse(argument.annotation) if argument.annotation else None,
        "default": _unparse(default) if default is not None else None,
    }


def _cli_options(tree: ast.AST) -> list[Mapping[str, object]]:
    options: list[Mapping[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags = [
            value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance((value := argument.value), str)
        ]
        keywords = {
            keyword.arg: _unparse(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        options.append({"line": node.lineno, "flags": flags, "keywords": keywords})
    return sorted(options, key=lambda item: (int(item["line"]), tuple(item["flags"])))


def _validation_evidence(tree: ast.AST, source: str) -> Mapping[str, object]:
    assertions = [
        {
            "line": node.lineno,
            "test": _unparse(node.test),
            "message": _unparse(node.msg) if node.msg is not None else None,
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]
    raises = [
        {
            "line": node.lineno,
            "expression": _unparse(node.exc) if node.exc is not None else None,
            "source": _source_segment(source, node),
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
    ]
    validation_functions = [
        {
            "name": node.name,
            "line": node.lineno,
            "parameters": _parameters(node.args),
        }
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(term in node.name.casefold() for term in ("valid", "check", "verify"))
    ]
    return {
        "assertions": sorted(assertions, key=lambda item: int(item["line"])),
        "raises": sorted(raises, key=lambda item: int(item["line"])),
        "validation_functions": sorted(
            validation_functions, key=lambda item: (int(item["line"]), str(item["name"]))
        ),
    }


def _candidate_controls(tree: ast.AST, source: str) -> Mapping[str, object]:
    relevant_nodes = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Call,
        ast.If,
        ast.Assert,
        ast.Return,
    )
    controls: dict[str, object] = {}
    for category, terms in _CONTROL_TERMS.items():
        identifiers: set[str] = set()
        evidence: dict[tuple[int, str], Mapping[str, object]] = {}
        definitions: list[Mapping[str, object]] = []
        for node in ast.walk(tree):
            names = _node_names(node)
            matches = {
                name
                for name in names
                if any(term in name.casefold() for term in terms)
            }
            identifiers.update(matches)
            if matches and isinstance(node, relevant_nodes) and hasattr(node, "lineno"):
                snippet = _source_segment(source, node)
                evidence[(node.lineno, snippet)] = {
                    "line": node.lineno,
                    "source": snippet,
                }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                term in node.name.casefold() for term in terms
            ):
                definitions.append(
                    {
                        "kind": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "parameters": _parameters(node.args),
                        "returns": (
                            _unparse(node.returns)
                            if node.returns is not None
                            else None
                        ),
                        "decorators": [_unparse(item) for item in node.decorator_list],
                        "return_expressions": [
                            _unparse(item.value)
                            for item in ast.walk(node)
                            if isinstance(item, ast.Return)
                            and item.value is not None
                        ],
                        "assignments": _function_assignments(node),
                        "conditions": [
                            {"line": item.lineno, "test": _unparse(item.test)}
                            for item in ast.walk(node)
                            if isinstance(item, (ast.If, ast.While))
                        ],
                    }
                )
        controls[category] = {
            "status": "evidence_only",
            "identifiers": sorted(identifiers),
            "definitions": sorted(
                definitions,
                key=lambda item: (int(item["line"]), str(item["name"])),
            ),
            "source_evidence": [
                evidence[key] for key in sorted(evidence)[:24]
            ],
        }
        if category == "pipeline":
            controls[category]["state_assignments"] = _named_assignments(
                tree, _PIPELINE_STATE_NAMES
            )
        elif category in {"wgmma", "warp_specialization"}:
            controls[category]["state_assignments"] = _named_assignments(
                tree, _WGMMA_STATE_NAMES
            )
    return controls


def _node_names(node: ast.AST) -> set[str]:
    names = {
        item.id for item in ast.walk(node) if isinstance(item, ast.Name)
    }
    names.update(item.arg for item in ast.walk(node) if isinstance(item, ast.arg))
    names.update(
        item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
    )
    return names


def _function_assignments(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[Mapping[str, object]]:
    assignments: list[Mapping[str, object]] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Assign):
            assignments.append(
                {
                    "line": item.lineno,
                    "targets": [_unparse(target) for target in item.targets],
                    "value": _unparse(item.value),
                }
            )
        elif isinstance(item, ast.AnnAssign):
            assignments.append(
                {
                    "line": item.lineno,
                    "targets": [_unparse(item.target)],
                    "value": _unparse(item.value) if item.value is not None else None,
                }
            )
    return sorted(assignments, key=lambda value: int(value["line"]))


def _named_assignments(
    tree: ast.AST, names: set[str]
) -> list[Mapping[str, object]]:
    assignments: list[Mapping[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        matched = [
            _unparse(target)
            for target in targets
            if _target_names(target) & names
        ]
        if matched:
            assignments.append(
                {
                    "line": node.lineno,
                    "targets": matched,
                    "value": _unparse(value),
                }
            )
    return sorted(assignments, key=lambda item: int(item["line"]))


def _target_names(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node) or _unparse(node)
    compact = " ".join(segment.split())
    return compact[:500]


def _unparse(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""
