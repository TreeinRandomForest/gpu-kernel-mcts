from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import ShapeCase, Strategy, WorkloadContract


def load_data(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text()
    if Path(path).suffix == ".json":
        return json.loads(source)
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("YAML configuration requires the 'config' optional dependency") from error
    result = yaml.safe_load(source)
    if not isinstance(result, dict):
        raise ValueError("configuration root must be a mapping")
    return result


def parse_strategies(data: dict[str, Any]) -> tuple[Strategy, ...]:
    return tuple(
        Strategy(item["id"], item["description"], item["prompts"])
        for item in data["strategies"]
    )


def parse_workload(data: dict[str, Any]) -> WorkloadContract:
    item = data["workload"]
    return WorkloadContract(
        benchmark_id=item["benchmark_id"],
        operation=item["operation"],
        dtype=item["dtype"],
        shapes=tuple(ShapeCase(shape["dimensions"], float(shape["weight"])) for shape in item["shapes"]),
        rtol=float(item["rtol"]),
        atol=float(item["atol"]),
        metadata=item.get("metadata", {}),
    )

