from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import ShapeCase, Strategy, WorkloadContract
from .strategy_applicability import TRITON_PARENT_STAGES


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
    strategies = []
    for item in data["strategies"]:
        applicability = item.get("applicability", {})
        if not isinstance(applicability, dict):
            raise ValueError("strategy applicability must be a mapping")
        allowed = {"backend", "operation", "gpu_model", "parent_stages"}
        if set(applicability) - allowed:
            raise ValueError("unknown strategy applicability field")
        for field in ("backend", "operation", "gpu_model"):
            if field in applicability and not isinstance(applicability[field], str):
                raise ValueError(f"strategy applicability {field} must be a string")
        if "parent_stages" in applicability:
            stages = applicability["parent_stages"]
            if not isinstance(stages, list) or not stages or not all(
                isinstance(stage, str) and stage in TRITON_PARENT_STAGES for stage in stages
            ):
                raise ValueError("strategy parent_stages must name known source stages")
        scaffold = item.get("scaffold_resource")
        if scaffold is not None and (
            not isinstance(scaffold, str) or not scaffold.endswith(".py")
            or Path(scaffold).name != scaffold
        ):
            raise ValueError("strategy scaffold_resource must be one Python filename")
        candidate_stage = item.get("candidate_stage")
        if candidate_stage is not None and (
            not isinstance(candidate_stage, str)
            or candidate_stage not in TRITON_PARENT_STAGES
        ):
            raise ValueError("strategy candidate_stage must name a known source stage")
        strategies.append(Strategy(item["id"], item["description"], item["prompts"],
                                   applicability, scaffold, candidate_stage))
    return tuple(strategies)


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

