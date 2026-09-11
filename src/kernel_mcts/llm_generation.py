from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .domain import EvaluationResult, KernelProgram
from .generation import GenerationRequest, GenerationResult
from .llm import LLMClient
from .serialization import serialize_evaluation, serialize_workload


class LLMKernelGenerator:
    """Generate one backend-aware kernel from one fresh LLM completion."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = build_generation_prompt(request)
        completion = self._client.complete(prompt)
        source = extract_kernel_source(completion.output_text)
        metadata = dict(completion.metadata)
        metadata.update(
            {
                "generator": "llm",
                "llm_call": True,
                "model": completion.model,
                "response_id": completion.response_id,
            }
        )
        return GenerationResult(
            generation_id=completion.response_id,
            raw_output=completion.output_text,
            program=(
                KernelProgram(source, backend=request.parent.backend)
                if source is not None
                else None
            ),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_text=prompt,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_seconds=completion.latency_seconds,
            metadata=metadata,
        )


def build_generation_prompt(request: GenerationRequest) -> str:
    payload: dict[str, Any] = {
        "task": "Produce a complete replacement GPU kernel preserving the workload contract.",
        "backend": request.parent.backend,
        "target_hardware": _json_value(request.hardware),
        "workload": serialize_workload(request.workload),
        "strategy": {
            "id": request.strategy.id,
            "description": request.strategy.description,
            "backend_prompt": request.strategy.prompt_for(request.parent.backend),
        },
        "parent_profile": _json_value(request.profile),
        "parent_kernel": request.parent.source,
        "constraints": [
            "Preserve the operation, tensor layouts, dtype, ABI, and entry point.",
            "Return a complete compilable source file, not a patch.",
            "Do not include Markdown fences or explanatory text.",
        ],
        "attempt": request.attempt,
    }
    if request.previous_program is not None or request.previous_result is not None:
        payload["repair"] = {
            "previous_candidate": (
                request.previous_program.source if request.previous_program else None
            ),
            "previous_evaluation": _repair_evidence(request.previous_result),
            "instruction": "Repair the failed candidate while preserving semantics.",
        }
    return json.dumps(payload, sort_keys=True, indent=2)


def extract_kernel_source(raw_output: str) -> str | None:
    source = raw_output.strip()
    if not source:
        return None
    if source.startswith("```") and source.endswith("```"):
        lines = source.splitlines()
        if len(lines) >= 3:
            source = "\n".join(lines[1:-1]).strip()
    return source or None


def _repair_evidence(result: EvaluationResult | None) -> Mapping[str, object] | None:
    if result is None:
        return None
    serialized = serialize_evaluation(result)
    compilation = serialized.get("compilation")
    if isinstance(compilation, dict):
        compilation = dict(compilation)
        for key in ("stdout", "stderr"):
            value = compilation.get(key)
            if isinstance(value, str):
                compilation[key] = value[-8_000:]
    return {
        "status": serialized["status"],
        "invalid_reason": serialized["invalid_reason"],
        "compile_status": serialized["compile_status"],
        "correctness_status": serialized["correctness_status"],
        "compilation": compilation,
        "correctness": serialized.get("correctness"),
    }


def _json_value(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError as error:
        raise TypeError("LLM prompt context must be JSON serializable") from error
    return value
