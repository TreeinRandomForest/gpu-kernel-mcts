"""Strategy-free CUDA generation for the sequential refinement experiment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .generation import GenerationRequest, GenerationResult
from .llm import LLMClient
from .llm_generation import _repair_evidence
from .serialization import serialize_workload
from .domain import KernelProgram


SEQUENTIAL_INSTRUCTIONS = (
    'Return only a JSON object with string fields "hypothesis" and "source". '
    'The source must be the complete replacement CUDA source; no Markdown.'
)


class SequentialKernelGenerator:
    def __init__(self, client: LLMClient, *, strategy_menu: tuple[str, ...] = ()) -> None:
        self._client = client
        self._strategy_menu = strategy_menu

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = build_sequential_prompt(request, self._strategy_menu)
        completion = self._client.complete(prompt)
        hypothesis: str | None = None
        source: str | None = None
        try:
            output = json.loads(completion.output_text)
            if isinstance(output, dict):
                hypothesis = output.get("hypothesis") if isinstance(output.get("hypothesis"), str) else None
                source = output.get("source") if isinstance(output.get("source"), str) else None
                if source is not None:
                    source = source.strip() or None
        except (TypeError, ValueError):
            pass
        metadata = dict(completion.metadata)
        metadata.update({
            "generator": "sequential_llm",
            "llm_call": True,
            "proposal_mechanism": "llm_generation",
            "model": completion.model,
            "response_id": completion.response_id,
            "hypothesis": hypothesis,
        })
        exact_input = metadata.get("conversation_input", prompt)
        exact_prompt = (
            json.dumps(exact_input, sort_keys=True)
            if not isinstance(exact_input, str) else exact_input
        )
        return GenerationResult(
            generation_id=completion.response_id,
            raw_output=completion.output_text,
            program=KernelProgram(source, backend=request.parent.backend) if source else None,
            prompt_hash=hashlib.sha256(exact_prompt.encode()).hexdigest(),
            prompt_text=exact_prompt,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_seconds=completion.latency_seconds,
            metadata=metadata,
            instructions_text=completion.instructions_text,
        )


def build_sequential_prompt(request: GenerationRequest, strategy_menu: tuple[str, ...] = ()) -> str:
    payload: dict[str, Any] = {
        "task": "Optimize the current verified CUDA kernel. State a concise hypothesis and return a complete replacement source.",
        "backend": request.parent.backend,
        "target_hardware": dict(request.hardware),
        "workload": serialize_workload(request.workload),
        "current_kernel": request.parent.source,
        "current_profile": request.profile,
        "current_benchmark": request.current_benchmark,
        "recent_feedback": request.recent_feedback,
        "current_compiler_summary": request.compiler_summary,
        "current_dataflow": request.dataflow_view,
        "constraints": [
            "Preserve the operation, tensor layouts, dtype, ABI, and entry point.",
            "Return a complete compilable source file, not a patch.",
            "Choose any optimization; a local slowdown is permitted for exploration.",
        ],
        "attempt": request.attempt,
    }
    if strategy_menu:
        payload["strategy_menu"] = list(strategy_menu)
        payload["strategy_instruction"] = "Choose one listed strategy in your hypothesis, then implement it."
    if request.previous_program is not None or request.previous_result is not None:
        payload["repair"] = {
            "previous_candidate": request.previous_program.source if request.previous_program else None,
            "previous_evaluation": _repair_evidence(request.previous_result),
            "instruction": "Repair the failed candidate while preserving semantics.",
        }
    return json.dumps(payload, sort_keys=True, indent=2)
