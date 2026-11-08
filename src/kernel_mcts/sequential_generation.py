"""CUDA generation for freeform and catalog-choice sequential refinement."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .generation import GenerationRequest, GenerationResult
from .llm import LLMClient
from .llm_generation import _prompt_workload, _repair_evidence
from .domain import KernelProgram, Strategy
from .strategy_applicability import applicable_strategies


SEQUENTIAL_INSTRUCTIONS = (
    'Return only a JSON object with string fields "selected_strategy_id", "hypothesis", and "source". '
    'The source must be the complete replacement CUDA source; no Markdown.'
)
SEQUENTIAL_TRITON_INSTRUCTIONS = SEQUENTIAL_INSTRUCTIONS.replace(
    "replacement CUDA source", "replacement Triton Python source"
)
SEQUENTIAL_PROMPT_SCHEMA_VERSION = 2
SEQUENTIAL_TRITON_PROMPT_SCHEMA_VERSION = 3


class SequentialKernelGenerator:
    def __init__(
        self, client: LLMClient, *, strategy_menu: tuple[str, ...] = (),
        strategy_ids: tuple[str, ...] | None = None,
        catalog_sha256: str | None = None,
        strategy_catalog: tuple[Strategy, ...] = (),
        instructions_text: str = SEQUENTIAL_INSTRUCTIONS,
        prompt_schema_version: int = SEQUENTIAL_PROMPT_SCHEMA_VERSION,
    ) -> None:
        self._client = client
        self._strategy_menu = strategy_menu
        self._strategy_ids = strategy_ids if strategy_ids is not None else tuple(
            entry.split(":", 1)[0].strip() for entry in strategy_menu
        )
        if (
            len(self._strategy_ids) != len(strategy_menu)
            or len(set(self._strategy_ids)) != len(self._strategy_ids)
        ):
            raise ValueError("strategy menu must have one unique ID per entry")
        if any(not item or item == "freeform" for item in self._strategy_ids):
            raise ValueError("strategy menu IDs must be nonempty and cannot use freeform")
        self._catalog_sha256 = catalog_sha256
        self.strategy_catalog = strategy_catalog
        self._instructions_text = instructions_text
        self._prompt_schema_version = prompt_schema_version

    @property
    def choice_provenance(self) -> dict[str, object]:
        menu_json = json.dumps(self._strategy_menu, ensure_ascii=False, separators=(",", ":"))
        return {
            "prompt_schema_version": self._prompt_schema_version,
            "menu_enabled": bool(self._strategy_menu),
            "ordered_strategy_ids": list(self._strategy_ids),
            "rendered_strategy_menu": list(self._strategy_menu),
            "catalog_sha256": self._catalog_sha256,
            "rendered_menu_sha256": hashlib.sha256(menu_json.encode()).hexdigest(),
            "instructions_sha256": hashlib.sha256(self._instructions_text.encode()).hexdigest(),
            "dynamic_applicability": bool(self.strategy_catalog),
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        eligible = applicable_strategies(
            self.strategy_catalog, request.parent, request.workload, request.hardware,
        ) if self.strategy_catalog else ()
        menu = (
            tuple(f"{item.id}: {item.description}; {item.prompt_for(request.parent.backend)}"
                  for item in eligible)
            if self.strategy_catalog else self._strategy_menu
        )
        prompt = build_sequential_prompt(request, menu)
        completion = self._client.complete(prompt)
        hypothesis: str | None = None
        source: str | None = None
        selected_id: str | None = None
        try:
            output = json.loads(completion.output_text)
            if isinstance(output, dict):
                raw_id = output.get("selected_strategy_id")
                selected_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None
                hypothesis = output.get("hypothesis") if isinstance(output.get("hypothesis"), str) else None
                source = output.get("source") if isinstance(output.get("source"), str) else None
                if source is not None:
                    source = source.strip() or None
        except (TypeError, ValueError):
            pass
        allowed_ids = (
            {item.id for item in eligible} if self.strategy_catalog else
            set(self._strategy_ids) if self._strategy_menu else {"freeform"}
        )
        if selected_id is None:
            status = "missing"
        elif selected_id not in allowed_ids:
            status = "out_of_catalog"
        elif request.strategy_choice_locked_id is not None and selected_id != request.strategy_choice_locked_id:
            status = "repair_mismatch"
        else:
            status = "valid"
        original_id = request.strategy_choice_original_id or selected_id
        lock_id = request.strategy_choice_locked_id or (
            selected_id if status == "valid" else None
        )
        metadata = dict(completion.metadata)
        metadata.update({
            "generator": "sequential_llm",
            "llm_call": True,
            "proposal_mechanism": "llm_generation",
            "model": completion.model,
            "response_id": completion.response_id,
            "hypothesis": hypothesis,
            "selected_strategy_id": selected_id,
            "strategy_choice_status": status,
            "strategy_choice_original_id": original_id,
            "strategy_choice_lock_id": lock_id,
            "strategy_choice_prompt_schema_version": self._prompt_schema_version,
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
    triton = request.parent.backend == "triton_h100"
    payload: dict[str, Any] = {
        "task": (
            "Optimize the current verified Triton kernel. State a concise hypothesis and return a complete replacement Python source."
            if triton else
            "Optimize the current verified CUDA kernel. State a concise hypothesis and return a complete replacement source."
        ),
        "backend": request.parent.backend,
        "target_hardware": dict(request.hardware),
        "workload": _prompt_workload(request.workload, request.parent.backend),
        "current_kernel": request.parent.source,
        "current_profile": request.profile,
        "current_benchmark": request.current_benchmark,
        "recent_feedback": request.recent_feedback,
        "current_compiler_summary": request.compiler_summary,
        "current_dataflow": request.dataflow_view,
        "constraints": [
            ("Preserve launch(A,B,C,M,N,K), tensor layouts, dtype, and correctness contract."
             if triton else
             "Preserve the operation, tensor layouts, dtype, ABI, and entry point."),
            "Return a complete compilable source file, not a patch.",
            "Choose any optimization; a local slowdown is permitted for exploration.",
        ],
        "attempt": request.attempt,
        "strategy_instruction": (
            "Choose exactly one listed strategy ID and implement it. Return that ID in selected_strategy_id."
            if strategy_menu else
            "Choose any optimization and return selected_strategy_id as freeform."
        ),
    }
    if strategy_menu:
        payload["strategy_menu"] = list(strategy_menu)
    if request.best_so_far is not None:
        payload["best_so_far"] = dict(request.best_so_far)
    if request.hardware_resource_graph is not None:
        payload["hardware_resource_graph"] = dict(request.hardware_resource_graph)
    if request.reference_baselines:
        payload["performance_references"] = dict(request.reference_baselines)
    if request.previous_program is not None or request.previous_result is not None:
        payload["repair"] = {
            "previous_candidate": request.previous_program.source if request.previous_program else None,
            "previous_evaluation": _repair_evidence(request.previous_result),
            "original_selected_strategy_id": request.strategy_choice_original_id,
            "locked_strategy_id": request.strategy_choice_locked_id,
            "instruction": (
                "Repair the failed candidate while preserving semantics and the locked strategy ID."
                if request.strategy_choice_locked_id else
                "Repair the failed candidate while preserving semantics."
            ),
        }
    return json.dumps(payload, sort_keys=True, indent=2)
