from __future__ import annotations

from dataclasses import replace
import json
from typing import Mapping

from .cute_program import (
    CuteGemmProgram,
    PinnedCuteGemmRenderer,
    cute_gemm_program_from_source,
    validate_cute_gemm_program,
)
from .generation import GenerationRequest, GenerationResult, KernelGenerator


class CuteTypedLLMGenerator:
    """Convert untrusted LLM output into one canonical rendered CuTe program."""

    def __init__(
        self,
        delegate: KernelGenerator,
        renderer: PinnedCuteGemmRenderer | None = None,
    ) -> None:
        self._delegate = delegate
        self._renderer = renderer or PinnedCuteGemmRenderer()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = self._delegate.generate(request)
        metadata = dict(result.metadata or {})
        metadata["proposal_mechanism"] = "llm_typed_representation"
        if result.program is None:
            return replace(result, metadata=metadata)

        try:
            candidate, output_format = _parse_representation(result.program.source)
        except (json.JSONDecodeError, SyntaxError, TypeError, ValueError) as error:
            metadata.update(
                {
                    "typed_output_format": "invalid",
                    "static_validation": {
                        "valid": False,
                        "violations": [
                            {
                                "code": "malformed_typed_representation",
                                "message": str(error),
                            }
                        ],
                    },
                }
            )
            return replace(result, program=None, metadata=metadata)

        validation = validate_cute_gemm_program(candidate)
        parent = cute_gemm_program_from_source(request.parent.source)
        metadata.update(
            {
                "typed_output_format": output_format,
                "representation": candidate.as_dict(),
                "representation_schema_version": candidate.schema_version,
                "configuration_hash": candidate.configuration_hash,
                "static_validation": validation.as_dict(),
                "transformation": {
                    "mechanism": "llm_typed_representation",
                    "strategy_id": request.strategy.id,
                    "parent_configuration_hash": parent.configuration_hash,
                    "child_configuration_hash": candidate.configuration_hash,
                    "changed_fields": _changed_fields(parent, candidate),
                },
            }
        )
        program = self._renderer.render(candidate) if validation.valid else None
        return replace(result, program=program, metadata=metadata)


def _parse_representation(source: str) -> tuple[CuteGemmProgram, str]:
    try:
        value = json.loads(source)
    except json.JSONDecodeError:
        return cute_gemm_program_from_source(source), "rendered_source"
    if not isinstance(value, Mapping):
        raise ValueError("CuTe typed output must be a JSON object")
    try:
        return CuteGemmProgram(**value), "json"
    except TypeError as error:
        raise ValueError("CuTe typed output has invalid fields") from error


def _changed_fields(
    parent: CuteGemmProgram, candidate: CuteGemmProgram
) -> dict[str, dict[str, object]]:
    before = parent.as_dict()
    after = candidate.as_dict()
    return {
        name: {"before": before[name], "after": after[name]}
        for name in before
        if before[name] != after[name]
    }
