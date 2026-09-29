from __future__ import annotations

from dataclasses import replace
import json
from typing import Mapping

from .cute_independent import (
    IndependentCuteGemmKernel,
    independent_cute_gemm_from_dict,
    validate_independent_cute_gemm,
)
from .cute_independent_program import (
    REPRESENTATION_NAME as INDEPENDENT_REPRESENTATION_NAME,
    IndependentCuteGemmRenderer,
    independent_cute_gemm_from_source,
)
from .cute_mutations import enumerate_independent_cute_mutations
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
        independent_renderer: IndependentCuteGemmRenderer | None = None,
    ) -> None:
        self._delegate = delegate
        self._renderer = renderer or PinnedCuteGemmRenderer()
        self._independent_renderer = independent_renderer or IndependentCuteGemmRenderer()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = self._delegate.generate(request)
        metadata = dict(result.metadata or {})
        metadata["proposal_mechanism"] = "llm_typed_representation"
        if result.program is None:
            return replace(result, metadata=metadata)

        if INDEPENDENT_REPRESENTATION_NAME in request.parent.source:
            return self._generate_independent(request, result, metadata)

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

    def _generate_independent(
        self,
        request: GenerationRequest,
        result: GenerationResult,
        metadata: dict[str, object],
    ) -> GenerationResult:
        try:
            candidate, output_format = _parse_independent_representation(
                result.program.source
            )
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

        parent = independent_cute_gemm_from_source(request.parent.source)
        validation = validate_independent_cute_gemm(candidate).as_dict()
        admissible = any(
            proposal.strategy_id == request.strategy.id
            and proposal.candidate.configuration_hash == candidate.configuration_hash
            for proposal in enumerate_independent_cute_mutations(parent)
        )
        if not admissible:
            validation = {
                "valid": False,
                "violations": [
                    *validation["violations"],
                    {
                        "code": "inadmissible_typed_transition",
                        "message": (
                            "candidate is not an admitted independent CuTe transition "
                            f"for strategy {request.strategy.id!r}"
                        ),
                    },
                ],
            }
        metadata.update(
            {
                "typed_output_format": output_format,
                "representation": candidate.as_dict(),
                "representation_schema_version": candidate.schema_version,
                "configuration_hash": candidate.configuration_hash,
                "static_validation": validation,
                "transformation": {
                    "mechanism": "llm_typed_representation",
                    "strategy_id": request.strategy.id,
                    "parent_configuration_hash": parent.configuration_hash,
                    "child_configuration_hash": candidate.configuration_hash,
                    "changed_fields": _changed_fields(parent, candidate),
                },
            }
        )
        program = self._independent_renderer.render(candidate) if validation["valid"] else None
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


def _parse_independent_representation(
    source: str,
) -> tuple[IndependentCuteGemmKernel, str]:
    try:
        value = json.loads(source)
    except json.JSONDecodeError:
        return independent_cute_gemm_from_source(source), "rendered_source"
    if not isinstance(value, dict):
        raise ValueError("independent CuTe typed output must be a JSON object")
    return independent_cute_gemm_from_dict(value), "json"


def _changed_fields(
    parent: CuteGemmProgram | IndependentCuteGemmKernel,
    candidate: CuteGemmProgram | IndependentCuteGemmKernel,
) -> dict[str, dict[str, object]]:
    before = parent.as_dict()
    after = candidate.as_dict()
    return {
        name: {"before": before[name], "after": after[name]}
        for name in before
        if before[name] != after[name]
    }
