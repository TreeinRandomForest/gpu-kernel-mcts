from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping
from uuid import uuid4

from .cute_program import (
    CuteGemmProgram,
    CuteLegalityResult,
    PinnedCuteGemmRenderer,
    cute_gemm_program_from_source,
    validate_cute_gemm_program,
)
from .cute_schedule import CLUSTER_SHAPE_CHOICES, CTA_TILE_CHOICES
from .domain import Strategy
from .generation import (
    GenerationRequest,
    GenerationResult,
    ProposalBudgetKind,
)


CHANGE_CTA_TILE = "change_cta_tile"
CHANGE_CLUSTER_SHAPE = "change_cluster_shape"
CHANGE_PIPELINE_STAGES = "change_pipeline_stages"
CHANGE_EPILOGUE_STAGES = "change_epilogue_stages"
CHANGE_SHARED_MEMORY_SWIZZLE = "change_shared_memory_swizzle"
CUTE_MUTATION_STRATEGY_IDS = (
    CHANGE_CTA_TILE,
    CHANGE_CLUSTER_SHAPE,
    CHANGE_PIPELINE_STAGES,
    CHANGE_EPILOGUE_STAGES,
    CHANGE_SHARED_MEMORY_SWIZZLE,
)
CUTE_MUTATION_STRATEGIES = (
    Strategy(
        CHANGE_CTA_TILE,
        "Change the CTA output-tile decomposition.",
        {
            "cute_dsl": (
                "Change only tile_m and tile_n. Choose one supported pair: "
                "(64,128), (128,128), or (128,256)."
            )
        },
    ),
    Strategy(
        CHANGE_CLUSTER_SHAPE,
        "Change the Hopper thread-block cluster geometry.",
        {
            "cute_dsl": (
                "Change only cluster_m and cluster_n. Choose one supported pair: "
                "(1,1), (1,2), or (2,1)."
            )
        },
    ),
    Strategy(
        CHANGE_PIPELINE_STAGES,
        "Change the A/B mainloop pipeline depth while preserving epilogue staging.",
        {
            "cute_dsl": (
                "Change only pipeline_stages. Choose one supported explicit value: "
                "2 or 3. None preserves the pinned heuristic and is not a proposal."
            )
        },
    ),
    Strategy(
        CHANGE_EPILOGUE_STAGES,
        "Change epilogue pipeline depth while preserving A/B mainloop staging.",
        {
            "cute_dsl": (
                "Change only epilogue_stages. Choose 2 or 3. None preserves the "
                "pinned depth of 4 and is not a proposal. This action applies only "
                "to tile (128,256) with cluster (2,1)."
            )
        },
    ),
    Strategy(
        CHANGE_SHARED_MEMORY_SWIZZLE,
        "Change the shared-memory layout swizzle while preserving operand majorness.",
        {
            "cute_dsl": (
                "Change only shared_memory_swizzle. Choose heuristic or sw64. "
                "The sw64 value applies only to tile (128,256) with cluster (2,1)."
            )
        },
    ),
)


@dataclass(frozen=True, slots=True)
class CuteMutationProposal:
    """One auditable deterministic transition between typed CuTe states."""

    parent: CuteGemmProgram
    candidate: CuteGemmProgram
    strategy_id: str
    parameters: Mapping[str, object]
    validation: CuteLegalityResult

    @property
    def changed_fields(self) -> Mapping[str, Mapping[str, object]]:
        parent = self.parent.as_dict()
        candidate = self.candidate.as_dict()
        return {
            name: {"before": parent[name], "after": candidate[name]}
            for name in parent
            if parent[name] != candidate[name]
        }

    def transformation_evidence(self) -> dict[str, object]:
        return {
            "mechanism": "typed_mutation",
            "strategy_id": self.strategy_id,
            "parameters": dict(self.parameters),
            "parent_configuration_hash": self.parent.configuration_hash,
            "child_configuration_hash": self.candidate.configuration_hash,
            "changed_fields": dict(self.changed_fields),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "proposal_mechanism": "typed_mutation",
            "strategy_id": self.strategy_id,
            "representation": self.candidate.as_dict(),
            "representation_schema_version": self.candidate.schema_version,
            "configuration_hash": self.candidate.configuration_hash,
            "static_validation": self.validation.as_dict(),
            "transformation": self.transformation_evidence(),
        }


def mutate_cute_program(
    parent: CuteGemmProgram,
    strategy_id: str,
    parameters: Mapping[str, object],
) -> CuteMutationProposal:
    """Apply one typed mutation without compiling, profiling, or consuming B_gen."""

    parent_validation = validate_cute_gemm_program(parent)
    if not parent_validation.valid:
        raise ValueError("typed CuTe mutations require a statically valid parent")

    if strategy_id == CHANGE_CTA_TILE:
        values = _exact_integer_parameters(parameters, ("tile_m", "tile_n"))
        candidate = replace(parent, **values)
    elif strategy_id == CHANGE_CLUSTER_SHAPE:
        values = _exact_integer_parameters(parameters, ("cluster_m", "cluster_n"))
        candidate = replace(parent, **values)
    elif strategy_id == CHANGE_PIPELINE_STAGES:
        values = _exact_integer_parameters(parameters, ("pipeline_stages",))
        candidate = replace(parent, **values)
    elif strategy_id == CHANGE_EPILOGUE_STAGES:
        values = _exact_integer_parameters(parameters, ("epilogue_stages",))
        candidate = replace(parent, **values)
    elif strategy_id == CHANGE_SHARED_MEMORY_SWIZZLE:
        values = _exact_string_parameters(parameters, ("shared_memory_swizzle",))
        candidate = replace(parent, **values)
    else:
        raise ValueError(f"unknown CuTe mutation strategy {strategy_id!r}")

    if candidate == parent:
        raise ValueError("typed CuTe mutation must change the parent representation")
    return CuteMutationProposal(
        parent,
        candidate,
        strategy_id,
        values,
        validate_cute_gemm_program(candidate),
    )


def enumerate_cute_mutations(
    parent: CuteGemmProgram,
) -> tuple[CuteMutationProposal, ...]:
    """Enumerate the stable one-hop neighborhood of the initial design space."""

    proposals = []
    for tile_m, tile_n in CTA_TILE_CHOICES:
        if (tile_m, tile_n) != (parent.tile_m, parent.tile_n):
            proposals.append(
                mutate_cute_program(
                    parent,
                    CHANGE_CTA_TILE,
                    {"tile_m": tile_m, "tile_n": tile_n},
                )
            )
    for cluster_m, cluster_n in CLUSTER_SHAPE_CHOICES:
        if (cluster_m, cluster_n) != (parent.cluster_m, parent.cluster_n):
            proposals.append(
                mutate_cute_program(
                    parent,
                    CHANGE_CLUSTER_SHAPE,
                    {"cluster_m": cluster_m, "cluster_n": cluster_n},
                )
            )
    for pipeline_stages in (2, 3):
        if pipeline_stages != parent.pipeline_stages:
            proposals.append(
                mutate_cute_program(
                    parent,
                    CHANGE_PIPELINE_STAGES,
                    {"pipeline_stages": pipeline_stages},
                )
            )
    if (
        parent.tile_m,
        parent.tile_n,
        parent.cluster_m,
        parent.cluster_n,
    ) == (128, 256, 2, 1):
        for epilogue_stages in (2, 3):
            if epilogue_stages != parent.epilogue_stages:
                proposals.append(
                    mutate_cute_program(
                        parent,
                        CHANGE_EPILOGUE_STAGES,
                        {"epilogue_stages": epilogue_stages},
                    )
                )
        for shared_memory_swizzle in ("heuristic", "sw64"):
            if shared_memory_swizzle != parent.shared_memory_swizzle:
                proposals.append(
                    mutate_cute_program(
                        parent,
                        CHANGE_SHARED_MEMORY_SWIZZLE,
                        {"shared_memory_swizzle": shared_memory_swizzle},
                    )
                )
    return tuple(proposals)


class CuteMutationGenerator:
    """Deterministically realize untried typed neighbors for a selected strategy."""

    def __init__(self, renderer: PinnedCuteGemmRenderer | None = None) -> None:
        self._renderer = renderer or PinnedCuteGemmRenderer()
        self._issued: set[tuple[str, str, str]] = set()

    @staticmethod
    def proposal_budget_kind(_request: GenerationRequest) -> ProposalBudgetKind:
        return ProposalBudgetKind.MUTATION

    def can_generate(self, request: GenerationRequest) -> bool:
        return bool(self._available(request))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        available = self._available(request)
        if not available:
            raise RuntimeError("selected CuTe strategy has no untried typed mutation")
        proposal = available[0]
        child_hash = proposal.candidate.configuration_hash
        key = (proposal.parent.configuration_hash, proposal.strategy_id, child_hash)
        self._issued.add(key)
        serialized = proposal.as_dict()
        identity = json.dumps(
            {
                "parent": proposal.parent.configuration_hash,
                "strategy": proposal.strategy_id,
                "child": child_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return GenerationResult(
            generation_id=f"mutation:{uuid4()}",
            raw_output=proposal.candidate.canonical_json(),
            program=(
                self._renderer.render(proposal.candidate)
                if proposal.validation.valid
                else None
            ),
            prompt_hash=hashlib.sha256(identity.encode()).hexdigest(),
            metadata={
                "generator": "typed_mutation",
                "llm_call": False,
                **serialized,
            },
        )

    def _available(
        self, request: GenerationRequest
    ) -> tuple[CuteMutationProposal, ...]:
        if request.parent.backend != "cute_dsl":
            return ()
        parent = cute_gemm_program_from_source(request.parent.source)
        return tuple(
            proposal
            for proposal in enumerate_cute_mutations(parent)
            if proposal.strategy_id == request.strategy.id
            and proposal.validation.valid
            and (
                parent.configuration_hash,
                proposal.strategy_id,
                proposal.candidate.configuration_hash,
            )
            not in self._issued
        )


def _exact_integer_parameters(
    parameters: Mapping[str, object], expected: tuple[str, ...]
) -> dict[str, int]:
    if set(parameters) != set(expected):
        raise ValueError(f"mutation parameters must be exactly {list(expected)}")
    values = {name: parameters[name] for name in expected}
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise TypeError("mutation parameters must be integers")
    return values


def _exact_string_parameters(
    parameters: Mapping[str, object], expected: tuple[str, ...]
) -> dict[str, str]:
    if set(parameters) != set(expected):
        raise ValueError(f"mutation parameters must be exactly {list(expected)}")
    values = {name: parameters[name] for name in expected}
    if any(not isinstance(value, str) for value in values.values()):
        raise TypeError("mutation parameters must be strings")
    return values
