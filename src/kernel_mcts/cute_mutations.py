from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .cute_program import (
    CuteGemmProgram,
    CuteLegalityResult,
    validate_cute_gemm_program,
)
from .cute_schedule import CLUSTER_SHAPE_CHOICES, CTA_TILE_CHOICES


CHANGE_CTA_TILE = "change_cta_tile"
CHANGE_CLUSTER_SHAPE = "change_cluster_shape"
CUTE_MUTATION_STRATEGIES = (CHANGE_CTA_TILE, CHANGE_CLUSTER_SHAPE)


@dataclass(frozen=True, slots=True)
class CuteMutationProposal:
    """One auditable deterministic transition between typed CuTe states."""

    parent: CuteGemmProgram
    candidate: CuteGemmProgram
    strategy_id: str
    parameters: Mapping[str, int]
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
    parameters: Mapping[str, int],
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
    return tuple(proposals)


def _exact_integer_parameters(
    parameters: Mapping[str, int], expected: tuple[str, ...]
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
