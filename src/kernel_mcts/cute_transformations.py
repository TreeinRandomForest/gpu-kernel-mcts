from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .cute_independent import IndependentCuteGemmKernel, make_independent_cute_gemm


@dataclass(frozen=True, slots=True)
class IndependentTransformationRealization:
    candidate: IndependentCuteGemmKernel
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IndependentStructuralTransformation:
    """Reusable intent plus one bounded, typed realization enumerator."""

    transformation_id: str
    semantic_strategy_id: str
    operation_families: tuple[str, ...]
    dtypes: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    parameter_schema: Mapping[str, str]
    coupled_fields: tuple[str, ...]
    _enumerate: Callable[
        [IndependentCuteGemmKernel], tuple[IndependentTransformationRealization, ...]
    ]

    def enumerate(
        self, parent: IndependentCuteGemmKernel
    ) -> tuple[IndependentTransformationRealization, ...]:
        return self._enumerate(parent)


def enumerate_independent_transformations(
    parent: IndependentCuteGemmKernel,
) -> tuple[
    tuple[IndependentStructuralTransformation, IndependentTransformationRealization],
    ...,
]:
    """Enumerate the admitted graph without changing its historical proposal order."""

    schedule = parent.mainloop.schedule
    mode = parent.mainloop.producer_consumer_mode
    if schedule == "prefetch":
        order = (
            (SPECIALIZATION, CTA_TILE)
            if mode == "warp_specialized"
            else (MAINLOOP_SCHEDULE, SPECIALIZATION)
        )
    else:
        order = (CTA_TILE, CLUSTER, SWIZZLE, PIPELINE, MAINLOOP_SCHEDULE)
    return tuple(
        (transformation, realization)
        for transformation in order
        for realization in transformation.enumerate(parent)
    )


def _candidate(parent: IndependentCuteGemmKernel, **changes: object) -> IndependentCuteGemmKernel:
    values = {
        "swizzle_bytes": parent.mainloop.a_copy.swizzle_bytes,
        "pipeline_stages": parent.mainloop.pipeline_stages,
        "tile_m": parent.mainloop.tile_m,
        "tile_n": parent.mainloop.tile_n,
        "cluster_m": parent.mainloop.cluster_m,
        "mainloop_schedule": parent.mainloop.schedule,
        "producer_consumer_mode": parent.mainloop.producer_consumer_mode,
    }
    values.update(changes)
    return make_independent_cute_gemm(**values)


def _cta_realizations(
    parent: IndependentCuteGemmKernel,
) -> tuple[IndependentTransformationRealization, ...]:
    tile = (parent.mainloop.tile_m, parent.mainloop.tile_n)
    root = (64, 256)
    wide = (128, 256)
    if parent.mainloop.schedule == "prefetch":
        if parent.mainloop.producer_consumer_mode != "warp_specialized":
            return ()
        targets = (wide,) if tile == root else ((root,) if tile == wide else ())
    else:
        if (
            parent.mainloop.a_copy.swizzle_bytes != 128
            or parent.mainloop.pipeline_stages != 3
            or (parent.mainloop.cluster_m, parent.mainloop.cluster_n) != (1, 1)
        ):
            return ()
        alternates = ((128, 256), (64, 128), (128, 128))
        targets = alternates if tile == root else ((root,) if tile in alternates else ())
    return tuple(
        IndependentTransformationRealization(
            _candidate(parent, tile_m=tile_m, tile_n=tile_n),
            {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "warp_groups_m": tile_m // 64,
                "instruction_n": tile_n,
                "epilogue_stages": (tile_m // 64) * (tile_n // 64),
            },
        )
        for tile_m, tile_n in targets
    )


def _cluster_realizations(parent: IndependentCuteGemmKernel) -> tuple[IndependentTransformationRealization, ...]:
    if (
        parent.mainloop.schedule != "serial"
        or parent.mainloop.a_copy.swizzle_bytes != 128
        or parent.mainloop.pipeline_stages != 3
        or (parent.mainloop.tile_m, parent.mainloop.tile_n) != (64, 256)
        or (parent.mainloop.cluster_m, parent.mainloop.cluster_n) not in ((1, 1), (2, 1))
    ):
        return ()
    cluster_m = 2 if parent.mainloop.cluster_m == 1 else 1
    return (
        IndependentTransformationRealization(
            _candidate(parent, cluster_m=cluster_m),
            {
                "cluster_m": cluster_m,
                "cluster_n": 1,
                "b_multicast_axis": "cluster_m" if cluster_m == 2 else "none",
            },
        ),
    )


def _swizzle_realizations(parent: IndependentCuteGemmKernel) -> tuple[IndependentTransformationRealization, ...]:
    if (
        parent.mainloop.schedule != "serial"
        or parent.mainloop.pipeline_stages != 3
        or (parent.mainloop.tile_m, parent.mainloop.tile_n) != (64, 256)
        or (parent.mainloop.cluster_m, parent.mainloop.cluster_n) != (1, 1)
    ):
        return ()
    swizzle = 64 if parent.mainloop.a_copy.swizzle_bytes == 128 else 128
    return (
        IndependentTransformationRealization(
            _candidate(parent, swizzle_bytes=swizzle),
            {"swizzle_bytes": swizzle},
        ),
    )


def _pipeline_realizations(parent: IndependentCuteGemmKernel) -> tuple[IndependentTransformationRealization, ...]:
    if (
        parent.mainloop.schedule != "serial"
        or parent.mainloop.a_copy.swizzle_bytes != 128
        or (parent.mainloop.tile_m, parent.mainloop.tile_n) != (64, 256)
        or (parent.mainloop.cluster_m, parent.mainloop.cluster_n) != (1, 1)
    ):
        return ()
    stages = parent.mainloop.pipeline_stages
    targets = (2, 4) if stages == 3 else ((3,) if stages in (2, 4) else ())
    return tuple(
        IndependentTransformationRealization(
            _candidate(parent, pipeline_stages=target),
            {"pipeline_stages": target},
        )
        for target in targets
    )


def _schedule_realizations(parent: IndependentCuteGemmKernel) -> tuple[IndependentTransformationRealization, ...]:
    root_shape = (
        (parent.mainloop.tile_m, parent.mainloop.tile_n) == (64, 256)
        and (parent.mainloop.cluster_m, parent.mainloop.cluster_n) == (1, 1)
        and parent.mainloop.a_copy.swizzle_bytes == 128
        and parent.mainloop.pipeline_stages == 3
    )
    if not root_shape or parent.mainloop.producer_consumer_mode != "cooperative":
        return ()
    target = "prefetch" if parent.mainloop.schedule == "serial" else "serial"
    return (
        IndependentTransformationRealization(
            _candidate(parent, mainloop_schedule=target),
            {"mainloop_schedule": target},
        ),
    )


def _specialization_realizations(parent: IndependentCuteGemmKernel) -> tuple[IndependentTransformationRealization, ...]:
    if parent.mainloop.schedule != "prefetch":
        return ()
    tile = (parent.mainloop.tile_m, parent.mainloop.tile_n)
    if parent.mainloop.producer_consumer_mode == "cooperative" and tile == (64, 256):
        target = "warp_specialized"
    elif parent.mainloop.producer_consumer_mode == "warp_specialized" and tile == (64, 256):
        target = "cooperative"
    else:
        return ()
    return (
        IndependentTransformationRealization(
            _candidate(parent, producer_consumer_mode=target),
            {"producer_consumer_mode": target},
        ),
    )


_COMMON = {
    "operation_families": ("gemm",),
    "dtypes": ("bfloat16",),
    "required_capabilities": ("sm90a", "tma", "wgmma"),
}

CTA_TILE = IndependentStructuralTransformation(
    "independent.change_cta_decomposition.v1",
    "change_cta_tile",
    parameter_schema={"tile_m": "int", "tile_n": "int"},
    coupled_fields=("mainloop", "consumer", "epilogue", "execution", "launch_grid"),
    _enumerate=_cta_realizations,
    **_COMMON,
)
CLUSTER = IndependentStructuralTransformation(
    "independent.change_cluster_multicast.v1",
    "change_cluster_shape",
    parameter_schema={"cluster_m": "int", "cluster_n": "int"},
    coupled_fields=("mainloop", "execution", "cluster_launch"),
    _enumerate=_cluster_realizations,
    **_COMMON,
)
SWIZZLE = IndependentStructuralTransformation(
    "independent.change_smem_swizzle.v1",
    "change_shared_memory_swizzle",
    parameter_schema={"swizzle_bytes": "int"},
    coupled_fields=("mainloop.a_copy", "mainloop.b_copy"),
    _enumerate=_swizzle_realizations,
    **_COMMON,
)
PIPELINE = IndependentStructuralTransformation(
    "independent.change_pipeline_depth.v1",
    "change_pipeline_stages",
    parameter_schema={"pipeline_stages": "int"},
    coupled_fields=("mainloop", "execution", "epilogue.storage_offset_bytes"),
    _enumerate=_pipeline_realizations,
    **_COMMON,
)
MAINLOOP_SCHEDULE = IndependentStructuralTransformation(
    "independent.change_mainloop_schedule.v1",
    "change_mainloop_schedule",
    parameter_schema={"mainloop_schedule": "serial|prefetch"},
    coupled_fields=("mainloop.schedule", "execution"),
    _enumerate=_schedule_realizations,
    **_COMMON,
)
SPECIALIZATION = IndependentStructuralTransformation(
    "independent.change_producer_consumer_ownership.v1",
    "change_producer_consumer_specialization",
    parameter_schema={"producer_consumer_mode": "cooperative|warp_specialized"},
    coupled_fields=("mainloop", "execution", "cta_threads"),
    _enumerate=_specialization_realizations,
    **_COMMON,
)

INDEPENDENT_STRUCTURAL_TRANSFORMATIONS = (
    CTA_TILE,
    CLUSTER,
    SWIZZLE,
    PIPELINE,
    MAINLOOP_SCHEDULE,
    SPECIALIZATION,
)
