from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .cute_schedule import CuteSchedule, validate_cute_schedule
from .domain import KernelProgram


CUTE_GEMM_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CuteGemmProgram:
    """Versioned structural configuration for the fixed BF16 GEMM workload.

    Workload shape, dtypes, layouts, tolerances, and measurement policy are
    deliberately absent: they belong to WorkloadContract rather than search state.
    """

    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    mainloop: str = "hopper_wgmma_tma"
    wgmma_configuration: str = "pinned_default"
    pipeline_stages: int | None = None
    tma_copy_layout: str = "pinned_default"
    shared_memory_swizzle: str = "pinned_default"
    warp_specialization: str = "pinned_default"
    epilogue_policy: str = "pinned_default"
    schema_version: int = CUTE_GEMM_SCHEMA_VERSION

    @property
    def schedule(self) -> CuteSchedule:
        return CuteSchedule(
            self.tile_m,
            self.tile_n,
            self.cluster_m,
            self.cluster_n,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


REFERENCE_CUTE_GEMM = CuteGemmProgram(128, 256, 1, 1)


@dataclass(frozen=True, slots=True)
class CuteLegalityViolation:
    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class CuteLegalityResult:
    violations: tuple[CuteLegalityViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "violations": [asdict(item) for item in self.violations],
        }


def validate_cute_gemm_program(program: CuteGemmProgram) -> CuteLegalityResult:
    violations: list[CuteLegalityViolation] = []
    if program.schema_version != CUTE_GEMM_SCHEMA_VERSION:
        violations.append(
            CuteLegalityViolation(
                "unsupported_schema_version",
                f"schema version must be {CUTE_GEMM_SCHEMA_VERSION}",
                "schema_version",
            )
        )
    for reason in validate_cute_schedule(program.schedule):
        violations.append(CuteLegalityViolation("invalid_schedule", reason, "schedule"))
    supported_defaults = {
        "mainloop": (program.mainloop, "hopper_wgmma_tma"),
        "wgmma_configuration": (program.wgmma_configuration, "pinned_default"),
        "tma_copy_layout": (program.tma_copy_layout, "pinned_default"),
        "shared_memory_swizzle": (program.shared_memory_swizzle, "pinned_default"),
        "warp_specialization": (program.warp_specialization, "pinned_default"),
        "epilogue_policy": (program.epilogue_policy, "pinned_default"),
    }
    for field, (value, supported) in supported_defaults.items():
        if value != supported:
            violations.append(
                CuteLegalityViolation(
                    "unsupported_structural_value",
                    f"{field}={value!r} is not yet supported by the pinned renderer",
                    field,
                )
            )
    if program.pipeline_stages is not None:
        violations.append(
            CuteLegalityViolation(
                "unsupported_structural_value",
                "explicit pipeline stages are not yet supported by the pinned renderer",
                "pipeline_stages",
            )
        )
    return CuteLegalityResult(tuple(violations))


class CuteProgramRenderer(Protocol):
    def render(self, program: CuteGemmProgram) -> KernelProgram: ...


@dataclass(frozen=True, slots=True)
class PinnedCuteGemmRenderer:
    """Deterministically render a launcher for the pinned NVIDIA kernel template."""

    example_path: str = (
        "/opt/cutlass/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py"
    )

    def render(self, program: CuteGemmProgram) -> KernelProgram:
        legality = validate_cute_gemm_program(program)
        if not legality.valid:
            messages = "; ".join(item.message for item in legality.violations)
            raise ValueError(f"illegal CuTe GEMM program: {messages}")
        source = f'''# Generated deterministically from CuteGemmProgram schema v{program.schema_version}.
import importlib.util

import cutlass

EXAMPLE_PATH = {self.example_path!r}
CONFIGURATION_HASH = {program.configuration_hash!r}
KERNEL_MCTS_REPRESENTATION = {program.as_dict()!r}


def _load_example():
    spec = importlib.util.spec_from_file_location("kernel_mcts_pinned_cute_gemm", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned CuTe DSL example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    return _load_example().run(
        mnkl=(4096, 4096, 4096, 1),
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        c_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mn=({program.tile_m}, {program.tile_n}),
        cluster_shape_mn=({program.cluster_m}, {program.cluster_n}),
        tolerance=2.0e-2,
        warmup_iterations=10,
        iterations=30,
        skip_ref_check=False,
        use_cold_l2=False,
    )
'''
        return KernelProgram(source, "cute_dsl")
