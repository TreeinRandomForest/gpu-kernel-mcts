from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .cute_schedule import CuteSchedule, validate_cute_schedule
from .domain import KernelProgram


CUTE_GEMM_SCHEMA_VERSION = 3
SUPPORTED_MAINLOOP_PIPELINE_STAGES = (2, 3, 4)
SUPPORTED_EPILOGUE_PIPELINE_STAGES = (2, 3)
SUPPORTED_WGMMA_CONFIGURATIONS = ("pinned_default", "single_warp_group")
SUPPORTED_SHARED_MEMORY_SWIZZLES = ("heuristic", "sw64")


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
    epilogue_stages: int | None = None
    tma_copy_layout: str = "pinned_default"
    shared_memory_swizzle: str = "heuristic"
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


def cute_gemm_program_from_source(source: str) -> CuteGemmProgram:
    tree = ast.parse(source)
    values = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "KERNEL_MCTS_REPRESENTATION"
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("program must define one literal KERNEL_MCTS_REPRESENTATION")
    try:
        return CuteGemmProgram(**values[0])
    except TypeError as error:
        raise ValueError("program contains an invalid CuTe representation") from error


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
        "tma_copy_layout": (program.tma_copy_layout, "pinned_default"),
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
    if program.shared_memory_swizzle not in SUPPORTED_SHARED_MEMORY_SWIZZLES:
        violations.append(
            CuteLegalityViolation(
                "unsupported_structural_value",
                "shared_memory_swizzle must be one of "
                f"{SUPPORTED_SHARED_MEMORY_SWIZZLES}",
                "shared_memory_swizzle",
            )
        )
    if program.shared_memory_swizzle == "sw64" and (
        program.tile_m,
        program.tile_n,
        program.cluster_m,
        program.cluster_n,
    ) != (128, 256, 2, 1):
        violations.append(
            CuteLegalityViolation(
                "incompatible_structural_values",
                "sw64 shared-memory layout is validated only for tile "
                "(128,256) with cluster (2,1)",
                "shared_memory_swizzle",
            )
        )
    if program.wgmma_configuration not in SUPPORTED_WGMMA_CONFIGURATIONS:
        violations.append(
            CuteLegalityViolation(
                "unsupported_structural_value",
                "wgmma_configuration must be one of "
                f"{SUPPORTED_WGMMA_CONFIGURATIONS}",
                "wgmma_configuration",
            )
        )
    if (
        program.pipeline_stages is not None
        and program.pipeline_stages not in SUPPORTED_MAINLOOP_PIPELINE_STAGES
    ):
        violations.append(
            CuteLegalityViolation(
                "unsupported_structural_value",
                "pipeline_stages must be one of "
                f"{SUPPORTED_MAINLOOP_PIPELINE_STAGES} or None",
                "pipeline_stages",
            )
        )
    if (
        program.epilogue_stages is not None
        and program.epilogue_stages not in SUPPORTED_EPILOGUE_PIPELINE_STAGES
    ):
        violations.append(
            CuteLegalityViolation(
                "unsupported_structural_value",
                "epilogue_stages must be one of "
                f"{SUPPORTED_EPILOGUE_PIPELINE_STAGES} or None",
                "epilogue_stages",
            )
        )
    if program.epilogue_stages is not None and (
        program.tile_m,
        program.tile_n,
        program.cluster_m,
        program.cluster_n,
    ) != (128, 256, 2, 1):
        violations.append(
            CuteLegalityViolation(
                "incompatible_structural_values",
                "explicit epilogue_stages are validated only for tile "
                "(128,256) with cluster (2,1)",
                "epilogue_stages",
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
import cutlass
import importlib.util

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
    module = _load_example()
    wgmma_configuration = {program.wgmma_configuration!r}
    if wgmma_configuration == "single_warp_group":
        pinned_init = module.HopperWgmmaGemmKernel.__init__

        def init_with_single_warp_group(self, *args, **kwargs):
            pinned_init(self, *args, **kwargs)
            self.atom_layout_mnk = (1, 1, 1)
            self.mma_warp_groups = 1
            self.threads_per_cta = self.num_threads_per_warp_group

        module.HopperWgmmaGemmKernel.__init__ = init_with_single_warp_group
    pipeline_stages = {program.pipeline_stages!r}
    if pipeline_stages is not None:
        pinned_mainloop_compute_stages = module.HopperWgmmaGemmKernel._compute_stages

        def compute_stages_with_mainloop_override(
            tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
        ):
            _, epi_stage = pinned_mainloop_compute_stages(
                tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
            )
            return pipeline_stages, epi_stage

        module.HopperWgmmaGemmKernel._compute_stages = staticmethod(
            compute_stages_with_mainloop_override
        )
    epilogue_stages = {program.epilogue_stages!r}
    if epilogue_stages is not None:
        pinned_epilogue_compute_stages = module.HopperWgmmaGemmKernel._compute_stages

        def compute_stages_with_epilogue_override(
            tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
        ):
            mainloop_stages, _ = pinned_epilogue_compute_stages(
                tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy
            )
            return mainloop_stages, epilogue_stages

        module.HopperWgmmaGemmKernel._compute_stages = staticmethod(
            compute_stages_with_epilogue_override
        )
    shared_memory_swizzle = {program.shared_memory_swizzle!r}
    if shared_memory_swizzle == "sw64":
        pinned_layout_atom_selector = module.sm90_utils.get_smem_layout_atom

        def select_sw64_layout_atom(
            layout, element_type, major_mode_size, *, loc=None, ip=None
        ):
            selected = pinned_layout_atom_selector(
                layout,
                element_type,
                major_mode_size,
                loc=loc,
                ip=ip,
            )
            selected_name = getattr(
                selected, "name", str(selected).rsplit(".", 1)[-1]
            )
            target_name = (
                "MN_SW64" if selected_name.startswith("MN_") else "K_SW64"
            )
            return getattr(type(selected), target_name)

        module.sm90_utils.get_smem_layout_atom = select_sw64_layout_atom
    return module.run(
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
