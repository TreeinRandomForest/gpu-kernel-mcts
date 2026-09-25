from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


INDEPENDENT_TMA_SMEM_SCHEMA_VERSION = 3
H100_MAX_SHARED_MEMORY_BYTES = 227_328
BF16_BYTES = 2

Operand = Literal["a", "b"]
MulticastAxis = Literal["none", "cluster_m"]


@dataclass(frozen=True, slots=True)
class IndependentTmaOperandCopy:
    """One operand's complete global-to-shared TMA copy contract."""

    operand: Operand
    global_major: str
    tile_rows: int
    tile_columns: int
    element_bytes: int
    alignment_bytes: int
    shared_major: str
    swizzle_bytes: int
    multicast_axis: MulticastAxis
    stage_stride_bytes: int
    storage_offset_bytes: int

    @property
    def tile_bytes(self) -> int:
        return self.tile_rows * self.tile_columns * self.element_bytes


@dataclass(frozen=True, slots=True)
class IndependentTmaSmemMainloop:
    """A bounded TMA/shared-memory contract for an independent Hopper GEMM.

    This is deliberately not the expert-template ``CuteGemmProgram``. It captures
    the coordinated producer-side memory contract that a future complete CuTe DSL
    renderer will combine with WGMMA consumers and an epilogue.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    cluster_m: int
    cluster_n: int
    pipeline_stages: int
    barrier_slots: int
    producer_warp_groups: int
    a_copy: IndependentTmaOperandCopy
    b_copy: IndependentTmaOperandCopy
    schema_version: int = INDEPENDENT_TMA_SMEM_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def shared_memory_bytes(self) -> int:
        return max(
            copy.storage_offset_bytes
            + copy.stage_stride_bytes * self.pipeline_stages
            for copy in (self.a_copy, self.b_copy)
        )


@dataclass(frozen=True, slots=True)
class IndependentWgmmaConsumer:
    """Warp-group ownership of one shared-memory mainloop tile."""

    instruction_m: int
    instruction_n: int
    instruction_k: int
    warp_groups_m: int
    warp_groups_n: int
    accumulator_dtype: str
    a_source: str
    b_source: str
    accumulator_owner: str


@dataclass(frozen=True, slots=True)
class IndependentGemmEpilogue:
    """Accumulator-to-output ownership and staging contract."""

    accumulator_source: str
    output_dtype: str
    output_major: str
    store_kind: str
    tile_m: int
    tile_n: int
    pipeline_stages: int
    barrier_slots: int
    alignment_bytes: int
    storage_offset_bytes: int
    stage_stride_bytes: int
    allocation_padding_bytes: int

    @property
    def shared_memory_bytes(self) -> int:
        return (
            self.pipeline_stages * self.stage_stride_bytes
            + self.allocation_padding_bytes
        )


@dataclass(frozen=True, slots=True)
class IndependentExecutionAgent:
    """A GPU agent assigned ownership of one or more execution phases."""

    agent_id: str
    scope: str
    count: int
    role: str


@dataclass(frozen=True, slots=True)
class IndependentBufferRing:
    """A statically allocated producer/consumer pipeline ring."""

    buffer_id: str
    memory_space: str
    stages: int
    producer_agent: str
    consumer_agent: str
    barrier_protocol: str


@dataclass(frozen=True, slots=True)
class IndependentExecutionPhase:
    """One dependency-ordered phase in the generated device kernel."""

    phase_id: str
    agent_id: str
    operation: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    waits_for: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndependentExecutionSchedule:
    """Agent, buffer, and dependency graph for dynamic kernel lowering."""

    agents: tuple[IndependentExecutionAgent, ...]
    buffers: tuple[IndependentBufferRing, ...]
    phases: tuple[IndependentExecutionPhase, ...]


@dataclass(frozen=True, slots=True)
class IndependentCuteGemmKernel:
    """Complete typed structure for the first independent Hopper GEMM."""

    mainloop: IndependentTmaSmemMainloop
    consumer: IndependentWgmmaConsumer
    epilogue: IndependentGemmEpilogue
    execution: IndependentExecutionSchedule
    schema_version: int = INDEPENDENT_TMA_SMEM_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def shared_memory_bytes(self) -> int:
        return max(
            self.mainloop.shared_memory_bytes,
            self.epilogue.storage_offset_bytes + self.epilogue.shared_memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class IndependentTmaSmemViolation:
    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class IndependentTmaSmemLegality:
    violations: tuple[IndependentTmaSmemViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "violations": [asdict(item) for item in self.violations],
        }


def make_independent_tma_smem_mainloop(
    *,
    swizzle_bytes: int = 128,
    pipeline_stages: int = 3,
    tile_m: int = 64,
) -> IndependentTmaSmemMainloop:
    """Build one of the two initial coordinated BF16 copy/layout variants."""

    tile_n, tile_k = 256, 64
    stages = pipeline_stages
    a_tile_bytes = tile_m * tile_k * BF16_BYTES
    b_tile_bytes = tile_k * tile_n * BF16_BYTES
    return IndependentTmaSmemMainloop(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=1,
        cluster_n=1,
        pipeline_stages=stages,
        barrier_slots=stages,
        producer_warp_groups=1,
        a_copy=IndependentTmaOperandCopy(
            operand="a",
            global_major="k",
            tile_rows=tile_m,
            tile_columns=tile_k,
            element_bytes=BF16_BYTES,
            alignment_bytes=16,
            shared_major="k",
            swizzle_bytes=swizzle_bytes,
            multicast_axis="none",
            stage_stride_bytes=a_tile_bytes,
            storage_offset_bytes=0,
        ),
        b_copy=IndependentTmaOperandCopy(
            operand="b",
            global_major="k",
            tile_rows=tile_k,
            tile_columns=tile_n,
            element_bytes=BF16_BYTES,
            alignment_bytes=16,
            shared_major="k",
            swizzle_bytes=swizzle_bytes,
            multicast_axis="none",
            stage_stride_bytes=b_tile_bytes,
            storage_offset_bytes=a_tile_bytes * stages,
        ),
    )


def make_independent_cute_gemm(
    *,
    swizzle_bytes: int = 128,
    pipeline_stages: int = 3,
    tile_m: int = 64,
) -> IndependentCuteGemmKernel:
    """Build the first complete structural GEMM contract.

    The contract is complete as typed design state, but does not become executable
    until a CuTe lowering implements every ownership and synchronization field.
    """

    mainloop = make_independent_tma_smem_mainloop(
        swizzle_bytes=swizzle_bytes,
        pipeline_stages=pipeline_stages,
        tile_m=tile_m,
    )
    consumer_warp_groups = tile_m // 64
    epilogue_stages = consumer_warp_groups * 4
    return IndependentCuteGemmKernel(
        mainloop=mainloop,
        consumer=IndependentWgmmaConsumer(
            instruction_m=64,
            instruction_n=256,
            instruction_k=16,
            warp_groups_m=consumer_warp_groups,
            warp_groups_n=1,
            accumulator_dtype="float32",
            a_source="shared_memory",
            b_source="shared_memory",
            accumulator_owner="consumer_warp_group",
        ),
        epilogue=IndependentGemmEpilogue(
            accumulator_source="registers",
            output_dtype="bfloat16",
            output_major="n",
            store_kind="tma",
            tile_m=64,
            tile_n=64,
            pipeline_stages=epilogue_stages,
            barrier_slots=epilogue_stages,
            alignment_bytes=16,
            storage_offset_bytes=mainloop.shared_memory_bytes,
            stage_stride_bytes=64 * 64 * BF16_BYTES,
            allocation_padding_bytes=64 * 64 * BF16_BYTES,
        ),
        execution=IndependentExecutionSchedule(
            agents=(
                IndependentExecutionAgent(
                    "tma_load_agent", "warp", 1, "mainloop_producer"
                ),
                IndependentExecutionAgent(
                    "wgmma_agents",
                    "warp_group",
                    consumer_warp_groups,
                    "mainloop_consumer",
                ),
                IndependentExecutionAgent(
                    "epilogue_agents",
                    "thread",
                    consumer_warp_groups * 128,
                    "epilogue_producer",
                ),
                IndependentExecutionAgent(
                    "tma_store_agent", "warp", 1, "epilogue_consumer"
                ),
            ),
            buffers=(
                IndependentBufferRing(
                    "shared_a",
                    "shared_memory",
                    mainloop.pipeline_stages,
                    "tma_load_agent",
                    "wgmma_agents",
                    "tma_async_full_empty",
                ),
                IndependentBufferRing(
                    "shared_b",
                    "shared_memory",
                    mainloop.pipeline_stages,
                    "tma_load_agent",
                    "wgmma_agents",
                    "tma_async_full_empty",
                ),
                IndependentBufferRing(
                    "shared_c",
                    "shared_memory",
                    epilogue_stages,
                    "epilogue_agents",
                    "tma_store_agent",
                    "tma_store_pipeline",
                ),
            ),
            phases=(
                IndependentExecutionPhase(
                    "tma_load",
                    "tma_load_agent",
                    "global_to_shared_tma",
                    ("global_a", "global_b"),
                    ("shared_a", "shared_b"),
                    signals=("mainloop_full",),
                ),
                IndependentExecutionPhase(
                    "wgmma",
                    "wgmma_agents",
                    "shared_to_register_wgmma",
                    ("shared_a", "shared_b"),
                    ("accumulator",),
                    waits_for=("mainloop_full",),
                    signals=("mainloop_empty", "accumulator_ready"),
                ),
                IndependentExecutionPhase(
                    "epilogue_stage",
                    "epilogue_agents",
                    "register_to_shared_convert",
                    ("accumulator",),
                    ("shared_c",),
                    waits_for=("accumulator_ready",),
                    signals=("epilogue_ready",),
                ),
                IndependentExecutionPhase(
                    "tma_store",
                    "tma_store_agent",
                    "shared_to_global_tma",
                    ("shared_c",),
                    ("global_c",),
                    waits_for=("epilogue_ready",),
                    signals=("epilogue_empty",),
                ),
            ),
        ),
    )


def independent_cute_gemm_from_dict(
    value: dict[str, object],
) -> IndependentCuteGemmKernel:
    """Reconstruct the versioned typed kernel from canonical JSON data."""

    try:
        mainloop_value = dict(value["mainloop"])
        mainloop = IndependentTmaSmemMainloop(
            **{
                **mainloop_value,
                "a_copy": IndependentTmaOperandCopy(**dict(mainloop_value["a_copy"])),
                "b_copy": IndependentTmaOperandCopy(**dict(mainloop_value["b_copy"])),
            }
        )
        execution_value = dict(value["execution"])
        execution = IndependentExecutionSchedule(
            agents=tuple(
                IndependentExecutionAgent(**dict(item))
                for item in execution_value["agents"]
            ),
            buffers=tuple(
                IndependentBufferRing(**dict(item))
                for item in execution_value["buffers"]
            ),
            phases=tuple(
                IndependentExecutionPhase(
                    **{
                        **dict(item),
                        "reads": tuple(dict(item)["reads"]),
                        "writes": tuple(dict(item)["writes"]),
                        "waits_for": tuple(dict(item).get("waits_for", ())),
                        "signals": tuple(dict(item).get("signals", ())),
                    }
                )
                for item in execution_value["phases"]
            ),
        )
        return IndependentCuteGemmKernel(
            mainloop=mainloop,
            consumer=IndependentWgmmaConsumer(**dict(value["consumer"])),
            epilogue=IndependentGemmEpilogue(**dict(value["epilogue"])),
            execution=execution,
            schema_version=int(value.get("schema_version", INDEPENDENT_TMA_SMEM_SCHEMA_VERSION)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid independent CuTe GEMM representation") from error


def validate_independent_tma_smem_mainloop(
    plan: IndependentTmaSmemMainloop,
) -> IndependentTmaSmemLegality:
    violations: list[IndependentTmaSmemViolation] = []

    def reject(code: str, message: str, field: str | None = None) -> None:
        violations.append(IndependentTmaSmemViolation(code, message, field))

    if plan.schema_version != INDEPENDENT_TMA_SMEM_SCHEMA_VERSION:
        reject(
            "unsupported_schema_version",
            f"schema version must be {INDEPENDENT_TMA_SMEM_SCHEMA_VERSION}",
            "schema_version",
        )
    if (plan.tile_m, plan.tile_n, plan.tile_k) not in (
        (64, 256, 64),
        (128, 256, 64),
    ):
        reject(
            "unsupported_tile",
            "the independent mainloop supports tiles (64,256,64) and (128,256,64)",
            "tile",
        )
    if (plan.cluster_m, plan.cluster_n) != (1, 1):
        reject(
            "unsupported_cluster",
            "the initial independent mainloop supports only cluster (1,1)",
            "cluster",
        )
    if plan.pipeline_stages not in (2, 3):
        reject(
            "unsupported_pipeline_depth",
            "the independent mainloop supports two or three pipeline stages",
            "pipeline_stages",
        )
    if plan.barrier_slots != plan.pipeline_stages:
        reject(
            "incomplete_barrier_ring",
            "one arrival-barrier slot is required per pipeline stage",
            "barrier_slots",
        )
    if plan.producer_warp_groups != 1:
        reject(
            "unsupported_producer_partition",
            "the initial contract has exactly one producer warp group",
            "producer_warp_groups",
        )

    expected_tiles = {
        "a": (plan.tile_m, plan.tile_k),
        "b": (plan.tile_k, plan.tile_n),
    }
    expected_multicast = {"a": "none", "b": "none"}
    copies = (plan.a_copy, plan.b_copy)
    if tuple(copy.operand for copy in copies) != ("a", "b"):
        reject(
            "incomplete_operand_contract",
            "the contract must contain ordered A and B operand copies",
            "copies",
        )
    for name, copy in (("a_copy", plan.a_copy), ("b_copy", plan.b_copy)):
        if copy.operand not in expected_tiles:
            reject("unknown_operand", f"unsupported operand {copy.operand!r}", name)
            continue
        if (copy.tile_rows, copy.tile_columns) != expected_tiles[copy.operand]:
            reject(
                "incomplete_tile_coverage",
                f"{copy.operand.upper()} copy does not cover its CTA K tile",
                name,
            )
        if copy.element_bytes != BF16_BYTES:
            reject("unsupported_element_size", "copies must contain BF16 elements", name)
        if copy.global_major != "k" or copy.shared_major != "k":
            reject(
                "unsupported_majorness",
                "the initial workload requires K-major global and shared operands",
                name,
            )
        if copy.alignment_bytes < 16 or copy.alignment_bytes % 16:
            reject(
                "insufficient_tma_alignment",
                "TMA operands require at least 16-byte alignment",
                name,
            )
        if copy.swizzle_bytes not in (64, 128):
            reject(
                "unsupported_shared_layout",
                "shared-memory swizzle must be 64 or 128 bytes",
                name,
            )
        if copy.multicast_axis != expected_multicast[copy.operand]:
            reject(
                "incompatible_multicast_partition",
                "the initial single-CTA contract does not multicast operands",
                name,
            )
        if copy.stage_stride_bytes < copy.tile_bytes:
            reject(
                "overlapping_pipeline_stages",
                "stage stride is smaller than one operand tile",
                name,
            )

    if plan.a_copy.swizzle_bytes != plan.b_copy.swizzle_bytes:
        reject(
            "partial_layout_transition",
            "the initial variants change A and B shared-memory swizzles together",
            "copies",
        )

    a_range = (
        plan.a_copy.storage_offset_bytes,
        plan.a_copy.storage_offset_bytes
        + plan.a_copy.stage_stride_bytes * plan.pipeline_stages,
    )
    b_range = (
        plan.b_copy.storage_offset_bytes,
        plan.b_copy.storage_offset_bytes
        + plan.b_copy.stage_stride_bytes * plan.pipeline_stages,
    )
    if max(a_range[0], b_range[0]) < min(a_range[1], b_range[1]):
        reject(
            "overlapping_operand_storage",
            "A and B pipeline storage regions overlap",
            "copies",
        )
    if plan.shared_memory_bytes > H100_MAX_SHARED_MEMORY_BYTES:
        reject(
            "shared_memory_capacity_exceeded",
            "pipeline storage exceeds the H100 per-CTA shared-memory bound",
            "copies",
        )
    return IndependentTmaSmemLegality(tuple(violations))


def validate_independent_cute_gemm(
    kernel: IndependentCuteGemmKernel,
) -> IndependentTmaSmemLegality:
    violations = list(
        validate_independent_tma_smem_mainloop(kernel.mainloop).violations
    )

    def reject(code: str, message: str, field: str | None = None) -> None:
        violations.append(IndependentTmaSmemViolation(code, message, field))

    if kernel.schema_version != INDEPENDENT_TMA_SMEM_SCHEMA_VERSION:
        reject(
            "unsupported_schema_version",
            f"schema version must be {INDEPENDENT_TMA_SMEM_SCHEMA_VERSION}",
            "schema_version",
        )

    consumer = kernel.consumer
    mainloop = kernel.mainloop
    if (consumer.instruction_m, consumer.instruction_n, consumer.instruction_k) != (
        64,
        256,
        16,
    ):
        reject(
            "unsupported_wgmma_instruction",
            "the first consumer uses a 64x256x16 Hopper WGMMA instruction shape",
            "consumer",
        )
    if (
        consumer.instruction_m * consumer.warp_groups_m != mainloop.tile_m
        or consumer.instruction_n * consumer.warp_groups_n != mainloop.tile_n
        or mainloop.tile_k % consumer.instruction_k
    ):
        reject(
            "incomplete_wgmma_tile_coverage",
            "WGMMA ownership must cover the complete CTA M/N/K tile",
            "consumer",
        )
    if consumer.accumulator_dtype != "float32":
        reject(
            "unsupported_accumulator_dtype",
            "the BF16 workload accumulates in float32",
            "consumer.accumulator_dtype",
        )
    if consumer.a_source != "shared_memory" or consumer.b_source != "shared_memory":
        reject(
            "incompatible_wgmma_source",
            "the initial WGMMA consumer reads A and B from shared memory",
            "consumer",
        )
    if consumer.accumulator_owner != "consumer_warp_group":
        reject(
            "unsupported_accumulator_owner",
            "consumer warp groups must own their register accumulators",
            "consumer.accumulator_owner",
        )

    epilogue = kernel.epilogue
    if epilogue.accumulator_source != "registers":
        reject(
            "incompatible_epilogue_source",
            "the epilogue must consume WGMMA register accumulators",
            "epilogue.accumulator_source",
        )
    if (
        epilogue.output_dtype,
        epilogue.output_major,
        epilogue.store_kind,
    ) != ("bfloat16", "n", "tma"):
        reject(
            "unsupported_epilogue_contract",
            "the initial epilogue performs an N-major BF16 TMA store",
            "epilogue",
        )
    if (
        mainloop.tile_m % epilogue.tile_m
        or mainloop.tile_n % epilogue.tile_n
    ):
        reject(
            "incomplete_epilogue_tile_coverage",
            "epilogue tiles must evenly cover the CTA output tile",
            "epilogue",
        )
    expected_epilogue_stages = (
        mainloop.tile_m // epilogue.tile_m
    ) * (mainloop.tile_n // epilogue.tile_n)
    if (
        epilogue.pipeline_stages != expected_epilogue_stages
        or epilogue.barrier_slots != expected_epilogue_stages
    ):
        reject(
            "incomplete_epilogue_pipeline",
            "the no-reuse TMA-store epilogue requires one stage and barrier per output tile",
            "epilogue",
        )
    if epilogue.alignment_bytes < 16 or epilogue.alignment_bytes % 16:
        reject(
            "insufficient_epilogue_alignment",
            "the output TMA store requires at least 16-byte alignment",
            "epilogue.alignment_bytes",
        )
    expected_epilogue_stage_bytes = (
        epilogue.tile_m * epilogue.tile_n * BF16_BYTES
    )
    if epilogue.stage_stride_bytes < expected_epilogue_stage_bytes:
        reject(
            "overlapping_epilogue_stages",
            "epilogue stage stride is smaller than one output tile",
            "epilogue.stage_stride_bytes",
        )
    if (
        epilogue.allocation_padding_bytes < expected_epilogue_stage_bytes
        or epilogue.allocation_padding_bytes % epilogue.alignment_bytes
    ):
        reject(
            "insufficient_epilogue_layout_padding",
            "the initial composed epilogue layout requires one aligned guard tile",
            "epilogue.allocation_padding_bytes",
        )
    if epilogue.storage_offset_bytes < mainloop.shared_memory_bytes:
        reject(
            "overlapping_mainloop_epilogue_storage",
            "dedicated epilogue storage overlaps the mainloop pipeline",
            "epilogue.storage_offset_bytes",
        )
    if kernel.shared_memory_bytes > H100_MAX_SHARED_MEMORY_BYTES:
        reject(
            "shared_memory_capacity_exceeded",
            "combined mainloop and epilogue storage exceeds the H100 CTA bound",
            "epilogue",
        )

    execution = kernel.execution
    agent_ids = [agent.agent_id for agent in execution.agents]
    buffer_ids = [buffer.buffer_id for buffer in execution.buffers]
    phase_ids = [phase.phase_id for phase in execution.phases]
    for field, identifiers in (
        ("execution.agents", agent_ids),
        ("execution.buffers", buffer_ids),
        ("execution.phases", phase_ids),
    ):
        if len(identifiers) != len(set(identifiers)):
            reject(
                "duplicate_execution_identifier",
                f"{field} must contain unique identifiers",
                field,
            )
    known_agents = set(agent_ids)
    known_buffers = set(buffer_ids)
    for agent in execution.agents:
        if agent.count <= 0:
            reject(
                "invalid_agent_count",
                f"agent {agent.agent_id!r} must have positive count",
                "execution.agents",
            )
    agent_counts = {agent.agent_id: agent.count for agent in execution.agents}
    expected_consumer_groups = consumer.warp_groups_m * consumer.warp_groups_n
    if agent_counts.get("wgmma_agents") != expected_consumer_groups:
        reject(
            "inconsistent_consumer_agents",
            "execution must contain one WGMMA agent per consumer warp group",
            "execution.agents",
        )
    if agent_counts.get("epilogue_agents") != expected_consumer_groups * 128:
        reject(
            "inconsistent_epilogue_agents",
            "epilogue ownership must include every consumer warp-group thread",
            "execution.agents",
        )
    for buffer in execution.buffers:
        if buffer.producer_agent not in known_agents:
            reject(
                "unknown_buffer_producer",
                f"buffer {buffer.buffer_id!r} names an unknown producer",
                "execution.buffers",
            )
        if buffer.consumer_agent not in known_agents:
            reject(
                "unknown_buffer_consumer",
                f"buffer {buffer.buffer_id!r} names an unknown consumer",
                "execution.buffers",
            )
        if buffer.stages <= 0:
            reject(
                "invalid_buffer_ring",
                f"buffer {buffer.buffer_id!r} must have positive stage count",
                "execution.buffers",
            )
    expected_buffer_stages = {
        "shared_a": mainloop.pipeline_stages,
        "shared_b": mainloop.pipeline_stages,
        "shared_c": epilogue.pipeline_stages,
    }
    actual_buffer_stages = {
        buffer.buffer_id: buffer.stages for buffer in execution.buffers
    }
    if actual_buffer_stages != expected_buffer_stages:
        reject(
            "inconsistent_buffer_rings",
            "execution buffer rings must match mainloop and epilogue stages",
            "execution.buffers",
        )

    signal_producers: dict[str, str] = {}
    for phase in execution.phases:
        if phase.agent_id not in known_agents:
            reject(
                "unknown_phase_agent",
                f"phase {phase.phase_id!r} names an unknown agent",
                "execution.phases",
            )
        for buffer_id in (*phase.reads, *phase.writes):
            if buffer_id.startswith("shared_") and buffer_id not in known_buffers:
                reject(
                    "unknown_phase_buffer",
                    f"phase {phase.phase_id!r} names unknown buffer {buffer_id!r}",
                    "execution.phases",
                )
        for signal in phase.signals:
            if signal in signal_producers:
                reject(
                    "duplicate_signal_producer",
                    f"signal {signal!r} has more than one producer",
                    "execution.phases",
                )
            signal_producers[signal] = phase.phase_id

    dependencies: dict[str, set[str]] = {phase_id: set() for phase_id in phase_ids}
    for phase in execution.phases:
        for signal in phase.waits_for:
            producer = signal_producers.get(signal)
            if producer is None:
                reject(
                    "unproduced_phase_signal",
                    f"phase {phase.phase_id!r} waits for unproduced signal {signal!r}",
                    "execution.phases",
                )
            else:
                dependencies[phase.phase_id].add(producer)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(phase_id: str) -> bool:
        if phase_id in visiting:
            return False
        if phase_id in visited:
            return True
        visiting.add(phase_id)
        if any(not visit(parent) for parent in dependencies.get(phase_id, ())):
            return False
        visiting.remove(phase_id)
        visited.add(phase_id)
        return True

    if any(not visit(phase_id) for phase_id in phase_ids):
        reject(
            "cyclic_execution_dependencies",
            "execution phase dependencies must form a DAG",
            "execution.phases",
        )
    return IndependentTmaSmemLegality(tuple(violations))


def render_independent_tma_smem_contract(
    plan: IndependentTmaSmemMainloop,
) -> str:
    """Render a deterministic, inspectable input for the future CuTe kernel renderer.

    The result is intentionally a structural fragment rather than an executable
    kernel. Promoting it to a backend state requires H100 validation of a complete
    renderer, WGMMA consumer, and epilogue.
    """

    legality = validate_independent_tma_smem_mainloop(plan)
    if not legality.valid:
        messages = "; ".join(item.message for item in legality.violations)
        raise ValueError(f"illegal independent TMA/shared-memory contract: {messages}")
    representation = plan.as_dict()
    return f'''# Deterministic independent CuTe TMA/shared-memory contract.
# This is a structural renderer input, not an executable kernel.
INDEPENDENT_TMA_SMEM_SCHEMA_VERSION = {plan.schema_version}
CONFIGURATION_HASH = {plan.configuration_hash!r}
KERNEL_MCTS_TMA_SMEM_CONTRACT = {representation!r}
'''


def render_independent_cute_gemm_contract(
    kernel: IndependentCuteGemmKernel,
) -> str:
    """Render the complete typed design as a deterministic lowering input."""

    legality = validate_independent_cute_gemm(kernel)
    if not legality.valid:
        messages = "; ".join(item.message for item in legality.violations)
        raise ValueError(f"illegal independent CuTe GEMM contract: {messages}")
    return f'''# Deterministic independent CuTe GEMM structural contract.
# A future lowering must implement every field before this can execute.
INDEPENDENT_CUTE_GEMM_SCHEMA_VERSION = {kernel.schema_version}
CONFIGURATION_HASH = {kernel.configuration_hash!r}
KERNEL_MCTS_INDEPENDENT_CUTE_GEMM = {kernel.as_dict()!r}
'''
