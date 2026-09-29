from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class PersistentOwnershipDiagnosticSource:
    source: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PersistentStageUse:
    work_index: int
    k_tile: int
    global_k_tile: int
    stage: int
    phase: int


def iter_persistent_stage_uses(
    *, work_items: int, k_tiles_per_work: int, pipeline_stages: int
) -> Iterator[PersistentStageUse]:
    """Yield the stage-ring state carried across persistent output tiles."""

    if work_items < 0:
        raise ValueError("persistent work-item count must be non-negative")
    if k_tiles_per_work <= 0:
        raise ValueError("K-tile count per work item must be positive")
    if pipeline_stages <= 0:
        raise ValueError("pipeline-stage count must be positive")
    for work_index in range(work_items):
        for k_tile in range(k_tiles_per_work):
            global_k_tile = work_index * k_tiles_per_work + k_tile
            yield PersistentStageUse(
                work_index=work_index,
                k_tile=k_tile,
                global_k_tile=global_k_tile,
                stage=global_k_tile % pipeline_stages,
                phase=(global_k_tile // pipeline_stages) % 2,
            )


def validate_persistent_stage_schedule(
    *, work_items: int, k_tiles_per_work: int, pipeline_stages: int
) -> bool:
    """Check producer/consumer stage reuse alternates phase without resetting."""

    previous_phase_by_stage: dict[int, int] = {}
    for use in iter_persistent_stage_uses(
        work_items=work_items,
        k_tiles_per_work=k_tiles_per_work,
        pipeline_stages=pipeline_stages,
    ):
        previous_phase = previous_phase_by_stage.get(use.stage)
        if previous_phase is not None and use.phase != 1 - previous_phase:
            return False
        previous_phase_by_stage[use.stage] = use.phase
    return True


def render_persistent_ownership_diagnostic(
    *,
    grid_m: int = 32,
    grid_n: int = 16,
) -> PersistentOwnershipDiagnosticSource:
    """Render an evidence-only persistent tile-ownership diagnostic.

    This intentionally excludes GEMM, TMA, and WGMMA.  It proves the scheduler's
    bounded, exactly-once logical tile assignment before it is composed with the
    independent kernel's producer/consumer barrier protocol.
    """

    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("persistent ownership grid extents must be positive")

    source = f'''# Generated persistent CTA tile-ownership diagnostic.
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

GRID_M = {grid_m}
GRID_N = {grid_n}
LOGICAL_TILE_COUNT = GRID_M * GRID_N
CLUSTER_SHAPE_MNL = (1, 1, 1)


class PersistentOwnershipKernel:
    @cute.jit
    def __call__(self, records, max_active_clusters: cutlass.Int32, stream: cuda.CUstream):
        params = utils.PersistentTileSchedulerParams(
            (GRID_M, GRID_N, 1), CLUSTER_SHAPE_MNL, 1, True
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            params, max_active_clusters
        )
        self.kernel(params, records).launch(
            grid=grid,
            block=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, params, records):
        scheduler = utils.StaticPersistentTileScheduler.create(
            params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work = scheduler.initial_work_tile_info()
        while work.is_valid_tile:
            tile_m, tile_n, tile_l = work.tile_idx
            worker = cute.arch.block_idx()[2]
            iteration = scheduler.num_tiles_executed
            records[(worker, iteration, 0)] = tile_m
            records[(worker, iteration, 1)] = tile_n
            records[(worker, iteration, 2)] = tile_l
            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()


def run_diagnostic():
    import ctypes
    import statistics
    import subprocess
    import tempfile
    from pathlib import Path
    import torch
    from kernel_mcts.cute_diagnostics import describe_kernel_callable

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_active_clusters = int(properties.multi_processor_count)
    if max_active_clusters <= 0:
        raise RuntimeError("GPU reports no streaming multiprocessors")
    persistent_ctas = min(LOGICAL_TILE_COUNT, max_active_clusters)
    max_tiles_per_cta = math.ceil(LOGICAL_TILE_COUNT / persistent_ctas)
    records = torch.full(
        (persistent_ctas, max_tiles_per_cta, 3),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    records_cute = from_dlpack(records, assumed_align=16).mark_layout_dynamic(
        leading_dim=2
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    diagnostic = PersistentOwnershipKernel()
    compiled = cute.compile(
        diagnostic,
        records_cute,
        cutlass.Int32(max_active_clusters),
        stream,
    )
    jit_diagnostics = describe_kernel_callable(compiled, {{}})
    compiled(
        records_cute,
        cutlass.Int32(max_active_clusters),
        stream,
    )
    torch.cuda.synchronize()

    host_records = records.cpu().tolist()
    assignments = []
    worker_tile_counts = []
    for worker_records in host_records:
        worker_assignments = [
            tuple(int(value) for value in coordinate)
            for coordinate in worker_records
            if coordinate[0] >= 0
        ]
        worker_tile_counts.append(len(worker_assignments))
        assignments.extend(worker_assignments)

    expected = {{(m, n, 0) for m in range(GRID_M) for n in range(GRID_N)}}
    observed = set(assignments)
    duplicate_count = len(assignments) - len(observed)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    exactly_once = (
        len(assignments) == LOGICAL_TILE_COUNT
        and duplicate_count == 0
        and not missing
        and not unexpected
    )
    return {{
        "status": "ok" if exactly_once else "ownership_failed",
        "logical_grid_mn": [GRID_M, GRID_N],
        "logical_tile_count": LOGICAL_TILE_COUNT,
        "max_active_clusters": max_active_clusters,
        "persistent_cta_count": persistent_ctas,
        "max_tiles_per_cta": max_tiles_per_cta,
        "worker_tile_counts": worker_tile_counts,
        "assignment_count": len(assignments),
        "unique_assignment_count": len(observed),
        "duplicate_count": duplicate_count,
        "missing_tiles": [list(item) for item in missing],
        "unexpected_tiles": [list(item) for item in unexpected],
        "exactly_once": exactly_once,
        "jit_diagnostics": jit_diagnostics,
    }}
'''
    return PersistentOwnershipDiagnosticSource(source)


def render_persistent_barrier_diagnostic(
    *,
    grid_m: int = 32,
    grid_n: int = 16,
    k_tiles_per_work: int = 64,
    pipeline_stages: int = 3,
) -> PersistentOwnershipDiagnosticSource:
    """Render a producer/consumer barrier-ring carry diagnostic.

    This is the synchronization gate between ownership-only scheduling and the
    complete persistent GEMM. It deliberately excludes TMA and WGMMA data movement.
    """

    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("persistent barrier grid extents must be positive")
    if k_tiles_per_work <= 0:
        raise ValueError("persistent barrier K-tile count must be positive")
    if pipeline_stages <= 0:
        raise ValueError("persistent barrier stage count must be positive")
    if not validate_persistent_stage_schedule(
        work_items=4,
        k_tiles_per_work=k_tiles_per_work,
        pipeline_stages=pipeline_stages,
    ):
        raise ValueError("persistent barrier phase schedule does not alternate")

    source = f'''# Generated persistent producer/consumer barrier-ring diagnostic.
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

GRID_M = {grid_m}
GRID_N = {grid_n}
LOGICAL_TILE_COUNT = GRID_M * GRID_N
K_TILES_PER_WORK = {k_tiles_per_work}
PIPELINE_STAGES = {pipeline_stages}
THREADS_PER_CTA = 256


class PersistentBarrierKernel:
    @cute.jit
    def __call__(self, completions, max_active_clusters: cutlass.Int32, stream: cuda.CUstream):
        params = utils.PersistentTileSchedulerParams(
            (GRID_M, GRID_N, 1), (1, 1, 1), 1, True
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            params, max_active_clusters
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES]
            empty: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES]

        self.shared_storage = SharedStorage
        self.kernel(params, completions).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, params, completions):
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        if tidx == 0:
            for stage in cutlass.range_constexpr(PIPELINE_STAGES):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if tidx == 0:
            for stage in cutlass.range_constexpr(PIPELINE_STAGES):
                cute.arch.mbarrier_arrive(empty + stage)
        pipeline.sync(barrier_id=1)

        worker = cute.arch.block_idx()[2]
        if tidx == 0:
            producer = utils.StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            producer_work = producer.initial_work_tile_info()
            while producer_work.is_valid_tile:
                for k_tile in cutlass.range(0, K_TILES_PER_WORK, 1, unroll=1):
                    global_k_tile = producer.num_tiles_executed * K_TILES_PER_WORK + k_tile
                    stage = global_k_tile % PIPELINE_STAGES
                    phase = (global_k_tile // PIPELINE_STAGES) % 2
                    cute.arch.mbarrier_wait(empty + stage, phase)
                    cute.arch.mbarrier_arrive(full + stage)
                producer.advance_to_next_work()
                producer_work = producer.get_current_work()
            completions[(worker, 0)] = producer.num_tiles_executed

        if tidx == 128:
            consumer = utils.StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            consumer_work = consumer.initial_work_tile_info()
            while consumer_work.is_valid_tile:
                for k_tile in cutlass.range(0, K_TILES_PER_WORK, 1, unroll=1):
                    global_k_tile = consumer.num_tiles_executed * K_TILES_PER_WORK + k_tile
                    stage = global_k_tile % PIPELINE_STAGES
                    phase = (global_k_tile // PIPELINE_STAGES) % 2
                    cute.arch.mbarrier_wait(full + stage, phase)
                    cute.arch.mbarrier_arrive(empty + stage)
                consumer.advance_to_next_work()
                consumer_work = consumer.get_current_work()
            completions[(worker, 1)] = consumer.num_tiles_executed


def run_diagnostic():
    import torch
    from kernel_mcts.cute_diagnostics import describe_kernel_callable

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_active_clusters = int(properties.multi_processor_count)
    persistent_ctas = min(LOGICAL_TILE_COUNT, max_active_clusters)
    completions = torch.full(
        (persistent_ctas, 2), -1, device="cuda", dtype=torch.int32
    )
    completions_cute = from_dlpack(
        completions, assumed_align=16
    ).mark_layout_dynamic(leading_dim=1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    diagnostic = PersistentBarrierKernel()
    compiled = cute.compile(
        diagnostic,
        completions_cute,
        cutlass.Int32(max_active_clusters),
        stream,
    )
    jit_diagnostics = describe_kernel_callable(compiled, {{}})
    compiled(
        completions_cute,
        cutlass.Int32(max_active_clusters),
        stream,
    )
    torch.cuda.synchronize()

    completion_rows = [[int(value) for value in row] for row in completions.cpu().tolist()]
    producer_counts = [row[0] for row in completion_rows]
    consumer_counts = [row[1] for row in completion_rows]
    lockstep = producer_counts == consumer_counts
    complete = sum(producer_counts) == LOGICAL_TILE_COUNT
    bounded = all(count in (3, 4) for count in producer_counts)
    passed = lockstep and complete and bounded
    return {{
        "status": "ok" if passed else "barrier_failed",
        "logical_tile_count": LOGICAL_TILE_COUNT,
        "persistent_cta_count": persistent_ctas,
        "k_tiles_per_work": K_TILES_PER_WORK,
        "pipeline_stages": PIPELINE_STAGES,
        "producer_work_count": sum(producer_counts),
        "consumer_work_count": sum(consumer_counts),
        "worker_completion_counts": completion_rows,
        "producer_consumer_lockstep": lockstep,
        "complete": complete,
        "bounded": bounded,
        "jit_diagnostics": jit_diagnostics,
    }}
'''
    return PersistentOwnershipDiagnosticSource(source)


def render_persistent_tma_diagnostic(
    *,
    grid_m: int = 32,
    grid_n: int = 16,
    pipeline_stages: int = 3,
    enable_wgmma_issue: bool = False,
    enable_epilogue: bool = False,
    full_k: bool = False,
) -> PersistentOwnershipDiagnosticSource:
    """Render a persistent TMA payload/addressing diagnostic.

    Each scheduled output tile loads its first A and B K tile through TMA, then the
    consumer records one value from each shared-memory tile for host verification.
    WGMMA and the epilogue remain intentionally excluded.
    """

    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("persistent TMA grid extents must be positive")
    if pipeline_stages <= 0:
        raise ValueError("persistent TMA stage count must be positive")
    if enable_epilogue and not enable_wgmma_issue:
        raise ValueError("persistent epilogue requires WGMMA issue")

    source = f'''# Generated persistent TMA payload diagnostic.
import math
import ctypes
from pathlib import Path
import statistics
import subprocess
import tempfile

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

GRID_M = {grid_m}
GRID_N = {grid_n}
LOGICAL_TILE_COUNT = GRID_M * GRID_N
TILE_SHAPE_MNK = (128, 256, 64)
PIPELINE_STAGES = {pipeline_stages}
THREADS_PER_CTA = 256
MAX_TILES_PER_CTA = 4
ENABLE_WGMMA_ISSUE = {enable_wgmma_issue!r}
ENABLE_EPILOGUE = {enable_epilogue!r}
FULL_K = {full_k!r}
K_TILES_PER_WORK = 64 if FULL_K else 1
PROBLEM_K = K_TILES_PER_WORK * TILE_SHAPE_MNK[2]
CONSUMER_THREADS = 256 if ENABLE_WGMMA_ISSUE else 1
THREADS_PER_CTA = 384 if ENABLE_WGMMA_ISSUE else 256
EPILOGUE_TILE = (64, 64)
EPILOGUE_STAGES = 8
EPILOGUE_STORAGE_ELEMENTS = 9 * EPILOGUE_TILE[0] * EPILOGUE_TILE[1]


class PersistentTmaKernel:
    def __init__(self):
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a, b, c, coordinates, samples, max_active_clusters: cutlass.Int32, stream: cuda.CUstream):
        self.dtype = a.element_type
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.from_tensor(b)
        c_layout = utils.LayoutEnum.from_tensor(c)
        self.c_layout = c_layout
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            cutlass.Float32,
            (2, 1, 1),
            tiler_mn=(64, TILE_SHAPE_MNK[1]),
        )
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, TILE_SHAPE_MNK, self.dtype, PIPELINE_STAGES
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, TILE_SHAPE_MNK, self.dtype, PIPELINE_STAGES
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.dtype, c_layout, EPILOGUE_TILE, EPILOGUE_STAGES
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES]
            empty: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.dtype, EPILOGUE_STORAGE_ELEMENTS],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_a, tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            a,
            a_smem_layout,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            num_multicast=1,
        )
        tma_b, tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            b,
            b_smem_layout,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
            num_multicast=1,
        )
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_c, tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            EPILOGUE_TILE,
        )
        params = utils.PersistentTileSchedulerParams(
            (GRID_M, GRID_N, 1), (1, 1, 1), 1, True
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            params, max_active_clusters
        )
        self.kernel(
            params,
            tma_a,
            tensor_a,
            tma_b,
            tensor_b,
            tma_c,
            tensor_c,
            tiled_mma,
            coordinates,
            samples,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
        ).launch(grid=grid, block=(THREADS_PER_CTA, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        params,
        tma_a,
        tensor_a,
        tma_b,
        tensor_b,
        tma_c,
        tensor_c,
        tiled_mma,
        coordinates,
        samples,
        a_smem_layout_staged,
        b_smem_layout_staged,
        epi_smem_layout_staged,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        a_smem = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem = cute.slice_(b_smem_layout_staged, (None, None, 0))
        transaction_bytes = cute.size_in_bytes(self.dtype, a_smem)
        transaction_bytes = transaction_bytes + cute.size_in_bytes(self.dtype, b_smem)
        if tidx == 0:
            for stage in cutlass.range_constexpr(PIPELINE_STAGES):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if tidx == 0:
            for stage in cutlass.range_constexpr(PIPELINE_STAGES):
                cute.arch.mbarrier_arrive(empty + stage)
        pipeline.sync(barrier_id=1)

        gA = cute.local_tile(
            tensor_a, (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]), (None, None)
        )
        gB = cute.local_tile(
            tensor_b, (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]), (None, None)
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2)
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2)
        )
        worker = cute.arch.block_idx()[2]

        if tidx == 0:
            producer = utils.StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work = producer.initial_work_tile_info()
            while work.is_valid_tile:
                tile_m, tile_n, _ = work.tile_idx
                for k_tile in cutlass.range(
                    0, K_TILES_PER_WORK, 1, unroll=1
                ):
                    global_k_tile = (
                        producer.num_tiles_executed * K_TILES_PER_WORK + k_tile
                    )
                    stage = global_k_tile % PIPELINE_STAGES
                    phase = (global_k_tile // PIPELINE_STAGES) % 2
                    cute.arch.mbarrier_wait(empty + stage, phase)
                    cute.arch.mbarrier_expect_tx(full + stage, transaction_bytes)
                    cute.copy(
                        tma_a,
                        tAgA[(None, tile_m, k_tile)],
                        tAsA[(None, stage)],
                        tma_bar_ptr=full + stage,
                    )
                    cute.copy(
                        tma_b,
                        tBgB[(None, tile_n, k_tile)],
                        tBsB[(None, stage)],
                        tma_bar_ptr=full + stage,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(full + stage)
                producer.advance_to_next_work()
                work = producer.get_current_work()

        if (
            (not ENABLE_WGMMA_ISSUE and tidx == 128)
            or (ENABLE_WGMMA_ISSUE and tidx >= 128)
        ):
            if cutlass.const_expr(ENABLE_WGMMA_ISSUE):
                warp_group_idx = cute.arch.make_warp_uniform((tidx - 128) // 128)
                warp_group_layout = cute.make_layout(2, stride=128)
                thr_mma = tiled_mma.get_slice(warp_group_layout(warp_group_idx))
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                consumer_barrier = pipeline.NamedBarrier(
                    barrier_id=2, num_threads=CONSUMER_THREADS
                )
                if cutlass.const_expr(ENABLE_EPILOGUE):
                    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                        self.c_layout,
                        elem_ty_d=self.dtype,
                        elem_ty_acc=cutlass.Float32,
                    )
                    copy_atom_c = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(), 4
                        ),
                        self.dtype,
                    )
                    tiled_copy_c_atom = cute.make_tiled_copy_C_atom(
                        copy_atom_c, tiled_mma
                    )
                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s, tiled_copy_c_atom
                    )
                    epilogue_thread_idx = (tidx - 128) % 128
                    thr_copy_r2s = tiled_copy_r2s.get_slice(epilogue_thread_idx)
                    tRS_sC = thr_copy_r2s.partition_D(sC)
            consumer = utils.StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work = consumer.initial_work_tile_info()
            while work.is_valid_tile:
                tile_m, tile_n, _ = work.tile_idx
                slot = consumer.num_tiles_executed
                if cutlass.const_expr(ENABLE_WGMMA_ISSUE):
                    gC = cute.local_tile(
                        tensor_c,
                        (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[1]),
                        (tile_m, tile_n),
                    )
                    tCgC = thr_mma.partition_C(gC)
                    accumulators = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)
                    accumulators.fill(0.0)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(
                    0, K_TILES_PER_WORK, 1, unroll=1
                ):
                    global_k_tile = slot * K_TILES_PER_WORK + k_tile
                    stage = global_k_tile % PIPELINE_STAGES
                    phase = (global_k_tile // PIPELINE_STAGES) % 2
                    cute.arch.mbarrier_wait(full + stage, phase)
                    if cutlass.const_expr(ENABLE_WGMMA_ISSUE):
                        cute.nvgpu.warpgroup.fence()
                        for k_block in cutlass.range(
                            cute.size(tCrA, mode=[2]), unroll_full=True
                        ):
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[(None, None, k_block, stage)],
                                tCrB[(None, None, k_block, stage)],
                                accumulators,
                            )
                            tiled_mma.set(
                                cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        consumer_barrier.arrive_and_wait()
                    if tidx == 128:
                        if k_tile == 0:
                            coordinates[(worker, slot, 0)] = tile_m
                            coordinates[(worker, slot, 1)] = tile_n
                            samples[(worker, slot, 0)] = sA[(0, 0, stage)]
                            samples[(worker, slot, 1)] = sB[(0, 0, stage)]
                        cute.arch.mbarrier_arrive(empty + stage)
                    if cutlass.const_expr(ENABLE_WGMMA_ISSUE):
                        consumer_barrier.arrive_and_wait()
                if cutlass.const_expr(ENABLE_EPILOGUE):
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                    rC_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    rC_layout = cute.make_layout(rC_shape[:3])
                    rC = cute.make_rmem_tensor_like(rC_layout, cutlass.Float32)
                    rC_out = cute.make_rmem_tensor_like(rC_layout, self.dtype)
                    rC_size = cute.size(rC)
                    gC_for_tma = cute.zipped_divide(gC, EPILOGUE_TILE)
                    tma_sC, tma_gC = cute.nvgpu.cpasync.tma_partition(
                        tma_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        gC_for_tma,
                    )
                    epi_tile_shape = gC_for_tma.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )
                    for local_epi_index in cutlass.range_constexpr(4):
                        for value_index in cutlass.range_constexpr(rC_size):
                            rC[value_index] = tRS_rAcc[
                                local_epi_index * rC_size + value_index
                            ]
                        rC_out.store(rC.load().to(self.dtype))
                        epi_buffer = warp_group_idx * 4 + local_epi_index
                        cute.copy(
                            tiled_copy_r2s,
                            rC_out,
                            tRS_sC[(None, None, None, epi_buffer)],
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    consumer_barrier.arrive_and_wait()
                    if warp_idx == 4:
                        c_pipeline = pipeline.PipelineTmaStore.create(
                            num_stages=EPILOGUE_STAGES,
                            producer_group=pipeline.CooperativeGroup(
                                pipeline.Agent.Thread, CONSUMER_THREADS
                            ),
                        )
                        for epi_index in cutlass.range_constexpr(8):
                            global_coord = epi_tile_layout.get_hier_coord(epi_index)
                            cute.copy(
                                tma_c,
                                tma_sC[(None, epi_index)],
                                tma_gC[(None, global_coord)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        c_pipeline.producer_tail()
                    consumer_barrier.arrive_and_wait()
                consumer.advance_to_next_work()
                work = consumer.get_current_work()


def run_diagnostic():
    import torch
    from kernel_mcts.cute_diagnostics import describe_kernel_callable

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_active_clusters = int(properties.multi_processor_count)
    persistent_ctas = min(LOGICAL_TILE_COUNT, max_active_clusters)
    input_directory = None
    if FULL_K:
        input_directory = tempfile.TemporaryDirectory(
            prefix="kernel-mcts-persistent-inputs-"
        )
        root = Path(input_directory.name)
        a_path = root / "a.bf16"
        b_path = root / "b.bf16"
        subprocess.run(
            [
                "/usr/local/bin/kernel-mcts-bf16-inputs",
                str(GRID_M * TILE_SHAPE_MNK[0]),
                str(GRID_N * TILE_SHAPE_MNK[1]),
                str(PROBLEM_K),
                "0",
                str(a_path),
                str(b_path),
            ],
            check=True,
            timeout=120,
        )
        a = torch.from_file(
            str(a_path),
            shared=False,
            size=GRID_M * TILE_SHAPE_MNK[0] * PROBLEM_K,
            dtype=torch.bfloat16,
        ).reshape(GRID_M * TILE_SHAPE_MNK[0], PROBLEM_K).to("cuda")
        b = torch.from_file(
            str(b_path),
            shared=False,
            size=GRID_N * TILE_SHAPE_MNK[1] * PROBLEM_K,
            dtype=torch.bfloat16,
        ).reshape(GRID_N * TILE_SHAPE_MNK[1], PROBLEM_K).to("cuda")
    else:
        a = torch.arange(
            GRID_M * TILE_SHAPE_MNK[0], device="cuda", dtype=torch.float32
        )
        a = a[:, None].expand(-1, PROBLEM_K).to(torch.bfloat16).contiguous()
        b = torch.arange(
            GRID_N * TILE_SHAPE_MNK[1], device="cuda", dtype=torch.float32
        )
        b = (
            (10000 + b[:, None])
            .expand(-1, PROBLEM_K)
            .to(torch.bfloat16)
            .contiguous()
        )
    c = torch.zeros(
        (GRID_M * TILE_SHAPE_MNK[0], GRID_N * TILE_SHAPE_MNK[1]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    coordinates = torch.full(
        (persistent_ctas, MAX_TILES_PER_CTA, 2), -1, device="cuda", dtype=torch.int32
    )
    samples = torch.zeros(
        (persistent_ctas, MAX_TILES_PER_CTA, 2), device="cuda", dtype=torch.bfloat16
    )
    tensors = [
        from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=value.ndim - 1)
        for value in (a, b, c, coordinates, samples)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    diagnostic = PersistentTmaKernel()
    compiled = cute.compile(
        diagnostic, *tensors, cutlass.Int32(max_active_clusters), stream
    )
    jit_diagnostics = describe_kernel_callable(compiled, {{}})
    compiled(*tensors, cutlass.Int32(max_active_clusters), stream)
    torch.cuda.synchronize()

    host_coordinates = coordinates.cpu().tolist()
    host_samples = samples.float().cpu().tolist()
    assignments = []
    mismatches = []
    for worker, rows in enumerate(host_coordinates):
        for slot, coordinate in enumerate(rows):
            tile_m, tile_n = (int(coordinate[0]), int(coordinate[1]))
            if tile_m < 0:
                continue
            assignments.append((tile_m, tile_n))
            observed_a, observed_b = host_samples[worker][slot]
            expected_a = float(a[tile_m * TILE_SHAPE_MNK[0], 0].float().item())
            expected_b = float(b[tile_n * TILE_SHAPE_MNK[1], 0].float().item())
            if observed_a != expected_a or observed_b != expected_b:
                mismatches.append({{
                    "tile": [tile_m, tile_n],
                    "observed": [observed_a, observed_b],
                    "expected": [expected_a, expected_b],
                }})
    expected_coordinates = {{(m, n) for m in range(GRID_M) for n in range(GRID_N)}}
    observed_coordinates = set(assignments)
    exactly_once = len(assignments) == LOGICAL_TILE_COUNT and observed_coordinates == expected_coordinates
    payload_exact = not mismatches
    numerical_correct = None
    maximum_error = None
    mean_error = None
    if ENABLE_EPILOGUE:
        if FULL_K:
            reference = torch.empty_like(c)
            library = ctypes.CDLL("/usr/local/lib/kernel-mcts-bf16-reference.so")
            reference_call = library.kernel_mcts_bf16_reference
            reference_call.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            reference_call.restype = ctypes.c_int
            reference_status = reference_call(
                a.data_ptr(),
                b.data_ptr(),
                reference.data_ptr(),
                GRID_M * TILE_SHAPE_MNK[0],
                GRID_N * TILE_SHAPE_MNK[1],
                PROBLEM_K,
            )
            if reference_status != 0:
                raise RuntimeError(
                    f"cuBLAS reference failed with status {{reference_status}}"
                )
            torch.cuda.synchronize()
        else:
            reference = torch.matmul(a.float(), b.float().transpose(0, 1)).to(
                torch.bfloat16
            )
        error = (c.float() - reference.float()).abs()
        maximum_error = float(error.max().item())
        mean_error = float(error.mean().item())
        numerical_correct = bool(
            torch.all(error <= 0.02 + 0.02 * reference.float().abs()).item()
        )
    benchmark = None
    if FULL_K and numerical_correct:
        for _ in range(10):
            compiled(*tensors, cutlass.Int32(max_active_clusters), stream)
        torch.cuda.synchronize()
        timings_us = []
        for _ in range(30):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(*tensors, cutlass.Int32(max_active_clusters), stream)
            end.record()
            end.synchronize()
            timings_us.append(float(start.elapsed_time(end) * 1000.0))
        benchmark = {{
            "warmup_count": 10,
            "measurement_count": 30,
            "timings_us": timings_us,
            "median_us": float(statistics.median(timings_us)),
            "mean_us": float(statistics.mean(timings_us)),
            "min_us": min(timings_us),
            "max_us": max(timings_us),
        }}
    passed = exactly_once and payload_exact and numerical_correct is not False
    result = {{
        "status": "ok" if passed else "tma_payload_failed",
        "logical_tile_count": LOGICAL_TILE_COUNT,
        "persistent_cta_count": persistent_ctas,
        "assignment_count": len(assignments),
        "unique_assignment_count": len(observed_coordinates),
        "exactly_once": exactly_once,
        "payload_exact": payload_exact,
        "wgmma_issued": ENABLE_WGMMA_ISSUE,
        "epilogue_stored": ENABLE_EPILOGUE,
        "full_k": FULL_K,
        "k_tiles_per_work": K_TILES_PER_WORK,
        "numerical_correct": numerical_correct,
        "maximum_error": maximum_error,
        "mean_error": mean_error,
        "benchmark": benchmark,
        "repository_contract": FULL_K,
        "mismatches": mismatches[:16],
        "jit_diagnostics": jit_diagnostics,
    }}
    if input_directory is not None:
        input_directory.cleanup()
    return result
'''
    return PersistentOwnershipDiagnosticSource(source)
