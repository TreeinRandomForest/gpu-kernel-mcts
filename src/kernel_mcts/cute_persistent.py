from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersistentOwnershipDiagnosticSource:
    source: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


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
