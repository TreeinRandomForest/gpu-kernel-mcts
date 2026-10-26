KERNEL_MCTS_INDEPENDENT_CUTE_GEMM = {'mainloop': {'tile_m': 64, 'tile_n': 256, 'tile_k': 64, 'cluster_m': 1, 'cluster_n': 1, 'pipeline_stages': 2, 'barrier_slots': 2, 'producer_warp_groups': 1, 'schedule': 'prefetch', 'producer_consumer_mode': 'cooperative', 'a_copy': {'operand': 'a', 'global_major': 'k', 'tile_rows': 64, 'tile_columns': 64, 'element_bytes': 2, 'alignment_bytes': 16, 'shared_major': 'k', 'swizzle_bytes': 128, 'multicast_axis': 'none', 'stage_stride_bytes': 8192, 'storage_offset_bytes': 0}, 'b_copy': {'operand': 'b', 'global_major': 'k', 'tile_rows': 64, 'tile_columns': 256, 'element_bytes': 2, 'alignment_bytes': 16, 'shared_major': 'k', 'swizzle_bytes': 128, 'multicast_axis': 'none', 'stage_stride_bytes': 32768, 'storage_offset_bytes': 16384}, 'schema_version': 5}, 'consumer': {'instruction_m': 64, 'instruction_n': 256, 'instruction_k': 16, 'warp_groups_m': 1, 'warp_groups_n': 1, 'accumulator_dtype': 'float32', 'a_source': 'shared_memory', 'b_source': 'shared_memory', 'accumulator_owner': 'consumer_warp_group'}, 'epilogue': {'accumulator_source': 'registers', 'output_dtype': 'bfloat16', 'output_major': 'n', 'store_kind': 'tma', 'tile_m': 64, 'tile_n': 64, 'pipeline_stages': 4, 'barrier_slots': 4, 'alignment_bytes': 16, 'storage_offset_bytes': 81920, 'stage_stride_bytes': 8192, 'allocation_padding_bytes': 8192}, 'execution': {'agents': ({'agent_id': 'tma_load_agent', 'scope': 'warp', 'count': 1, 'role': 'cooperative_mainloop_producer'}, {'agent_id': 'wgmma_agents', 'scope': 'warp_group', 'count': 1, 'role': 'mainloop_consumer'}, {'agent_id': 'epilogue_agents', 'scope': 'thread', 'count': 128, 'role': 'epilogue_producer'}, {'agent_id': 'tma_store_agent', 'scope': 'warp', 'count': 1, 'role': 'epilogue_consumer'}), 'buffers': ({'buffer_id': 'shared_a', 'memory_space': 'shared_memory', 'stages': 2, 'producer_agent': 'tma_load_agent', 'consumer_agent': 'wgmma_agents', 'barrier_protocol': 'tma_async_full_empty'}, {'buffer_id': 'shared_b', 'memory_space': 'shared_memory', 'stages': 2, 'producer_agent': 'tma_load_agent', 'consumer_agent': 'wgmma_agents', 'barrier_protocol': 'tma_async_full_empty'}, {'buffer_id': 'shared_c', 'memory_space': 'shared_memory', 'stages': 4, 'producer_agent': 'epilogue_agents', 'consumer_agent': 'tma_store_agent', 'barrier_protocol': 'tma_store_pipeline'}), 'phases': ({'phase_id': 'tma_load', 'agent_id': 'tma_load_agent', 'operation': 'global_to_shared_tma', 'reads': ('global_a', 'global_b'), 'writes': ('shared_a', 'shared_b'), 'waits_for': (), 'signals': ('mainloop_full',)}, {'phase_id': 'wgmma', 'agent_id': 'wgmma_agents', 'operation': 'shared_to_register_wgmma', 'reads': ('shared_a', 'shared_b'), 'writes': ('accumulator',), 'waits_for': ('mainloop_full',), 'signals': ('mainloop_empty', 'accumulator_ready')}, {'phase_id': 'epilogue_stage', 'agent_id': 'epilogue_agents', 'operation': 'register_to_shared_convert', 'reads': ('accumulator',), 'writes': ('shared_c',), 'waits_for': ('accumulator_ready',), 'signals': ('epilogue_ready',)}, {'phase_id': 'tma_store', 'agent_id': 'tma_store_agent', 'operation': 'shared_to_global_tma', 'reads': ('shared_c',), 'writes': ('global_c',), 'waits_for': ('epilogue_ready',), 'signals': ('epilogue_empty',)})}, 'schema_version': 5}
# Generated independent Hopper TMA copy-only diagnostic.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

CONFIGURATION_HASH = 'c5eca57e9aabfb0ae7229d70e5ee4518c70e21caf20e55bde1169cdbb08022c2'
DEBUG_STAGE = 'wgmma_full_workload'
TILE_SHAPE_MNK = (64, 256, 64)
PROBLEM_K = 4096
PROBLEM_M = 4096
PROBLEM_N = 4096
FULL_K = True
FULL_WORKLOAD = True
OVERLAPPED_MAINLOOP = True
REPOSITORY_CONTRACT = True
PROFILE_SINGLE_LAUNCH = False
FULL_K_TILE_COUNT = PROBLEM_K // TILE_SHAPE_MNK[2]
GRID_M = 64
GRID_N = 16
CLUSTER_SHAPE_MN = (1, 1)
MAINLOOP_STAGES = 2
SWIZZLE_BYTES = 128
THREADS_PER_CTA = 128
CONSUMER_THREADS = 128
DMA_WARP_GROUPS = 0
WARP_SPECIALIZED = False
REGISTER_REPARTITION = False
MMA_WARP_GROUPS = 1
COOPERATIVE_EPILOGUE = False
EPILOGUE_TILES_PER_WARP_GROUP = 4
EPILOGUE_TILE = (64, 64)
EPILOGUE_STAGES = 4
EPILOGUE_STORAGE_ELEMENTS = 20480
ENABLE_B = True
ENABLE_MULTICAST = False
LAUNCH_KERNEL = True
EMPTY_KERNEL = False
LOAD_ONLY = False
ENABLE_WGMMA = True
WGMMA_ISSUE_ONLY = False
WGMMA_R2S_ONLY = False
WGMMA_R2S_FIRST_TILE = False
WGMMA_R2S_TWO_TILES = False
WGMMA_R2S_THREE_TILES = False
WGMMA_R2S_FOUR_REUSE_ZERO = False
WGMMA_R2S_FOUR_PADDED = False
WGMMA_R2S_FOUR_TILES = False


def _select_layout_atom(layout, element_type, major_mode_size, *, loc=None, ip=None):
    selected = _PINNED_LAYOUT_SELECTOR(
        layout, element_type, major_mode_size, loc=loc, ip=ip
    )
    if SWIZZLE_BYTES == 128:
        return selected
    selected_name = getattr(selected, "name", str(selected).rsplit(".", 1)[-1])
    target_name = "MN_SW64" if selected_name.startswith("MN_") else "K_SW64"
    return getattr(type(selected), target_name)


_PINNED_LAYOUT_SELECTOR = sm90_utils.get_smem_layout_atom


class IndependentTmaCopyKernel:
    def __init__(self):
        self.tile_shape_mnk = TILE_SHAPE_MNK
        self.cluster_shape_mn = CLUSTER_SHAPE_MN
        self.threads_per_cta = THREADS_PER_CTA
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a, b, c, a_out, b_out, stream: cuda.CUstream):
        self.dtype = a.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.epi_tile = (64, 64)
        self.epi_stage = EPILOGUE_STAGES
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            cutlass.Float32,
            (MMA_WARP_GROUPS, 1, 1),
            tiler_mn=(64, TILE_SHAPE_MNK[1]),
        )

        pinned_selector = sm90_utils.get_smem_layout_atom
        sm90_utils.get_smem_layout_atom = _select_layout_atom
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, self.tile_shape_mnk, self.dtype, MAINLOOP_STAGES
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, self.tile_shape_mnk, self.dtype, MAINLOOP_STAGES
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.epi_stage,
        )
        sm90_utils.get_smem_layout_atom = pinned_selector

        @cute.struct
        class SharedStorage:
            load_barrier: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES]
            empty_barrier: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    EPILOGUE_STORAGE_ELEMENTS,
                ],
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
        b_load_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if ENABLE_MULTICAST
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        tma_b, tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_load_op,
            b,
            b_smem_layout,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
            num_multicast=CLUSTER_SHAPE_MN[0] if ENABLE_MULTICAST else 1,
        )
        tma_a_out, tensor_a_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            a_out,
            a_smem_layout,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
        )
        tma_b_out, tensor_b_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            b_out,
            b_smem_layout,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
        )
        epi_smem_layout = cute.slice_(
            epi_smem_layout_staged, (None, None, 0)
        )
        tma_c, tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )
        cta_layout_mnk = cute.make_layout((*CLUSTER_SHAPE_MN, 1))
        self.kernel(
            tma_a,
            tensor_a,
            tma_b,
            tensor_b,
            tma_a_out,
            tensor_a_out,
            tma_b_out,
            tensor_b_out,
            tma_c,
            tensor_c,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
        ).launch(
            grid=(GRID_M, GRID_N, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(*CLUSTER_SHAPE_MN, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_a,
        tensor_a,
        tma_b,
        tensor_b,
        tma_a_out,
        tensor_a_out,
        tma_b_out,
        tensor_b_out,
        tma_c,
        tensor_c,
        tiled_mma,
        cta_layout_mnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        epi_smem_layout_staged,
    ):
        if cutlass.const_expr(EMPTY_KERNEL):
            return
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

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
        load_barriers = storage.load_barrier.data_ptr()
        empty_barriers = storage.empty_barrier.data_ptr()
        a_smem = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem = cute.slice_(b_smem_layout_staged, (None, None, 0))
        transaction_bytes = cute.size_in_bytes(self.dtype, a_smem)
        if cutlass.const_expr(ENABLE_B):
            transaction_bytes = transaction_bytes + cute.size_in_bytes(
                self.dtype, b_smem
            )

        if tidx == 0:
            for barrier_index in cutlass.range_constexpr(MAINLOOP_STAGES):
                cute.arch.mbarrier_init(load_barriers + barrier_index, 1)
                cute.arch.mbarrier_init(empty_barriers + barrier_index, 1)
        cute.arch.mbarrier_init_fence()
        if tidx == 0:
            for barrier_index in cutlass.range_constexpr(MAINLOOP_STAGES):
                cute.arch.mbarrier_arrive(empty_barriers + barrier_index)
        pipeline.sync(barrier_id=1)
        if cute.size(CLUSTER_SHAPE_MN) > 1:
            # Every destination barrier must exist before a multicast can target it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        gA = cute.local_tile(
            tensor_a,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        gB = cute.local_tile(
            tensor_b,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        a_cta_layout = cute.make_layout(1)
        b_cta_layout = cute.make_layout(1)
        b_cta_coord = 0
        if cutlass.const_expr(ENABLE_MULTICAST):
            b_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
            )
            b_cta_coord = cluster_coord[0]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        b_mcast_mask = 0
        if cutlass.const_expr(ENABLE_MULTICAST):
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord, mode=0
            )
        load_barrier = load_barriers
        if cutlass.const_expr(not FULL_K) and tidx == 0:
            cute.arch.mbarrier_expect_tx(load_barrier, transaction_bytes)
        if cutlass.const_expr(not FULL_K) and warp_idx == 0:
            cute.copy(
                tma_a,
                tAgA[(None, bidx, 0)],
                tAsA[(None, 0)],
                tma_bar_ptr=load_barrier,
            )
            if cutlass.const_expr(ENABLE_B):
                cute.copy(
                    tma_b,
                    tBgB[(None, 0, 0)],
                    tBsB[(None, 0)],
                    tma_bar_ptr=load_barrier,
                    mcast_mask=b_mcast_mask,
                )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(load_barrier)

        if cutlass.const_expr(not FULL_K):
            cute.arch.mbarrier_wait(load_barrier, 0)
        if cutlass.const_expr(LOAD_ONLY):
            return
        if cutlass.const_expr(ENABLE_WGMMA):
            warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
            if cutlass.const_expr(WARP_SPECIALIZED) and warp_group_idx == 0:
                cute.arch.setmaxregister_decrease(40)
                for producer_tile in cutlass.range(
                    0, FULL_K_TILE_COUNT, 1, unroll=1
                ):
                    producer_stage = producer_tile % MAINLOOP_STAGES
                    producer_phase = (producer_tile // MAINLOOP_STAGES) % 2
                    producer_empty = empty_barriers + producer_stage
                    producer_full = load_barriers + producer_stage
                    cute.arch.mbarrier_wait(producer_empty, producer_phase)
                    if tidx == 0:
                        cute.arch.mbarrier_expect_tx(
                            producer_full, transaction_bytes
                        )
                    if warp_idx == 0:
                        cute.copy(
                            tma_a,
                            tAgA[(None, bidx, producer_tile)],
                            tAsA[(None, producer_stage)],
                            tma_bar_ptr=producer_full,
                        )
                        cute.copy(
                            tma_b,
                            tBgB[(None, bidy, producer_tile)],
                            tBsB[(None, producer_stage)],
                            tma_bar_ptr=producer_full,
                            mcast_mask=b_mcast_mask,
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(producer_full)
            if not WARP_SPECIALIZED or warp_group_idx >= DMA_WARP_GROUPS:
                if cutlass.const_expr(WARP_SPECIALIZED and REGISTER_REPARTITION):
                    cute.arch.setmaxregister_increase(232)
                gC = cute.local_tile(
                    tensor_c,
                    (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[1]),
                    (bidx, bidy),
                )
                warp_group_idx = warp_group_idx - DMA_WARP_GROUPS
                warp_group_thread_layout = cute.make_layout(
                    MMA_WARP_GROUPS, stride=128
                )
                thr_mma = tiled_mma.get_slice(
                    warp_group_thread_layout(warp_group_idx)
                )
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCgC = thr_mma.partition_C(gC)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                if cutlass.const_expr(FULL_K):
                    if cutlass.const_expr(
                        OVERLAPPED_MAINLOOP and not WARP_SPECIALIZED
                    ):
                        # Fill the complete stage ring before consuming stage zero.
                        # Later iterations refill a stage only after WGMMA has finished
                        # reading it, so TMA for tile k+stages can overlap consumption
                        # of the already-full intervening stages.
                        for prefetch_tile in cutlass.range_constexpr(MAINLOOP_STAGES):
                            prefetch_barrier = load_barriers + prefetch_tile
                            if tidx == 0:
                                cute.arch.mbarrier_expect_tx(
                                    prefetch_barrier, transaction_bytes
                                )
                            if warp_idx == 0:
                                cute.copy(
                                    tma_a,
                                    tAgA[(None, bidx, prefetch_tile)],
                                    tAsA[(None, prefetch_tile)],
                                    tma_bar_ptr=prefetch_barrier,
                                )
                                cute.copy(
                                    tma_b,
                                    tBgB[(None, bidy, prefetch_tile)],
                                    tBsB[(None, prefetch_tile)],
                                    tma_bar_ptr=prefetch_barrier,
                                    mcast_mask=b_mcast_mask,
                                )
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive(prefetch_barrier)
                    for k_tile in cutlass.range(0, FULL_K_TILE_COUNT, 1, unroll=1):
                        stage = k_tile % MAINLOOP_STAGES
                        phase = (k_tile // MAINLOOP_STAGES) % 2
                        stage_barrier = load_barriers + stage
                        if cutlass.const_expr(not OVERLAPPED_MAINLOOP):
                            if tidx == 0:
                                cute.arch.mbarrier_expect_tx(
                                    stage_barrier, transaction_bytes
                                )
                            if warp_idx == 0:
                                cute.copy(
                                    tma_a,
                                    tAgA[(None, bidx, k_tile)],
                                    tAsA[(None, stage)],
                                    tma_bar_ptr=stage_barrier,
                                )
                                cute.copy(
                                    tma_b,
                                    tBgB[(None, bidy, k_tile)],
                                    tBsB[(None, stage)],
                                    tma_bar_ptr=stage_barrier,
                                    mcast_mask=b_mcast_mask,
                                )
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive(stage_barrier)
                        cute.arch.mbarrier_wait(stage_barrier, phase)
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
                        if cutlass.const_expr(WARP_SPECIALIZED):
                            if tidx == DMA_WARP_GROUPS * 128:
                                cute.arch.mbarrier_arrive(empty_barriers + stage)
                        if cutlass.const_expr(
                            OVERLAPPED_MAINLOOP and not WARP_SPECIALIZED
                        ):
                            future_tile = k_tile + MAINLOOP_STAGES
                            if future_tile < FULL_K_TILE_COUNT:
                                if tidx == 0:
                                    cute.arch.mbarrier_expect_tx(
                                        stage_barrier, transaction_bytes
                                    )
                                if warp_idx == 0:
                                    cute.copy(
                                        tma_a,
                                        tAgA[(None, bidx, future_tile)],
                                        tAsA[(None, stage)],
                                        tma_bar_ptr=stage_barrier,
                                    )
                                    cute.copy(
                                        tma_b,
                                        tBgB[(None, bidy, future_tile)],
                                        tBsB[(None, stage)],
                                        tma_bar_ptr=stage_barrier,
                                        mcast_mask=b_mcast_mask,
                                    )
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive(stage_barrier)
                else:
                    cute.nvgpu.warpgroup.fence()
                    for k_block in cutlass.range(
                        cute.size(tCrA, mode=[2]), unroll_full=True
                    ):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[(None, None, k_block, 0)],
                            tCrB[(None, None, k_block, 0)],
                            accumulators,
                        )
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                if cutlass.const_expr(not WGMMA_ISSUE_ONLY):
                    consumer_barrier = pipeline.NamedBarrier(
                        barrier_id=1, num_threads=CONSUMER_THREADS
                    )
                    if cutlass.const_expr(WARP_SPECIALIZED):
                        consumer_barrier.arrive_and_wait()
                    else:
                        cute.arch.sync_threads()
                    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                        self.c_layout,
                        elem_ty_d=self.c_dtype,
                        elem_ty_acc=self.acc_dtype,
                    )
                    copy_atom_c = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(), 4
                        ),
                        self.c_dtype,
                    )
                    tiled_copy_c_atom = cute.make_tiled_copy_C_atom(
                        copy_atom_c, tiled_mma
                    )
                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s, tiled_copy_c_atom
                    )
                    epilogue_thread_idx = tidx
                    if cutlass.const_expr(WARP_SPECIALIZED):
                        epilogue_thread_idx = tidx - DMA_WARP_GROUPS * 128
                    if cutlass.const_expr(COOPERATIVE_EPILOGUE):
                        epilogue_thread_idx = tidx % 128
                    thr_copy_r2s = tiled_copy_r2s.get_slice(epilogue_thread_idx)
                    tRS_sC = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                    rC_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    rC_layout = cute.make_layout(rC_shape[:3])
                    rC = cute.make_rmem_tensor_like(rC_layout, self.acc_dtype)
                    rC_out = cute.make_rmem_tensor_like(rC_layout, self.c_dtype)
                    rC_size = cute.size(rC)

                    sC_for_tma = cute.group_modes(sC, 0, 2)
                    gC_for_tma = cute.zipped_divide(gC, self.epi_tile)
                    tma_sC, tma_gC = cute.nvgpu.cpasync.tma_partition(
                        tma_c,
                        0,
                        cute.make_layout(1),
                        sC_for_tma,
                        gC_for_tma,
                    )
                    epi_tile_count = cute.size(gC_for_tma, mode=[1])
                    if cutlass.const_expr(WGMMA_R2S_FIRST_TILE):
                        epi_tile_count = 1
                    if cutlass.const_expr(WGMMA_R2S_TWO_TILES):
                        epi_tile_count = 2
                    if cutlass.const_expr(WGMMA_R2S_THREE_TILES):
                        epi_tile_count = 3
                    if cutlass.const_expr(WGMMA_R2S_FOUR_REUSE_ZERO):
                        epi_tile_count = 4
                    if cutlass.const_expr(WGMMA_R2S_FOUR_PADDED):
                        epi_tile_count = 4
                    if cutlass.const_expr(WGMMA_R2S_FOUR_TILES):
                        epi_tile_count = 4
                    epi_tile_shape = gC_for_tma.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )
                    c_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=pipeline.CooperativeGroup(
                            pipeline.Agent.Thread, CONSUMER_THREADS
                        ),
                    )
                    if cutlass.const_expr(COOPERATIVE_EPILOGUE):
                        # Each warp group owns one 64-row accumulator partition. Stage its
                        # four N tiles into disjoint shared-memory buffers before one elected
                        # warp issues all CTA-level TMA stores. This avoids treating the
                        # second warp group's local accumulator as a continuation of the
                        # first group's register fragment.
                        for local_epi_index in cutlass.range_constexpr(
                            EPILOGUE_TILES_PER_WARP_GROUP
                        ):
                            for value_index in cutlass.range_constexpr(rC_size):
                                rC[value_index] = tRS_rAcc[
                                    local_epi_index * rC_size + value_index
                                ]
                            rC_out.store(rC.load().to(self.c_dtype))
                            epi_buffer = (
                                warp_group_idx * EPILOGUE_TILES_PER_WARP_GROUP
                                + local_epi_index
                            )
                            cute.copy(
                                tiled_copy_r2s,
                                rC_out,
                                tRS_sC[(None, None, None, epi_buffer)],
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(WARP_SPECIALIZED):
                            consumer_barrier.arrive_and_wait()
                        else:
                            pipeline.sync(barrier_id=1)
                        if cutlass.const_expr(not WGMMA_R2S_ONLY):
                            if warp_idx == DMA_WARP_GROUPS * 4:
                                for epi_index in cutlass.range_constexpr(epi_tile_count):
                                    global_coord = epi_tile_layout.get_hier_coord(epi_index)
                                    cute.copy(
                                        tma_c,
                                        tma_sC[(None, epi_index)],
                                        tma_gC[(None, global_coord)],
                                    )
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                        if cutlass.const_expr(WARP_SPECIALIZED):
                            consumer_barrier.arrive_and_wait()
                        else:
                            pipeline.sync(barrier_id=1)
                        if (
                            cutlass.const_expr(not WGMMA_R2S_ONLY)
                            and warp_idx == DMA_WARP_GROUPS * 4
                        ):
                            c_pipeline.producer_tail()
                    else:
                        for epi_index in cutlass.range_constexpr(epi_tile_count):
                            for value_index in cutlass.range_constexpr(rC_size):
                                rC[value_index] = tRS_rAcc[
                                    epi_index * rC_size + value_index
                                ]
                            rC_out.store(rC.load().to(self.c_dtype))
                            epi_buffer = epi_index % cute.size(tRS_sC, mode=[3])
                            if cutlass.const_expr(WGMMA_R2S_FOUR_REUSE_ZERO):
                                epi_buffer = 0
                            cute.copy(
                                tiled_copy_r2s,
                                rC_out,
                                tRS_sC[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            if cutlass.const_expr(WARP_SPECIALIZED):
                                consumer_barrier.arrive_and_wait()
                            else:
                                pipeline.sync(barrier_id=1)
                            if cutlass.const_expr(not WGMMA_R2S_ONLY):
                                global_coord = epi_tile_layout.get_hier_coord(epi_index)
                                if warp_idx == DMA_WARP_GROUPS * 4:
                                    cute.copy(
                                        tma_c,
                                        tma_sC[(None, epi_buffer)],
                                        tma_gC[(None, global_coord)],
                                    )
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                            if cutlass.const_expr(WARP_SPECIALIZED):
                                consumer_barrier.arrive_and_wait()
                            else:
                                pipeline.sync(barrier_id=1)
                        if (
                            cutlass.const_expr(not WGMMA_R2S_ONLY)
                            and warp_idx == DMA_WARP_GROUPS * 4
                        ):
                            c_pipeline.producer_tail()
            return
        cute.arch.fence_proxy("async.shared", space="cta")
        pipeline.sync(barrier_id=1)

        gA_out = cute.local_tile(
            tensor_a_out,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        tAo_s, tAo_g = cute.nvgpu.cpasync.tma_partition(
            tma_a_out,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_out, 0, 2),
        )
        if warp_idx == 0:
            cute.copy(tma_a_out, tAo_s[(None, 0)], tAo_g[(None, bidx, 0)])

        if cutlass.const_expr(ENABLE_B) and cta_rank == 0:
            gB_out = cute.local_tile(
                tensor_b_out,
                (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
                (None, None),
            )
            tBo_s, tBo_g = cute.nvgpu.cpasync.tma_partition(
                tma_b_out,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_out, 0, 2),
            )
            if warp_idx == 0:
                cute.copy(tma_b_out, tBo_s[(None, 0)], tBo_g[(None, 0, 0)])

        if warp_idx == 0:
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)


def run_diagnostic():
    import ctypes
    import itertools
    import subprocess
    import statistics
    import tempfile
    from pathlib import Path
    import torch
    from kernel_mcts.cute_diagnostics import describe_kernel_callable

    print(f"Independent TMA {DEBUG_STAGE}: allocating tensors", flush=True)
    input_directory = None
    if REPOSITORY_CONTRACT:
        input_directory = tempfile.TemporaryDirectory(
            prefix="kernel-mcts-independent-inputs-"
        )
        root = Path(input_directory.name)
        a_path = root / "a.bf16"
        b_path = root / "b.bf16"
        subprocess.run(
            [
                "/usr/local/bin/kernel-mcts-bf16-inputs",
                str(PROBLEM_M), str(PROBLEM_N), str(PROBLEM_K), "0",
                str(a_path), str(b_path),
            ],
            check=True,
            timeout=120,
        )
        a = torch.from_file(
            str(a_path), shared=False, size=PROBLEM_M * PROBLEM_K,
            dtype=torch.bfloat16,
        ).reshape(PROBLEM_M, PROBLEM_K).to("cuda")
        b = torch.from_file(
            str(b_path), shared=False, size=PROBLEM_N * PROBLEM_K,
            dtype=torch.bfloat16,
        ).reshape(PROBLEM_N, PROBLEM_K).to("cuda")
    else:
        torch.manual_seed(0)
        a = torch.randn(
            (PROBLEM_M, PROBLEM_K), device="cuda", dtype=torch.bfloat16
        )
        b = torch.randn(
            (PROBLEM_N, PROBLEM_K), device="cuda", dtype=torch.bfloat16
        )
    a_out = torch.zeros_like(a)
    b_out = torch.zeros_like(b)
    c = torch.zeros(
        (PROBLEM_M, PROBLEM_N),
        device="cuda",
        dtype=torch.bfloat16,
    )
    tensors = [
        from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        for value in (a, b, c, a_out, b_out)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    diagnostic = IndependentTmaCopyKernel()
    print(f"Independent TMA {DEBUG_STAGE}: JIT started", flush=True)
    compiled = cute.compile(diagnostic, *tensors, stream)
    jit_diagnostics = describe_kernel_callable(compiled, {})
    print(f"Independent TMA {DEBUG_STAGE}: JIT completed", flush=True)
    if not LAUNCH_KERNEL:
        return {
            "status": "compile_only_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }
    print(f"Independent TMA {DEBUG_STAGE}: kernel launch started", flush=True)
    compiled(*tensors, stream)
    print(f"Independent TMA {DEBUG_STAGE}: kernel launch returned", flush=True)
    print(f"Independent TMA {DEBUG_STAGE}: synchronize started", flush=True)
    torch.cuda.synchronize()
    print(f"Independent TMA {DEBUG_STAGE}: synchronize completed", flush=True)
    if EMPTY_KERNEL:
        return {
            "status": "empty_launch_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }
    if LOAD_ONLY:
        return {
            "status": "load_only_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }
    if WGMMA_ISSUE_ONLY:
        return {
            "status": "wgmma_issue_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }
    if WGMMA_R2S_ONLY:
        return {
            "status": "wgmma_r2s_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }
    if ENABLE_WGMMA:
        if REPOSITORY_CONTRACT:
            reference = torch.empty_like(c)
            library = ctypes.CDLL(
                "/usr/local/lib/kernel-mcts-bf16-reference.so"
            )
            reference_call = library.kernel_mcts_bf16_reference
            reference_call.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            reference_call.restype = ctypes.c_int
            status = reference_call(
                a.data_ptr(), b.data_ptr(), reference.data_ptr(),
                PROBLEM_M, PROBLEM_N, PROBLEM_K,
            )
            if status != 0:
                raise RuntimeError(f"cuBLAS reference failed with status {status}")
            torch.cuda.synchronize()
        else:
            reference = torch.matmul(a.float(), b.float().transpose(0, 1)).to(
                torch.bfloat16
            )
        output_float = c.float()
        reference_float = reference.float()
        maximum_error = float((c.float() - reference.float()).abs().max().item())
        mean_error = float((c.float() - reference.float()).abs().mean().item())
        output_abs_max = float(c.float().abs().max().item())
        output_nonzero = int(torch.count_nonzero(c).item())
        reference_abs_max = float(reference.float().abs().max().item())
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                output_float.reshape(1, -1),
                reference_float.reshape(1, -1),
            ).item()
        )
        best_tile_mean_error = None
        best_tile_permutation = None
        best_transposed_tile_mean_error = None
        best_transposed_tile_permutation = None
        if not FULL_WORKLOAD:
            output_tiles = [
                output_float[
                    tile_m * EPILOGUE_TILE[0] : (tile_m + 1) * EPILOGUE_TILE[0],
                    tile_n * EPILOGUE_TILE[1] : (tile_n + 1) * EPILOGUE_TILE[1],
                ]
                for tile_m in range(TILE_SHAPE_MNK[0] // EPILOGUE_TILE[0])
                for tile_n in range(TILE_SHAPE_MNK[1] // EPILOGUE_TILE[1])
            ]
            reference_tiles = [
                reference_float[
                    tile_m * EPILOGUE_TILE[0] : (tile_m + 1) * EPILOGUE_TILE[0],
                    tile_n * EPILOGUE_TILE[1] : (tile_n + 1) * EPILOGUE_TILE[1],
                ]
                for tile_m in range(TILE_SHAPE_MNK[0] // EPILOGUE_TILE[0])
                for tile_n in range(TILE_SHAPE_MNK[1] // EPILOGUE_TILE[1])
            ]
            tile_costs = [
                [float((out - ref).abs().mean().item()) for ref in reference_tiles]
                for out in output_tiles
            ]
            transposed_tile_costs = [
                [
                    float((out - ref.transpose(0, 1)).abs().mean().item())
                    for ref in reference_tiles
                ]
                for out in output_tiles
            ]

            def best_assignment(costs):
                best_cost = float("inf")
                best_permutation = None
                for permutation in itertools.permutations(range(len(costs))):
                    cost = sum(
                        costs[index][source]
                        for index, source in enumerate(permutation)
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_permutation = permutation
                return best_cost / len(costs), list(best_permutation)

            best_tile_mean_error, best_tile_permutation = best_assignment(tile_costs)
            (
                best_transposed_tile_mean_error,
                best_transposed_tile_permutation,
            ) = best_assignment(transposed_tile_costs)
        error = (c.float() - reference.float()).abs()
        correct = bool(torch.all(torch.isfinite(c) & (
            error <= 0.02 + 0.02 * reference.float().abs()
        )).item())
        benchmark = None
        if FULL_WORKLOAD:
            warmup_count = 0 if PROFILE_SINGLE_LAUNCH else 10
            measurement_count = 1 if PROFILE_SINGLE_LAUNCH else 30
            for _ in range(warmup_count):
                compiled(*tensors, stream)
            torch.cuda.synchronize()
            timings_us = []
            for _ in range(measurement_count):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                compiled(*tensors, stream)
                end.record()
                end.synchronize()
                timings_us.append(float(start.elapsed_time(end) * 1000.0))
            benchmark = {
                "warmup_count": 10,
                "measurement_count": 30,
                "timings_us": timings_us,
                "median_us": float(statistics.median(timings_us)),
                "mean_us": float(statistics.mean(timings_us)),
                "min_us": min(timings_us),
                "max_us": max(timings_us),
            }
        result = {
            "status": "ok" if correct else "correctness_failed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
            "correct": correct,
            "maximum_error": maximum_error,
            "mean_error": mean_error,
            "output_abs_max": output_abs_max,
            "output_nonzero": output_nonzero,
            "reference_abs_max": reference_abs_max,
            "cosine_similarity": cosine_similarity,
            "best_tile_mean_error": best_tile_mean_error,
            "best_tile_permutation": best_tile_permutation,
            "best_transposed_tile_mean_error": best_transposed_tile_mean_error,
            "best_transposed_tile_permutation": best_transposed_tile_permutation,
            "benchmark": benchmark,
            "jit_diagnostics": jit_diagnostics,
        }
        if REPOSITORY_CONTRACT:
            result.update({
                "implementation": "independent_cute_gemm_v1",
                "comparable_to_repository_baselines": True,
                "comparability_blockers": [],
                "correctness": {
                    "success": correct,
                    "maximum_error": maximum_error,
                    "mean_error": mean_error,
                    "failed_test_id": None if correct else "fixed_shape",
                    "reference_metadata": {
                        "implementation": "cuBLAS",
                        "compute_type": "CUBLAS_COMPUTE_32F",
                        "seed": 0,
                    },
                },
                "contract": {
                    "mnkl": [PROBLEM_M, PROBLEM_N, PROBLEM_K, 1],
                    "rtol": 0.02,
                    "atol": 0.02,
                    "seed": 0,
                    "warmup_count": warmup_count,
                    "measurement_count": measurement_count,
                },
            })
        if input_directory is not None:
            input_directory.cleanup()
        return result
    a_equal = bool(torch.equal(a, a_out))
    b_equal = bool(torch.equal(b, b_out)) if ENABLE_B else None
    passed = a_equal and (b_equal if ENABLE_B else True)
    return {
        "status": "ok" if passed else "correctness_failed",
        "debug_stage": DEBUG_STAGE,
        "configuration_hash": CONFIGURATION_HASH,
        "a_exact": a_equal,
        "b_exact": b_equal,
        "a_max_error": float((a.float() - a_out.float()).abs().max().item()),
        "b_max_error": (
            float((b.float() - b_out.float()).abs().max().item())
            if ENABLE_B
            else None
        ),
        "cluster_shape_mn": list(CLUSTER_SHAPE_MN),
        "tile_shape_mnk": list(TILE_SHAPE_MNK),
        "swizzle_bytes": SWIZZLE_BYTES,
    }
