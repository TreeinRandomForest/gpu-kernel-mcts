#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

using AccFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::accumulator, 16, 16, 16, float>;
using AFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_a, 16, 16, 16,
    __nv_bfloat16, nvcuda::wmma::row_major>;
using BFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_b, 16, 16, 16,
    __nv_bfloat16, nvcuda::wmma::col_major>;

constexpr int DIM = 4096;
constexpr int K_PANEL = 32;
constexpr int PANEL_COUNT = DIM / K_PANEL;
constexpr int SHARED_LD = 40;

union __align__(32) SharedStorage {
    struct {
        __nv_bfloat16 a[2][128 * SHARED_LD];
        __nv_bfloat16 b[2][128 * SHARED_LD];
    } input;
    float output[8][256];
};

union Bfloat2Bits {
    __nv_bfloat162 value;
    uint32_t bits;
};

__device__ __forceinline__ void cp_async_ca_16(
    void* shared_ptr,
    const void* global_ptr) {
    const uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(shared_address), "l"(global_ptr)
        : "memory");
}

__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ void async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
}

__device__ __forceinline__ void async_load_panel(
    __nv_bfloat16* __restrict__ shared_a,
    __nv_bfloat16* __restrict__ shared_b,
    const __nv_bfloat16* __restrict__ global_a,
    const __nv_bfloat16* __restrict__ global_b,
    int tid) {
    const int row = tid >> 2;
    const int offset = (tid & 3) << 3;

    __nv_bfloat16* const shared_a0 =
        shared_a + row * SHARED_LD + offset;
    __nv_bfloat16* const shared_b0 =
        shared_b + row * SHARED_LD + offset;
    const __nv_bfloat16* const global_a0 =
        global_a + (row << 12) + offset;
    const __nv_bfloat16* const global_b0 =
        global_b + (row << 12) + offset;

    cp_async_ca_16(shared_a0, global_a0);
    cp_async_ca_16(shared_b0, global_b0);
    cp_async_ca_16(
        shared_a0 + 64 * SHARED_LD,
        global_a0 + (64 << 12));
    cp_async_ca_16(
        shared_b0 + 64 * SHARED_LD,
        global_b0 + (64 << 12));
    async_commit();
}

__device__ __forceinline__ void compute_k_slice(
    const __nv_bfloat16* warp_a,
    const __nv_bfloat16* warp_b,
    int k_offset,
    AccFragment (&accum)[2][4]) {
    AFragment af0;
    AFragment af1;
    BFragment bf;

    nvcuda::wmma::load_matrix_sync(
        af0, warp_a + k_offset, SHARED_LD);
    nvcuda::wmma::load_matrix_sync(
        af1, warp_a + 16 * SHARED_LD + k_offset, SHARED_LD);

    nvcuda::wmma::load_matrix_sync(
        bf, warp_b + k_offset, SHARED_LD);
    nvcuda::wmma::mma_sync(accum[0][0], af0, bf, accum[0][0]);
    nvcuda::wmma::mma_sync(accum[1][0], af1, bf, accum[1][0]);

    nvcuda::wmma::load_matrix_sync(
        bf, warp_b + 16 * SHARED_LD + k_offset, SHARED_LD);
    nvcuda::wmma::mma_sync(accum[0][1], af0, bf, accum[0][1]);
    nvcuda::wmma::mma_sync(accum[1][1], af1, bf, accum[1][1]);

    nvcuda::wmma::load_matrix_sync(
        bf, warp_b + 32 * SHARED_LD + k_offset, SHARED_LD);
    nvcuda::wmma::mma_sync(accum[0][2], af0, bf, accum[0][2]);
    nvcuda::wmma::mma_sync(accum[1][2], af1, bf, accum[1][2]);

    nvcuda::wmma::load_matrix_sync(
        bf, warp_b + 48 * SHARED_LD + k_offset, SHARED_LD);
    nvcuda::wmma::mma_sync(accum[0][3], af0, bf, accum[0][3]);
    nvcuda::wmma::mma_sync(accum[1][3], af1, bf, accum[1][3]);
}

__device__ __forceinline__ void compute_panel(
    const __nv_bfloat16* warp_a,
    const __nv_bfloat16* warp_b,
    AccFragment (&accum)[2][4]) {
    compute_k_slice(warp_a, warp_b, 0, accum);
    compute_k_slice(warp_a, warp_b, 16, accum);
}

__device__ __forceinline__ void store_bf16_tile(
    float* __restrict__ staging,
    const AccFragment& fragment,
    __nv_bfloat16* __restrict__ C,
    int row,
    int col,
    int lane) {
    nvcuda::wmma::store_matrix_sync(
        staging, fragment, 16, nvcuda::wmma::mem_row_major);

    const int local_row = lane >> 1;
    const int local_col = (lane & 1) << 3;
    const float* const src = staging + (local_row << 4) + local_col;

    Bfloat2Bits p0;
    Bfloat2Bits p1;
    Bfloat2Bits p2;
    Bfloat2Bits p3;
    p0.value = __floats2bfloat162_rn(src[0], src[1]);
    p1.value = __floats2bfloat162_rn(src[2], src[3]);
    p2.value = __floats2bfloat162_rn(src[4], src[5]);
    p3.value = __floats2bfloat162_rn(src[6], src[7]);

    *reinterpret_cast<uint4*>(
        C + ((row + local_row) << 12) + col + local_col) =
        make_uint4(p0.bits, p1.bits, p2.bits, p3.bits);
}

extern "C" __global__ __launch_bounds__(256, 2) void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    const bool fast_path =
        M == DIM && N == DIM && K == DIM &&
        ((reinterpret_cast<uintptr_t>(A) & 31u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(B) & 31u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(C) & 15u) == 0u);

    if (!fast_path) {
        const int tx = static_cast<int>(threadIdx.x);
        const int ty = static_cast<int>(threadIdx.y);
        const int row = static_cast<int>(blockIdx.y) * 16 + ty;
        const int col = static_cast<int>(blockIdx.x) * 16 + tx;
        if (row < M && col < N) {
            float accumulator = 0.0f;
            for (int k = 0; k < K; ++k) {
                accumulator += __bfloat162float(A[row * K + k]) *
                               __bfloat162float(B[col * K + k]);
            }
            C[row * N + col] = __float2bfloat16(accumulator);
        }
        return;
    }

    const int linear_block =
        (static_cast<int>(blockIdx.y) << 8) + static_cast<int>(blockIdx.x);
    if (linear_block >= 1024) {
        return;
    }

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = (ty << 4) + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int macro_row = (linear_block >> 5) << 7;
    const int macro_col = (linear_block & 31) << 7;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int local_row_base = warp_row << 5;
    const int local_col_base = warp_col << 6;
    const int local_a_offset = local_row_base * SHARED_LD;
    const int local_b_offset = local_col_base * SHARED_LD;

    AccFragment accum[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            nvcuda::wmma::fill_fragment(accum[i][j], 0.0f);
        }
    }

    __shared__ SharedStorage shared;

    const __nv_bfloat16* const macro_a = A + (macro_row << 12);
    const __nv_bfloat16* const macro_b = B + (macro_col << 12);

    async_load_panel(
        shared.input.a[0],
        shared.input.b[0],
        macro_a,
        macro_b,
        tid);
    async_wait_all();
    __syncthreads();

#pragma unroll 1
    for (int panel = 0; panel < PANEL_COUNT - 1; ++panel) {
        const int current = panel & 1;
        const int next = current ^ 1;
        const int next_k = (panel + 1) * K_PANEL;

        async_load_panel(
            shared.input.a[next],
            shared.input.b[next],
            macro_a + next_k,
            macro_b + next_k,
            tid);

        compute_panel(
            shared.input.a[current] + local_a_offset,
            shared.input.b[current] + local_b_offset,
            accum);

        async_wait_all();
        __syncthreads();
    }

    compute_panel(
        shared.input.a[1] + local_a_offset,
        shared.input.b[1] + local_b_offset,
        accum);

    __syncthreads();

    float* const tile = shared.output[warp_id];
    const int row0 = macro_row + local_row_base;
    const int row1 = row0 + 16;
    const int col0 = macro_col + local_col_base;

    store_bf16_tile(tile, accum[0][0], C, row0, col0, lane_id);
    store_bf16_tile(tile, accum[0][1], C, row0, col0 + 16, lane_id);
    store_bf16_tile(tile, accum[0][2], C, row0, col0 + 32, lane_id);
    store_bf16_tile(tile, accum[0][3], C, row0, col0 + 48, lane_id);
    store_bf16_tile(tile, accum[1][0], C, row1, col0, lane_id);
    store_bf16_tile(tile, accum[1][1], C, row1, col0 + 16, lane_id);
    store_bf16_tile(tile, accum[1][2], C, row1, col0 + 32, lane_id);
    store_bf16_tile(tile, accum[1][3], C, row1, col0 + 48, lane_id);
}