#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int FIXED_K = 4096;
constexpr int K_STAGE = 32;
constexpr int K_STAGES = FIXED_K / K_STAGE;
constexpr int SHARED_LD = 40;
constexpr int CTA_ROWS = 128;
constexpr int CTA_COLS = 128;
constexpr int A_STAGE_ELEMENTS = CTA_ROWS * SHARED_LD;
constexpr int B_STAGE_ELEMENTS = CTA_COLS * SHARED_LD;

using AccFragment =
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using AFragment =
    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   __nv_bfloat16, wmma::row_major>;
using BFragment =
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   __nv_bfloat16, wmma::col_major>;

struct PipeStorage {
    __nv_bfloat16 a[2][A_STAGE_ELEMENTS];
    __nv_bfloat16 b[2][B_STAGE_ELEMENTS];
};

union SharedStorage {
    PipeStorage pipe;
    float output[8][4][256];
};

union Bfloat2Bits {
    __nv_bfloat162 value;
    unsigned int bits;
};

__device__ __forceinline__ void copy_16_async(
    void* shared_destination,
    const void* global_source) {
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_destination));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shared_address), "l"(global_source)
        : "memory");
}

__device__ __forceinline__ void commit_async_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void wait_for_async_groups() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void issue_stage(
    PipeStorage& pipe,
    int stage,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    int tile_row,
    int tile_col,
    int k_base,
    int tid) {
#pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        const int vector_index = tid + iteration * 256;
        const int row = vector_index >> 2;
        const int k_offset = (vector_index & 3) << 3;
        const int global_index =
            ((tile_row + row) << 12) + k_base + k_offset;

        copy_16_async(
            pipe.a[stage] + row * SHARED_LD + k_offset,
            A + global_index);
    }

#pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        const int vector_index = tid + iteration * 256;
        const int column = vector_index >> 2;
        const int k_offset = (vector_index & 3) << 3;
        const int global_index =
            ((tile_col + column) << 12) + k_base + k_offset;

        copy_16_async(
            pipe.b[stage] + column * SHARED_LD + k_offset,
            B + global_index);
    }
}

__device__ __forceinline__ unsigned int pack_two_bf16(
    float x0,
    float x1) {
    Bfloat2Bits packed;
    packed.value = __floats2bfloat162_rn(x0, x1);
    return packed.bits;
}

__device__ __forceinline__ void convert_and_store_tile(
    const float* scratch,
    __nv_bfloat16* C,
    int output_row,
    int output_column,
    int lane_id) {
    const int row = lane_id >> 1;
    const int column = (lane_id & 1) << 3;
    const int element = (row << 4) + column;

    __nv_bfloat16* const destination =
        C + ((output_row + row) << 12) + output_column + column;

    const uint4 packed = make_uint4(
        pack_two_bf16(scratch[element], scratch[element + 1]),
        pack_two_bf16(scratch[element + 2], scratch[element + 3]),
        pack_two_bf16(scratch[element + 4], scratch[element + 5]),
        pack_two_bf16(scratch[element + 6], scratch[element + 7]));

    *reinterpret_cast<uint4*>(destination) = packed;
}

}  // namespace

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    (void)M;
    (void)N;
    (void)K;

    if ((blockIdx.x & 7u) != 0u || (blockIdx.y & 7u) != 0u) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.y) * 16 +
                    static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int tile_row = static_cast<int>(blockIdx.y) << 4;
    const int tile_col = static_cast<int>(blockIdx.x) << 4;
    const int local_row = (warp_id & 1) << 6;
    const int local_column = (warp_id >> 1) << 5;

    AFragment a_fragment0;
    AFragment a_fragment1;
    AFragment a_fragment2;
    AFragment a_fragment3;
    BFragment b_fragment0;
    BFragment b_fragment1;

    AccFragment accumulator00;
    AccFragment accumulator01;
    AccFragment accumulator10;
    AccFragment accumulator11;
    AccFragment accumulator20;
    AccFragment accumulator21;
    AccFragment accumulator30;
    AccFragment accumulator31;

    wmma::fill_fragment(accumulator00, 0.0f);
    wmma::fill_fragment(accumulator01, 0.0f);
    wmma::fill_fragment(accumulator10, 0.0f);
    wmma::fill_fragment(accumulator11, 0.0f);
    wmma::fill_fragment(accumulator20, 0.0f);
    wmma::fill_fragment(accumulator21, 0.0f);
    wmma::fill_fragment(accumulator30, 0.0f);
    wmma::fill_fragment(accumulator31, 0.0f);

    __shared__ __align__(16) SharedStorage shared;

    issue_stage(shared.pipe, 0, A, B, tile_row, tile_col, 0, tid);
    commit_async_group();
    wait_for_async_groups();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < K_STAGES; ++iteration) {
        const int stage = iteration & 1;
        const int next_iteration = iteration + 1;

        if (next_iteration < K_STAGES) {
            issue_stage(
                shared.pipe,
                stage ^ 1,
                A,
                B,
                tile_row,
                tile_col,
                next_iteration << 5,
                tid);
            commit_async_group();
        }

#pragma unroll
        for (int k_inner = 0; k_inner < K_STAGE; k_inner += 16) {
            const __nv_bfloat16* const a_base =
                shared.pipe.a[stage] + local_row * SHARED_LD + k_inner;
            const __nv_bfloat16* const b_base =
                shared.pipe.b[stage] + local_column * SHARED_LD + k_inner;

            wmma::load_matrix_sync(a_fragment0, a_base, SHARED_LD);
            wmma::load_matrix_sync(
                a_fragment1, a_base + 16 * SHARED_LD, SHARED_LD);
            wmma::load_matrix_sync(
                a_fragment2, a_base + 32 * SHARED_LD, SHARED_LD);
            wmma::load_matrix_sync(
                a_fragment3, a_base + 48 * SHARED_LD, SHARED_LD);

            wmma::load_matrix_sync(b_fragment0, b_base, SHARED_LD);
            wmma::load_matrix_sync(
                b_fragment1, b_base + 16 * SHARED_LD, SHARED_LD);

            wmma::mma_sync(
                accumulator00, a_fragment0, b_fragment0, accumulator00);
            wmma::mma_sync(
                accumulator01, a_fragment0, b_fragment1, accumulator01);
            wmma::mma_sync(
                accumulator10, a_fragment1, b_fragment0, accumulator10);
            wmma::mma_sync(
                accumulator11, a_fragment1, b_fragment1, accumulator11);
            wmma::mma_sync(
                accumulator20, a_fragment2, b_fragment0, accumulator20);
            wmma::mma_sync(
                accumulator21, a_fragment2, b_fragment1, accumulator21);
            wmma::mma_sync(
                accumulator30, a_fragment3, b_fragment0, accumulator30);
            wmma::mma_sync(
                accumulator31, a_fragment3, b_fragment1, accumulator31);
        }

        if (next_iteration < K_STAGES) {
            wait_for_async_groups();
            __syncthreads();
        }
    }

    __syncthreads();

    float (*const scratch)[256] = shared.output[warp_id];
    const int output_row = tile_row + local_row;
    const int output_column = tile_col + local_column;

    wmma::store_matrix_sync(
        scratch[0], accumulator00, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[1], accumulator01, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[2], accumulator10, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[3], accumulator11, 16, wmma::mem_row_major);

    convert_and_store_tile(
        scratch[0], C, output_row, output_column, lane_id);
    convert_and_store_tile(
        scratch[1], C, output_row, output_column + 16, lane_id);
    convert_and_store_tile(
        scratch[2], C, output_row + 16, output_column, lane_id);
    convert_and_store_tile(
        scratch[3], C, output_row + 16, output_column + 16, lane_id);

    wmma::store_matrix_sync(
        scratch[0], accumulator20, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[1], accumulator21, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[2], accumulator30, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch[3], accumulator31, 16, wmma::mem_row_major);

    convert_and_store_tile(
        scratch[0], C, output_row + 32, output_column, lane_id);
    convert_and_store_tile(
        scratch[1], C, output_row + 32, output_column + 16, lane_id);
    convert_and_store_tile(
        scratch[2], C, output_row + 48, output_column, lane_id);
    convert_and_store_tile(
        scratch[3], C, output_row + 48, output_column + 16, lane_id);
}