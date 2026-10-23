#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int TILE = 16;
    constexpr int WARPS = 8;
    constexpr int SPLIT_K = 4;
    constexpr int OUTPUT_TILES = 8;
    constexpr int ROW_TILES = 4;
    constexpr int ACCUMULATORS = 4;
    constexpr int FRAGMENT_STRIDE = TILE * TILE;

    const int tid = static_cast<int>(threadIdx.y) * TILE +
                    static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int block_row = by * TILE;
    const int group_bx = bx & ~(OUTPUT_TILES - 1);
    const int group_col = group_bx * TILE;
    const int group_by = by & ~(ROW_TILES - 1);
    const int group_row = group_by * TILE;

    __shared__ __align__(32) unsigned char storage[
        ACCUMULATORS * WARPS * FRAGMENT_STRIDE * sizeof(float)];

    const bool grouped_tensor_path =
        group_row + ROW_TILES * TILE <= M &&
        group_col + OUTPUT_TILES * TILE <= N &&
        (K & (TILE - 1)) == 0;

    if (grouped_tensor_path) {
        if (bx != group_bx || by != group_by) {
            return;
        }

        float* partial = reinterpret_cast<float*>(storage);
        const int output_group = warp >> 2;
        const int split = warp & (SPLIT_K - 1);
        const int output_col0 = group_col + output_group * TILE;
        const int output_col1 = output_col0 + 2 * TILE;
        const int output_col2 = output_col0 + 4 * TILE;
        const int output_col3 = output_col0 + 6 * TILE;

        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::col_major> b_frag0;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::col_major> b_frag1;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::col_major> b_frag2;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::col_major> b_frag3;

        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float>
            acc[ROW_TILES][ACCUMULATORS];

#pragma unroll
        for (int r = 0; r < ROW_TILES; ++r) {
#pragma unroll
            for (int a = 0; a < ACCUMULATORS; ++a) {
                wmma::fill_fragment(acc[r][a], 0.0f);
            }
        }

        const __nv_bfloat16* a_base0 =
            A + static_cast<long long>(group_row) * K;
        const __nv_bfloat16* a_base1 = a_base0 + static_cast<long long>(TILE) * K;
        const __nv_bfloat16* a_base2 = a_base1 + static_cast<long long>(TILE) * K;
        const __nv_bfloat16* a_base3 = a_base2 + static_cast<long long>(TILE) * K;
        const __nv_bfloat16* b_base0 =
            B + static_cast<long long>(output_col0) * K;
        const __nv_bfloat16* b_base1 =
            B + static_cast<long long>(output_col1) * K;
        const __nv_bfloat16* b_base2 =
            B + static_cast<long long>(output_col2) * K;
        const __nv_bfloat16* b_base3 =
            B + static_cast<long long>(output_col3) * K;

        for (int k = split * TILE; k < K; k += SPLIT_K * TILE) {
            wmma::load_matrix_sync(b_frag0, b_base0 + k, K);
            wmma::load_matrix_sync(b_frag1, b_base1 + k, K);
            wmma::load_matrix_sync(b_frag2, b_base2 + k, K);
            wmma::load_matrix_sync(b_frag3, b_base3 + k, K);

            wmma::load_matrix_sync(a_frag, a_base0 + k, K);
            wmma::mma_sync(acc[0][0], a_frag, b_frag0, acc[0][0]);
            wmma::mma_sync(acc[0][1], a_frag, b_frag1, acc[0][1]);
            wmma::mma_sync(acc[0][2], a_frag, b_frag2, acc[0][2]);
            wmma::mma_sync(acc[0][3], a_frag, b_frag3, acc[0][3]);

            wmma::load_matrix_sync(a_frag, a_base1 + k, K);
            wmma::mma_sync(acc[1][0], a_frag, b_frag0, acc[1][0]);
            wmma::mma_sync(acc[1][1], a_frag, b_frag1, acc[1][1]);
            wmma::mma_sync(acc[1][2], a_frag, b_frag2, acc[1][2]);
            wmma::mma_sync(acc[1][3], a_frag, b_frag3, acc[1][3]);

            wmma::load_matrix_sync(a_frag, a_base2 + k, K);
            wmma::mma_sync(acc[2][0], a_frag, b_frag0, acc[2][0]);
            wmma::mma_sync(acc[2][1], a_frag, b_frag1, acc[2][1]);
            wmma::mma_sync(acc[2][2], a_frag, b_frag2, acc[2][2]);
            wmma::mma_sync(acc[2][3], a_frag, b_frag3, acc[2][3]);

            wmma::load_matrix_sync(a_frag, a_base3 + k, K);
            wmma::mma_sync(acc[3][0], a_frag, b_frag0, acc[3][0]);
            wmma::mma_sync(acc[3][1], a_frag, b_frag1, acc[3][1]);
            wmma::mma_sync(acc[3][2], a_frag, b_frag2, acc[3][2]);
            wmma::mma_sync(acc[3][3], a_frag, b_frag3, acc[3][3]);
        }

        const int local_row = tid >> 4;
        const int local_col = tid & 15;

#pragma unroll
        for (int r = 0; r < ROW_TILES; ++r) {
#pragma unroll
            for (int a = 0; a < ACCUMULATORS; ++a) {
                wmma::store_matrix_sync(
                    partial + (a * WARPS + warp) * FRAGMENT_STRIDE,
                    acc[r][a], TILE, wmma::mem_row_major);
            }
            __syncthreads();

            const long long c_row =
                static_cast<long long>(group_row + r * TILE + local_row) * N;

#pragma unroll
            for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
                const int accumulator_id = tile >> 1;
                const int output_group_id = tile & 1;
                const int base_warp =
                    accumulator_id * WARPS + output_group_id * SPLIT_K;
                float sum =
                    partial[(base_warp + 0) * FRAGMENT_STRIDE + tid];
                sum += partial[(base_warp + 1) * FRAGMENT_STRIDE + tid];
                sum += partial[(base_warp + 2) * FRAGMENT_STRIDE + tid];
                sum += partial[(base_warp + 3) * FRAGMENT_STRIDE + tid];
                C[c_row + group_col + tile * TILE + local_col] =
                    __float2bfloat16_rn(sum);
            }
            __syncthreads();
        }
        return;
    }

    __nv_bfloat16* shared_bf16 =
        reinterpret_cast<__nv_bfloat16*>(storage);
    __nv_bfloat16* a_tile = shared_bf16;
    __nv_bfloat16* b_tile = shared_bf16 + TILE * TILE;

    const int block_col = bx * TILE;
    const int row = block_row + static_cast<int>(threadIdx.y);
    const int col = block_col + static_cast<int>(threadIdx.x);
    const int local = static_cast<int>(threadIdx.y) * TILE +
                      static_cast<int>(threadIdx.x);
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + static_cast<int>(threadIdx.x);
        const int b_row = tile_k + static_cast<int>(threadIdx.y);

        a_tile[local] =
            (row < M && a_col < K)
                ? A[static_cast<long long>(row) * K + a_col]
                : __float2bfloat16_rn(0.0f);
        b_tile[local] =
            (b_row < K && col < N)
                ? B[static_cast<long long>(col) * K + b_row]
                : __float2bfloat16_rn(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator +=
                __bfloat162float(
                    a_tile[static_cast<int>(threadIdx.y) * TILE + inner]) *
                __bfloat162float(
                    b_tile[inner * TILE + static_cast<int>(threadIdx.x)]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<long long>(row) * N + col] =
            __float2bfloat16_rn(accumulator);
    }
}