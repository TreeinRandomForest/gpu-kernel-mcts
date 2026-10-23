#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    constexpr int TILE = 16;
    constexpr int WARPS = 8;

    __shared__ __align__(16) float output_tiles[WARPS][2][TILE * TILE];
    __shared__ __nv_bfloat16 fallback_a[TILE][TILE];
    __shared__ __nv_bfloat16 fallback_b[TILE][TILE];

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int warp = tid >> 5;

    const bool grouped_shape =
        ((M & 127) == 0) &&
        ((N & 31) == 0) &&
        ((K & 15) == 0);

    if (grouped_shape) {
        if ((blockIdx.x & 1u) != 0u || (blockIdx.y & 7u) != 0u) {
            return;
        }

        const int block_row = static_cast<int>(blockIdx.y) * TILE;
        const int block_col = static_cast<int>(blockIdx.x) * TILE;
        const int warp_row = block_row + warp * TILE;

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> b1_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;

        wmma::fill_fragment(acc0, 0.0f);
        wmma::fill_fragment(acc1, 0.0f);

        for (int k0 = 0; k0 < K; k0 += TILE) {
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k0, K);
            wmma::load_matrix_sync(b0_frag, B + block_col * K + k0, K);
            wmma::load_matrix_sync(b1_frag,
                                   B + (block_col + TILE) * K + k0, K);
            wmma::mma_sync(acc0, a_frag, b0_frag, acc0);
            wmma::mma_sync(acc1, a_frag, b1_frag, acc1);
        }

        wmma::store_matrix_sync(output_tiles[warp][0], acc0, TILE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(output_tiles[warp][1], acc1, TILE,
                                wmma::mem_row_major);
        __syncthreads();

#pragma unroll
        for (int q = 0; q < 16; ++q) {
            const int linear = tid + q * 256;
            const int owner_warp = linear >> 9;
            const int within_warp = linear & 511;
            const int column_tile = within_warp >> 8;
            const int element = within_warp & 255;
            const int local_row = element >> 4;
            const int local_col = element & 15;

            const float value =
                output_tiles[owner_warp][column_tile][element];
            const int row = block_row + owner_warp * TILE + local_row;
            const int col = block_col + column_tile * TILE + local_col;
            C[row * N + col] = __float2bfloat16_rn(value);
        }
        return;
    }

    const int tile_row = static_cast<int>(blockIdx.y) * TILE;
    const int tile_col = static_cast<int>(blockIdx.x) * TILE;
    const int row = tile_row + threadIdx.y;
    const int col = tile_col + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;

        fallback_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K)
                ? A[row * K + a_col]
                : __float2bfloat16_rn(0.0f);
        fallback_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? B[col * K + b_row]
                : __float2bfloat16_rn(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator +=
                __bfloat162float(fallback_a[threadIdx.y][inner]) *
                __bfloat162float(fallback_b[inner][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16_rn(accumulator);
    }
}