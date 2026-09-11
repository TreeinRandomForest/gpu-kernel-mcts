#include <cuda_bf16.h>

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int TILE = 16;

    __shared__ __nv_bfloat16 a_tile[TILE][TILE];
    __shared__ __nv_bfloat16 b_tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE + tx;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    const int b_k_lane = tid & (TILE - 1);
    const int b_col_lane = tid >> 4;

    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + b_k_lane;
        const int b_col = blockIdx.x * TILE + b_col_lane;

        a_tile[ty][tx] =
            (row < M && a_col < K)
                ? A[row * K + a_col]
                : __float2bfloat16(0.0f);

        b_tile[b_k_lane][b_col_lane] =
            (b_row < K && b_col < N)
                ? B[b_col * K + b_row]
                : __float2bfloat16(0.0f);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            accumulator += __bfloat162float(a_tile[ty][k]) *
                           __bfloat162float(b_tile[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(accumulator);
    }
}