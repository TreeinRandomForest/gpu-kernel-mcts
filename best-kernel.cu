#include <cuda_bf16.h>

// Correctness-first baseline for the fixed benchmark contract:
//   A: [M, K], row-major BF16
//   B: [K, N], column-major BF16
//   C: [M, N], row-major BF16
// Launch with block=(16, 16, 1) and grid=(ceil(N/16), ceil(M/16), 1).
extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    constexpr int TILE = 16;
    __shared__ __nv_bfloat16 a_tile[TILE][TILE];
    __shared__ __nv_bfloat16 b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            row < M && a_col < K ? A[row * K + a_col] : __float2bfloat16(0.0f);
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < K && col < N ? B[col * K + b_row] : __float2bfloat16(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator += __bfloat162float(a_tile[threadIdx.y][inner]) *
                           __bfloat162float(b_tile[inner][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(accumulator);
    }
}
