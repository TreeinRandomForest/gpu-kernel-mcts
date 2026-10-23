#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    __shared__ __align__(32) __nv_bfloat16 a_stage[64][16];
    __shared__ __align__(32) __nv_bfloat16 b_stage[64][16];
    __shared__ __align__(32) float output_stage[8][2][16 * 16];

    const int tid = threadIdx.y * 16 + threadIdx.x;

    if (M == 4096 && N == 4096 && K == 4096) {
        // One owner block covers a 4x4 group of logical 16x16 blocks.
        if ((blockIdx.x & 3u) != 0u || (blockIdx.y & 3u) != 0u) {
            return;
        }

        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int warp_tile_row = warp_id >> 1;
        const int warp_tile_col_pair = warp_id & 1;
        const int block_row = static_cast<int>(blockIdx.y) * 16;
        const int block_col = static_cast<int>(blockIdx.x) * 16;
        const int output_row = block_row + warp_tile_row * 16;
        const int output_col = block_col + warp_tile_col_pair * 32;

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;

        wmma::fill_fragment(acc0, 0.0f);
        wmma::fill_fragment(acc1, 0.0f);

#pragma unroll 1
        for (int k0 = 0; k0 < 4096; k0 += 16) {
            // Each thread moves one aligned 16-byte vector. Threads 0..127
            // stage A and threads 128..255 stage B.
            if (tid < 128) {
                const int row = tid >> 1;
                const int half = tid & 1;
                const uint4* src = reinterpret_cast<const uint4*>(
                    A + static_cast<long long>(block_row + row) * 4096 +
                    k0 + half * 8);
                uint4* dst = reinterpret_cast<uint4*>(&a_stage[row][half * 8]);
                *dst = *src;
            } else {
                const int local = tid - 128;
                const int col = local >> 1;
                const int half = local & 1;
                const uint4* src = reinterpret_cast<const uint4*>(
                    B + static_cast<long long>(block_col + col) * 4096 +
                    k0 + half * 8);
                uint4* dst = reinterpret_cast<uint4*>(&b_stage[col][half * 8]);
                *dst = *src;
            }
            __syncthreads();

            wmma::load_matrix_sync(
                a_frag, &a_stage[warp_tile_row * 16][0], 16);

            wmma::load_matrix_sync(
                b_frag, &b_stage[warp_tile_col_pair * 32][0], 16);
            wmma::mma_sync(acc0, a_frag, b_frag, acc0);

            wmma::load_matrix_sync(
                b_frag, &b_stage[warp_tile_col_pair * 32 + 16][0], 16);
            wmma::mma_sync(acc1, a_frag, b_frag, acc1);

            __syncthreads();
        }

        wmma::store_matrix_sync(
            &output_stage[warp_id][0][0], acc0, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(
            &output_stage[warp_id][1][0], acc1, 16, wmma::mem_row_major);
        __syncwarp();

#pragma unroll
        for (int index = lane_id; index < 256; index += 32) {
            const int local_row = index >> 4;
            const int local_col = index & 15;
            C[static_cast<long long>(output_row + local_row) * 4096 +
              output_col + local_col] =
                __float2bfloat16(output_stage[warp_id][0][index]);
            C[static_cast<long long>(output_row + local_row) * 4096 +
              output_col + 16 + local_col] =
                __float2bfloat16(output_stage[warp_id][1][index]);
        }
        return;
    }

    // General-shape correctness fallback using the original logical tiling.
    const int row = static_cast<int>(blockIdx.y) * 16 + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * 16 + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += 16) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;

        a_stage[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K)
                ? A[static_cast<long long>(row) * K + a_col]
                : __float2bfloat16(0.0f);
        b_stage[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? B[static_cast<long long>(col) * K + b_row]
                : __float2bfloat16(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            accumulator +=
                __bfloat162float(a_stage[threadIdx.y][inner]) *
                __bfloat162float(b_stage[threadIdx.x][inner]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<long long>(row) * N + col] =
            __float2bfloat16(accumulator);
    }
}