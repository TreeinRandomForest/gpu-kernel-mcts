#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

__device__ __forceinline__ void cp_async_16_ca(
    void* shared_dst,
    const void* global_src) {
    const unsigned int shared_addr =
        static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(shared_addr), "l"(global_src)
        : "memory");
}

extern "C" __global__ __launch_bounds__(256, 2) void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    namespace wmma = nvcuda::wmma;

    constexpr int TILE = 16;
    constexpr int MACRO = 128;
    constexpr int K_STAGE = 64;
    constexpr int SMEM_LD = 72;
    constexpr int PANEL_ELEMENTS = MACRO * SMEM_LD;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * TILE + tx;

    if (M != 4096 || N != 4096 || K != 4096) {
        const int row = static_cast<int>(blockIdx.y) * TILE + ty;
        const int col = static_cast<int>(blockIdx.x) * TILE + tx;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __bfloat162float(A[row * K + k]) *
                       __bfloat162float(B[col * K + k]);
            }
            C[row * N + col] = __float2bfloat16(sum);
        }
        return;
    }

    const unsigned int linear_block =
        static_cast<unsigned int>(blockIdx.y) *
            static_cast<unsigned int>(gridDim.x) +
        static_cast<unsigned int>(blockIdx.x);

    if (linear_block >= 1024u) {
        return;
    }

    const int macro_x = static_cast<int>(linear_block & 31u);
    const int macro_y = static_cast<int>(linear_block >> 5);
    const int row0 = macro_y * MACRO;
    const int col0 = macro_x * MACRO;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(32) unsigned char shared_storage[
        2 * PANEL_ELEMENTS * sizeof(__nv_bfloat16)];

    __nv_bfloat16* smem_a =
        reinterpret_cast<__nv_bfloat16*>(shared_storage);
    __nv_bfloat16* smem_b = smem_a + PANEL_ELEMENTS;

    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[8];

#pragma unroll
    for (int n = 0; n < 8; ++n) {
        wmma::fill_fragment(accum[n], 0.0f);
    }

    const int copy_row = tid >> 3;
    const int copy_element = (tid & 7) << 3;

    __nv_bfloat16* sa0 = smem_a + copy_row * SMEM_LD + copy_element;
    __nv_bfloat16* sa1 = sa0 + 32 * SMEM_LD;
    __nv_bfloat16* sa2 = sa1 + 32 * SMEM_LD;
    __nv_bfloat16* sa3 = sa2 + 32 * SMEM_LD;

    __nv_bfloat16* sb0 = smem_b + copy_row * SMEM_LD + copy_element;
    __nv_bfloat16* sb1 = sb0 + 32 * SMEM_LD;
    __nv_bfloat16* sb2 = sb1 + 32 * SMEM_LD;
    __nv_bfloat16* sb3 = sb2 + 32 * SMEM_LD;

    const __nv_bfloat16* ga0 =
        A + (row0 + copy_row) * 4096 + copy_element;
    const __nv_bfloat16* ga1 = ga0 + 32 * 4096;
    const __nv_bfloat16* ga2 = ga1 + 32 * 4096;
    const __nv_bfloat16* ga3 = ga2 + 32 * 4096;

    const __nv_bfloat16* gb0 =
        B + (col0 + copy_row) * 4096 + copy_element;
    const __nv_bfloat16* gb1 = gb0 + 32 * 4096;
    const __nv_bfloat16* gb2 = gb1 + 32 * 4096;
    const __nv_bfloat16* gb3 = gb2 + 32 * 4096;

    for (int k0 = 0; k0 < 4096; k0 += K_STAGE) {
        cp_async_16_ca(sa0, ga0 + k0);
        cp_async_16_ca(sa1, ga1 + k0);
        cp_async_16_ca(sa2, ga2 + k0);
        cp_async_16_ca(sa3, ga3 + k0);

        cp_async_16_ca(sb0, gb0 + k0);
        cp_async_16_ca(sb1, gb1 + k0);
        cp_async_16_ca(sb2, gb2 + k0);
        cp_async_16_ca(sb3, gb3 + k0);

        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < K_STAGE; ks += TILE) {
            wmma::load_matrix_sync(
                a_frag,
                smem_a + warp_id * TILE * SMEM_LD + ks,
                SMEM_LD);

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::load_matrix_sync(
                    b_frag,
                    smem_b + n * TILE * SMEM_LD + ks,
                    SMEM_LD);
                wmma::mma_sync(
                    accum[n], a_frag, b_frag, accum[n]);
            }
        }

        if (k0 + K_STAGE < 4096) {
            __syncthreads();
        }
    }

    float* smem_out = reinterpret_cast<float*>(shared_storage);
    float* warp_out = smem_out + warp_id * TILE * TILE;

#pragma unroll
    for (int n = 0; n < 8; ++n) {
        wmma::store_matrix_sync(
            warp_out, accum[n], TILE, wmma::mem_row_major);
        __syncwarp();

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int element = lane + i * 32;
            const int local_row = element >> 4;
            const int local_col = element & 15;

            C[(row0 + warp_id * TILE + local_row) * 4096 +
              col0 + n * TILE + local_col] =
                __float2bfloat16(warp_out[element]);
        }

        if (n != 7) {
            __syncwarp();
        }
    }
}