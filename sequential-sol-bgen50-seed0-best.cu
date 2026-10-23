#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void store_fragment_bf16(
    __nv_bfloat16* __restrict__ C, int row, int col, const Acc& f) {
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int lane = tid & 31;
    const int r = lane >> 2;
    const int c = (lane & 3) << 1;

    __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(
        C + ((long long)(row + r) << 12) + col + c);
    __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(
        C + ((long long)(row + r + 8) << 12) + col + c);
    __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(
        C + ((long long)(row + r) << 12) + col + c + 8);
    __nv_bfloat162* p3 = reinterpret_cast<__nv_bfloat162*>(
        C + ((long long)(row + r + 8) << 12) + col + c + 8);

    *p0 = __floats2bfloat162_rn(f.x[0], f.x[1]);
    *p1 = __floats2bfloat162_rn(f.x[2], f.x[3]);
    *p2 = __floats2bfloat162_rn(f.x[4], f.x[5]);
    *p3 = __floats2bfloat162_rn(f.x[6], f.x[7]);
}

extern "C" __global__ __launch_bounds__(256, 2) void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 48;
    constexpr int LD = 4096;
    constexpr int TAIL = 4080;
    constexpr int AE = BM * BK;
    constexpr int BE = BN * BK;
    constexpr int STAGE_BYTES = (AE + BE) * sizeof(__nv_bfloat16);

    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int warp = tid >> 5;
    const int warp_row = warp >> 1;
    const int warp_half = warp & 1;
    const int native_row = (int)blockIdx.y * 16;
    const int native_col = (int)blockIdx.x * 16;

    if (M == LD && N == LD && K == LD) {
        const unsigned linear_block =
            (unsigned)blockIdx.y * (unsigned)gridDim.x + (unsigned)blockIdx.x;
        if (linear_block >= 1024u) return;

        const int tile_row = (int)(linear_block >> 5);
        const int tile_col = (int)(linear_block & 31u);
        const int block_row = tile_row * BM;
        const int block_col = tile_col * BN;

        __shared__ __align__(16) unsigned char storage[49152];

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> b;
        Acc c0, c1, c2, c3, c4, c5, c6, c7;

        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        wmma::fill_fragment(c2, 0.0f);
        wmma::fill_fragment(c3, 0.0f);
        wmma::fill_fragment(c4, 0.0f);
        wmma::fill_fragment(c5, 0.0f);
        wmma::fill_fragment(c6, 0.0f);
        wmma::fill_fragment(c7, 0.0f);

        {
            __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(storage);
            __nv_bfloat16* sa = stage;
            __nv_bfloat16* sb = stage + AE;

#pragma unroll
            for (int n = 0; n < 3; ++n) {
                const int e = (tid + n * 256) * 8;
                const int r = e / BK;
                const int k = e - r * BK;
                cp16(sa + e,
                     A + ((long long)(block_row + r) << 12) + k);
            }
#pragma unroll
            for (int n = 0; n < 3; ++n) {
                const int e = (tid + n * 256) * 8;
                const int c = e / BK;
                const int k = e - c * BK;
                cp16(sb + e,
                     B + ((long long)(block_col + c) << 12) + k);
            }
            cp_commit();
            cp_wait();
            __syncthreads();
        }

        int read_stage = 0;
        for (int kb = 0; kb < TAIL; kb += BK) {
            const int next = kb + BK;
            const int write_stage = read_stage ^ 1;
            __nv_bfloat16* write = reinterpret_cast<__nv_bfloat16*>(
                storage + write_stage * STAGE_BYTES);
            __nv_bfloat16* wa = write;
            __nv_bfloat16* wb = write + AE;

            if (next < TAIL) {
#pragma unroll
                for (int n = 0; n < 3; ++n) {
                    const int e = (tid + n * 256) * 8;
                    const int r = e / BK;
                    const int k = e - r * BK;
                    cp16(wa + e,
                         A + ((long long)(block_row + r) << 12) + next + k);
                }
#pragma unroll
                for (int n = 0; n < 3; ++n) {
                    const int e = (tid + n * 256) * 8;
                    const int c = e / BK;
                    const int k = e - c * BK;
                    cp16(wb + e,
                         B + ((long long)(block_col + c) << 12) + next + k);
                }
            } else {
                const int r = tid >> 1;
                const int k = (tid & 1) << 3;
                cp16(wa + r * BK + k,
                     A + ((long long)(block_row + r) << 12) + TAIL + k);
                cp16(wb + r * BK + k,
                     B + ((long long)(block_col + r) << 12) + TAIL + k);
            }
            cp_commit();

            const __nv_bfloat16* stage =
                reinterpret_cast<const __nv_bfloat16*>(
                    storage + read_stage * STAGE_BYTES);
            const __nv_bfloat16* sa = stage;
            const __nv_bfloat16* sb = stage + AE;
            const int ar = warp_row * 32;
            const int bc = warp_half * 64;

#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::load_matrix_sync(a0, sa + ar * BK + kk, BK);
                wmma::load_matrix_sync(a1, sa + (ar + 16) * BK + kk, BK);

                wmma::load_matrix_sync(b, sb + (bc + 0) * BK + kk, BK);
                wmma::mma_sync(c0, a0, b, c0);
                wmma::mma_sync(c4, a1, b, c4);

                wmma::load_matrix_sync(b, sb + (bc + 16) * BK + kk, BK);
                wmma::mma_sync(c1, a0, b, c1);
                wmma::mma_sync(c5, a1, b, c5);

                wmma::load_matrix_sync(b, sb + (bc + 32) * BK + kk, BK);
                wmma::mma_sync(c2, a0, b, c2);
                wmma::mma_sync(c6, a1, b, c6);

                wmma::load_matrix_sync(b, sb + (bc + 48) * BK + kk, BK);
                wmma::mma_sync(c3, a0, b, c3);
                wmma::mma_sync(c7, a1, b, c7);
            }

            cp_wait();
            __syncthreads();
            read_stage = write_stage;
        }

        {
            const __nv_bfloat16* stage =
                reinterpret_cast<const __nv_bfloat16*>(
                    storage + read_stage * STAGE_BYTES);
            const __nv_bfloat16* sa = stage;
            const __nv_bfloat16* sb = stage + AE;
            const int ar = warp_row * 32;
            const int bc = warp_half * 64;

            wmma::load_matrix_sync(a0, sa + ar * BK, BK);
            wmma::load_matrix_sync(a1, sa + (ar + 16) * BK, BK);

            wmma::load_matrix_sync(b, sb + (bc + 0) * BK, BK);
            wmma::mma_sync(c0, a0, b, c0);
            wmma::mma_sync(c4, a1, b, c4);

            wmma::load_matrix_sync(b, sb + (bc + 16) * BK, BK);
            wmma::mma_sync(c1, a0, b, c1);
            wmma::mma_sync(c5, a1, b, c5);

            wmma::load_matrix_sync(b, sb + (bc + 32) * BK, BK);
            wmma::mma_sync(c2, a0, b, c2);
            wmma::mma_sync(c6, a1, b, c6);

            wmma::load_matrix_sync(b, sb + (bc + 48) * BK, BK);
            wmma::mma_sync(c3, a0, b, c3);
            wmma::mma_sync(c7, a1, b, c7);
        }

        const int out_row = block_row + warp_row * 32;
        const int out_col = block_col + warp_half * 64;

        store_fragment_bf16(C, out_row, out_col + 0, c0);
        store_fragment_bf16(C, out_row, out_col + 16, c1);
        store_fragment_bf16(C, out_row, out_col + 32, c2);
        store_fragment_bf16(C, out_row, out_col + 48, c3);
        store_fragment_bf16(C, out_row + 16, out_col + 0, c4);
        store_fragment_bf16(C, out_row + 16, out_col + 16, c5);
        store_fragment_bf16(C, out_row + 16, out_col + 32, c6);
        store_fragment_bf16(C, out_row + 16, out_col + 48, c7);
        return;
    }

    const int row = native_row + (int)threadIdx.y;
    const int col = native_col + (int)threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum = fmaf(__bfloat162float(A[(long long)row * K + k]),
                   __bfloat162float(B[(long long)col * K + k]), sum);
    }
    C[(long long)row * N + col] = __float2bfloat16(sum);
}