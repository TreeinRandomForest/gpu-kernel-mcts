#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

static __device__ __forceinline__ void copy_async_16(
    void* shared_dst, const void* global_src) {
    const unsigned shared_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::
                 "r"(shared_addr), "l"(global_src) : "memory");
}

static __device__ __forceinline__ void copy_async_16_zfill(
    void* shared_dst, const void* global_src, int valid_bytes) {
    const unsigned shared_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::
                 "r"(shared_addr), "l"(global_src), "r"(valid_bytes) : "memory");
}

static __device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

static __device__ __forceinline__ void async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

static __device__ __forceinline__ uint32_t pack_bf16x2(float x, float y) {
    union Pack {
        __nv_bfloat162 bf;
        uint32_t u;
    } value;
    value.bf = __floats2bfloat162_rn(x, y);
    return value.u;
}

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    constexpr int BM = 64;
    constexpr int BN = 96;
    constexpr int BK = 64;
    constexpr int LD = 72;
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WK = 16;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int groups_m = (M + BM - 1) / BM;
    const int groups_n = (N + BN - 1) / BN;
    const unsigned long long linear_block =
        static_cast<unsigned long long>(blockIdx.y) * gridDim.x + blockIdx.x;
    const unsigned long long group_count =
        static_cast<unsigned long long>(groups_m) * groups_n;

    if (linear_block >= group_count)
        return;

    const int group_m = static_cast<int>(linear_block / groups_n);
    const int group_n = static_cast<int>(linear_block -
        static_cast<unsigned long long>(group_m) * groups_n);
    const int block_row = group_m * BM;
    const int block_col = group_n * BN;

    struct __align__(16) Stage {
        __nv_bfloat16 a[BM][LD];
        __nv_bfloat16 b[BN][LD];
    };

    struct __align__(16) Panels {
        Stage stage[2];
    };

    union __align__(16) SharedStorage {
        Panels panels;
        float output[8][WM * WN];
    };

    __shared__ SharedStorage shared;

    const bool fast_path =
        M == 4096 && N == 4096 && K == 4096 &&
        ((reinterpret_cast<uintptr_t>(A) & 15u) == 0) &&
        ((reinterpret_cast<uintptr_t>(B) & 15u) == 0) &&
        ((reinterpret_cast<uintptr_t>(C) & 7u) == 0);

    if (fast_path) {
        const int warp_row = warp >> 1;
        const int warp_col_group = warp & 1;
        const int local_row = warp_row * WM;
        const int local_col0 = warp_col_group * 48;
        const int local_col1 = local_col0 + 16;
        const int local_col2 = local_col0 + 32;

        wmma::fragment<wmma::matrix_a, WM, WN, WK,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK,
                       __nv_bfloat16, wmma::col_major> b_frag0;
        wmma::fragment<wmma::matrix_b, WM, WN, WK,
                       __nv_bfloat16, wmma::col_major> b_frag1;
        wmma::fragment<wmma::matrix_b, WM, WN, WK,
                       __nv_bfloat16, wmma::col_major> b_frag2;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc0;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc1;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc2;

        wmma::fill_fragment(acc0, 0.0f);
        wmma::fill_fragment(acc1, 0.0f);
        wmma::fill_fragment(acc2, 0.0f);

        constexpr int A_VECTORS = BM * BK / 8;
        constexpr int B_VECTORS = BN * BK / 8;

#pragma unroll
        for (int vector_index = tid;
             vector_index < A_VECTORS;
             vector_index += 256) {
            const int line = vector_index >> 3;
            const int segment = (vector_index & 7) << 3;
            const __nv_bfloat16* src =
                A + static_cast<unsigned long long>(block_row + line) * K +
                segment;
            copy_async_16(&shared.panels.stage[0].a[line][segment], src);
        }

#pragma unroll
        for (int vector_index = tid;
             vector_index < B_VECTORS;
             vector_index += 256) {
            const int line = vector_index >> 3;
            const int segment = (vector_index & 7) << 3;
            const bool valid = block_col + line < N;
            const int safe_line = valid ? block_col + line : block_col;
            const __nv_bfloat16* src =
                B + static_cast<unsigned long long>(safe_line) * K + segment;
            copy_async_16_zfill(
                &shared.panels.stage[0].b[line][segment], src,
                valid ? 16 : 0);
        }

        async_commit();
        async_wait_all();
        __syncthreads();

        constexpr int PANEL_COUNT = 4096 / BK;

        for (int panel = 0; panel < PANEL_COUNT; ++panel) {
            const int stage = panel & 1;
            const int next_panel = panel + 1;

            if (next_panel < PANEL_COUNT) {
                const int next_stage = stage ^ 1;
                const int next_k = next_panel * BK;

#pragma unroll
                for (int vector_index = tid;
                     vector_index < A_VECTORS;
                     vector_index += 256) {
                    const int line = vector_index >> 3;
                    const int segment = (vector_index & 7) << 3;
                    const __nv_bfloat16* src =
                        A + static_cast<unsigned long long>(block_row + line) * K +
                        next_k + segment;
                    copy_async_16(
                        &shared.panels.stage[next_stage].a[line][segment], src);
                }

#pragma unroll
                for (int vector_index = tid;
                     vector_index < B_VECTORS;
                     vector_index += 256) {
                    const int line = vector_index >> 3;
                    const int segment = (vector_index & 7) << 3;
                    const bool valid = block_col + line < N;
                    const int safe_line = valid ? block_col + line : block_col;
                    const __nv_bfloat16* src =
                        B + static_cast<unsigned long long>(safe_line) * K +
                        next_k + segment;
                    copy_async_16_zfill(
                        &shared.panels.stage[next_stage].b[line][segment], src,
                        valid ? 16 : 0);
                }

                async_commit();
            }

#pragma unroll
            for (int kk = 0; kk < BK; kk += WK) {
                wmma::load_matrix_sync(
                    a_frag, &shared.panels.stage[stage].a[local_row][kk], LD);
                wmma::load_matrix_sync(
                    b_frag0, &shared.panels.stage[stage].b[local_col0][kk], LD);
                wmma::load_matrix_sync(
                    b_frag1, &shared.panels.stage[stage].b[local_col1][kk], LD);
                wmma::load_matrix_sync(
                    b_frag2, &shared.panels.stage[stage].b[local_col2][kk], LD);
                wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
                wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
                wmma::mma_sync(acc2, a_frag, b_frag2, acc2);
            }

            if (next_panel < PANEL_COUNT) {
                async_wait_all();
                __syncthreads();
            }
        }

#define STORE_FRAGMENT(ACC, LOCAL_COL)                                           \
        wmma::store_matrix_sync(                                                \
            shared.output[warp], ACC, WN, wmma::mem_row_major);                 \
        __syncwarp();                                                           \
        _Pragma("unroll")                                                       \
        for (int quad = lane; quad < (WM * WN) / 4; quad += 32) {              \
            const int index = quad * 4;                                         \
            const int r = index >> 4;                                           \
            const int c = index & 15;                                           \
            const int output_col = block_col + (LOCAL_COL);                     \
            if (output_col < N) {                                               \
                const unsigned long long output_row =                           \
                    static_cast<unsigned long long>(                            \
                        block_row + local_row + r) * N;                         \
                const float4 values = *reinterpret_cast<const float4*>(         \
                    &shared.output[warp][index]);                               \
                const uint2 packed = make_uint2(                                \
                    pack_bf16x2(values.x, values.y),                            \
                    pack_bf16x2(values.z, values.w));                           \
                *reinterpret_cast<uint2*>(                                      \
                    C + output_row + output_col + c) = packed;                  \
            }                                                                   \
        }

        STORE_FRAGMENT(acc0, local_col0);
        __syncwarp();
        STORE_FRAGMENT(acc1, local_col1);
        __syncwarp();
        STORE_FRAGMENT(acc2, local_col2);

#undef STORE_FRAGMENT
        return;
    }

    for (int index = tid; index < BM * BN; index += 256) {
        const int row = block_row + index / BN;
        const int col = block_col + index % BN;
        if (row < M && col < N) {
            const __nv_bfloat16* a_row =
                A + static_cast<unsigned long long>(row) * K;
            const __nv_bfloat16* b_col =
                B + static_cast<unsigned long long>(col) * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum = fmaf(__bfloat162float(a_row[k]),
                           __bfloat162float(b_col[k]), sum);
            }
            C[static_cast<unsigned long long>(row) * N + col] =
                __float2bfloat16(sum);
        }
    }
}