#include <cuda_bf16.h>
#include <mma.h>
#include <vector_types.h>

union Bfloat2Bits {
    __nv_bfloat162 value;
    unsigned int bits;
};

__device__ __forceinline__ void copy_async_cg_16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(address), "l"(src) : "memory");
#else
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ __forceinline__ void copy_async_ca_16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(address), "l"(src) : "memory");
#else
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ __forceinline__ void async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

__device__ __forceinline__ int4 pack_bf16x8(const float* values) {
    Bfloat2Bits p0, p1, p2, p3;
    p0.value = __floats2bfloat162_rn(values[0], values[1]);
    p1.value = __floats2bfloat162_rn(values[2], values[3]);
    p2.value = __floats2bfloat162_rn(values[4], values[5]);
    p3.value = __floats2bfloat162_rn(values[6], values[7]);
    return make_int4(static_cast<int>(p0.bits), static_cast<int>(p1.bits),
                     static_cast<int>(p2.bits), static_cast<int>(p3.bits));
}

extern "C" __global__ __launch_bounds__(256, 2) void bf16_gemm_root(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    namespace wmma = nvcuda::wmma;

    __align__(16) __shared__ unsigned char workspace[40960];

    if (M == 4096 && N == 4096 && K == 4096) {
        constexpr int ACTIVE_BLOCKS = 1024;
        constexpr int K_TILE = 32;
        constexpr int K_STAGES = 128;
        constexpr int AB_STRIDE = 40;
        constexpr int A_ELEMENTS = 128 * AB_STRIDE;
        constexpr int B_ELEMENTS = 128 * AB_STRIDE;
        constexpr int STAGE_ELEMENTS = A_ELEMENTS + B_ELEMENTS;
        constexpr int OUT_STRIDE = 16;
        constexpr int OUT_ELEMENTS = 256;

        const int linear_block =
            static_cast<int>(blockIdx.y) * 256 + static_cast<int>(blockIdx.x);
        if (linear_block >= ACTIVE_BLOCKS) return;

        const int logical_x = linear_block >> 5;
        const int logical_y = linear_block & 31;
        const int tid = threadIdx.y * 16 + threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;

        const int block_row = logical_y << 7;
        const int block_col = logical_x << 7;
        const int tile_row = block_row + (warp_id << 4);

        __nv_bfloat16* shared_base =
            reinterpret_cast<__nv_bfloat16*>(workspace);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc3;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc5;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc6;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc7;
        wmma::fill_fragment(acc0, 0.0f);
        wmma::fill_fragment(acc1, 0.0f);
        wmma::fill_fragment(acc2, 0.0f);
        wmma::fill_fragment(acc3, 0.0f);
        wmma::fill_fragment(acc4, 0.0f);
        wmma::fill_fragment(acc5, 0.0f);
        wmma::fill_fragment(acc6, 0.0f);
        wmma::fill_fragment(acc7, 0.0f);

        {
            __nv_bfloat16* stage_a = shared_base;
            __nv_bfloat16* stage_b = stage_a + A_ELEMENTS;

#pragma unroll
            for (int offset = 0; offset < 512; offset += 256) {
                const int unit = tid + offset;
                const int row_or_col = unit >> 2;
                const int vector_index = unit & 3;

                copy_async_cg_16(
                    stage_a + row_or_col * AB_STRIDE + (vector_index << 3),
                    A + ((block_row + row_or_col) << 12) +
                        (vector_index << 3));
                copy_async_ca_16(
                    stage_b + row_or_col * AB_STRIDE + (vector_index << 3),
                    B + ((block_col + row_or_col) << 12) +
                        (vector_index << 3));
            }
            async_commit();
            async_wait_all();
            __syncthreads();
        }

#pragma unroll 1
        for (int stage_index = 0; stage_index < K_STAGES; ++stage_index) {
            const int current_buffer = stage_index & 1;
            const int next_buffer = current_buffer ^ 1;
            __nv_bfloat16* current_a =
                shared_base + current_buffer * STAGE_ELEMENTS;
            __nv_bfloat16* current_b = current_a + A_ELEMENTS;

            if (stage_index != K_STAGES - 1) {
                __nv_bfloat16* next_a =
                    shared_base + next_buffer * STAGE_ELEMENTS;
                __nv_bfloat16* next_b = next_a + A_ELEMENTS;
                const int next_k = (stage_index + 1) << 5;

#pragma unroll
                for (int offset = 0; offset < 512; offset += 256) {
                    const int unit = tid + offset;
                    const int row_or_col = unit >> 2;
                    const int vector_index = unit & 3;

                    copy_async_cg_16(
                        next_a + row_or_col * AB_STRIDE +
                            (vector_index << 3),
                        A + ((block_row + row_or_col) << 12) + next_k +
                            (vector_index << 3));
                    copy_async_ca_16(
                        next_b + row_or_col * AB_STRIDE +
                            (vector_index << 3),
                        B + ((block_col + row_or_col) << 12) + next_k +
                            (vector_index << 3));
                }
                async_commit();
            }

#pragma unroll
            for (int inner_k = 0; inner_k < K_TILE; inner_k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, wmma::col_major> b_frag;

                const __nv_bfloat16* a_ptr =
                    current_a + warp_id * 16 * AB_STRIDE + inner_k;
                const __nv_bfloat16* b_ptr = current_b + inner_k;

                wmma::load_matrix_sync(a_frag, a_ptr, AB_STRIDE);

                wmma::load_matrix_sync(b_frag, b_ptr, AB_STRIDE);
                wmma::mma_sync(acc0, a_frag, b_frag, acc0);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 16 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc1, a_frag, b_frag, acc1);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 32 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc2, a_frag, b_frag, acc2);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 48 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc3, a_frag, b_frag, acc3);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 64 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc4, a_frag, b_frag, acc4);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 80 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc5, a_frag, b_frag, acc5);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 96 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc6, a_frag, b_frag, acc6);
                wmma::load_matrix_sync(b_frag,
                    b_ptr + 112 * AB_STRIDE, AB_STRIDE);
                wmma::mma_sync(acc7, a_frag, b_frag, acc7);
            }

            async_wait_all();
            __syncthreads();
        }

        float* shared_out = reinterpret_cast<float*>(workspace);
        float* out0 = shared_out + (warp_id * 4 + 0) * OUT_ELEMENTS;
        float* out1 = shared_out + (warp_id * 4 + 1) * OUT_ELEMENTS;
        float* out2 = shared_out + (warp_id * 4 + 2) * OUT_ELEMENTS;
        float* out3 = shared_out + (warp_id * 4 + 3) * OUT_ELEMENTS;

        wmma::store_matrix_sync(out0, acc0, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out1, acc1, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out2, acc2, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out3, acc3, OUT_STRIDE,
                                wmma::mem_row_major);
        __syncwarp();

        const int local_row = lane_id >> 1;
        const int half_col = (lane_id & 1) << 3;
        __nv_bfloat16* output = C + ((tile_row + local_row) << 12) + block_col;

        *reinterpret_cast<int4*>(output + half_col) =
            pack_bf16x8(out0 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 16 + half_col) =
            pack_bf16x8(out1 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 32 + half_col) =
            pack_bf16x8(out2 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 48 + half_col) =
            pack_bf16x8(out3 + local_row * OUT_STRIDE + half_col);

        wmma::store_matrix_sync(out0, acc4, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out1, acc5, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out2, acc6, OUT_STRIDE,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(out3, acc7, OUT_STRIDE,
                                wmma::mem_row_major);
        __syncwarp();

        *reinterpret_cast<int4*>(output + 64 + half_col) =
            pack_bf16x8(out0 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 80 + half_col) =
            pack_bf16x8(out1 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 96 + half_col) =
            pack_bf16x8(out2 + local_row * OUT_STRIDE + half_col);
        *reinterpret_cast<int4*>(output + 112 + half_col) =
            pack_bf16x8(out3 + local_row * OUT_STRIDE + half_col);
        return;
    }

    constexpr int TILE = 16;
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* b_tile = a_tile + TILE * TILE;

    const int row = static_cast<int>(blockIdx.y) * TILE + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * TILE + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        const int index = threadIdx.y * TILE + threadIdx.x;

        a_tile[index] =
            (row < M && a_col < K) ? A[row * K + a_col]
                                   : __float2bfloat16(0.0f);
        b_tile[index] =
            (b_row < K && col < N) ? B[col * K + b_row]
                                   : __float2bfloat16(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator +=
                __bfloat162float(a_tile[threadIdx.y * TILE + inner]) *
                __bfloat162float(b_tile[inner * TILE + threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(accumulator);
    }
}