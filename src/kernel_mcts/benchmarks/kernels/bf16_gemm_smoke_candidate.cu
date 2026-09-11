#include <cuda_bf16.h>

// Deliberately simple, correctness-first candidate for one-iteration remote
// orchestration tests. B is column-major under the fixed workload contract.
extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float accumulator = 0.0f;
    for (int k = 0; k < K; ++k) {
        accumulator += __bfloat162float(A[row * K + k]) *
                       __bfloat162float(B[col * K + k]);
    }
    C[row * N + col] = __float2bfloat16(accumulator);
}
