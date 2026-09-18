#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" int kernel_mcts_bf16_reference(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b,
        CUDA_R_16BF,
        k,
        a,
        CUDA_R_16BF,
        k,
        &beta,
        c,
        CUDA_R_16BF,
        n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const cudaError_t synchronization = cudaDeviceSynchronize();
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) return 2;
    if (synchronization != cudaSuccess) return 3;
    return 0;
}
