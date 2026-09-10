#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

extern "C" __global__ void bf16_gemm_root(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K);

static void infrastructure_failure(const char* operation) {
    std::printf("{\"status\":\"infrastructure_failure\",\"operation\":\"%s\"}\n", operation);
    std::exit(2);
}

static void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) infrastructure_failure(operation);
}

static void require_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) infrastructure_failure(operation);
}

static int int_arg(int argc, char** argv, const char* name, int fallback) {
    const std::string prefix = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        std::string value(argv[i]);
        if (value.rfind(prefix, 0) == 0) return std::stoi(value.substr(prefix.size()));
    }
    return fallback;
}

static float float_arg(int argc, char** argv, const char* name, float fallback) {
    const std::string prefix = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        std::string value(argv[i]);
        if (value.rfind(prefix, 0) == 0) return std::stof(value.substr(prefix.size()));
    }
    return fallback;
}

static std::string string_arg(int argc, char** argv, const char* name) {
    const std::string prefix = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        std::string value(argv[i]);
        if (value.rfind(prefix, 0) == 0) return value.substr(prefix.size());
    }
    return "";
}

static bool launch_candidate(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    bf16_gemm_root<<<grid, block>>>(A, B, C, M, N, K);
    return cudaGetLastError() == cudaSuccess;
}

int main(int argc, char** argv) {
    const std::string mode = string_arg(argc, argv, "mode");
    const int M = int_arg(argc, argv, "M", 0);
    const int N = int_arg(argc, argv, "N", 0);
    const int K = int_arg(argc, argv, "K", 0);
    const int seed = int_arg(argc, argv, "seed", 0);
    const int warmups = int_arg(argc, argv, "warmups", 10);
    const int measurements = int_arg(argc, argv, "measurements", 30);
    const float rtol = float_arg(argc, argv, "rtol", 0.02f);
    const float atol = float_arg(argc, argv, "atol", 0.02f);
    if ((mode != "correctness" && mode != "benchmark") || M <= 0 || N <= 0 || K <= 0) {
        infrastructure_failure("arguments");
    }

    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    std::vector<__nv_bfloat16> hA(a_count), hB(b_count), hC(c_count), hReference(c_count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (auto& value : hA) value = __float2bfloat16(distribution(generator));
    for (auto& value : hB) value = __float2bfloat16(distribution(generator));

    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dReference = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), a_count * sizeof(*dA)), "cudaMalloc_A");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&dB), b_count * sizeof(*dB)), "cudaMalloc_B");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&dC), c_count * sizeof(*dC)), "cudaMalloc_C");
    require_cuda(cudaMemcpy(dA, hA.data(), a_count * sizeof(*dA), cudaMemcpyHostToDevice), "copy_A");
    require_cuda(cudaMemcpy(dB, hB.data(), b_count * sizeof(*dB), cudaMemcpyHostToDevice), "copy_B");

    if (!launch_candidate(dA, dB, dC, M, N, K)) {
        std::printf("{\"status\":\"launch_failure\"}\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("{\"status\":\"launch_failure\"}\n");
        return 1;
    }

    if (mode == "correctness") {
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&dReference), c_count * sizeof(*dReference)),
            "cudaMalloc_reference");
        cublasHandle_t handle;
        require_cublas(cublasCreate(&handle), "cublasCreate");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major C is viewed as column-major C^T = B^T * A^T.
        require_cublas(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                dB,
                CUDA_R_16BF,
                K,
                dA,
                CUDA_R_16BF,
                K,
                &beta,
                dReference,
                CUDA_R_16BF,
                N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
        require_cuda(cudaDeviceSynchronize(), "reference_synchronize");
        require_cuda(cudaMemcpy(hC.data(), dC, c_count * sizeof(*dC), cudaMemcpyDeviceToHost), "copy_C");
        require_cuda(
            cudaMemcpy(
                hReference.data(),
                dReference,
                c_count * sizeof(*dReference),
                cudaMemcpyDeviceToHost),
            "copy_reference");

        double maximum_error = 0.0;
        double error_sum = 0.0;
        bool success = true;
        for (size_t index = 0; index < c_count; ++index) {
            const double actual = __bfloat162float(hC[index]);
            const double reference = __bfloat162float(hReference[index]);
            const double error = std::abs(actual - reference);
            maximum_error = std::max(maximum_error, error);
            error_sum += error;
            if (!std::isfinite(actual) || error > atol + rtol * std::abs(reference)) success = false;
        }
        std::printf(
            "{\"status\":\"ok\",\"success\":%s,\"maximum_error\":%.9g,"
            "\"mean_error\":%.9g,\"failed_test_id\":%s,"
            "\"reference_metadata\":{\"implementation\":\"cuBLAS\","
            "\"compute_type\":\"CUBLAS_COMPUTE_32F\",\"seed\":%d}}\n",
            success ? "true" : "false",
            maximum_error,
            error_sum / c_count,
            success ? "null" : "\"fixed_shape\"",
            seed);
        cublasDestroy(handle);
        cudaFree(dReference);
    } else {
        for (int iteration = 0; iteration < warmups; ++iteration) {
            if (!launch_candidate(dA, dB, dC, M, N, K)) {
                std::printf("{\"status\":\"launch_failure\"}\n");
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("{\"status\":\"launch_failure\"}\n");
            return 1;
        }
        cudaEvent_t start, stop;
        require_cuda(cudaEventCreate(&start), "event_start");
        require_cuda(cudaEventCreate(&stop), "event_stop");
        std::vector<float> timings(measurements);
        for (int iteration = 0; iteration < measurements; ++iteration) {
            require_cuda(cudaEventRecord(start), "record_start");
            if (!launch_candidate(dA, dB, dC, M, N, K)) {
                std::printf("{\"status\":\"launch_failure\"}\n");
                return 1;
            }
            require_cuda(cudaEventRecord(stop), "record_stop");
            if (cudaEventSynchronize(stop) != cudaSuccess) {
                std::printf("{\"status\":\"launch_failure\"}\n");
                return 1;
            }
            float milliseconds = 0.0f;
            require_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "event_elapsed");
            timings[iteration] = milliseconds * 1000.0f;
        }
        std::printf("{\"status\":\"ok\",\"timings_us\":[");
        for (int index = 0; index < measurements; ++index) {
            if (index) std::printf(",");
            std::printf("%.9g", timings[index]);
        }
        std::printf("]}\n");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
