#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Element = cutlass::bfloat16_t;
using CutlassGemm = cutlass::gemm::device::Gemm<
    Element,
    cutlass::layout::RowMajor,
    Element,
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>>;

static void fail(const char* operation) {
    std::printf("{\"status\":\"infrastructure_failure\",\"operation\":\"%s\"}\n", operation);
    std::exit(2);
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

static void launch_cublas(
    cublasHandle_t handle,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
            B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta,
            C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
        fail("cublasGemmEx");
    }
}

static void launch_cutlass(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    CutlassGemm operation;
    CutlassGemm::Arguments arguments(
        {M, N, K},
        {reinterpret_cast<const Element*>(A), K},
        {reinterpret_cast<const Element*>(B), K},
        {reinterpret_cast<const Element*>(C), N},
        {reinterpret_cast<Element*>(C), N},
        {1.0f, 0.0f});
    if (operation(arguments, nullptr, stream) != cutlass::Status::kSuccess)
        fail("cutlass_gemm");
}

int main(int argc, char** argv) {
    const std::string implementation = string_arg(argc, argv, "implementation");
    const int M = int_arg(argc, argv, "M", 0);
    const int N = int_arg(argc, argv, "N", 0);
    const int K = int_arg(argc, argv, "K", 0);
    const int seed = int_arg(argc, argv, "seed", 0);
    const int warmups = int_arg(argc, argv, "warmups", 10);
    const int measurements = int_arg(argc, argv, "measurements", 30);
    const std::string timing_contract = string_arg(argc, argv, "timing-contract");
    const float rtol = float_arg(argc, argv, "rtol", 0.02f);
    const float atol = float_arg(argc, argv, "atol", 0.02f);
    if ((implementation != "cublas" && implementation != "cutlass") ||
        M <= 0 || N <= 0 || K <= 0 || measurements <= 0 ||
        (timing_contract != "cuda_event_per_call_v1" &&
         timing_contract != "triton_h100_cuda_graph_v1")) fail("arguments");

    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    std::vector<__nv_bfloat16> hA(a_count), hB(b_count), hC(c_count), hReference(c_count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (auto& value : hA) value = __float2bfloat16(distribution(generator));
    for (auto& value : hB) value = __float2bfloat16(distribution(generator));

    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dReference = nullptr;
    if (cudaMalloc(&dA, a_count * sizeof(*dA)) != cudaSuccess ||
        cudaMalloc(&dB, b_count * sizeof(*dB)) != cudaSuccess ||
        cudaMalloc(&dC, c_count * sizeof(*dC)) != cudaSuccess ||
        cudaMalloc(&dReference, c_count * sizeof(*dReference)) != cudaSuccess) fail("cudaMalloc");
    if (cudaMemcpy(dA, hA.data(), a_count * sizeof(*dA), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, hB.data(), b_count * sizeof(*dB), cudaMemcpyHostToDevice) != cudaSuccess) fail("cudaMemcpy");

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) fail("cublasCreate");
    cudaStream_t launch_stream = nullptr;
    auto launch = [&]() {
        if (implementation == "cublas") launch_cublas(handle, dA, dB, dC, M, N, K);
        else launch_cutlass(dA, dB, dC, M, N, K, launch_stream);
    };

    launch();
    if (cudaDeviceSynchronize() != cudaSuccess) fail("initial_launch");
    launch_cublas(handle, dA, dB, dReference, M, N, K);
    if (cudaDeviceSynchronize() != cudaSuccess) fail("reference_launch");
    if (cudaMemcpy(hC.data(), dC, c_count * sizeof(*dC), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(hReference.data(), dReference, c_count * sizeof(*dReference), cudaMemcpyDeviceToHost) != cudaSuccess) fail("copy_results");

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

    const bool graph_mode = timing_contract == "triton_h100_cuda_graph_v1";
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    if (graph_mode) {
        if (cudaStreamCreateWithFlags(&launch_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cublasSetStream(handle, launch_stream) != CUBLAS_STATUS_SUCCESS)
            fail("graph_stream_setup");
        launch();
        if (cudaStreamSynchronize(launch_stream) != cudaSuccess)
            fail("graph_stream_warmup");
        if (cudaStreamBeginCapture(launch_stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            fail("graph_begin_capture");
        launch();
        if (cudaStreamEndCapture(launch_stream, &graph) != cudaSuccess || graph == nullptr)
            fail("graph_end_capture");
        if (cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) != cudaSuccess)
            fail("graph_instantiate");
        if (cudaMemset(dC, 0, c_count * sizeof(*dC)) != cudaSuccess ||
            cudaGraphLaunch(graph_exec, launch_stream) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(hC.data(), dC, c_count * sizeof(*dC), cudaMemcpyDeviceToHost) != cudaSuccess)
            fail("graph_correctness_replay");
        double graph_error_sum = 0.0;
        for (size_t index = 0; index < c_count; ++index) {
            const double actual = __bfloat162float(hC[index]);
            const double reference = __bfloat162float(hReference[index]);
            const double error = std::abs(actual - reference);
            maximum_error = std::max(maximum_error, error);
            graph_error_sum += error;
            if (!std::isfinite(actual) || error > atol + rtol * std::abs(reference))
                success = false;
        }
        error_sum = std::max(error_sum, graph_error_sum);
    }
    auto timed_launch = [&]() {
        if (graph_mode) {
            if (cudaGraphLaunch(graph_exec, launch_stream) != cudaSuccess) fail("graph_launch");
        } else launch();
    };
    for (int iteration = 0; iteration < warmups; ++iteration) timed_launch();
    if (cudaDeviceSynchronize() != cudaSuccess) fail("warmup");
    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) fail("events");
    std::vector<float> timings(measurements);
    for (int iteration = 0; iteration < measurements; ++iteration) {
        if (cudaEventRecord(start, launch_stream) != cudaSuccess) fail("record_start");
        timed_launch();
        if (cudaEventRecord(stop, launch_stream) != cudaSuccess ||
            cudaEventSynchronize(stop) != cudaSuccess) fail("record_stop");
        float milliseconds = 0.0f;
        if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess) fail("elapsed");
        timings[iteration] = milliseconds * 1000.0f;
    }
    std::vector<float> ordinary_timings;
    if (graph_mode) {
        ordinary_timings.resize(measurements);
        for (int iteration = 0; iteration < measurements; ++iteration) {
            if (cudaEventRecord(start, launch_stream) != cudaSuccess)
                fail("ordinary_record_start");
            launch();
            if (cudaEventRecord(stop, launch_stream) != cudaSuccess ||
                cudaEventSynchronize(stop) != cudaSuccess) fail("ordinary_record_stop");
            float milliseconds = 0.0f;
            if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess)
                fail("ordinary_elapsed");
            ordinary_timings[iteration] = milliseconds * 1000.0f;
        }
    }

    std::printf(
        "{\"status\":\"ok\",\"success\":%s,\"maximum_error\":%.9g,"
        "\"mean_error\":%.9g,\"timings_us\":[",
        success ? "true" : "false", maximum_error, error_sum / c_count);
    for (int index = 0; index < measurements; ++index) {
        if (index) std::printf(",");
        std::printf("%.9g", timings[index]);
    }
    std::printf("],\"timing_contract\":\"%s\",\"ordinary_launch_timings_us\":[",
                timing_contract.c_str());
    for (int index = 0; index < static_cast<int>(ordinary_timings.size()); ++index) {
        if (index) std::printf(",");
        std::printf("%.9g", ordinary_timings[index]);
    }
    std::printf("]}\n");

    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (launch_stream != nullptr) cudaStreamDestroy(launch_stream);
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dReference);
    return success ? 0 : 1;
}
