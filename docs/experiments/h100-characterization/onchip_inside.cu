// Explicit shared-load and register-only controls for one H100 worker.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kThreads = 256;
constexpr int kBlocks = 528;
constexpr int kLaunches = 64;
constexpr int kRounds = 15;

static void checked(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }
}

__global__ void shared_load_probe(uint32_t* output, int repeats) {
  __shared__ volatile uint32_t shared_values[kThreads];
  const int lane = threadIdx.x;
  shared_values[lane] = static_cast<uint32_t>(lane + 1);
  __syncthreads();
  uint32_t accumulated = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    accumulated += shared_values[lane];
  }
  output[blockIdx.x * blockDim.x + lane] = accumulated;
}

__global__ void register_control_probe(uint32_t* output, int repeats) {
  const int lane = threadIdx.x;
  const uint32_t seed = static_cast<uint32_t>(lane + 1);
  uint32_t accumulated = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    asm volatile("add.u32 %0, %0, %1;" : "+r"(accumulated) : "r"(seed));
  }
  output[blockIdx.x * blockDim.x + lane] = accumulated;
}

static void launch(const std::string& kind, uint32_t* output, int repeats,
                   cudaStream_t stream) {
  if (kind == "shared") {
    shared_load_probe<<<kBlocks, kThreads, 0, stream>>>(output, repeats);
  } else if (kind == "register") {
    register_control_probe<<<kBlocks, kThreads, 0, stream>>>(output, repeats);
  } else {
    throw std::invalid_argument("unknown probe kind");
  }
  checked(cudaGetLastError(), "kernel launch");
}

static void check_output(uint32_t* device, int repeats) {
  std::vector<uint32_t> host(kBlocks * kThreads);
  checked(cudaMemcpy(host.data(), device, host.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), "copy output");
  for (int block = 0; block < kBlocks; ++block) {
    for (int lane = 0; lane < kThreads; ++lane) {
      if (host[block * kThreads + lane] != static_cast<uint32_t>((lane + 1) * repeats)) {
        throw std::runtime_error("on-chip probe failed correctness");
      }
    }
  }
}

static double quantile(std::vector<double> samples, double fraction) {
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<size_t>(std::lround((samples.size() - 1) * fraction));
  return samples[index];
}

struct Result {
  std::string kind;
  int repeats;
  std::vector<double> samples;
  double median;
  double p10;
  double p90;
};

static Result benchmark(const std::string& kind, uint32_t* output, int repeats) {
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  cudaEvent_t start, end;
  checked(cudaStreamCreate(&stream), "create stream");
  checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
  for (int i = 0; i < kLaunches; ++i) launch(kind, output, repeats, stream);
  checked(cudaStreamEndCapture(stream, &graph), "end capture");
  checked(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
          "instantiate graph");
  for (int i = 0; i < 3; ++i) checked(cudaGraphLaunch(graph_exec, stream), "warm graph");
  checked(cudaStreamSynchronize(stream), "synchronize warmup");
  checked(cudaEventCreate(&start), "create start event");
  checked(cudaEventCreate(&end), "create end event");
  std::vector<double> samples;
  for (int i = 0; i < kRounds; ++i) {
    checked(cudaEventRecord(start, stream), "record start");
    checked(cudaGraphLaunch(graph_exec, stream), "replay graph");
    checked(cudaEventRecord(end, stream), "record end");
    checked(cudaEventSynchronize(end), "synchronize end");
    float milliseconds = 0;
    checked(cudaEventElapsedTime(&milliseconds, start, end), "elapsed time");
    samples.push_back(static_cast<double>(milliseconds) * 1000.0 / kLaunches);
  }
  check_output(output, repeats);
  checked(cudaEventDestroy(start), "destroy start event");
  checked(cudaEventDestroy(end), "destroy end event");
  checked(cudaGraphExecDestroy(graph_exec), "destroy graph exec");
  checked(cudaGraphDestroy(graph), "destroy graph");
  checked(cudaStreamDestroy(stream), "destroy stream");
  return {kind, repeats, samples, quantile(samples, 0.5),
          quantile(samples, 0.1), quantile(samples, 0.9)};
}

static void write_results(const std::string& path, const std::vector<Result>& results) {
  cudaDeviceProp properties;
  checked(cudaGetDeviceProperties(&properties, 0), "device properties");
  std::ofstream file(path);
  if (!file) throw std::runtime_error("could not open output file");
  file << std::fixed << std::setprecision(9);
  file << "{\n  \"schema_version\": 1,\n  \"device_name\": \"" << properties.name
       << "\",\n  \"compute_capability\": \"" << properties.major << "."
       << properties.minor << "\",\n  \"timing_method\": "
       << "\"CUDA events around CUDA Graph replay; no NCU during timing\",\n"
       << "  \"blocks\": " << kBlocks << ",\n  \"threads_per_block\": " << kThreads
       << ",\n  \"launches_per_graph_replay\": " << kLaunches
       << ",\n  \"rounds\": " << kRounds << ",\n  \"cases\": [\n";
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    const uint64_t shared_bytes = result.kind == "shared"
        ? static_cast<uint64_t>(kBlocks) * kThreads * sizeof(uint32_t) * result.repeats : 0;
    file << "    {\"kind\": \"" << result.kind << "\", \"repeats\": "
         << result.repeats << ", \"raw_us_per_launch\": [";
    for (size_t i = 0; i < result.samples.size(); ++i) {
      if (i) file << ", ";
      file << result.samples[i];
    }
    file << "], \"median_us_per_launch\": " << result.median
         << ", \"p10_us_per_launch\": " << result.p10
         << ", \"p90_us_per_launch\": " << result.p90
         << ", \"requested_shared_read_bytes_per_launch\": " << shared_bytes
         << ", \"measurement_scope\": \"on_chip_kernel_path\", "
         << "\"physical_link_attribution\": \"unverified\"}";
    if (index + 1 != results.size()) file << ",";
    file << "\n";
  }
  file << "  ]\n}\n";
}

int main(int argc, char** argv) {
  try {
    std::string mode = "bench";
    std::string kind = "shared";
    std::string output_path;
    int repeats = 256;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--mode" && i + 1 < argc) mode = argv[++i];
      else if (argument == "--kind" && i + 1 < argc) kind = argv[++i];
      else if (argument == "--repeats" && i + 1 < argc) repeats = std::stoi(argv[++i]);
      else if (argument == "--output" && i + 1 < argc) output_path = argv[++i];
      else throw std::invalid_argument("unknown or incomplete argument: " + argument);
    }
    cudaDeviceProp properties;
    checked(cudaGetDeviceProperties(&properties, 0), "device properties");
    if (properties.major != 9 || properties.minor != 0)
      throw std::runtime_error("on-chip probe requires compute capability 9.0");
    if (repeats <= 0) throw std::invalid_argument("repeats must be positive");
    uint32_t* device_output = nullptr;
    checked(cudaMalloc(reinterpret_cast<void**>(&device_output),
                       kBlocks * kThreads * sizeof(uint32_t)), "allocate output");
    if (mode == "profile") {
      for (int i = 0; i < 11; ++i) launch(kind, device_output, repeats, 0);
      checked(cudaDeviceSynchronize(), "profile synchronization");
      check_output(device_output, repeats);
      std::cout << "PROFILE_PROBE_OK\n";
    } else if (mode == "bench") {
      if (output_path.empty()) throw std::invalid_argument("--output is required");
      std::vector<Result> results;
      for (const std::string& probe : {"shared", "register"}) {
        for (const int count : {1, 16, 64, 256, 1024}) {
          results.push_back(benchmark(probe, device_output, count));
        }
      }
      write_results(output_path, results);
      std::cout << "{\"cases\": " << results.size() << "}\n";
    } else {
      throw std::invalid_argument("unknown mode");
    }
    checked(cudaFree(device_output), "free output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
