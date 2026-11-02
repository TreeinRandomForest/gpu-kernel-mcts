// Dependent-load latency probes. Values are observed paths, not physical links.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t kSectorStrideWords = 32;  // one pointer per 128-byte region
constexpr int kRounds = 15;
constexpr int kWarmups = 3;
constexpr int kSharedNodes = 1024;

static void checked(cudaError_t status, const char* action) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

struct Record {
  unsigned long long cycles;
  unsigned long long nanoseconds;
  uint32_t final_index;
};

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

template <bool CacheAll>
__device__ __forceinline__ void global_chase(const uint32_t* data,
                                              uint32_t first_index,
                                              uint32_t steps, Record* output) {
  if (threadIdx.x != 0) return;
  uint32_t index = first_index;
  const unsigned long long before_ns = global_nanoseconds();
  const unsigned long long before_cycles = clock64();
  for (uint32_t iteration = 0; iteration < steps; ++iteration) {
    const uint32_t* address = data + static_cast<size_t>(index) * kSectorStrideWords;
    uint32_t next;
    if constexpr (CacheAll) {
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(next) : "l"(address) : "memory");
    } else {
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(next) : "l"(address) : "memory");
    }
    index = next;
  }
  const unsigned long long after_cycles = clock64();
  const unsigned long long after_ns = global_nanoseconds();
  output[0] = {after_cycles - before_cycles, after_ns - before_ns, index};
}

__global__ void global_chase_ca(const uint32_t* data, uint32_t first_index,
                                uint32_t steps, Record* output) {
  global_chase<true>(data, first_index, steps, output);
}

__global__ void global_chase_cg(const uint32_t* data, uint32_t first_index,
                                uint32_t steps, Record* output) {
  global_chase<false>(data, first_index, steps, output);
}

__global__ void shared_chase_probe(uint32_t steps, Record* output) {
  __shared__ volatile uint32_t links[kSharedNodes];
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < kSharedNodes; ++i)
      links[i] = (5 * i + 1) & (kSharedNodes - 1);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  uint32_t index = 0;
  const unsigned long long before_ns = global_nanoseconds();
  const unsigned long long before_cycles = clock64();
  for (uint32_t iteration = 0; iteration < steps; ++iteration)
    index = links[index];
  const unsigned long long after_cycles = clock64();
  const unsigned long long after_ns = global_nanoseconds();
  output[0] = {after_cycles - before_cycles, after_ns - before_ns, index};
}

__global__ void register_chase_control(uint32_t steps, Record* output) {
  if (threadIdx.x != 0) return;
  uint32_t index = 0;
  const unsigned long long before_ns = global_nanoseconds();
  const unsigned long long before_cycles = clock64();
  for (uint32_t iteration = 0; iteration < steps; ++iteration) {
    uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 5, 1;" : "=r"(next) : "r"(index));
    index = next;
  }
  const unsigned long long after_cycles = clock64();
  const unsigned long long after_ns = global_nanoseconds();
  output[0] = {after_cycles - before_cycles, after_ns - before_ns, index};
}

struct Config {
  const char* label;
  const char* kind;
  size_t working_set_bytes;
  uint32_t steps;
};

static std::vector<Config> configs() {
  return {
      {"l1_ca_4k", "ca", 4 << 10, 8192},
      {"l2_cg_4k", "cg", 4 << 10, 8192},
      {"l2_cg_1m", "cg", 1 << 20, (1 << 20) / 128},
      {"l2_cg_8m", "cg", 8 << 20, (8 << 20) / 128},
      {"hbm_cg_128m", "cg", 128 << 20, (128 << 20) / 128},
      {"shared_4k", "shared", 4 << 10, 8192},
      {"register_8k", "register", 0, 8192},
      {"register_64k", "register", 0, (8 << 20) / 128},
      {"register_1m", "register", 0, (128 << 20) / 128},
  };
}

struct Ring {
  uint32_t* device = nullptr;
  uint32_t first_index = 0;
  uint32_t nodes = 0;
};

static Ring make_ring(size_t bytes) {
  const uint32_t nodes = static_cast<uint32_t>(bytes / 128);
  std::vector<uint32_t> order(nodes);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(bytes) ^ 0x5a17u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint32_t> host(bytes / sizeof(uint32_t), 0);
  for (uint32_t i = 0; i < nodes; ++i)
    host[static_cast<size_t>(order[i]) * kSectorStrideWords] = order[(i + 1) % nodes];
  uint32_t* device = nullptr;
  checked(cudaMalloc(reinterpret_cast<void**>(&device), bytes), "allocate ring");
  checked(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "copy ring");
  return {device, order[0], nodes};
}

static void launch(const Config& config, const Ring& ring, Record* output) {
  if (std::string(config.kind) == "ca") {
    global_chase_ca<<<1, 32>>>(ring.device, ring.first_index, config.steps, output);
  } else if (std::string(config.kind) == "cg") {
    global_chase_cg<<<1, 32>>>(ring.device, ring.first_index, config.steps, output);
  } else if (std::string(config.kind) == "shared") {
    shared_chase_probe<<<1, 32>>>(config.steps, output);
  } else if (std::string(config.kind) == "register") {
    register_chase_control<<<1, 32>>>(config.steps, output);
  } else {
    throw std::invalid_argument("unknown probe kind");
  }
  checked(cudaGetLastError(), "launch latency probe");
}

static uint32_t expected_index(const Config& config, const Ring& ring) {
  if (std::string(config.kind) == "ca" || std::string(config.kind) == "cg") {
    if (config.steps % ring.nodes != 0)
      throw std::runtime_error("global ring steps must complete full cycles");
    return ring.first_index;
  }
  if (std::string(config.kind) == "shared") return 0;
  uint32_t index = 0;
  for (uint32_t i = 0; i < config.steps; ++i) index = index * 5u + 1u;
  return index;
}

static double quantile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>(std::lround((values.size() - 1) * fraction))];
}

struct Result {
  Config config;
  std::vector<double> cycles_per_hop;
  std::vector<double> nanoseconds_per_hop;
  double median_cycles;
  double median_nanoseconds;
  double p10_nanoseconds;
  double p90_nanoseconds;
};

static Result benchmark_case(const Config& config) {
  const bool global = std::string(config.kind) == "ca" || std::string(config.kind) == "cg";
  Ring ring = global ? make_ring(config.working_set_bytes) : Ring{};
  Record* output = nullptr;
  checked(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(Record) * kRounds),
          "allocate output");
  for (int i = 0; i < kWarmups; ++i) launch(config, ring, output);
  checked(cudaDeviceSynchronize(), "synchronize warmup");
  for (int i = 0; i < kRounds; ++i) launch(config, ring, output + i);
  checked(cudaDeviceSynchronize(), "synchronize latency rounds");
  std::vector<Record> host(kRounds);
  checked(cudaMemcpy(host.data(), output, sizeof(Record) * kRounds,
                     cudaMemcpyDeviceToHost), "copy latency samples");
  std::vector<double> cycles, nanoseconds;
  const uint32_t expected = expected_index(config, ring);
  for (const Record& record : host) {
    if (record.final_index != expected || record.cycles == 0 || record.nanoseconds == 0)
      throw std::runtime_error(std::string(config.label) + ": incorrect pointer chain");
    cycles.push_back(static_cast<double>(record.cycles) / config.steps);
    nanoseconds.push_back(static_cast<double>(record.nanoseconds) / config.steps);
  }
  checked(cudaFree(output), "free output");
  if (ring.device) checked(cudaFree(ring.device), "free ring");
  return {config, cycles, nanoseconds, quantile(cycles, 0.5),
          quantile(nanoseconds, 0.5), quantile(nanoseconds, 0.1),
          quantile(nanoseconds, 0.9)};
}

static void profile_case(const Config& config) {
  const bool global = std::string(config.kind) == "ca" || std::string(config.kind) == "cg";
  Ring ring = global ? make_ring(config.working_set_bytes) : Ring{};
  Record* output = nullptr;
  checked(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(Record)), "allocate output");
  for (int i = 0; i < 11; ++i) launch(config, ring, output);
  checked(cudaDeviceSynchronize(), "synchronize profile launches");
  Record host{};
  checked(cudaMemcpy(&host, output, sizeof(Record), cudaMemcpyDeviceToHost),
          "copy profile result");
  if (host.final_index != expected_index(config, ring))
    throw std::runtime_error("profile pointer chain incorrect");
  checked(cudaFree(output), "free output");
  if (ring.device) checked(cudaFree(ring.device), "free ring");
  std::cout << "PROFILE_PROBE_OK\n";
}

static void write_results(const std::string& path, const std::vector<Result>& results) {
  cudaDeviceProp properties;
  checked(cudaGetDeviceProperties(&properties, 0), "device properties");
  std::ofstream file(path);
  if (!file) throw std::runtime_error("could not open output file");
  file << std::fixed << std::setprecision(9);
  file << "{\n  \"schema_version\": 1,\n  \"device_name\": \""
       << properties.name << "\",\n  \"compute_capability\": \""
       << properties.major << "." << properties.minor << "\",\n"
       << "  \"timer\": \"clock64 cycles and globaltimer nanoseconds inside one thread\",\n"
       << "  \"sector_stride_bytes\": 128,\n  \"rounds\": " << kRounds
       << ",\n  \"cases\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const Result& result = results[i];
    file << "    {\"label\": \"" << result.config.label << "\", \"kind\": \""
         << result.config.kind << "\", \"working_set_bytes\": "
         << result.config.working_set_bytes << ", \"steps\": " << result.config.steps
         << ", \"raw_cycles_per_hop\": [";
    for (size_t j = 0; j < result.cycles_per_hop.size(); ++j) {
      if (j) file << ", ";
      file << result.cycles_per_hop[j];
    }
    file << "], \"raw_nanoseconds_per_hop\": [";
    for (size_t j = 0; j < result.nanoseconds_per_hop.size(); ++j) {
      if (j) file << ", ";
      file << result.nanoseconds_per_hop[j];
    }
    file << "], \"median_cycles_per_hop\": " << result.median_cycles
         << ", \"median_nanoseconds_per_hop\": " << result.median_nanoseconds
         << ", \"p10_nanoseconds_per_hop\": " << result.p10_nanoseconds
         << ", \"p90_nanoseconds_per_hop\": " << result.p90_nanoseconds
         << ", \"measurement_scope\": \"dependent_read_chain\", "
         << "\"physical_link_attribution\": \"unverified\"}";
    if (i + 1 < results.size()) file << ",";
    file << "\n";
  }
  file << "  ]\n}\n";
}

int main(int argc, char** argv) {
  try {
    std::string mode = "bench";
    std::string label;
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
      else if (arg == "--label" && i + 1 < argc) label = argv[++i];
      else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
      else throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    cudaDeviceProp properties;
    checked(cudaGetDeviceProperties(&properties, 0), "device properties");
    if (properties.major != 9 || properties.minor != 0)
      throw std::runtime_error("latency probe requires compute capability 9.0");
    if (mode == "profile") {
      for (const Config& config : configs()) {
        if (label == config.label) {
          profile_case(config);
          return 0;
        }
      }
      throw std::invalid_argument("unsupported profile label");
    }
    if (mode != "bench" || output_path.empty())
      throw std::invalid_argument("bench mode requires --output");
    std::vector<Result> results;
    for (const Config& config : configs()) results.push_back(benchmark_case(config));
    write_results(output_path, results);
    std::cout << "{\"cases\": " << results.size() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
