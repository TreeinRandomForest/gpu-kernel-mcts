#include <cuda_bf16.h>

#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

static void write_values(
    const char* path,
    std::size_t count,
    std::mt19937& generator,
    std::uniform_real_distribution<float>& distribution) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output file");
    for (std::size_t index = 0; index < count; ++index) {
        const __nv_bfloat16 value = __float2bfloat16(distribution(generator));
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    if (!output) throw std::runtime_error("cannot write output file");
}

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    try {
        const std::size_t m = std::stoull(argv[1]);
        const std::size_t n = std::stoull(argv[2]);
        const std::size_t k = std::stoull(argv[3]);
        const unsigned int seed = static_cast<unsigned int>(std::stoul(argv[4]));
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
        write_values(argv[5], m * k, generator, distribution);
        write_values(argv[6], k * n, generator, distribution);
    } catch (...) {
        return 1;
    }
    return 0;
}
