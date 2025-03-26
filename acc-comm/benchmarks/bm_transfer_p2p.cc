#include <cstdint>
#include <cstdio>
#include <iostream>

#include "bm.h"

constexpr size_t problem_size{1024 * 1024 * 1024}; // Bytes

int
main(void)
{
    int device_count{-1};
    ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));

    for (int i{0}; i < device_count; ++i) {
        ERRCHK_CUDA_API(cudaSetDevice(i));

        ac::device_buffer<uint8_t> din{problem_size};
        ac::device_buffer<uint8_t> dout{problem_size};

        constexpr size_t len{128};
        char             label[len];
        snprintf(label, len, "Local D2D memcpy (device %d)", i);

        const auto ns_elapsed{
            benchmark_ns(label, []() {}, [&din, &dout]() { ac::mr::copy(din.get(), dout.get()); })};
        const auto bw{(2 * problem_size) / (1e-9 * ns_elapsed) / (1024ul * 1024 * 1024)};
        std::cout << "\tMedian bandwidth: " << bw << " GiB/s" << std::endl;
    }

    for (int i{0}; i < device_count; ++i) {
        ERRCHK_CUDA_API(cudaSetDevice(i));
        ac::device_buffer<uint8_t> din{problem_size};

        for (int j{i + 1}; j < device_count; ++j) {
            ERRCHK_CUDA_API(cudaSetDevice(j));
            ac::device_buffer<uint8_t> dout{problem_size};
            ERRCHK(din.size() == dout.size());

            ERRCHK_CUDA_API(cudaSetDevice(i));
            constexpr size_t len{128};
            char             label[len];
            snprintf(label, len, "D2D memcpy peer (device %d -> %d)", i, j);

            ERRCHK_CUDA_API(cudaDeviceEnablePeerAccess(j, 0));

            const auto ns_elapsed{benchmark_ns(
                label,
                []() {},
                [&i, &din, &j, &dout]() {
                    ERRCHK_CUDA_API(cudaMemcpyPeer(dout.data(), j, din.data(), i, din.size()));
                })};
            const auto bw{(2 * problem_size) / (1e-9 * ns_elapsed) / (1024ul * 1024 * 1024)};
            std::cout << "\tMedian bandwidth: " << bw << " GiB/s" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
