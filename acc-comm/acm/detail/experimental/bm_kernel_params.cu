#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define ACM_HIP_ENABLED
#define ACM_DEVICE_ENABLED
#include "acm/detail/cuda_utils.h"
#include "acm/detail/timer.h"

template <size_t N> struct Array {
    uint8_t* data[N];
};

template <size_t N>
__global__ void
fn(const Array<N> input, Array<N> output)
{
    output.data[0] = input.data[0];
}

template <typename Duration>
static auto
to_ns(const Duration& d)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

int
main()
{
    // Setup
    const size_t                                     nsamples{10000};
    std::vector<std::chrono::steady_clock::duration> measurements;
    ac::timer                                        timer;

    // Memory
    uint8_t* data;
    cudaMalloc(&data, sizeof(data[0]));

    // IO
    std::ofstream os{"bm-kernel-params.csv"};
    os << "bytes, ns" << std::endl;

    // Benchmark
    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<1> input{data};
            Array<1> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 1 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<2> input{data};
            Array<2> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 2 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<4> input{data};
            Array<4> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 4 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<8> input{data};
            Array<8> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 8 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<16> input{data};
            Array<16> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 16 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<32> input{data};
            Array<32> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 32 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<64> input{data};
            Array<64> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 64 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<128> input{data};
            Array<128> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 128 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<256> input{data};
            Array<256> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 256 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<512> input{data};
            Array<512> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 512 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<1024> input{data};
            Array<1024> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 1024 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<2048> input{data};
            Array<2048> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 2048 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<4096> input{data};
            Array<4096> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 4096 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<8192> input{data};
            Array<8192> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 8192 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<16384> input{data};
            Array<16384> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 16384 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<32768> input{data};
            Array<32768> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 32768 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    {
        for (size_t i{0}; i < nsamples; ++i) {
            Array<65536> input{data};
            Array<65536> output{data};
            cudaDeviceSynchronize();
            timer.reset();
            fn<<<1, 1>>>(input, output);
            cudaDeviceSynchronize();
            os << 65536 << "," << to_ns(timer.lap()) << std::endl;
        }
    }

    cudaFree(data);
    return EXIT_SUCCESS;
}