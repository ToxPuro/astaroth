#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#if USE_HIP
#include <hipcub/hipcub.hpp>

#define cub hipcub
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#else
#include <cub/cub.cuh>
#endif

#include "timer_hires.h"

#define ARRAY_SIZE(x) (sizeof(x) / (sizeof(x[0])))
#define NUM_SAMPLES (10)

int
main(void)
{
    // Allocate
    const size_t nn           = 256;
    const size_t np           = 32;
    const size_t count        = nn * nn * nn * np;
    const size_t num_segments = nn * np;

    double* in = (double*)malloc(sizeof(in[0]) * count);
    assert(in);
    for (size_t i = 0; i < count; ++i)
        in[i] = !(i % (nn * nn)) ? i / (nn * nn) : 0;

    size_t* offsets = (size_t*)malloc(sizeof(offsets[0]) * (num_segments + 1));
    assert(offsets);
    for (size_t i = 0; i <= num_segments; ++i)
        offsets[i] = i * (count / num_segments);

    double* out = (double*)malloc(sizeof(out[0]) * num_segments);
    assert(out);

    double* d_in;
    cudaMalloc(&d_in, sizeof(d_in[0]) * count);
    assert(d_in);
    cudaMemcpy(d_in, in, sizeof(d_in[0]) * count, cudaMemcpyHostToDevice);

    size_t* d_offsets;
    cudaMalloc(&d_offsets, sizeof(d_offsets[0]) * (num_segments + 1));
    assert(d_offsets);
    cudaMemcpy(d_offsets, offsets, sizeof(d_offsets[0]) * (num_segments + 1),
               cudaMemcpyHostToDevice);

    double* d_out;
    cudaMalloc(&d_out, sizeof(d_out[0]) * num_segments);
    assert(d_out);

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments,
                                    d_offsets, d_offsets + 1);
    // cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, count);
    printf("Temp storage: %zu bytes\n", temp_storage_bytes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    assert(d_temp_storage);

    // Warmup
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments,
                                    d_offsets, d_offsets + 1);

    // Benchmark and compute
    double time_elapsed = 0;
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        cudaDeviceSynchronize();
        Timer t;
        timer_reset(&t);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                        num_segments, d_offsets, d_offsets + 1);
        // cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, count);
        cudaDeviceSynchronize();
        time_elapsed += timer_diff_nsec(t) / 1e6;
        // timer_diff_print(t);
    }
    time_elapsed /= NUM_SAMPLES;
    printf("Average time elapsed: %g ms\n", time_elapsed);

    // Get results
    cudaMemcpy(out, d_out, sizeof(out[0]) * num_segments, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_segments; ++i) {
        if (out[i] != i)
            printf("%zu: %g\n", i, out[i]);
        assert(out[i] == i);
    }

    // Deallocate
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    cudaFree(d_offsets);
    cudaFree(d_in);
    free(out);
    free(offsets);
    free(in);

    return EXIT_SUCCESS;
}