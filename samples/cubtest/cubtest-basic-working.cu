#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cub/cub.cuh>

#include "timer_hires.h"

#define ARRAY_SIZE(x) (sizeof(x) / (sizeof(x[0])))
#define NUM_SAMPLES (10)

int
main(void)
{
    // Allocate
    const double in[1024]     = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const size_t num_segments = 2;
    const size_t offsets[]    = {0, 5, ARRAY_SIZE(in)}; //{0, 3, 3, count};
    assert(num_segments + 1 == ARRAY_SIZE(offsets));
    double out[num_segments] = {0};

    double* d_in;
    cudaMalloc(&d_in, sizeof(d_in[0]) * ARRAY_SIZE(in));
    cudaMemcpy(d_in, in, sizeof(d_in[0]) * ARRAY_SIZE(in), cudaMemcpyHostToDevice);

    size_t* d_offsets;
    cudaMalloc(&d_offsets, sizeof(d_offsets[0]) * ARRAY_SIZE(offsets));
    cudaMemcpy(d_offsets, offsets, sizeof(d_offsets[0]) * ARRAY_SIZE(offsets),
               cudaMemcpyHostToDevice);

    double* d_out;
    cudaMalloc(&d_out, sizeof(d_out[0]) * ARRAY_SIZE(out));

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments,
                                    d_offsets, d_offsets + 1);
    // cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, count);
    printf("Temp storage: %zu bytes\n", temp_storage_bytes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

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
    for (size_t i = 0; i < ARRAY_SIZE(out); ++i)
        printf("%zu: %g\n", i, out[i]);

    // Deallocate
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    cudaFree(d_offsets);
    cudaFree(d_in);

    return EXIT_SUCCESS;
}