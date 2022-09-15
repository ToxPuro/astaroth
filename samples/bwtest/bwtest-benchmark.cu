/**
    Microbenchmark the GPU caches in 1D stencil computations and generate a plottable .csv output
*/
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime.h> // Needed in files that include kernels
#endif

#include "errchk.h"

#define NUM_FIELDS (8)
#define NUM_STENCILS (10)
#define NUM_POINTS_PER_STENCIL (55)
#define WORKING_SET_SIZE (NUM_FIELDS * NUM_STENCILS * NUM_POINTS_PER_STENCIL)
//#define HALO ((WORKING_SET_SIZE - 1) / 2)
//#define HALO (8)
#define HALO ((size_t)0)

static const char* benchmark_dir = "bwtest-benchmark.csv";

typedef struct {
    size_t count;
    double* data;
    bool on_device;
} Array;

static Array
arrayCreate(const size_t count, const bool on_device)
{
    Array a = (Array){
        .count     = count,
        .data      = NULL,
        .on_device = on_device,
    };

    const size_t bytes = count * sizeof(a.data[0]);
    if (on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&a.data, bytes));
    }
    else {
        a.data = (double*)malloc(bytes);
        ERRCHK_ALWAYS(a.data);
    }

    return a;
}

static void
arrayDestroy(Array* a)
{
    if (a->on_device)
        cudaFree(a->data);
    else
        free(a->data);
    a->data  = NULL;
    a->count = 0;
}

/**
    Simple rng for doubles in range [0...1].
    Not suitable for generating full-precision f64 randoms.
*/
static double
randd(void)
{
    return (double)rand() / RAND_MAX;
}

static void
arrayRandomize(Array* a)
{
    if (!a->on_device) {
        for (size_t i = 0; i < a->count; ++i)
            a->data[i] = randd();
    }
    else {
        Array b = arrayCreate(a->count, false);
        arrayRandomize(&b);
        const size_t bytes = a->count * sizeof(b.data[0]);
        cudaMemcpy(a->data, b.data, bytes, cudaMemcpyHostToDevice);
        arrayDestroy(&b);
    }
}

__global__ void
kernel(const Array in, Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (HALO <= tid && tid < in.count - HALO) {
        double tmp = 0.0;

        //#pragma unroll
        for (int i = -HALO; i <= HALO; ++i)
            tmp += in.data[tid + i];

        out.data[tid] = tmp;
    }
}

void
model_kernel(const Array in, Array out)
{
    for (size_t tid = 0; tid < in.count; ++tid) {
        if (HALO <= tid && tid < in.count - HALO) {
            double tmp = 0.0;

#pragma unroll
            for (int i = -HALO; i <= HALO; ++i)
                tmp += in.data[tid + i];

            out.data[tid] = tmp;
        }
    }
}

typedef struct {
    size_t count;
    size_t tpb;
    size_t bpg;
} KernelConfig;

/** Returns the optimal threadblock dimensions for a given problem size */
static KernelConfig
autotune(const size_t count)
{
    Array a = arrayCreate(count, true);
    Array b = arrayCreate(count, true);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const size_t warp_size             = (size_t)props.warpSize;
    const size_t max_threads_per_block = (size_t)props.maxThreadsPerBlock;

    // Warmup
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart); // Timing start
    for (size_t i = 0; i < 1; ++i)
        kernel<<<1, 1>>>(a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaDeviceSynchronize();

    // Tune
    KernelConfig c  = {.count = count, .tpb = 0, .bpg = 0};
    float best_time = INFINITY;
    for (size_t tpb = 1; tpb <= max_threads_per_block; ++tpb) {
        if (tpb > max_threads_per_block)
            break;

        const size_t bpg = (size_t)ceil(1. * count / tpb);

        if (tpb % warp_size)
            continue;

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < 5; ++i)
            kernel<<<bpg, tpb>>>(a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
            // Exit in case of unrecoverable error that needs a device reset
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Unrecoverable CUDA error\n");
                exit(EXIT_FAILURE);
            }
            continue;
        }

        // printf("KernelConfig {.tpb = %lu, .bpg = %lu}\n", tpb, bpg);
        // printf("\tTime elapsed: %g ms\n", (double)milliseconds);
        if (milliseconds < best_time) {
            best_time = milliseconds;
            c.tpb     = tpb;
            c.bpg     = bpg;
        }
    }
    printf("KernelConfig {.count = %lu, .tpb = %lu, .bpg = %lu}\n", c.count, c.tpb, c.bpg);

    arrayDestroy(&a);
    arrayDestroy(&b);

    return c;
}

void
verify(const KernelConfig c)
{
    const size_t count = c.count;
    const size_t tpb   = c.tpb;
    const size_t bpg   = c.bpg;

    Array ahost = arrayCreate(count, false);
    Array bhost = arrayCreate(count, false);
    Array a     = arrayCreate(count, true);
    Array b     = arrayCreate(count, true);

    arrayRandomize(&ahost);
    model_kernel(ahost, bhost);

    const size_t bytes = count * sizeof(ahost.data[0]);
    cudaMemcpy(a.data, ahost.data, bytes, cudaMemcpyHostToDevice);
    kernel<<<bpg, tpb>>>(a, b);
    cudaMemcpy(ahost.data, b.data, bytes, cudaMemcpyDeviceToHost);

    const double* candidate = ahost.data;
    const double* model     = bhost.data;

    for (size_t i = HALO; i < ahost.count - HALO; ++i) {
        if (model[i] != candidate[i]) {
            fprintf(stderr, "Failure at %lu: %g (host) and %g (device)\n", i, model[i],
                    candidate[i]);
        }
    }

    arrayDestroy(&a);
    arrayDestroy(&b);
    arrayDestroy(&ahost);
    arrayDestroy(&bhost);

    printf("Results verified\n");
}

static void
benchmark(const KernelConfig c)
{
    const size_t num_iters = 10;

    // Allocate
    Array a = arrayCreate(c.count, true);
    Array b = arrayCreate(c.count, true);

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart); // Timing start
    for (size_t i = 0; i < num_iters; ++i)
        kernel<<<c.bpg, c.tpb>>>(a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    ERRCHK_CUDA_KERNEL_ALWAYS();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tstart, tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);

    const size_t bytes   = num_iters * sizeof(a.data[0]) * (a.count + b.count - 2 * HALO);
    const double seconds = (double)milliseconds / 1e3;
    printf("Effective bandwidth: %g GiB/s\n", bytes / seconds / pow(1024, 3));
    printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
    printf("\tTime elapsed: %g ms\n", (double)milliseconds);

    FILE* fp = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);
    fprintf(fp, "%lu, %g\n", HALO, (double)milliseconds);
    fclose(fp);

    // Free
    arrayDestroy(&a);
    arrayDestroy(&b);
}

void
printDeviceInfo(const int device_id)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    printf("--------------------------------------------------\n");
    printf("Device Number: %d\n", device_id);
    const size_t bus_id_max_len = 128;
    char bus_id[bus_id_max_len];
    cudaDeviceGetPCIBusId(bus_id, bus_id_max_len, device_id);
    printf("  PCI bus ID: %s\n", bus_id);
    printf("    Device name: %s\n", props.name);
    printf("    Compute capability: %d.%d\n", props.major, props.minor);

    // Compute
    printf("  Compute\n");
    printf("    Clock rate (GHz): %g\n", props.clockRate / 1e6); // KHz -> GHz
    printf("    Stream processors: %d\n", props.multiProcessorCount);
    printf(
        "    Compute mode: %d\n",
        (int)props
            .computeMode); // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
    // Memory
    printf("  Global memory\n");
    printf("    Memory Clock Rate (MHz): %d\n", props.memoryClockRate / (1000));
    printf("    Memory Bus Width (bits): %d\n", props.memoryBusWidth);
    printf("    Peak Memory Bandwidth (GiB/s): %f\n",
           2 * (props.memoryClockRate * 1e3) * props.memoryBusWidth / (8. * 1024. * 1024. * 1024.));
    printf("    ECC enabled: %d\n", props.ECCEnabled);

    // Memory usage
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes = total_bytes - free_bytes;
    printf("    Total global mem: %.2f GiB\n", props.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("    Gmem used (GiB): %.2f\n", used_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory free (GiB): %.2f\n", free_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory total (GiB): %.2f\n", total_bytes / (1024.0 * 1024 * 1024));
    printf("  Caches\n");
#if !AC_USE_HIP
    printf("    Local L1 cache supported: %d\n", props.localL1CacheSupported);
    printf("    Global L1 cache supported: %d\n", props.globalL1CacheSupported);
#endif
    printf("    L2 size: %d KiB\n", props.l2CacheSize / (1024));
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    printf("--------------------------------------------------\n");
}

int
main(void)
{
    printDeviceInfo(0);

    const size_t count = NUM_FIELDS * (size_t)pow(32, 3);
    KernelConfig c     = autotune(count);
    verify(c);
    benchmark(c);
    return EXIT_SUCCESS;
}