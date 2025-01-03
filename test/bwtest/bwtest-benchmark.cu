/**
    Note: deprecated. Up-to-date microbenchmarks in samples/microbenchmark.

    Microbenchmark the GPU caches in 1D stencil computations and generate a plottable .csv output

    Examples:
        # Usage
        ./bwtest-benchmark <problem size in bytes> <working set size in bytes>

        # 256 MiB problem size and working set of size 8 (one double), i.e. halo r=0
        ./bwtest-benchmark 268435456 8

        # 3-point von Neumann stencil
        ./bwtest-benchmark 268435456 24

        # Profiling
        cmake -DUSE_HIP=ON .. &&\
        make -j &&\
        rocprof --trace-start off -i ~/rocprof-input-metrics.txt ./bwtest-benchmark 268435456 256

cat ~/rocprof-input-metrics.txt
```
# Perf counters group 1
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
# Perf counters group 2
pmc : TCC_HIT[0], TCC_MISS[0], TCC_HIT_sum, TCC_MISS_sum
# Perf counters group 3
pmc: L2CacheHit MemUnitBusy LDSBankConflict

# Filter by dispatches range, GPU index and kernel names
# supported range formats: "3:9", "3:", "3"
#range: 0 : 16
gpu: 0 1 2 3
#kernel: singlepass_solve
```
*/
#include <stdio.h>
#include <stdlib.h>

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <roctracer/roctracer_ext.h>   // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // cudaStream_t
#endif

#include "common.h"

// #define USE_SMEM (0) // Set with cmake
// #define MAX_THREADS_PER_BLOCK (0) // Set with cmake

#if USE_SMEM
static size_t
get_smem(const int tpb, const int radius)
{
    return (tpb + 2 * radius) * sizeof(double);
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const int radius, const size_t pad, const Array in,
           Array out)
{
    extern __shared__ double smem[];

    const int base_idx = blockIdx.x * blockDim.x + pad - radius;
    for (int sid = threadIdx.x; sid < (int)(blockDim.x + 2 * radius); sid += blockDim.x)
        if (sid + base_idx < in.count)
            smem[sid] = in.data[sid + base_idx];
    __syncthreads();

    double tmp = 0.0;
    for (int i = 0; i < 2 * radius + 1; ++i)
        tmp += smem[threadIdx.x + i];

    const size_t tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < domain_length)
        out.data[tid + pad] = tmp;
}
#else
static size_t
get_smem(const int tpb, const int halo)
{
    (void)tpb;  // Unused
    (void)halo; // Unused
    return 0;
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const int radius, const size_t pad, const Array in,
           Array out)
{
    const size_t tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < domain_length) {

        double tmp = 0.0;
        for (int i = -radius; i <= radius; ++i)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}
#endif

void
model_kernel(const size_t domain_length, const int radius, const size_t pad, const Array in,
             Array out)
{
    for (size_t tid = 0; tid < domain_length; ++tid) {
        double tmp = 0.0;
        for (int i = -radius; i <= radius; ++i)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}

typedef struct {
    size_t array_length;
    size_t domain_length;
    int radius;
    size_t pad;
    size_t tpb;
    size_t bpg;
    size_t smem;
} KernelConfig;

/** Returns the optimal threadblock dimensions for a given problem size */
static KernelConfig
autotune(const size_t array_length, const size_t domain_length, const int radius, const size_t pad)
{
    // File
    const char* benchmark_dir = "microbenchmark-autotune.csv";
    FILE* fp                  = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);

    // Arrays
    Array a = arrayCreate(array_length, true);
    Array b = arrayCreate(array_length, true);

    cudaDeviceProp props;
    ERRCHK_CUDA(cudaGetDeviceProperties(&props, 0));
    const size_t warp_size             = (size_t)props.warpSize;
    const size_t max_smem              = (size_t)props.sharedMemPerBlock;
    const size_t max_threads_per_block = MAX_THREADS_PER_BLOCK
                                             ? (size_t)min(props.maxThreadsPerBlock,
                                                           MAX_THREADS_PER_BLOCK)
                                             : (size_t)props.maxThreadsPerBlock;

    // Warmup
    cudaEvent_t tstart, tstop;
    ERRCHK_CUDA(cudaEventCreate(&tstart));
    ERRCHK_CUDA(cudaEventCreate(&tstop));
    ERRCHK_CUDA(cudaEventRecord(tstart)); // Timing start
    for (size_t i = 0; i < 1; ++i)
        kernel<<<1, 1, max_smem>>>(domain_length, radius, pad, a, b);
    ERRCHK_CUDA(cudaEventRecord(tstop)); // Timing stop
    ERRCHK_CUDA(cudaEventSynchronize(tstop));
    ERRCHK_CUDA(cudaEventDestroy(tstart));
    ERRCHK_CUDA(cudaEventDestroy(tstop));
    ERRCHK_CUDA(cudaDeviceSynchronize());

    // Tune
    KernelConfig c = {
        .array_length  = array_length,
        .domain_length = domain_length,
        .radius        = radius,
        .pad           = pad,
        .tpb           = 0,
        .bpg           = 0,
        .smem          = 0,
    };
    float best_time = INFINITY;
    for (size_t tpb = 1; tpb <= max_threads_per_block; ++tpb) {

        if (tpb > max_threads_per_block)
            break;

        if (tpb % warp_size)
            continue;

        const size_t bpg  = (size_t)ceil(1. * c.domain_length / tpb);
        const size_t smem = get_smem(tpb, c.radius);

        if (smem > max_smem)
            continue;

        printf("Current KernelConfig {.array_length = %zu, .domain_length = %zu, .radius = %d, "
               ".pad = %zu, .tpb = %zu, .bpg = %zu, .smem = "
               "%zu}\n",
               c.array_length, c.domain_length, c.radius, c.pad, tpb, bpg, smem);

        ERRCHK_CUDA(cudaEventCreate(&tstart));
        ERRCHK_CUDA(cudaEventCreate(&tstop));

        kernel<<<bpg, tpb, smem>>>(domain_length, radius, pad, a, b);
        ERRCHK_CUDA(cudaDeviceSynchronize());
        ERRCHK_CUDA(cudaEventRecord(tstart)); // Timing start
        for (int i = 0; i < 10; ++i)
            kernel<<<bpg, tpb, smem>>>(domain_length, radius, pad, a, b);
        ERRCHK_CUDA(cudaEventRecord(tstop)); // Timing stop
        ERRCHK_CUDA(cudaEventSynchronize(tstop));

        float milliseconds = 0;
        ERRCHK_CUDA(cudaEventElapsedTime(&milliseconds, tstart, tstop));

        ERRCHK_CUDA(cudaEventDestroy(tstart));
        ERRCHK_CUDA(cudaEventDestroy(tstop));

        ERRCHK_CUDA_KERNEL_ALWAYS();
        //  Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
            // Exit in case of unrecoverable error that needs a device reset
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Unrecoverable CUDA error\n");
                exit(EXIT_FAILURE);
            }
            continue;
        }

        printf("\tTime elapsed: %g ms\n", (double)milliseconds);
        if (milliseconds < best_time) {
            best_time = milliseconds;
            c.tpb     = tpb;
            c.bpg     = bpg;
            c.smem    = smem;
        }

        // format
        // 'usesmem, maxthreadsperblock, problemsize, workingsetsize, milliseconds,
        // tpb, bpg, smem'
        fprintf(fp, "%d,%d,%lu,%lu,%g,%zu,%zu,%zu\n", USE_SMEM, MAX_THREADS_PER_BLOCK,
                c.domain_length * sizeof(double), (2 * c.radius + 1) * sizeof(double),
                (double)milliseconds, c.tpb, c.bpg, c.smem);
    }
    printf("KernelConfig {.array_length = %zu, .domain_length = %zu, .radius = %d, "
           ".pad = %zu, .tpb = %zu, .bpg = %zu, .smem = "
           "%zu}\n",
           c.array_length, c.domain_length, c.radius, c.pad, c.tpb, c.bpg, c.smem);

    arrayDestroy(&a);
    arrayDestroy(&b);

#if USE_SMEM
    ERRCHK_ALWAYS(c.smem);
#endif
    ERRCHK_ALWAYS(c.tpb > 0);
    ERRCHK_ALWAYS(c.bpg > 0);

    fclose(fp);
    fflush(stdout);
    return c;
}

void
verify(const KernelConfig c)
{
    Array ahost = arrayCreate(c.array_length, false);
    Array bhost = arrayCreate(c.array_length, false);
    Array a     = arrayCreate(c.array_length, true);
    Array b     = arrayCreate(c.array_length, true);

    arrayRandomize(&ahost);
    model_kernel(c.domain_length, c.radius, c.pad, ahost, bhost);

    const size_t bytes = c.array_length * sizeof(ahost.data[0]);
    ERRCHK_CUDA(cudaMemcpy(a.data, ahost.data, bytes, cudaMemcpyHostToDevice));
    kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.radius, c.pad, a, b);
    ERRCHK_CUDA(cudaMemcpy(ahost.data, b.data, bytes, cudaMemcpyDeviceToHost));

    const double* candidate = ahost.data;
    const double* model     = bhost.data;

    size_t failure_count       = 0;
    const size_t failure_limit = 100;
    for (size_t i = c.pad; i < c.pad + c.domain_length; ++i) {
        if (model[i] != candidate[i]) {
            fprintf(stderr, "Failure at %lu: %g (host) and %g (device)\n", i, model[i],
                    candidate[i]);
            ++failure_count;
        }
        if (failure_count > failure_limit) {
            fprintf(stderr, "Failure limit reached, exiting...\n");
            break;
        }
    }

    arrayDestroy(&a);
    arrayDestroy(&b);
    arrayDestroy(&ahost);
    arrayDestroy(&bhost);

    printf("Results verified: %s\n", failure_count ? "Failures found" : "OK!");
}

static void
benchmark(const KernelConfig c)
{
    const size_t num_iters = 100;

    // Allocate
    Array a = arrayCreate(c.array_length, true);
    Array b = arrayCreate(c.array_length, true);

    // Benchmark
    cudaEvent_t tstart, tstop;
    ERRCHK_CUDA(cudaEventCreate(&tstart));
    ERRCHK_CUDA(cudaEventCreate(&tstop));

    ERRCHK_CUDA(cudaEventRecord(tstart)); // Timing start
    for (size_t i = 0; i < num_iters; ++i)
        kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.radius, c.pad, a, b);
    ERRCHK_CUDA(cudaEventRecord(tstop)); // Timing stop
    ERRCHK_CUDA(cudaEventSynchronize(tstop));
    ERRCHK_CUDA_KERNEL_ALWAYS();

    float milliseconds = 0;
    ERRCHK_CUDA(cudaEventElapsedTime(&milliseconds, tstart, tstop));
    ERRCHK_CUDA(cudaEventDestroy(tstart));
    ERRCHK_CUDA(cudaEventDestroy(tstop));

    const size_t bytes     = num_iters * sizeof(a.data[0]) * (2 * c.domain_length + 2 * c.radius);
    const double seconds   = (double)milliseconds / 1e3;
    const double bandwidth = bytes / seconds;
    printf("Effective bandwidth: %g GiB/s\n", bandwidth / pow(1024, 3));
    printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
    printf("\tTime elapsed: %g ms\n", (double)milliseconds);

    // File
    const char* benchmark_dir = "microbenchmark.csv";
    FILE* fp                  = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);

    // format
    // 'usesmem, maxthreadsperblock, problemsize, workingsetsize, milliseconds, effectivebandwidth,
    // tpb'
    fprintf(fp, "%d,%d,%lu,%lu,%g,%g,%zu\n", USE_SMEM, MAX_THREADS_PER_BLOCK,
            c.domain_length * sizeof(double), (2 * c.radius + 1) * sizeof(double),
            (double)milliseconds, bandwidth, c.tpb);
    fclose(fp);

    // Free
    arrayDestroy(&a);
    arrayDestroy(&b);
    fflush(stdout);
}

void
printDeviceInfo(const int device_id)
{
    cudaDeviceProp props;
    ERRCHK_CUDA(cudaGetDeviceProperties(&props, device_id));
    printf("--------------------------------------------------\n");
    printf("Device Number: %d\n", device_id);
    const size_t bus_id_max_len = 128;
    char bus_id[bus_id_max_len];
    ERRCHK_CUDA(cudaDeviceGetPCIBusId(bus_id, bus_id_max_len, device_id));
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
    ERRCHK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
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
    printf("    Shared memory per block: %lu\n", props.sharedMemPerBlock);
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    printf("--------------------------------------------------\n");
}

// Used for aligning the array to 256 bytes
// Returns the padding size in elements
static size_t
get_pad(const size_t r)
{
    size_t pad = r;
    while (pad * sizeof(double) % 256) {
        ++pad;
        ERRCHK_ALWAYS(pad <= 10000);
    }

    return pad;
}

int
main(int argc, char* argv[])
{
    /*
    TODO: add padding: go through the whole thing carefully!

    we have
        max_count = the array size incl. padding
        count = size of the computational domain
        pad = size of the padding in the beginning (incl. halo)
    */
    ERRCHK_CUDA(cudaProfilerStop());
    if (argc != 3) {
        fprintf(stderr, "Usage: ./benchmark <problem size> <working set size>\n");
        fprintf(stderr, "       ./benchmark 0 0 # To use the defaults\n");
        return EXIT_FAILURE;
    }
    const size_t arg0 = (size_t)atol(argv[1]);
    const size_t arg1 = (size_t)atol(argv[2]);

    // Input values
    const size_t problem_size     = arg0 ? arg0 : 268435456; // 256 MiB default, bytes
    const size_t working_set_size = arg1 ? arg1 : 8;         // 8 byte default (r=0), bytes

    // Derived values
    const int radius           = ((working_set_size / sizeof(double)) - 1) / 2;
    const size_t pad           = get_pad(radius);
    const size_t domain_length = problem_size / sizeof(double);
    const size_t array_length  = pad + domain_length + radius;
    ERRCHK((2 * radius + 1) * sizeof(double) == working_set_size);
    ERRCHK(domain_length * sizeof(double) == problem_size);

    if (working_set_size > problem_size) {
        fprintf(stderr, "Invalid working set size: %lu > %lu\n", working_set_size, problem_size);
        return EXIT_FAILURE;
    }

    printDeviceInfo(0);
    printf("USE_SMEM=%d\n", USE_SMEM);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);

    // ERRCHK_CUDA(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    // ERRCHK_CUDA(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    KernelConfig c = autotune(array_length, domain_length, radius, pad);
    verify(c);
    ERRCHK_CUDA(cudaProfilerStart());
    benchmark(c);
    ERRCHK_CUDA(cudaProfilerStop());
    return EXIT_SUCCESS;
}
