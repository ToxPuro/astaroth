/**
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
#include <roctracer_ext.h>   // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // cudaStream_t
#endif

#include "array.h"

typedef struct {
    size_t array_length;
    size_t domain_length;
    size_t pad;
    int radius;
    int stride;
    size_t tpb;
    size_t bpg;
    size_t smem;
} KernelConfig;

static void
print(const KernelConfig c)
{
    printf("KernelConfig {.array_length = %zu, .domain_length = %zu, .pad = %zu, .radius = %d, "
           ".stride = %d, .tpb = %zu, .bpg = %zu, .smem = "
           "%zu}\n",
           c.array_length, c.domain_length, c.pad, c.radius, c.stride, c.tpb, c.bpg, c.smem);
}

#if USE_SMEM
// With shared memory
static size_t
get_smem(const int tpb, const int radius)
{
    return (tpb + 2 * radius) * sizeof(double);
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
           const Array in, Array out)
{
    extern __shared__ double smem[];

    const int base_idx = blockIdx.x * blockDim.x + pad - radius;
    for (int sid = threadIdx.x; sid < (int)(blockDim.x + 2 * radius); sid += blockDim.x)
        if (sid + base_idx < in.length)
            smem[sid] = in.data[sid + base_idx];
    __syncthreads();

    double tmp = 0.0;
    for (int i = -radius; i <= radius; i += stride)
        tmp += smem[threadIdx.x + i + radius];

    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < domain_length)
        out.data[tid + pad] = tmp;
}
#else
// Without shared memory
static size_t
get_smem(const size_t tpb, const int radius)
{
    (void)tpb;    // Unused
    (void)radius; // Unused
    return 0;
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
           const Array in, Array out)
{
    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < domain_length) {

        double tmp = 0.0;
        for (int i = -radius; i <= radius; i += stride)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}
#endif

void
model_kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
             const Array in, Array out)
{
    for (int tid = 0; tid < domain_length; ++tid) {
        double tmp = 0.0;
        for (int i = -radius; i <= radius; i += stride)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}

/** Returns the optimal threadblock dimensions for a given problem size */
static KernelConfig
autotune(const size_t array_length, const size_t domain_length, const size_t pad, const int radius,
         const int stride)
{
    // File
    // const char* benchmark_dir = "microbenchmark-autotune.csv";
    // FILE* fp                  = fopen(benchmark_dir, "a");
    // ERRCHK_ALWAYS(fp);

    // Arrays
    Array a = arrayCreate(array_length, true);
    Array b = arrayCreate(array_length, true);

    // Randomize
    arrayRandomize(&a);
    arrayRandomize(&b);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    // const size_t warp_size             = (size_t)props.warpSize;
    const size_t max_smem              = (size_t)props.sharedMemPerBlock;
    const size_t max_threads_per_block = MAX_THREADS_PER_BLOCK
                                             ? (size_t)min(props.maxThreadsPerBlock,
                                                           MAX_THREADS_PER_BLOCK)
                                             : (size_t)props.maxThreadsPerBlock;

    // Warmup
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart); // Timing start
    kernel<<<1, 1, max_smem>>>(domain_length, pad, radius, stride, a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaDeviceSynchronize();

    // Tune
    KernelConfig c = {
        .array_length  = array_length,
        .domain_length = domain_length,
        .pad           = pad,
        .radius        = radius,
        .stride        = stride,
        .tpb           = 0,
        .bpg           = 0,
        .smem          = 0,
    };

    // 64 bytes on NVIDIA but the minimum L1 cache transaction is 32
    const int minimum_transaction_size_in_elems = 32 / sizeof(double);

    float best_time = INFINITY;
    for (size_t tpb = minimum_transaction_size_in_elems; tpb <= max_threads_per_block;
         tpb += minimum_transaction_size_in_elems) {

        if (tpb > max_threads_per_block)
            break;

        // if (tpb % warp_size)
        //     continue;

        const size_t bpg  = (size_t)ceil(1. * c.domain_length / tpb);
        const size_t smem = get_smem(tpb, c.radius);

        if (smem > max_smem)
            continue;

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        kernel<<<bpg, tpb, smem>>>(domain_length, pad, radius, stride, a, b);
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < 3; ++i)
            kernel<<<bpg, tpb, smem>>>(domain_length, pad, radius, stride, a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

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

        // printf("Threads per block: %zu.\tTime elapsed: %g ms\n", tpb, (double)milliseconds);
        if (milliseconds < best_time) {
            best_time = milliseconds;
            c.tpb     = tpb;
            c.bpg     = bpg;
            c.smem    = smem;
        }

        // format
        // 'usesmem, maxthreadsperblock, problemsize, workingsetsize, milliseconds,
        // tpb, bpg, smem'
        // fprintf(fp, "%d,%d,%lu,%lu,%g,%zu,%zu,%zu\n", USE_SMEM, MAX_THREADS_PER_BLOCK,
        //         c.domain_length * sizeof(double), (2 * c.radius + 1) * sizeof(double),
        //         (double)milliseconds, c.tpb, c.bpg, c.smem);
    }
    arrayDestroy(&a);
    arrayDestroy(&b);

#if USE_SMEM
    ERRCHK_ALWAYS(c.smem);
#endif
    ERRCHK_ALWAYS(c.tpb > 0);
    ERRCHK_ALWAYS(c.bpg > 0);

    print(c);
    // fclose(fp);
    fflush(stdout);
    return c;
}

void
verify(const KernelConfig c)
{
    // Allocate
    Array ahost = arrayCreate(c.array_length, false);
    Array bhost = arrayCreate(c.array_length, false);
    Array a     = arrayCreate(c.array_length, true);
    Array b     = arrayCreate(c.array_length, true);

    // Randomize
    arrayRandomize(&ahost);
    arrayRandomize(&bhost);

    // Model
    model_kernel(c.domain_length, c.pad, c.radius, c.stride, ahost, bhost);

    // Candidate
    const size_t bytes = c.array_length * sizeof(ahost.data[0]);
    cudaMemcpy(a.data, ahost.data, bytes, cudaMemcpyHostToDevice);
    kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.pad, c.radius, c.stride, a, b);
    cudaMemcpy(ahost.data, b.data, bytes, cudaMemcpyDeviceToHost);

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
benchmark(const KernelConfig c, const size_t jobid, const size_t seed, const size_t num_samples)
{
    // Allocate
    Array a = arrayCreate(c.array_length, true);
    Array b = arrayCreate(c.array_length, true);

    // Randomize
    arrayRandomize(&a);
    arrayRandomize(&b);

    // File
    const size_t buflen = 4096;
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "microbenchmark-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp, "caching,maxthreadsperblock,problemsize,workingsetsize,stride,milliseconds,"
                "effectivebandwidth,tpb,jobid,seed,iteration\n");

    // Dryrun
    kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.pad, c.radius, c.stride, a, b);

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    for (size_t i = 0; i < num_samples; ++i) {
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.pad, c.radius, c.stride, a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);
        ERRCHK_CUDA_KERNEL_ALWAYS();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        // Derived statistics
        const size_t bytes   = sizeof(a.data[0]) * (2 * c.domain_length + 2 * c.radius / c.stride);
        const double seconds = (double)milliseconds / 1e3;
        const double bandwidth = bytes / seconds;

        if (i == num_samples - 1) {
            printf("Effective bandwidth: %g GiB/s\n", bandwidth / pow(1024, 3));
            printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
            printf("\tTime elapsed: %g ms\n", (double)milliseconds);
        }

        // Write to file
        fprintf(fp, "%s,%d,%zu,%zu,%d,%g,%g,%zu,%zu,%zu,%zu\n",
                USE_SMEM ? "\"explicit\"" : "\"implicit\"", MAX_THREADS_PER_BLOCK,
                c.domain_length * sizeof(double), (2 * c.radius / c.stride + 1) * sizeof(double),
                c.stride, (double)milliseconds, bandwidth, c.tpb, jobid, seed, i);
    }
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);

    // Free
    fclose(fp);
    arrayDestroy(&a);
    arrayDestroy(&b);
    fflush(stdout);
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
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
    printf("    Compute mode: %d\n", (int)props.computeMode);
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
    printf("    Shared memory per block: %lu\n", props.sharedMemPerBlock);
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    printf("--------------------------------------------------\n");
}

// Used for aligning the array to 256 bytes
// Returns the padding size in elements
static size_t
get_pad(const size_t radius)
{
    size_t pad = radius;
    while (pad * sizeof(double) % 256) {
        ++pad;
        ERRCHK_ALWAYS(pad <= 10000);
    }

    return pad;
}

int
main(int argc, char* argv[])
{

    cudaProfilerStop();

    // Input parameters
    fprintf(stderr, "Usage: ./benchmark <problem size> <working set size> <stride> <jobid>\n");
    const size_t problem_size     = (argc > 1) ? (size_t)atol(argv[1]) : 268435456;
    const size_t working_set_size = (argc > 2) ? (size_t)atol(argv[2]) : 8;
    const int stride              = (argc > 3) ? (size_t)atol(argv[3]) : 1;
    const size_t jobid            = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples      = (argc > 5) ? (size_t)atol(argv[5]) : 100;

    // Derived values
    const int radius           = (((working_set_size / sizeof(double)) - 1) / 2) * stride;
    const size_t pad           = get_pad(radius);
    const size_t domain_length = problem_size / sizeof(double);
    const size_t array_length  = pad + domain_length + radius;
    const size_t seed          = 12345 + time(NULL) + jobid * time(NULL);
    ERRCHK((2 * radius / stride + 1) * sizeof(double) == working_set_size);
    ERRCHK(domain_length * sizeof(double) == problem_size);

    printf("Input parameters:\n");
    printf("\tproblem_size: %zu\n", problem_size);
    printf("\tpad: %zu\n", pad);
    printf("\tradius: %d\n", radius);
    printf("\tstride: %d\n", stride);
    printf("\tjobid: %zu\n", jobid);
    printf("\tnum_samples: %zu\n", num_samples);
    printf("\tseed: %zu\n", seed);
    fflush(stdout);

    if (working_set_size > problem_size) {
        fprintf(stderr, "Invalid working set size: %lu > %lu\n", working_set_size, problem_size);
        return EXIT_FAILURE;
    }

    # if USE_SMEM
    const size_t required_smem = get_smem(1, radius);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const size_t max_smem              = (size_t)props.sharedMemPerBlock;
    if (required_smem > max_smem) {
        fprintf(stderr, "Too large radius for a non-streaming shared memory implementation: %lu > %lu\n", required_smem, max_smem);
        return EXIT_FAILURE;
    }
    #endif

    printDeviceInfo(0);
    printf("USE_SMEM=%d\n", USE_SMEM);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Random
    srand(seed);

    // Benchmark pipeline
    KernelConfig c = autotune(array_length, domain_length, pad, radius, stride);
    verify(c);
    cudaProfilerStart();
    benchmark(c, jobid, seed, num_samples);
    cudaProfilerStop();
    return EXIT_SUCCESS;
}