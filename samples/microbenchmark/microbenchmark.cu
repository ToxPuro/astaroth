/**
    Microbenchmark the GPU caches in 1D stencil computations and generate a plottable .csv output

    Examples:
        # Usage (see the main invocation for an up-to-date list of input parameters)
        ./microbenchmark <computational domain length in elements> <stencil radius in elements>

        # Default problem size and radius
        ./microbenchmark

        # Bandwidth test with r=0 stencil
        ./microbenchmark 33554432 0

        # Profiling
        cmake -DUSE_HIP=ON .. &&\
        make -j &&\
        rocprof --trace-start off -i ~/rocprof-input-metrics.txt ./microbenchmark

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
#include <limits.h>
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

#include "array.h"
#include "timer_hires.h"

// MB_IMPLEMENTATION
//      1: 1 element per thread
//      2: 4 elements per thread
//      3: templated unrolling
#if MB_IMPLEMENTATION == 1
    #define USE_TEMPLATED_IMPLEMENTATION (0)
    #define ELEMS_PER_THREAD (1)
#elif MB_IMPLEMENTATION == 2
    #define USE_TEMPLATED_IMPLEMENTATION (0)
    #define ELEMS_PER_THREAD (4)
#elif MB_IMPLEMENTATION == 3
    #define USE_TEMPLATED_IMPLEMENTATION (1)
    #define ELEMS_PER_THREAD (1)
#else
    invalid implementation
#endif

// AMD performance pitfall with templates. Gets around compute bound on NVIDIA
// Works properly on MI100 and Triton (HIP 5.0 and Clang 14.0)
// #define USE_TEMPLATED_IMPLEMENTATION (1)

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

#if USE_TEMPLATED_IMPLEMENTATION
#if USE_SMEM
// With shared memory
static size_t
get_smem(const int tpb, const int radius)
{
    return (tpb + 2 * radius) * sizeof(real);
}

template <int radius, int stride>
__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const size_t pad, const Array in, Array out)
{
    extern __shared__ real smem[];

    const int base_idx = blockIdx.x * blockDim.x + pad - radius;
    for (int sid = threadIdx.x; sid < (int)(blockDim.x + 2 * radius); sid += blockDim.x)
        if (sid + base_idx < in.length)
            smem[sid] = in.data[sid + base_idx];
    __syncthreads();

    real tmp = 0.0;
#pragma unroll
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

template <int radius, int stride>
__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const size_t pad, const Array in, Array out)
{
    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < domain_length) {

        real tmp = 0.0;
#pragma unroll
        for (int i = -radius; i <= radius; i += stride)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}
#endif

static void
unrolled_kernel(const dim3 bpg, const dim3 tpb, const size_t smem, const size_t domain_length,
                const size_t pad, const int radius, const int stride, const Array in, Array out)
{
    ERRCHK_ALWAYS(stride == 1);
    switch (radius) {
    case 0:
        return kernel<0, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 1:
        return kernel<1, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 2:
        return kernel<2, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 4:
        return kernel<4, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 8:
        return kernel<8, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 16:
        return kernel<16, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 32:
        return kernel<32, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 64:
        return kernel<64, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 128:
        return kernel<128, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 256:
        return kernel<256, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 512:
        return kernel<512, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    case 1024:
        return kernel<1024, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    // case 2048:
    //     return kernel<2048, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    // case 4096:
    //     return kernel<4096, 1><<<bpg, tpb, smem>>>(domain_length, pad, in, out);
    default:
        fprintf(stderr, "Invalid radius %d\n", radius);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}
#else // NOT USE_TEMPLATED_IMPLEMENTATION
// #if USE_SMEM
// // With shared memory
// static size_t
// get_smem(const int tpb, const int radius)
// {
//     return (tpb + 2 * radius) * sizeof(real);
// }

// __global__ void
// #if MAX_THREADS_PER_BLOCK
// __launch_bounds__(MAX_THREADS_PER_BLOCK)
// #endif
//     kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
//            const Array in, Array out)
// {
//     extern __shared__ real smem[];

//     const int base_idx = blockIdx.x * blockDim.x + pad - radius;
//     for (int sid = threadIdx.x; sid < (int)(blockDim.x + 2 * radius); sid += blockDim.x)
//         if (sid + base_idx < in.length)
//             smem[sid] = in.data[sid + base_idx];
//     __syncthreads();

//     real tmp = 0.0;
//     for (int i = -radius; i <= radius; i += stride)
//         tmp += smem[threadIdx.x + i + radius];

//     const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);
//     if (tid < domain_length)
//         out.data[tid + pad] = tmp;
// }
#if USE_SMEM
// With shared memory
static size_t
get_smem(const int tpb, const int radius)
{
    return (tpb * ELEMS_PER_THREAD + 2 * radius) * sizeof(real);
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
           const Array in, Array out)
{
    extern __shared__ real smem[];

    const int base_idx = blockIdx.x * blockDim.x * ELEMS_PER_THREAD + pad - radius;
    for (int sid = threadIdx.x; sid < (int)(blockDim.x * ELEMS_PER_THREAD + 2 * radius);
         sid += blockDim.x)
        if (sid + base_idx < in.length)
            smem[sid] = in.data[sid + base_idx];
    __syncthreads();

    real tmp[ELEMS_PER_THREAD] = {0};
    for (int i = -radius; i <= radius; i += stride) {
#pragma unroll
        for (int block = 0; block < ELEMS_PER_THREAD; ++block)
            tmp[block] += smem[radius + threadIdx.x + i + block * blockDim.x];
    }
    const int base_tid = (int)(threadIdx.x + blockIdx.x * blockDim.x * ELEMS_PER_THREAD);
#pragma unroll
    for (int block = 0; block < ELEMS_PER_THREAD; ++block)
        out.data[pad + base_tid + block * blockDim.x] = tmp[block];
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
    const int tid              = (int)(threadIdx.x + blockIdx.x * blockDim.x * ELEMS_PER_THREAD);
    real tmp[ELEMS_PER_THREAD] = {0};

    for (int i = -radius; i <= radius; i += stride) {
#pragma unroll
        for (int block = 0; block < ELEMS_PER_THREAD; ++block)
            tmp[block] += in.data[pad + tid + block * blockDim.x + i];
        // tmp[0] += in.data[pad + tid + 0 * blockDim.x + i];
        // tmp[1] += in.data[pad + tid + 1 * blockDim.x + i];
        // tmp[2] += in.data[pad + tid + 2 * blockDim.x + i];
        // tmp[3] += in.data[pad + tid + 3 * blockDim.x + i];
    }
#pragma unroll
    for (int block = 0; block < ELEMS_PER_THREAD; ++block)
        out.data[pad + tid + block * blockDim.x] = tmp[block];
    // out.data[pad + tid + 0 * blockDim.x] = tmp[0];
    // out.data[pad + tid + 1 * blockDim.x] = tmp[1];
    // out.data[pad + tid + 2 * blockDim.x] = tmp[2];
    // out.data[pad + tid + 3 * blockDim.x] = tmp[3];
}
#endif
static void
unrolled_kernel(const dim3 bpg, const dim3 tpb, const size_t smem, const size_t domain_length,
                const size_t pad, const int radius, const int stride, const Array in, Array out)
{
    ERRCHK_ALWAYS(stride == 1);
    return kernel<<<bpg, tpb, smem>>>(domain_length, pad, radius, stride, in, out);
}
#endif

void
model_kernel(const size_t domain_length, const size_t pad, const int radius, const int stride,
             const Array in, Array out)
{
    ERRCHK_ALWAYS(domain_length < INT_MAX)
    for (int tid = 0; tid < domain_length; ++tid) {
        real tmp = 0.0;
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
    unrolled_kernel(1, 1, max_smem, domain_length, pad, radius, stride, a, b);
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
    const int minimum_transaction_size_in_elems = 32 / sizeof(real);

    float best_time = INFINITY;
    for (size_t tpb = minimum_transaction_size_in_elems; tpb <= max_threads_per_block;
         tpb += minimum_transaction_size_in_elems) {

        if (tpb > max_threads_per_block)
            break;

        // if (tpb % warp_size)
        //     continue;

        const size_t bpg  = (size_t)ceil(1. * c.domain_length / tpb / ELEMS_PER_THREAD);
        const size_t smem = get_smem(tpb, c.radius);

        if (smem > max_smem)
            continue;

        // Require that tbdims is divisible by domain length
        if (bpg * tpb * ELEMS_PER_THREAD != c.domain_length)
            continue;

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        unrolled_kernel(bpg, tpb, smem, domain_length, pad, radius, stride, a, b);
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < 3; ++i)
            unrolled_kernel(bpg, tpb, smem, domain_length, pad, radius, stride, a, b);
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
        //         c.domain_length * sizeof(real), (2 * c.radius + 1) * sizeof(real),
        //         (real)milliseconds, c.tpb, c.bpg, c.smem);
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
    unrolled_kernel(c.bpg, c.tpb, c.smem, c.domain_length, c.pad, c.radius, c.stride, a, b);
    cudaMemcpy(ahost.data, b.data, bytes, cudaMemcpyDeviceToHost);

    const real* candidate = ahost.data;
    const real* model     = bhost.data;

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
    if (failure_count > 0) {
        fprintf(stderr, "Failures found. Exiting without benchmarking.\n");
        exit(EXIT_FAILURE);
    }
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
    fprintf(fp, "implementation,maxthreadsperblock,domainlength,radius,stride,milliseconds,"
                "effectivebandwidth,tpb,jobid,seed,iteration,double_precision,mbimplementation\n");

    // Dryrun
    unrolled_kernel(c.bpg, c.tpb, c.smem, c.domain_length, c.pad, c.radius, c.stride, a, b);

    // Benchmark
    cudaProfilerStart();
    Timer t;
    //cudaEvent_t tstart, tstop;
    //cudaEventCreate(&tstart);
    //cudaEventCreate(&tstop);

    for (size_t i = 0; i < num_samples; ++i) {
        cudaDeviceSynchronize();
	timer_reset(&t);
        //cudaEventRecord(tstart); // Timing start
        unrolled_kernel(c.bpg, c.tpb, c.smem, c.domain_length, c.pad, c.radius, c.stride, a, b);
        //cudaEventRecord(tstop); // Timing stop
        //cudaEventSynchronize(tstop);
	cudaDeviceSynchronize();
	const long double milliseconds = timer_diff_nsec(t)/1e6l;
        ERRCHK_CUDA_KERNEL_ALWAYS();

        //float milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, tstart, tstop);

        // Derived statistics
        const size_t bytes   = sizeof(a.data[0]) * (2 * c.domain_length + 2 * c.radius / c.stride);
        const long double seconds = (long double)milliseconds / 1e3l;
        const long double bandwidth = bytes / seconds;

        if (i == num_samples - 1) {
            printf("Effective bandwidth: %Lg GiB/s\n", bandwidth / pow(1024, 3));
            printf("\tBytes transferred: %Lg GiB\n", (long double)bytes / pow(1024, 3));
            //printf("\tTime elapsed: %Lg ms (CUDA)\n", (long double)milliseconds);
            printf("\tTime elapsed: %Lg ms (POSIX)\n", milliseconds);
        }

        // Write to file
        fprintf(fp, "%s,%d,%zu,%zu,%d,%Lg,%Lg,%zu,%zu,%zu,%zu,%u,%u\n",
                USE_SMEM ? "\"explicit\"" : "\"implicit\"", MAX_THREADS_PER_BLOCK, c.domain_length,
                c.radius, c.stride, milliseconds, bandwidth, c.tpb, jobid, seed, i,
                DOUBLE_PRECISION,MB_IMPLEMENTATION);
    }
    //cudaEventDestroy(tstart);
    //cudaEventDestroy(tstop);
    cudaProfilerStop();

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
    while (pad * sizeof(real) % 256) {
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
    fprintf(stderr, "Usage: ./benchmark <computational domain length> <radius> <stride> <jobid> "
                    "<num_samples> <salt>\n");
    const size_t domain_length = (argc > 1) ? (size_t)atol(argv[1])
                                            : 128 * pow(1024, 2) / sizeof(real);
    const size_t radius        = (argc > 2) ? (size_t)atol(argv[2]) : 1;
    const int stride           = (argc > 3) ? (size_t)atol(argv[3]) : 1;
    const size_t jobid         = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples   = (argc > 5) ? (size_t)atol(argv[5]) : 100;
    const size_t salt          = (argc > 6) ? (size_t)atol(argv[6]) : 42;

    // Derived values
    ERRCHK_ALWAYS(stride == 1); // Not implemented yet
    // const int radius           = (((working_set_size / sizeof(real)) - 1) / 2) * stride;
    const size_t pad = get_pad(radius);
    // const size_t domain_length = problem_size / sizeof(real);
    const size_t array_length = pad + domain_length + radius;
    const size_t seed         = 12345 + salt +
                        (1 + domain_length + radius + stride + jobid + num_samples) * time(NULL);

    printf("Input parameters:\n");
    printf("\tdomain_length: %zu\n", domain_length);
    printf("\tradius: %d\n", radius);
    printf("\tpad: %zu\n", pad);
    printf("\tstride: %d\n", stride);
    printf("\tjobid: %zu\n", jobid);
    printf("\tnum_samples: %zu\n", num_samples);
    printf("\tseed: %zu\n", seed);
    fflush(stdout);

    // if (working_set_size > domain_length) {
    //     fprintf(stderr, "Invalid working set size: %lu > %lu\n", working_set_size, problem_size);
    //     return EXIT_FAILURE;
    // }

#if USE_SMEM
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    const size_t required_smem = get_smem(1, radius);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const size_t max_smem = (size_t)props.sharedMemPerBlock;
    if (required_smem > max_smem) {
        fprintf(stderr,
                "Too large radius for a non-streaming shared memory implementation: %lu > %lu\n",
                required_smem, max_smem);
        return EXIT_FAILURE;
    }
#endif

    printDeviceInfo(0);
    printf("USE_SMEM=%d\n", USE_SMEM);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);
    printf("DOUBLE_PRECISION=%u\n", DOUBLE_PRECISION);
    printf("MB_IMPLEMENTATION=%d\n", MB_IMPLEMENTATION);
    printf("ELEMS_PER_THREAD=%d\n", ELEMS_PER_THREAD);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Random
    srand(seed);

    // Benchmark pipeline
    KernelConfig c = autotune(array_length, domain_length, pad, radius, stride);
    verify(c);
    benchmark(c, jobid, seed, num_samples);
    return EXIT_SUCCESS;
}
