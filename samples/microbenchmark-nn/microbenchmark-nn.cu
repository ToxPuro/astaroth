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
#include "backend.h"

void
model_kernel(const size_t domain_length, const int radius, const int stride, const Array in,
             Array out)
{
    ERRCHK_ALWAYS(domain_length < INT_MAX)
    for (int i = 0; i < (int)domain_length; ++i) {
        real tmp = 0;
        for (int r = -radius; r <= radius; r += stride) {
            const size_t idx = i + r;
            if (idx >= 0 && idx < domain_length) // implicit zero pad
                tmp += in.data[idx];
        }
        out.data[i] = tmp;
    }
}

bool
verify(const size_t domain_length, const size_t radius, const size_t stride)
{
    // Allocate
    Array host_input  = arrayCreate(domain_length, false);
    Array host_output = arrayCreate(domain_length, false);

    // Randomize
    arrayRandomize(&host_input);
    arrayRandomize(&host_output);
    for (size_t i = 0; i < domain_length; ++i)
        host_input.data[i] = 1;

    // Model
    model_kernel(domain_length, radius, stride, host_input, host_output);

    // Candidate
    backendInit(domain_length, radius, stride);
    Array input  = backendGetInputTensor();
    Array output = backendGetOutputTensor();
    ERRCHK_ALWAYS(input.bytes == host_input.bytes);

    cudaMemcpy(input.data, host_input.data, host_input.bytes, cudaMemcpyHostToDevice);
    backendConvolutionFwd();
    cudaDeviceSynchronize();
    cudaMemcpy(host_input.data, output.data, output.bytes, cudaMemcpyDeviceToHost);

    const real* candidate = host_input.data;
    const real* model     = host_output.data;

    size_t failure_count       = 0;
    const size_t failure_limit = 100;
    for (size_t i = 0; i < domain_length; ++i) {
        // printf("%zu: %g and %g\n", i, model[i], candidate[i]);
        if (model[i] != candidate[i]) {
            fprintf(stderr, "Failure at %lu: %g (host) and %g (device)\n", i, (double)model[i],
                    (double)candidate[i]);
            ++failure_count;
        }
        if (failure_count > failure_limit) {
            fprintf(stderr, "Failure limit reached, exiting...\n");
            break;
        }
    }

    backendQuit();
    arrayDestroy(&host_input);
    arrayDestroy(&host_output);

    printf("Results verified: %s\n", failure_count ? "Failures found" : "OK!");
    return failure_count == 0;
}

static void
benchmark(const size_t domain_length, const size_t radius, const size_t stride, const size_t jobid,
          const size_t seed, const size_t num_samples)
{
    // Allocate
    Array a = arrayCreate(domain_length, true);
    Array b = arrayCreate(domain_length, true);

    // Randomize
    arrayRandomize(&a);
    arrayRandomize(&b);

    // File
    const size_t buflen = 4096;
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "microbenchmark-nn-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp, "implementation,problemsize,workingsetsize,stride,milliseconds,"
                "effectivebandwidth,jobid,seed,iteration,double_precision\n");

    // Dryrun
    // kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.pad, c.radius, c.stride, a, b);

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    for (size_t i = 0; i < num_samples; ++i) {
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        // kernel<<<c.bpg, c.tpb, c.smem>>>(c.domain_length, c.pad, c.radius, c.stride, a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);
        ERRCHK_CUDA_KERNEL_ALWAYS();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        // Derived statistics
        const size_t bytes     = sizeof(a.data[0]) * (2 * domain_length + 2 * radius / stride);
        const double seconds   = (double)milliseconds / 1e3;
        const double bandwidth = bytes / seconds;

        if (i == num_samples - 1) {
            printf("Effective bandwidth: %g GiB/s\n", bandwidth / pow(1024, 3));
            printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
            printf("\tTime elapsed: %g ms\n", (double)milliseconds);
        }

        // Write to file
        fprintf(fp, "%s,%zu,%zu,%zu,%g,%g,%zu,%zu,%zu,%u\n", "cudnn-miopen",
                domain_length * sizeof(a.data[0]), (2 * radius / stride + 1) * sizeof(a.data[0]),
                stride, (double)milliseconds, bandwidth, jobid, seed, i, DOUBLE_PRECISION);
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

int
main(int argc, char* argv[])
{
    cudaProfilerStop();

    // Input parameters
    fprintf(stderr, "Usage: ./benchmark-nn <problem size> <working set size> <stride> <jobid> "
                    "<num_samples> <salt>\n");
    const size_t problem_size     = (argc > 1) ? (size_t)atol(argv[1]) : 268435456;
    const size_t working_set_size = (argc > 2) ? (size_t)atol(argv[2]) : 3 * sizeof(real);
    const int stride              = (argc > 3) ? (size_t)atol(argv[3]) : 1;
    const size_t jobid            = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples      = (argc > 5) ? (size_t)atol(argv[5]) : 100;
    const size_t salt             = (argc > 6) ? (size_t)atol(argv[6]) : 42;

    // Derived values
    const int radius           = (((working_set_size / sizeof(real)) - 1) / 2) * stride;
    const size_t domain_length = problem_size / sizeof(real);
    const size_t seed          = 12345 + salt +
                        (1 + problem_size + working_set_size + stride + jobid + num_samples) *
                            time(NULL);
    ERRCHK((2 * radius / stride + 1) * sizeof(real) == working_set_size);
    ERRCHK(domain_length * sizeof(real) == problem_size);

    printf("Input parameters:\n");
    printf("\tproblem_size: %zu\n", problem_size);
    // printf("\tpad: %zu\n", pad);
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

    printDeviceInfo(0);
    // printf("USE_SMEM=%d\n", USE_SMEM);
    // printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);
    printf("DOUBLE_PRECISION=%u\n", DOUBLE_PRECISION);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Random
    srand(seed);

    // Benchmark pipeline
    // KernelConfig c = autotune(array_length, domain_length, pad, radius, stride);
    ERRCHK_ALWAYS(verify(domain_length, radius, stride));
    cudaProfilerStart();
    // benchmark(c, jobid, seed, num_samples);
    cudaProfilerStop();
    return EXIT_SUCCESS;
}