/**
    Microbenchmark the GPU caches in 1D stencil computations and generate a plottable .csv output

    Examples:
        # Usage (see the main invocation for an up-to-date list of input parameters)
        ./microbenchmark-nn <computational domain length in elements> <stencil radius in elements>

        # Default problem size and radius
        ./microbenchmark-nn

        # Bandwidth test with r=0 stencil
        ./microbenchmark-nn 33554432 0

        # Profiling
        cmake -DUSE_HIP=ON .. &&\
        make -j &&\
        rocprof --trace-start off -i ~/rocprof-input-metrics.txt ./microbenchmark-nn

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
#include <roctracer_ext.h>   // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // cudaStream_t
#endif

#include "array.h"
#include "backend.h"
#include "timer_hires.h"

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
        host_input.data[i] = (real)rand() / (real)RAND_MAX;

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
    backendInit(domain_length, radius, stride);
    Array input  = backendGetInputTensor();
    Array output = backendGetOutputTensor();

    // Randomize
    arrayRandomize(&input);
    arrayRandomize(&output);

    // Dryrun
    for (size_t i = 0; i < 10; ++i)
        backendConvolutionFwd();
    cudaDeviceSynchronize();
    ERRCHK_CUDA_KERNEL_ALWAYS();

    // File
    const size_t buflen = 4096;
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "microbenchmark-nn-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp, "implementation,maxthreadsperblock,domainlength,radius,stride,milliseconds,"
                "effectivebandwidth,tpb,jobid,seed,iteration,double_precision\n");

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
        backendConvolutionFwd();
        //cudaEventRecord(tstop); // Timing stop
        //cudaEventSynchronize(tstop);
	cudaDeviceSynchronize();
	const long double milliseconds = timer_diff_nsec(t)/1e6l;
        ERRCHK_CUDA_KERNEL_ALWAYS();

        //float milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, tstart, tstop);

        // Derived statistics
        const size_t bytes     = sizeof(input.data[0]) * (2 * domain_length + 2 * radius / stride);
        const long double seconds = (long double)milliseconds / 1e3l;
        const long double bandwidth = bytes / seconds;

        if (i == num_samples - 1) {
		printf("Effective bandwidth: %Lg GiB/s\n", bandwidth / pow(1024, 3));
		printf("\tBytes transferred: %Lg GiB\n", (long double)bytes / pow(1024, 3));
		//printf("\tTime elapsed: %Lg ms (CUDA)\n", (long double)milliseconds);
		printf("\tTime elapsed: %Lg ms (POSIX)\n", milliseconds);
	}
		

        // Write to file
        fprintf(fp, "%s,%d,%zu,%zu,%zu,%Lg,%Lg,%d,%zu,%zu,%zu,%u\n",
#if AC_USE_HIP
                "\"miopen\"",
#else
                "\"cudnn\"",
#endif
                -1, domain_length, radius, stride, milliseconds, bandwidth, -1, jobid, seed,
                i, DOUBLE_PRECISION);
    }
    //cudaEventDestroy(tstart);
    //cudaEventDestroy(tstop);
    cudaProfilerStop();

    // Free
    fclose(fp);
    backendQuit();
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
    const size_t seed = 12345 + salt +
                        (1 + domain_length + radius + stride + jobid + num_samples) * time(NULL);

    printf("Input parameters:\n");
    printf("\tdomain_length: %zu\n", domain_length);
    printf("\tradius: %d\n", radius);
    printf("\tstride: %d\n", stride);
    printf("\tjobid: %zu\n", jobid);
    printf("\tnum_samples: %zu\n", num_samples);
    printf("\tseed: %zu\n", seed);
    printf("\tsizeof(real): %zu\n", sizeof(real));
    fflush(stdout);

    // if (working_set_size > domain_length) {
    //     fprintf(stderr, "Invalid working set size: %lu > %lu\n", working_set_size, problem_size);
    //     return EXIT_FAILURE;
    // }

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
    benchmark(domain_length, radius, stride, jobid, seed, num_samples);
    return EXIT_SUCCESS;
}
