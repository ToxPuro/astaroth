#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <cuda.h> // CUDA driver API
#include <cuda_runtime_api.h>

#include "timer_hires.h" // From src/common

#define BLOCK_SIZE (256 * 256 * 3 * 8 * 8)
#define NGPUS_PER_NODE (4)

#define errchk(x)                                                                                  \
    {                                                                                              \
        if (!(x)) {                                                                                \
            fprintf(stderr, "errchk(%s) failed", #x);                                              \
            assert(x);                                                                             \
        }                                                                                          \
    }

static uint8_t*
allocHost(const size_t bytes)
{
    uint8_t* arr = malloc(bytes);
    errchk(arr);
    return arr;
}

static void
freeHost(uint8_t* arr)
{
    free(arr);
}

static uint8_t*
allocDevice(const size_t bytes)
{
    uint8_t* arr;
    // Standard (20 GiB/s internode, 85 GiB/s intranode)
    const cudaError_t retval = cudaMalloc((void**)&arr, bytes);
    // Unified mem (5 GiB/s internode, 6 GiB/s intranode)
    // const cudaError_t retval = cudaMallocManaged((void**)&arr, bytes, cudaMemAttachGlobal);
    // Pinned (40 GiB/s internode, 10 GiB/s intranode)
    // const cudaError_t retval = cudaMallocHost((void**)&arr, bytes);
    errchk(retval == cudaSuccess);
    return arr;
}

static void
freeDevice(uint8_t* arr)
{
    cudaFree(arr);
}

static Timer
measure_start(void)
{
    Timer t;
    MPI_Barrier(MPI_COMM_WORLD);
    timer_reset(&t);
    MPI_Barrier(MPI_COMM_WORLD);
    return t;
}

static void
measure_end(const Timer t, const size_t bytes)
{
    MPI_Barrier(MPI_COMM_WORLD);
    const long double time_elapsed = timer_diff_nsec(t) / 1e9l; // seconds

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (!pid) {
        printf("%Lg GiB/s\n", bytes / time_elapsed / pow(1024, 3));
    }
}

static int
min(const int a, const int b)
{
    return a < b ? a : b;
}

int
main(void)
{
    MPI_Init(NULL, NULL);

    // Disable stdout buffering (no need to flush)
    setbuf(stdout, NULL);

    int pid, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    cudaSetDevice(pid % NGPUS_PER_NODE);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    uint8_t* hdata = allocHost(BLOCK_SIZE);
    uint8_t* ddata = allocDevice(BLOCK_SIZE);

    cudaStream_t d2h, h2d;
    cudaStreamCreate(&d2h);
    cudaStreamCreate(&h2d);

    const size_t nsamples = 1000;
    if (!pid)
        printf("D2H unidirectional: ");
    Timer t = measure_start();
    for (size_t i = 0; i < nsamples; ++i) {
        cudaMemcpyAsync(hdata, ddata, BLOCK_SIZE, cudaMemcpyDeviceToHost, d2h);
    }
    cudaStreamSynchronize(d2h);
    measure_end(t, nsamples * BLOCK_SIZE * min(nprocs, NGPUS_PER_NODE));

    if (!pid)
        printf("H2D unidirectional: ");
    t = measure_start();
    for (size_t i = 0; i < nsamples; ++i) {
        cudaMemcpyAsync(ddata, hdata, BLOCK_SIZE, cudaMemcpyHostToDevice, h2d);
    }
    cudaStreamSynchronize(h2d);
    measure_end(t, nsamples * BLOCK_SIZE * min(nprocs, NGPUS_PER_NODE));

    if (!pid)
        printf("D2H2D bidirectional: ");
    t = measure_start();
    for (size_t i = 0; i < nsamples; ++i) {
        cudaMemcpyAsync(ddata, hdata, BLOCK_SIZE, cudaMemcpyHostToDevice, h2d);
        cudaMemcpyAsync(hdata, ddata, BLOCK_SIZE, cudaMemcpyDeviceToHost, d2h);
    }
    cudaStreamSynchronize(d2h);
    cudaStreamSynchronize(h2d);
    measure_end(t, 2 * nsamples * BLOCK_SIZE * min(nprocs, NGPUS_PER_NODE));

    cudaStreamDestroy(d2h);
    cudaStreamDestroy(h2d);

    freeHost(hdata);
    freeDevice(hdata);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
