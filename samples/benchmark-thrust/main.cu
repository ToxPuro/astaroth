#include <stdio.h>
#include <stdlib.h>

#include "acc_runtime.h"
#include "timer_hires.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "astaroth_utils.h"

template <typename T> struct square {
    __host__ __device__ T operator()(const T& x) const { return x * x; }
};

static void
benchmark_thrust(const size_t mx, const size_t my, const size_t mz)
{
    std::vector<thrust::device_vector<double>> inputs(mz);
    for (size_t k = 0; k < mz; ++k)
        inputs[k] = thrust::device_vector<double>(mx * my, 1);

    thrust::device_vector<double> results(mz);

    for (size_t k = 0; k < mz; ++k)
        results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());

    Timer t;
    cudaDeviceSynchronize();
    timer_reset(&t);
    for (size_t k = 0; k < mz; ++k)
        results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());
    cudaDeviceSynchronize();
    timer_diff_print(t);
}

int
main(int argc, char** argv)
{
    cudaProfilerStop();

    fprintf(stderr, "Usage: ./benchmark-thrust <nx> <ny> <nz>\n");
    const size_t nx     = (argc > 1) ? (size_t)atol(argv[1]) : 32;
    const size_t ny     = (argc > 2) ? (size_t)atol(argv[2]) : 32;
    const size_t nz     = (argc > 3) ? (size_t)atol(argv[3]) : 32;
    const size_t radius = STENCIL_ORDER / 2;
    const size_t mx     = nx + 2 * radius;
    const size_t my     = ny + 2 * radius;
    const size_t mz     = nz + 2 * radius;

    printf("Input parameters:\n");
    printf("\tnx: %zu\n", nx);
    printf("\tny: %zu\n", ny);
    printf("\tnz: %zu\n", nz);
    printf("\tradius: %zu\n", radius);
    printf("\tmx: %zu\n", mx);
    printf("\tmy: %zu\n", my);
    printf("\tmz: %zu\n", mz);

    // Device
    cudaSetDevice(0);

#if 1
    cudaProfilerStart();
    benchmark_thrust(mx, my, mz);
    cudaProfilerStop();
#else
    // Mesh info
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(nx, ny, nz, &info);
    acLoadMeshInfo(info, 0);

    // Random
    const size_t seed  = 12345;
    const size_t pid   = 0;
    const size_t count = acVertexBufferCompdomainSize(info);
    acRandInitAlt(seed, count, pid);

    AcMeshDims dims       = acGetMeshDims(info);
    VertexBufferArray vba = acVBACreate(mx, my, mz);
    acVBAReset(0, &vba);
    acLaunchKernel(randomize, 0, dims.n0, dims.n1, vba);
    acVBASwapBuffers(&vba);

    ProfileBufferArray pba = acPBACreate(mz);
    AcBufferArray ba0      = acBufferArrayCreate(12, nx * ny);
    AcBufferArray ba1      = acBufferArrayCreate(12, nx * ny);
    AcBufferArray ba2      = acBufferArrayCreate(12, nx * ny);
    // acMapCross(vba, 0, (int3){radius, radius, 0}, (int3){radius + nx, radius + ny, 1},
    // scratchpad);
    acMapCrossReduce(vba, 0, ba0, ba1, ba2, pba);

    Timer t;

    cudaProfilerStart();
    cudaDeviceSynchronize();
    timer_reset(&t);
    // for (size_t k = 0; k < mz; ++k)
    // acMapCross(vba, 0, (int3){radius, radius, k}, (int3){radius + nx, radius + ny, k +
    // 1},
    //            scratchpad);
    acMapCrossReduce(vba, 0, ba0, ba1, ba2, pba);

    cudaDeviceSynchronize();
    timer_diff_print(t);
    cudaProfilerStop();
    cudaDeviceSynchronize();

    ERRCHK_CUDA_KERNEL_ALWAYS();
    acRandQuit();
    acBufferArrayDestroy(&ba2);
    acBufferArrayDestroy(&ba1);
    acBufferArrayDestroy(&ba0);
    acPBADestroy(&pba);
    acVBADestroy(&vba);
#endif
    return EXIT_SUCCESS;
}