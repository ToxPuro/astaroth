#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "timer_hires.h"

int
main(int argc, char* argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: ./heat <nx> <ny> <nz> <jobid>\n");
        fprintf(stderr, "       ./heat 0 0 0 0 # To use the defaults\n");
        return EXIT_FAILURE;
    }

    const size_t arg1 = (size_t)atol(argv[1]);
    const size_t arg2 = (size_t)atol(argv[2]);
    const size_t arg3 = (size_t)atol(argv[3]);
    const size_t arg4 = (size_t)atol(argv[4]);

    // Input values
    const size_t nx    = arg1 ? arg1 : 256;
    const size_t ny    = arg2 ? arg2 : 256;
    const size_t nz    = arg3 ? arg3 : 256;
    const size_t jobid = arg4 ? arg4 : 0;

    // Derived values
    const size_t seed = 12345 + time(NULL) + jobid * time(NULL);

    // Print basic information
    printf("jobid: %zu\n", jobid);
    printf("seed: %zu\n", seed);

    AcMeshInfo info;
    acSetMeshDims(nx, ny, nz, &info);

    Device device;
    acDeviceCreate(0, info, &device);

    const AcReal dsx = 2 * AC_REAL_PI / nx;
    const AcReal dsy = 2 * AC_REAL_PI / ny;
    const AcReal dsz = 2 * AC_REAL_PI / nz;
    // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dx, 2 * AC_REAL_PI / nx);
    // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dy, 2 * AC_REAL_PI / ny);
    // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dz, 2 * AC_REAL_PI / nz);

    ERRCHK_ALWAYS(NUM_STENCILS == 1);
    ERRCHK_ALWAYS(STENCIL_DEPTH == STENCIL_HEIGHT && STENCIL_HEIGHT == STENCIL_WIDTH);

    const size_t mid = (STENCIL_WIDTH - 1) / 2;
    AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] = {{{{0}}}};
    for (size_t i = 0; i < STENCIL_WIDTH; ++i) {
        const AcReal dt       = (AcReal)1e-3;
        const AcReal coeffs[] = {1 / 90., -3 / 20., 3 / 2., -49 / 18., 3 / 2., -3 / 20., 1 / 90.};
        stencils[0][mid][mid][i] += (1.0 / dsx) * (1.0 / dsx) * coeffs[i] * dt;
        stencils[0][mid][i][mid] += (1.0 / dsy) * (1.0 / dsy) * coeffs[i] * dt;
        stencils[0][i][mid][mid] += (1.0 / dsz) * (1.0 / dsz) * coeffs[i] * dt;
    }
    stencils[0][mid][mid][mid] += (AcReal)1.0;
    acGridLoadStencils(STREAM_DEFAULT, stencils);

    AcMeshDims dims = acGetMeshDims(info);
    acPrintIntParams(AC_mx, AC_my, AC_mz, info);
    acPrintIntParams(AC_nx, AC_ny, AC_nz, info);

    // Init & dryrun
    const size_t pid   = 0;
    const size_t count = acVertexBufferSize(info);
    acRandInitAlt(1234UL, count, pid);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, init, dims.n0, dims.n1);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);
    // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dt, 1e-3);

    // File
    const size_t buflen = 4096;
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "heat-benchmark-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp, "implementation,maxthreadsperblock,nx,ny,nz,milliseconds,jobid,seed\n");

    acDeviceSynchronizeStream(device, STREAM_ALL);
    Timer t;
    const size_t num_iters = 100;
    for (size_t i = 0; i < num_iters; ++i) {
        acDeviceSynchronizeStream(device, STREAM_ALL);
        timer_reset(&t);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const double milliseconds = timer_diff_nsec(t) / 1e6;

        ERRCHK_CUDA_KERNEL_ALWAYS();
        fprintf(fp, "%d,%d,%zu,%zu,%zu,%g,%zu,%zu\n", IMPLEMENTATION, MAX_THREADS_PER_BLOCK, nx, ny,
                nz, milliseconds, jobid, seed);
    }
    const double elems_per_second = (nx * ny * nz) / (1e-9 * timer_diff_nsec(t));
    timer_diff_print(t);
    printf("%g M elements per second\n", elems_per_second / 1e6);

    // Deallocate resources
    fclose(fp);
    acRandQuit();
    acDeviceDestroy(device);

    return EXIT_SUCCESS;
}