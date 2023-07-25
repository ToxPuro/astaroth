#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

#if AC_DOUBLE_PRECISION
#define DOUBLE_PRECISION (1)
#else
#define DOUBLE_PRECISION (0)
#endif

#ifdef AC_INTEGRATION_ENABLED
int
main(int argc, char** argv)
{
    cudaProfilerStop();

    fprintf(stderr, "Usage: ./benchmark-device <nx> <ny> <nz> <jobid> <num_samples> <verify> <salt>\n");
    const size_t nx          = (argc > 1) ? (size_t)atol(argv[1]) : 256;
    const size_t ny          = (argc > 2) ? (size_t)atol(argv[2]) : 256;
    const size_t nz          = (argc > 3) ? (size_t)atol(argv[3]) : 256;
    const size_t jobid       = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples = (argc > 5) ? (size_t)atol(argv[5]) : 100;
    const size_t verify      = (argc > 6) ? (size_t)atol(argv[6]) : 0;
    const size_t salt        = (argc > 7) ? (size_t)atol(argv[7]) : 42;
    const size_t seed        = 12345 + salt + (1 + nx + ny + nz + jobid + num_samples + verify) * time(NULL);

    printf("Input parameters:\n");
    printf("\tnx: %zu\n", nx);
    printf("\tny: %zu\n", ny);
    printf("\tnz: %zu\n", nz);
    printf("\tjobid: %zu\n", jobid);
    printf("\tnum_samples: %zu\n", num_samples);
    printf("\tverify: %zu\n", verify);
    printf("\tseed: %zu\n", seed);

    printf("IMPLEMENTATION=%d\n", IMPLEMENTATION);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);
    printf("DOUBLE_PRECISION=%u\n", DOUBLE_PRECISION);
    fflush(stdout);

    // Mesh configuration
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(nx, ny, nz, &info);
    acPrintMeshInfo(info);

    // Mesh dimensions
    const AcMeshDims dims = acGetMeshDims(info);

    // Simulation parameters
    const AcReal dt = (AcReal)FLT_EPSILON;

    // Host memory
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    // Device memory
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    // Random numbers
    const size_t pid   = 0;
    const size_t count = acVertexBufferCompdomainSize(info);
    acRandInitAlt(seed, count, pid);
    srand(seed);

    // Verify
    if (verify) {
        // Dryrun and autotune
        for (int i = 0; i < 3; ++i)
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, dims.n0, dims.n1, dt);
        acDeviceResetMesh(device, STREAM_DEFAULT);

        // Verify: load/store
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(device, STREAM_DEFAULT, model);
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acVerifyMesh("Load/Store", model, candidate);

        // Verify: boundconds
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(device, STREAM_DEFAULT, model);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acHostMeshApplyPeriodicBounds(&model);
        acVerifyMesh("Boundconds", model, candidate);

        // Verify: integration
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(device, STREAM_DEFAULT, model);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

        const size_t num_verification_steps = 5;
        for (size_t j = 0; j < num_verification_steps; ++j) {
            for (int i = 0; i < 3; ++i) {
                acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, dims.n0, dims.n1, dt);
                acDeviceSwapBuffers(device);
                acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
            }
            acHostIntegrateStep(model, dt);
            acHostMeshApplyPeriodicBounds(&model);
        }
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acVerifyMesh("Integration", model, candidate);
    }

    // File
    const size_t buflen = 4096;
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "benchmark-device-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp, "implementation,maxthreadsperblock,nx,ny,nz,milliseconds,tpbx,tpby,tpbz,jobid,seed,"
                "iteration,double_precision\n");

    // Benchmark configuration
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, dt);
    acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, 2);

    // Benchmark
    Timer t;
    for (size_t j = 0; j < num_samples; ++j) {
        // Dryrun and randomize
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1);
        acDeviceResetMesh(device, STREAM_DEFAULT);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL);

        // Benchmark
        timer_reset(&t);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1);
        // acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 2, dims.n0, dims.n1, dt);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const double milliseconds = timer_diff_nsec(t) / 1e6;

        const Volume tpb = acKernelLaunchGetLastTPB();
        fprintf(fp, "%d,%d,%zu,%zu,%zu,%g,%zu,%zu,%zu,%zu,%zu,%zu,%u\n", IMPLEMENTATION,
                MAX_THREADS_PER_BLOCK, nx, ny, nz, milliseconds, tpb.x, tpb.y, tpb.z, jobid, seed,
                j, DOUBLE_PRECISION);

        if (j == num_samples - 1) {
            fprintf(stdout, "implementation,maxthreadsperblock,nx,ny,nz,milliseconds,tpbx,tpby,"
                            "tpbz,jobid,seed,"
                            "iteration,double_precision\n");
            fprintf(stdout, "%d,%d,%zu,%zu,%zu,%g,%zu,%zu,%zu,%zu,%zu,%zu,%u\n", IMPLEMENTATION,
                    MAX_THREADS_PER_BLOCK, nx, ny, nz, milliseconds, tpb.x, tpb.y, tpb.z, jobid,
                    seed, j, DOUBLE_PRECISION);
            printf("Milliseconds per kernel launch: %g\n", milliseconds);
            printf("Optimal tpb: (%zu, %zu, %zu)\n", tpb.x, tpb.y, tpb.z);
        }
    }

    // Free
    fclose(fp);
    acDeviceDestroy(device);
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
#else
int
main(void)
{
    fprintf(stderr, "AC_INTEGRATION was not enabled, cannot run benchmark-device\n");
    return EXIT_FAILURE;
}
#endif