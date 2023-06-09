#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

// Simulation parameters
static const AcReal DT = (AcReal)FLT_EPSILON;

// Finite-difference coefficients
static const AcReal coeffs0[] = {0.};
const AcReal coeffs2[]        = {1., -2., 1.};
const AcReal coeffs4[]        = {-1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12};
const AcReal coeffs6[]        = {1. / 90, -3. / 20, 3. / 2, -49. / 18, 3. / 2, -3. / 20, 1. / 90};
const AcReal coeffs8[]        = {
    -1. / 560, 8. / 315, -1. / 5, 8. / 5, -205. / 72, 8. / 5, -1. / 5, 8. / 315, -1. / 560,
};
static const AcReal* coeff_list[] = {coeffs0, coeffs2, coeffs4, coeffs6, coeffs8};

#define ARRAY_SIZE(x) (sizeof(x) / (sizeof(x[0])))

static void
acHostSolveHeat(AcMesh mesh, const size_t radius)
{
    AcMesh out;
    acHostMeshCreate(mesh.info, &out);

    const AcMeshDims domain = acGetMeshDims(mesh.info); // Is initialized to zero with calloc
    const size_t field      = 0;

    // Stencil loop
    for (size_t k = as_size_t(domain.n0.z); k < as_size_t(domain.n1.z); ++k) {
        for (size_t j = as_size_t(domain.n0.y); j < as_size_t(domain.n1.y); ++j) {
            for (size_t i = as_size_t(domain.n0.x); i < as_size_t(domain.n1.x); ++i) {
                const size_t idx = acVertexBufferIdx(i, j, k, mesh.info);

                const AcReal dsx = 2 * AC_REAL_PI / domain.nn.x;
                const AcReal dsy = 2 * AC_REAL_PI / domain.nn.y;
                const AcReal dsz = 2 * AC_REAL_PI / domain.nn.z;

                ERRCHK_ALWAYS(radius < ARRAY_SIZE(coeff_list));
                const AcReal* coeffs = coeff_list[radius];

                long double tmp = mesh.vertex_buffer[field][idx];
                if (domain.nn.x > 1) {
                    for (int r = -(int)radius; r <= (int)radius; ++r)
                        tmp += DT * (1 / (dsx * dsx)) * coeffs[radius + r] *
                               mesh.vertex_buffer[field][acVertexBufferIdx(i + r, j, k, mesh.info)];
                }
                if (domain.nn.y > 1) {
                    for (int r = -(int)radius; r <= (int)radius; ++r)
                        tmp += DT * (1 / (dsy * dsy)) * coeffs[radius + r] *
                               mesh.vertex_buffer[field][acVertexBufferIdx(i, j + r, k, mesh.info)];
                }
                if (domain.nn.z > 1) {
                    for (int r = -(int)radius; r <= (int)radius; ++r)
                        tmp += DT * (1 / (dsz * dsz)) * coeffs[radius + r] *
                               mesh.vertex_buffer[field][acVertexBufferIdx(i, j, k + r, mesh.info)];
                }
                out.vertex_buffer[field][idx] = tmp;
            }
        }
    }

    // Copy results to the initial buffer
    for (size_t k = as_size_t(domain.n0.z); k < as_size_t(domain.n1.z); ++k) {
        for (size_t j = as_size_t(domain.n0.y); j < as_size_t(domain.n1.y); ++j) {
            for (size_t i = as_size_t(domain.n0.x); i < as_size_t(domain.n1.x); ++i) {
                const size_t idx               = acVertexBufferIdx(i, j, k, mesh.info);
                mesh.vertex_buffer[field][idx] = out.vertex_buffer[field][idx];
            }
        }
    }

    acHostMeshDestroy(&out);
}

static Stencil
acGetStencilByName(const char* name)
{
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        if (!strcmp(stencil_names[i], name))
            return (Stencil)i;
    fprintf(stderr, "acGetStencilByName failed: no stencil named '%s' found\n", name);
    return NUM_STENCILS;
}

static void
load_stencil_coefficients(const Device device, const AcMeshDims dims, const size_t radius)
{
    ERRCHK_ALWAYS(STENCIL_DEPTH == STENCIL_HEIGHT && STENCIL_HEIGHT == STENCIL_WIDTH);

    const size_t mid = (STENCIL_WIDTH - 1) / 2;
    const AcReal dsx = 2 * AC_REAL_PI / dims.nn.x;
    const AcReal dsy = 2 * AC_REAL_PI / dims.nn.y;
    const AcReal dsz = 2 * AC_REAL_PI / dims.nn.z;

    ERRCHK_ALWAYS(radius < ARRAY_SIZE(coeff_list));
    const AcReal* coeffs = coeff_list[radius];

    Stencil stencil_list[3] = {0};
    for (size_t dim = 0; dim < 3; ++dim) {
        const size_t ord = 2 * radius;

        const size_t buflen = 4096;
        char stencil_name[buflen];
        snprintf(stencil_name, buflen, "heat%zud_ord%zu", dim + 1, ord);
        stencil_list[dim] = acGetStencilByName(stencil_name);
    }
    const Stencil stencil_heat1d = stencil_list[0];
    const Stencil stencil_heat2d = stencil_list[1];
    const Stencil stencil_heat3d = stencil_list[2];

    AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] = {{{{0}}}};
    for (int i = -(int)radius; i <= (int)radius; ++i) {
        // 1D
        stencils[stencil_heat1d][mid][mid][mid + i] += (1.0 / dsx) * (1.0 / dsx) *
                                                       coeffs[radius + i] * DT;

        // 2D
        stencils[stencil_heat2d][mid][mid][mid + i] += (1.0 / dsx) * (1.0 / dsx) *
                                                       coeffs[radius + i] * DT;
        stencils[stencil_heat2d][mid][mid + i][mid] += (1.0 / dsy) * (1.0 / dsy) *
                                                       coeffs[radius + i] * DT;

        // 3D
        stencils[stencil_heat3d][mid][mid][mid + i] += (1.0 / dsx) * (1.0 / dsx) *
                                                       coeffs[radius + i] * DT;
        stencils[stencil_heat3d][mid][mid + i][mid] += (1.0 / dsy) * (1.0 / dsy) *
                                                       coeffs[radius + i] * DT;
        stencils[stencil_heat3d][mid + i][mid][mid] += (1.0 / dsz) * (1.0 / dsz) *
                                                       coeffs[radius + i] * DT;
    }

    stencils[stencil_heat1d][mid][mid][mid] += (AcReal)1.0;
    stencils[stencil_heat2d][mid][mid][mid] += (AcReal)1.0;
    stencils[stencil_heat3d][mid][mid][mid] += (AcReal)1.0;
    acDeviceLoadStencils(device, STREAM_DEFAULT, stencils);
}

int
main(int argc, char** argv)
{
    cudaProfilerStop();

    fprintf(stderr,
            "Usage: ./heat-equation <nx> <ny> <nz> <jobid> <num_samples> <verify> <radius>\n");
    const size_t nx          = (argc > 1) ? (size_t)atol(argv[1]) : 256;
    const size_t ny          = (argc > 2) ? (size_t)atol(argv[2]) : 256;
    const size_t nz          = (argc > 3) ? (size_t)atol(argv[3]) : 256;
    const size_t jobid       = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples = (argc > 5) ? (size_t)atol(argv[5]) : 100;
    const size_t verify      = (argc > 6) ? (size_t)atol(argv[6]) : 0;
    const size_t radius      = (argc > 7) ? (size_t)atol(argv[7]) : STENCIL_ORDER / 2;
    const size_t seed        = 12345 + time(NULL) + jobid * time(NULL);
    // const Kernel kernel      = (nz > 1) ? solve3d : (ny > 1) ? solve2d : solve1d;
    // const size_t kernel_id   = acGetKernelId(kernel);
    // const char* kernel_name  = kernel_names[kernel_id];
    const size_t num_dims = (nz > 1) ? 3 : (ny > 1) ? 2 : 1;
    const size_t buflen   = 4096;
    char kernel_name[buflen];
    snprintf(kernel_name, buflen, "solve%zud_ord%zu", num_dims, 2 * radius);
    const size_t kernel_id = acGetKernelIdByName(kernel_name);
    const Kernel kernel    = kernels[kernel_id];

    // Input parameters
    printf("Input parameters:\n");
    printf("\tnx: %zu\n", nx);
    printf("\tny: %zu\n", ny);
    printf("\tnz: %zu\n", nz);
    printf("\tjobid: %zu\n", jobid);
    printf("\tnum_samples: %zu\n", num_samples);
    printf("\tverify: %zu\n", verify);
    printf("\tradius: %zu\n", radius);

    printf("IMPLEMENTATION=%d\n", IMPLEMENTATION);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);
    printf("STENCIL_ORDER=%d\n", STENCIL_ORDER);

    // Derived parameters
    printf("Derived parameters:\n");
    printf("\tseed: %zu\n", seed);
    printf("\tkernel: %s (kernels[%zu]: %p)\n", kernel_name, kernel_id, kernel);

    fflush(stdout);

    // Mesh configuration
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(nx, ny, nz, &info);
    acPrintMeshInfo(info);

    // Mesh dimensions
    const AcMeshDims dims = acGetMeshDims(info);

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
    const size_t count = acVertexBufferSize(info);
    acRandInitAlt(seed, count, pid);
    srand(seed);

    // Benchmark configuration
    load_stencil_coefficients(device, dims, radius);

    // Verify
    if (verify) {
        // Dryrun and autotune
        // for (size_t i = 0; i < NUM_KERNELS; ++i)
        //    acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1);
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
        acHostMeshApplyConstantBounds((AcReal)0.0, &model);
        acDeviceLoadMesh(device, STREAM_DEFAULT, model);
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acVerifyMesh("Boundconds", model, candidate);

        // Verify: integration
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);

        acHostMeshApplyConstantBounds((AcReal)0.0, &model);
        acDeviceLoadMesh(device, STREAM_DEFAULT, model);

        const size_t num_verification_steps = 3;
        for (size_t j = 0; j < num_verification_steps; ++j) {

            acDeviceFlushOutputBuffers(device, STREAM_DEFAULT);
            acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1);
            acDeviceSwapBuffers(device);

            acHostSolveHeat(model, radius);
        }
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acVerifyMesh("Kernel", model, candidate);
    }

    // File
    char benchmark_dir[buflen];
    snprintf(benchmark_dir, buflen, "heat-equation-%zu-%zu.csv", jobid, seed);
    FILE* fp = fopen(benchmark_dir, "w");
    ERRCHK_ALWAYS(fp);

    // File format
    fprintf(fp,
            "kernel,implementation,maxthreadsperblock,nx,ny,nz,radius,milliseconds,tpbx,tpby,tpbz,"
            "jobid,seed,"
            "iteration\n");

    // Benchmark
    Timer t;
    for (size_t j = 0; j < num_samples; ++j) {
        // Dryrun and randomize
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1);
        acDeviceResetMesh(device, STREAM_DEFAULT);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL);

        // Benchmark
        timer_reset(&t);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const double milliseconds = timer_diff_nsec(t) / 1e6;

        acDeviceBenchmarkKernel(device, kernel, dims.n0, dims.n1);

        const Volume tpb = acKernelLaunchGetLastTPB();
        fprintf(fp, "%s,%d,%d,%zu,%zu,%zu,%zu,%g,%zu,%zu,%zu,%zu,%zu,%zu\n", kernel_name,
                IMPLEMENTATION, MAX_THREADS_PER_BLOCK, nx, ny, nz, radius, milliseconds, tpb.x,
                tpb.y, tpb.z, jobid, seed, j);

        if (j == num_samples - 1) {
            fprintf(stdout, "kernel,implementation,maxthreadsperblock,nx,ny,nz,radius,"
                            "milliseconds,tpbx,tpby,"
                            "tpbz,jobid,seed,"
                            "iteration\n");
            fprintf(stdout, "%s,%d,%d,%zu,%zu,%zu,%zu,%g,%zu,%zu,%zu,%zu,%zu,%zu\n", kernel_name,
                    IMPLEMENTATION, MAX_THREADS_PER_BLOCK, nx, ny, nz, radius, milliseconds, tpb.x,
                    tpb.y, tpb.z, jobid, seed, j);
            printf("Milliseconds per kernel '%s' launch: %g\n", kernel_name, milliseconds);
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