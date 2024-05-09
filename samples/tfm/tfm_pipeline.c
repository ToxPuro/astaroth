/*
    Copyright (C) 2024, Johannes Pekkila.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
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
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
static AcReal
max(const AcReal a, const AcReal b)
{
    return a > b ? a : b;
}

static AcReal
min(const AcReal a, const AcReal b)
{
    return a < b ? a : b;
}

static long double
maxl(const long double a, const long double b)
{
    return a > b ? a : b;
}

static long double
minl(const long double a, const long double b)
{
    return a < b ? a : b;
}

AcReal
calc_timestep(const Device device, const AcMeshInfo info)
{
    AcReal uumax     = 0.0;
    AcReal vAmax     = 0.0;
    AcReal shock_max = 0.0;
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                      &uumax);

    const long double cdt  = (long double)info.real_params[AC_cdt];
    const long double cdtv = (long double)info.real_params[AC_cdtv];
    // const long double cdts     = (long double)info.real_params[AC_cdts];
    const long double cs2_sound = (long double)info.real_params[AC_cs2_sound];
    const long double nu_visc   = (long double)info.real_params[AC_nu_visc];
    const long double eta       = (long double)info.real_params[AC_eta];
    const long double chi      = 0; // (long double)info.real_params[AC_chi]; // TODO not calculated
    const long double gamma    = (long double)info.real_params[AC_gamma];
    const long double dsmin    = (long double)min(info.real_params[AC_dsx],
                                               min(info.real_params[AC_dsy],
                                                   info.real_params[AC_dsz]));
    const long double nu_shock = (long double)info.real_params[AC_nu_shock];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt = cdt * dsmin /
                              (fabsl((long double)uumax) +
                               sqrtl(cs2_sound + (long double)vAmax * (long double)vAmax));
    const long double visc_dt = cdtv * dsmin * dsmin /
                                (maxl(maxl(nu_visc, eta), gamma * chi) +
                                 nu_shock * (long double)shock_max);

    const long double dt = minl(uu_dt, visc_dt);
    // ERRCHK_ALWAYS(is_valid((AcReal)dt));
    return (AcReal)(dt);
}

void
tfm_pipeline(const Device device, const AcMeshInfo info)
{
    const AcMeshDims dims = acGetMeshDims(info);
    const AcReal dt       = calc_timestep(device, info);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, dt);

    for (size_t step_number = 0; step_number < 3; ++step_number) {
        acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, step_number);

        // Compute: hydrodynamics
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1);

        // Boundary conditions: hydrodynamics
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_LNRHO, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUX, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUY, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUZ, dims.m0, dims.m1);

        // Profile averages
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUX, PROFILE_Umean_x);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUY, PROFILE_Umean_y);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUZ, PROFILE_Umean_z);

        // TODO proper \overline{u x b^{pq}} calculation
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUX, PROFILE_ucrossb11mean_x);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUY, PROFILE_ucrossb11mean_y);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUZ, PROFILE_ucrossb11mean_z);

        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUX, PROFILE_ucrossb12mean_x);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUY, PROFILE_ucrossb12mean_y);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUZ, PROFILE_ucrossb12mean_z);

        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUX, PROFILE_ucrossb21mean_x);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUY, PROFILE_ucrossb21mean_y);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUZ, PROFILE_ucrossb21mean_z);

        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUX, PROFILE_ucrossb22mean_x);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUY, PROFILE_ucrossb22mean_y);
        acDeviceReduceXYAverage(device, STREAM_DEFAULT, VTXBUF_UUZ, PROFILE_ucrossb22mean_z);

        // Compute: test fields
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b11, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b12, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b21, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b22, dims.n0, dims.n1);

        // Boundary conditions: test fields
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b11_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b11_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b11_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b12_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b12_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b12_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b21_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b21_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b21_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b22_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b22_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_b22_z, dims.m0, dims.m1);
    }
}

void
tfm_dryrun(const Device device, const AcMeshInfo info)
{
    const AcMeshDims dims = acGetMeshDims(info);
    tfm_pipeline(device, info);
    acDeviceSynchronizeStream(device, STREAM_ALL);

    acDeviceResetMesh(device, STREAM_DEFAULT);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, init_profiles, dims.m0, dims.m1);
    acDeviceSwapBuffers(device);
    acDeviceSynchronizeStream(device, STREAM_ALL);
}

int
main(int argc, char** argv)
{
    cudaProfilerStop();

    fprintf(stderr,
            "Usage: ./benchmark-device <nx> <ny> <nz> <jobid> <num_samples> <verify> <salt>\n");
    const size_t nx          = (argc > 1) ? (size_t)atol(argv[1]) : 32;
    const size_t ny          = (argc > 2) ? (size_t)atol(argv[2]) : 32;
    const size_t nz          = (argc > 3) ? (size_t)atol(argv[3]) : 32;
    const size_t jobid       = (argc > 4) ? (size_t)atol(argv[4]) : 0;
    const size_t num_samples = (argc > 5) ? (size_t)atol(argv[5]) : 100;
    const size_t verify      = (argc > 6) ? (size_t)atol(argv[6]) : 0;
    const size_t salt        = (argc > 7) ? (size_t)atol(argv[7]) : 42;
    const size_t seed        = 12345 + salt +
                        (1 + nx + ny + nz + jobid + num_samples + verify) * time(NULL);

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
    fprintf(fp, "kernel,implementation,maxthreadsperblock,nx,ny,nz,milliseconds,"
                "jobid,seed,"
                "iteration,double_precision\n");

    // Warmup
    tfm_dryrun(device, info);

    // Benchmark
    Timer t;
    for (size_t j = 0; j < num_samples; ++j) {
        // Benchmark
        timer_reset(&t);
        tfm_pipeline(device, info);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const double milliseconds = timer_diff_nsec(t) / 1e6;

        fprintf(fp, "%s,%d,%d,%zu,%zu,%zu,%g,%zu,%zu,%zu,%u\n", "tfm-pipeline", IMPLEMENTATION,
                MAX_THREADS_PER_BLOCK, nx, ny, nz, milliseconds, jobid, seed, j, DOUBLE_PRECISION);

        if (j == num_samples - 1) {
            fprintf(stdout,
                    "kernel,implementation,maxthreadsperblock,nx,ny,nz,milliseconds,jobid,seed,"
                    "iteration,double_precision\n");
            fprintf(stdout, "%s,%d,%d,%zu,%zu,%zu,%g,%zu,%zu,%zu,%u\n", "tfm-pipeline",
                    IMPLEMENTATION, MAX_THREADS_PER_BLOCK, nx, ny, nz, milliseconds, jobid, seed, j,
                    DOUBLE_PRECISION);
            printf("Milliseconds per kernel launch: %g\n", milliseconds);
        }
    }
    // Profile
    cudaProfilerStart();
    tfm_pipeline(device, info);
    acDeviceSynchronizeStream(device, STREAM_ALL);
    cudaProfilerStop();

    acDeviceTest(device);

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