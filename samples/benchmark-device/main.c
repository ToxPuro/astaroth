#include <float.h> // FLT_EPSILON
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

#define NSAMPLES (100)

int
main(int argc, char** argv)
{
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Set mesh dimensions
    if (argc != 4) {
        fprintf(stderr, "Usage: ./benchmark-device <nx> <ny> <nz>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = atoi(argv[3]);
        acHostUpdateBuiltinParams(&info);
    }

    // Alloc
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    // Init
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);
    acHostMeshApplyPeriodicBounds(&model);

    // Verify that the mesh was loaded and stored correctly
    Device device;
    acDeviceCreate(0, info, &device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acVerifyMesh("Load/Store", model, candidate);

    // Verify that boundconds work correctly
    const int3 m_min = (int3){0, 0, 0};
    const int3 m_max = (int3){
        info.int_params[AC_mx],
        info.int_params[AC_my],
        info.int_params[AC_mz],
    };
    const int3 n_min = (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2};
    const int3 n_max = (int3){
        n_min.x + info.int_params[AC_nx],
        n_min.y + info.int_params[AC_ny],
        n_min.z + info.int_params[AC_nz],
    };
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    acHostMeshApplyPeriodicBounds(&model);
    acVerifyMesh("Boundconds", model, candidate);

    // Verify that integration works correctly
    const AcReal dt = FLT_EPSILON;
    for (int i = 0; i < 3; ++i) {
        acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, n_min, n_max, dt);
        acDeviceSwapBuffers(device);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
    }
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    acHostIntegrateStep(model, dt);
    acHostMeshApplyPeriodicBounds(&model);
    acVerifyMesh("Integration", model, candidate);

    // Warmup
    for (int j = 0; j < NSAMPLES / 10; ++j) {
        for (int step = 0; step < 3; ++step) {
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, step, n_min, n_max, dt);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
        }
    }
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    // Benchmark
    Timer t;
    timer_reset(&t);
    for (int i = 0; i < NSAMPLES; ++i) {
        for (int step = 0; step < 3; ++step) {
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, step, n_min, n_max, dt);
            acDeviceSwapBuffers(device);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
        }
    }
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    const double ms_elapsed = timer_diff_nsec(t) / 1e6;
    printf("Average integration time: %.4g ms\n", ms_elapsed / NSAMPLES);

    // Destroy
    acDeviceDestroy(device);
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
