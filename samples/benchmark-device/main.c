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
    acInit(info);
    acLoad(model);
    acStore(&candidate);
    acVerifyMesh("Load/Store", model, candidate);

    // Verify that boundconds work correctly
    acHostMeshRandomize(&model);
    acLoad(model);
    acHostMeshApplyPeriodicBounds(&model);

    acBoundcondStep();
    acStore(&candidate);
    acVerifyMesh("Boundconds", model, candidate);

    // Verify that integration works correctly
    const AcReal dt = FLT_EPSILON;

    acHostMeshRandomize(&model);
    acLoad(model);
    acHostIntegrateStep(model, dt);
    acHostMeshApplyPeriodicBounds(&model);

    acIntegrate(dt);
    acBoundcondStep();
    acStore(&candidate);
    acVerifyMesh("Integration", model, candidate);

    // Warmup
    for (int i = 0; i < NSAMPLES / 10; ++i)
        acIntegrate(dt);
    acSynchronize();

    // Benchmark
    Timer t;
    timer_reset(&t);
    for (int i = 0; i < NSAMPLES; ++i)
        acIntegrate(dt);
    acSynchronize();
    const double ms_elapsed = timer_diff_nsec(t) / 1e6;
    printf("Average integration time: %.4g ms\n", ms_elapsed / NSAMPLES);

    // Destroy
    acQuit();
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
