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

    // Set mesh dimensions
    if (argc != 3) {
        fprintf(stderr, "Usage: ./blur <nx> <ny>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = 1;
        acHostUpdateBuiltinParams(&info);
    }

    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh);

    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);

    // Benchmark
    Timer t;
    timer_reset(&t);

    const int3 nn_min = (int3){info.int_params[AC_nx_min], info.int_params[AC_ny_min],
                               info.int_params[AC_nz_min]};
    const int3 nn_max = (int3){info.int_params[AC_nx_max], info.int_params[AC_ny_max],
                               info.int_params[AC_nz_max]};
    const int3 start  = nn_min;
    const int3 end    = nn_max;
    acDeviceLaunchKernel(device, STREAM_DEFAULT, blur_kernel, start, end);
    acDeviceSwapBuffers(device);

    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);

    printf("Store done\n");

    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);

    return EXIT_SUCCESS;
}
