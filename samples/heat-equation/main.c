#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "timer_hires.h"

static const size_t nx = 128;
static const size_t ny = 128;
static const size_t nz = 128;

int
main(void)
{
    AcMeshInfo info;
    acSetMeshDims(nx, ny, nz, &info);

    Device device;
    acDeviceCreate(0, info, &device);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dx, 2 * AC_REAL_PI / nx);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dy, 2 * AC_REAL_PI / ny);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dz, 2 * AC_REAL_PI / nz);

    AcMeshDims dims = acGetMeshDims(info);

    // Init & dryrun
    acDeviceLaunchKernel(device, STREAM_DEFAULT, init, dims.m0, dims.m1);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);

    acDeviceSynchronizeStream(device, STREAM_ALL);
    Timer t;
    timer_reset(&t);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);
    acDeviceSynchronizeStream(device, STREAM_ALL);
    timer_diff_print(t);

    acDeviceDestroy(device);

    return EXIT_SUCCESS;
}