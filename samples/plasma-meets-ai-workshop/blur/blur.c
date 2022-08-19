#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

typedef struct {
    int3 n0, n1;
    int3 m0, m1;
} AcMeshDims;

static AcMeshDims
acGetAcMeshDims(const AcMeshInfo info)
{
    const int3 n0 = (int3){
        info.int_params[AC_nx_min],
        info.int_params[AC_ny_min],
        info.int_params[AC_nz_min],
    };
    const int3 n1 = (int3){
        info.int_params[AC_nx_max],
        info.int_params[AC_ny_max],
        info.int_params[AC_nz_max],
    };
    const int3 m0 = (int3){0, 0, 0};
    const int3 m1 = (int3){
        info.int_params[AC_mx],
        info.int_params[AC_my],
        info.int_params[AC_mz],
    };

    return (AcMeshDims){
        .n0 = n0,
        .n1 = n1,
        .m0 = m0,
        .m1 = m1,
    };
}

int
main(void)
{
    // Setup the mesh configuration
    AcMeshInfo info;
    acLoadConfig("../plasma-meets-ai-workshop/astaroth.conf");

    // Allocate memory on the GPU
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    const AcMeshDims dims = acGetMeshDims(info);

    // Setup initial conditions
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh) acHostMeshSet(1, &mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

    // Compute
    // acDeviceLaunchKernel(device, STREAM_DEFAULT, ..., dims.n0, dims.n1);
    acDeviceSwapBuffers(device);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    acDeviceSynchronizeStream(STREAM_DEFAULT);

    // Store to host memory and write to a file
    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    acDeviceSynchronizeStream(STREAM_DEFAULT);

    // Deallocate memory on the GPU
    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);
    return EXIT_SUCCESS;
}