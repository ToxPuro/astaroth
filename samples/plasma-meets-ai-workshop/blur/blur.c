#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

int
main(void)
{
    // Setup the mesh configuration
    AcMeshInfo info;
    acLoadConfig("../samples/plasma-meets-ai-workshop/astaroth.conf", &info);

    // Allocate memory on the GPU
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    const AcMeshDims dims = acGetMeshDims(info);

    // Setup initial conditions
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh);
    acHostMeshSet(1, &mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

    // Compute
    // acDeviceLaunchKernel(device, STREAM_DEFAULT, ..., dims.n0, dims.n1);
    acDeviceSwapBuffers(device);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    // Store to host memory and write to a file
    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    // Deallocate memory on the GPU
    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);
    return EXIT_SUCCESS;
}