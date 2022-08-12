/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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

#include "astaroth.h"
#include "astaroth_utils.h"
#include "user_defines.h"

void
save_slice(const Device device, const AcMeshInfo info, const size_t id)
{
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);

    acHostMeshWriteToFile(mesh, id);
    acHostMeshDestroy(&mesh);

#define WRITE_FILES_WITH_PYTHON (0)
#if WRITE_FILES_WITH_PYTHON
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        const size_t len = 4096;
        char buf[len];
        snprintf(buf, len, "../samples/les/analysis.py data-format.csv %s-%lu.dat", field_names[i],
                 id);

        FILE* proc = popen(buf, "r");
        ERRCHK_ALWAYS(proc);
        fclose(proc);
    }
#endif
}

int
main(void)
{
    ERRCHK_ALWAYS(acCheckDeviceAvailability() == AC_SUCCESS);

    AcMeshInfo info;
    const int nn = 64;
    acSetMeshDims(nn, nn, nn, &info);
    acPrintMeshInfo(info);

    // Alloc
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh);
    acHostMeshApplyPeriodicBounds(&mesh);

    // Init device
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);

    // Verify that the mesh was loaded and stored correctly
    AcMesh candidate;
    acHostMeshCreate(info, &candidate);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acVerifyMesh("Load/Store", mesh, candidate);

    // Verify that reading and writing to file works correctly
    acHostMeshWriteToFile(candidate, 0);
    acHostMeshReadFromFile(0, &candidate);
    acVerifyMesh("Read/Write", mesh, candidate);
    acHostMeshDestroy(&candidate);

    // Warmup
    const int3 start = (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2};
    const int3 end = (int3){STENCIL_ORDER / 2 + nn, STENCIL_ORDER / 2 + nn, STENCIL_ORDER / 2 + nn};
    for (size_t i = 0; i < NUM_KERNELS; ++i) {
        printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], start, end);
    }

    // Benchmark
    cudaProfilerStart();
    for (size_t i = 0; i < NUM_KERNELS; ++i) {
        printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], start, end);
    }
    cudaProfilerStop();
    acDeviceSwapBuffers(device);

    // Simulate
    acHostVertexBufferSet(RHO, 1, &mesh);
    acHostMeshApplyPeriodicBounds(&mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
    save_slice(device, info, 0);

    printf("VTXBUF ranges before integration:\n");
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        AcReal min, max;
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
        printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
    }

    for (size_t step = 1; step < 10; ++step) {
        /*
        for (size_t i = 0; i < NUM_KERNELS; ++i) {
            printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
            acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], start, end);
            acDeviceSwapBuffers(device);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
        }
        */
        for (size_t substep = 0; substep < 3; ++substep) {
            acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-3);
            acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep);

            acDeviceLaunchKernel(device, STREAM_DEFAULT, compute_stress_tensor_tau, start, end);
            acDeviceSwapBuffer(device, T00);
            acDeviceSwapBuffer(device, T01);
            acDeviceSwapBuffer(device, T02);
            acDeviceSwapBuffer(device, T11);
            acDeviceSwapBuffer(device, T12);
            acDeviceSwapBuffer(device, T22);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
            // Note: the above boundcond step does all fields instead of just the stress tensor
            acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, start, end);
            acDeviceSwapBuffer(device, UUX);
            acDeviceSwapBuffer(device, UUY);
            acDeviceSwapBuffer(device, UUZ);
            acDeviceSwapBuffer(device, RHO);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
        }

        // Write to disk
        acDeviceSynchronizeStream(device, STREAM_ALL);
        save_slice(device, info, step);

        printf("Step %lu\n", step);
        for (size_t i = 0; i < NUM_FIELDS; ++i) {
            AcReal min, max;
            acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
            acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
            printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
        }
    }

    printf("Done.\nVTXBUF ranges after integration:\n");
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        AcReal min, max;
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
        printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
    }

    acHostMeshDestroy(&mesh);
    acDeviceDestroy(device);

    return EXIT_SUCCESS;
}
