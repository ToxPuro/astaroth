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
/**
    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (100)

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void) {
    if (!finalized)
        MPI_Abort(MPI_COMM_WORLD, 0);
}

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(32, 32, 32, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // Boundconds
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
        acHostMeshRandomize(&model);
    }

    // Dryrun
    const AcReal dt = (AcReal)FLT_EPSILON;
    acGridIntegrate(STREAM_DEFAULT, dt);

    // Integration
    acGridLoadMesh(STREAM_DEFAULT, model);

    // Device integrate
    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        acGridIntegrate(STREAM_DEFAULT, dt);

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {

        // Host integrate
        for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
            acHostIntegrateStep(model, dt);

        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
        acHostMeshRandomize(&model);
    }

    // Scalar reductions
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, (ReductionType)i, v0, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceScal(model, (ReductionType)i, v0);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);

            if (!acEvalError(rtype_names[i], error)) {
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        const VertexBufferHandle v1 = VTXBUF_UUY;
        const VertexBufferHandle v2 = VTXBUF_UUZ;
        AcReal candval;
        acGridReduceVec(STREAM_DEFAULT, (ReductionType)i, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceVec(model, (ReductionType)i, v0, v1, v2);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);
            
            if (!acEvalError(rtype_names[i], error)) {
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "MPITEST complete: %s\n", retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return EXIT_SUCCESS;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
