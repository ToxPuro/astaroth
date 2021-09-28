/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

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
    Running: benchmark -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"

#include "errchk.h"
#include "timer_hires.h"

#if AC_MPI_ENABLED

#include <mpi.h>


int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    AcMesh model, candidate;
    if (!pid) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);
        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, model);

    const AcReal dt = FLT_EPSILON;
    const size_t nsteps = 10;
    for (size_t step = 0; step < nsteps; ++step) {
        acGridIntegrate(STREAM_DEFAULT, dt);
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        acGridStoreMesh(STREAM_DEFAULT, &candidate);

        if (!pid) {
            acModelIntegrateStep(model, dt);
            acMeshApplyPeriodicBounds(&model);

            acMeshWriteErrorSlice("errors.out", model, candidate, info.int_params[AC_nz]/2);
        }
        
    }

    acGridQuit();
    MPI_Finalize();
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
