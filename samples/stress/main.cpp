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

#include "math_utils.h"
#include "user_constants.h"


static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}


int
main(void)
{

    MPI_Init(NULL,NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    atexit(acAbort);
    int retval = 0;
    // Set random seed for reproducibility
    srand(321654987);
    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG,&info,NULL);;

    info.real_params[AC_dsx] = dsx;
    info.real_params[AC_dsy] = dsy;
    //info.real_params[AC_dsz] = d;

    const int max_devices = 2 * 2 * 4;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d). Please modify "
                "mpitest/main.cc to use a larger mesh.\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(npointsx_grid, npointsy_grid, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // Load/Store
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        const AcResult res = acVerifyMesh("Load/Store", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    std::vector<Field> all_fields {};
    for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i) all_fields.push_back((Field)i);
    auto initialize = acGetDSLTaskGraph(AC_initialize);
    auto update     = acGetDSLTaskGraph(AC_update);
    // Boundconds
    if (pid == 0)
        acHostMeshRandomize(&model);


    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Periodic boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);
    acGridExecuteTaskGraph(initialize, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acHostMeshApplyPeriodicBounds(&candidate);
    std::vector<AcReal> y(npointsx_grid);
    acGridSynchronizeStream(STREAM_ALL);

    for (int i = 0; i < nsteps; ++i)
    {
	    acGridExecuteTaskGraph(update, 1);
	    acGridSynchronizeStream(STREAM_ALL);
    }
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_ALL);
    acHostMeshApplyPeriodicBounds(&candidate);


    /**
    if(pid  == 0)
    {
            std::vector<AcReal> x(npointsx_grid);
            std::vector<AcReal> u(npointsx_grid);
            std::vector<AcReal> analytical(npointsx_grid);
	    FILE* fp_a = fopen("a.dat","w");
	    FILE* fp_u = fopen("u.dat","w");
	    FILE* fp_x = fopen("x.dat","w");
	    const int end = npointsx_grid;
            for(int i = 0; i< end; ++i)
            {
		const int idx = IDX_COMP_DOMAIN(i, npointsy_grid/2, npointsz_grid/2);
            	x[i] = candidate.vertex_buffer[COORDS_X][idx];
                u[i] = candidate.vertex_buffer[U][idx];
                analytical[i] = candidate.vertex_buffer[SOLUTION][idx];

		fprintf(fp_a,"%.14e",analytical[i]);
		if(i < end -1 ) fprintf(fp_a,"%s",",");

		fprintf(fp_u,"%.14e",u[i]);
		if(i < end -1 ) fprintf(fp_u,"%s",",");

		fprintf(fp_x,"%.14e",x[i]);
		if(i < end -1 ) fprintf(fp_x,"%s",",");
            }
	    fclose(fp_a);
	    fclose(fp_x);
	    fclose(fp_u);
    }
    **/

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "MPITEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

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
