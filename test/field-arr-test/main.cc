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
#include "user_constants.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define NUM_INTEGRATION_STEPS (2)

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}
double
drand()
{
	return (double)(rand()) / (double)(rand());
}

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(2 * 9, 2 * 11, 4 * 7, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    acGridInit(info);

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }
    auto null_loader = [&](ParamLoadingInfo l){(void)l;};
    AcTaskDefinition ops[] = {
	    acComputeWithParams(test_arr,all_fields,null_loader)
    };
    AcTaskGraph* graph = acGridBuildTaskGraph(ops);

    acGridExecuteTaskGraph(graph,3);

    // arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    	acGridExecuteTaskGraph(graph,1);

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);


    const int nx_min = model.info.int_params[AC_nx_min];
    const int nx_max = model.info.int_params[AC_nx_max];

    const int ny_min = model.info.int_params[AC_ny_min];
    const int ny_max = model.info.int_params[AC_ny_max];

    const int nz_min = model.info.int_params[AC_nz_min];
    const int nz_max = model.info.int_params[AC_nz_max];
    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };

    std::array<AcReal,3> arr = {1.0,2.0,3.0};
    std::array<AcReal3,3> vec_arr = {
	    				(AcReal3){1.0,2.0,3.0},
	    				(AcReal3){4.0,5.0,6.0},
	    				(AcReal3){7.0,8.0,9.0},
    				    };
    for (int step_number = 0; step_number < NUM_INTEGRATION_STEPS; ++step_number) {
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
			for (size_t field = 0; field < Field_arr.size(); ++field)
			{
				model.vertex_buffer[Field_arr[field]][IDX(i,j,k)] += arr[field];
			}
			for(size_t field = 0; field < DUST.size(); ++field)
			{

				model.vertex_buffer[DUST[field].x][IDX(i,j,k)] += vec_arr[field].x;
				model.vertex_buffer[DUST[field].y][IDX(i,j,k)] += vec_arr[field].y;
				model.vertex_buffer[DUST[field].z][IDX(i,j,k)] += vec_arr[field].z;
			}
                }
            }
        }
    }

    const AcResult res = acVerifyMesh("field-arrays", model, candidate);
    if (res != AC_SUCCESS) {
        retval = res;
        WARNCHK_ALWAYS(retval);
    }

    fflush(stdout);
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "FIELD_ARR_TEST complete: %s\n",
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
