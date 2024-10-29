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

#define NUM_INTEGRATION_STEPS (1)

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

    auto test_bc = [&](const auto& orig, const auto& dsl, const char* text)
    {
    	std::vector<Field> all_fields{};
    	for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
    	    all_fields.push_back(Field(i));
    	}

    	AcTaskGraph* graph = acGridBuildTaskGraph({acBoundaryCondition(BOUNDARY_XYZ,orig,all_fields)});
    	AcTaskGraph* graph_2 = acGridBuildTaskGraph({acBoundaryCondition(BOUNDARY_XYZ,dsl,all_fields)});
    	acGridExecuteTaskGraph(graph,1);
    	acGridExecuteTaskGraph(graph_2,1);

    	// arr test
    	if (pid == 0)
    	    acHostMeshRandomize(&model);

    	acGridLoadMesh(STREAM_DEFAULT, model);
    	for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    		acGridExecuteTaskGraph(graph,1);
    	acGridStoreMesh(STREAM_DEFAULT, &candidate);

    	acGridLoadMesh(STREAM_DEFAULT, model);
    	for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    		acGridExecuteTaskGraph(graph_2,1);
    	acGridStoreMesh(STREAM_DEFAULT, &model);


    	return acVerifyMesh(text, model, candidate);
    };
    auto test_bc_with_param = [&](const auto& orig, const auto& dsl, const auto param, const char* text)
    {
    	std::vector<Field> all_fields{};
	Field all_fields_ptr[NUM_VTXBUF_HANDLES];
    	for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
    	    all_fields.push_back(Field(i));
	    all_fields_ptr[i] = Field(i);
    	}

	AcRealParam params[] = {param};
    	AcTaskGraph* graph = acGridBuildTaskGraph({acBoundaryCondition(BOUNDARY_XYZ,orig,all_fields_ptr,params)});
    	AcTaskGraph* graph_2 = acGridBuildTaskGraph({acBoundaryCondition(BOUNDARY_XYZ,dsl,all_fields,param)});
    	acGridExecuteTaskGraph(graph,1);
    	acGridExecuteTaskGraph(graph_2,1);

    	// arr test
    	if (pid == 0)
    	    acHostMeshRandomize(&model);

    	acGridLoadMesh(STREAM_DEFAULT, model);
    	for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    		acGridExecuteTaskGraph(graph,1);
    	acGridStoreMesh(STREAM_DEFAULT, &candidate);

    	acGridLoadMesh(STREAM_DEFAULT, model);
    	for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    		acGridExecuteTaskGraph(graph_2,1);
    	acGridStoreMesh(STREAM_DEFAULT, &model);


    	return acVerifyMesh(text, model, candidate);
    };
    AcResult res = AC_SUCCESS;
    res = test_bc(BOUNDCOND_SYMMETRIC,KERNEL_BOUNDCOND_SYMMETRIC_DSL,"symmetric")
	    ? AC_FAILURE : res;
    res = test_bc(BOUNDCOND_ANTISYMMETRIC,KERNEL_BOUNDCOND_ANTI_SYMMETRIC_DSL,"antisymmetric")
	    ? res : AC_FAILURE;
    res = test_bc(BOUNDCOND_A2,KERNEL_BOUNDCOND_A2_DSL,"a2")
	    ? AC_FAILURE: res;
    res = test_bc_with_param(BOUNDCOND_CONST,KERNEL_BOUNDCOND_CONST_DSL,AC_mu0,"const")
	    ? AC_FAILURE: res;
    res = test_bc(BOUNDCOND_INFLOW,KERNEL_BOUNDCOND_INFLOW_DSL,"inflow")
	    ? AC_FAILURE: res;
    res = test_bc(BOUNDCOND_OUTFLOW,KERNEL_BOUNDCOND_OUTFLOW_DSL,"outflow")
	    ? AC_FAILURE: res;
    res = test_bc_with_param(BOUNDCOND_PRESCRIBED_DERIVATIVE,KERNEL_BOUNDCOND_PRESCRIBED_DERIVATIVE_DSL,AC_mu0,"prescribed derivative")
	    ? AC_FAILURE: res;
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
        fprintf(stderr, "BOUNDCOND-MIGRATION complete: %s\n",
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
