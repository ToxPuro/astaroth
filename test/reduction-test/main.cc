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
    AcCompInfo comp_info = acInitCompInfo();
    acLoadConfig(AC_DEFAULT_CONFIG, &info, &comp_info);

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

    // GPU alloc & compute
    acGridInit(info);

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }
    AcTaskDefinition ops[] = {
	    acCompute(KERNEL_test_reduce,all_fields)
    };
    AcTaskGraph* graph = acGridBuildTaskGraph(ops);

    acGridExecuteTaskGraph(graph,1);

    // arr test
    if (pid == 0)
        acHostMeshRandomize(&model);
    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };
    constexpr int RAND_RANGE = 100;
    for(int k = dims.n0.x; k < dims.n1.x;  ++k)
    	for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		for(int i = dims.n0.z; i < dims.n1.z; ++i)
			model.vertex_buffer[FIELD][IDX(i,j,k)] = (model.vertex_buffer[FIELD][IDX(i,j,k)]-0.5)*RAND_RANGE;


    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);

    acGridExecuteTaskGraph(graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridFinalizeReduceLocal(graph);
    acGridSynchronizeStream(STREAM_ALL);
    const auto gpu_max_val = acDeviceGetOutput(acGridGetDevice(), AC_max_val);
    const auto gpu_min_val = acDeviceGetOutput(acGridGetDevice(), AC_min_val);
    const auto gpu_sum_val = acDeviceGetOutput(acGridGetDevice(), AC_sum_val);
    const auto gpu_int_sum = acDeviceGetOutput(acGridGetDevice(), AC_int_sum_val);
    AcReal cpu_max_val = -1000000.000000000;
    AcReal cpu_min_val = +1000000.000000000;
    int cpu_int_sum = 0;
    long double long_cpu_sum_val = (long double)0.0;
    for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    {
    	for(int j = dims.n0.y; j < dims.n1.y; ++j)
	{
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
		{
			auto val = model.vertex_buffer[FIELD][IDX(i,j,k)];
			cpu_max_val = (val > cpu_max_val)  ? val : cpu_max_val;
			if(val < cpu_min_val)
			{
				cpu_min_val = val;
			}
			long_cpu_sum_val += (long double)val;
			cpu_int_sum += (int)val;
		}
	}
    }
    AcReal cpu_sum_val = (AcReal)long_cpu_sum_val;
    AcReal epsilon  = pow(10.0,-12.0);
    auto in_eps_threshold = [&](const auto a, const auto b)
    {
	    const auto abs_diff = fabs(a-b);
	    const auto relative_diff = abs_diff/a;
	    return relative_diff < epsilon;
    };

    fprintf(stderr,"MAX REDUCTION... %s %14e %14e\n", cpu_max_val == gpu_max_val ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_max_val,gpu_max_val);
    fprintf(stderr,"MIN REDUCTION... %s %14e %14e\n", cpu_min_val == gpu_min_val ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_min_val,gpu_min_val);
    fprintf(stderr,"SUM REDUCTION... %s %14e %14e\n", in_eps_threshold(cpu_sum_val,gpu_sum_val) ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_sum_val,gpu_sum_val);
    fprintf(stderr,"INT SUM REDUCTION... %s %d %d\n", cpu_int_sum == gpu_int_sum ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_int_sum,gpu_int_sum);
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "REDUCTION_TEST complete: %s\n",
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
