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
main(int argc, char* argv[])
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
    const int nx = argc > 1 ? atoi(argv[1]) : 2*9;
    const int ny = argc > 2 ? atoi(argv[2]) : 2*11;
    const int nz = argc > 3 ? atoi(argv[3]) : 4*7;
    acSetMeshDims(nx, ny, nz, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);
    auto graph = acGetDSLTaskGraph(rhs);
    acGridExecuteTaskGraph(graph,1);
    acGridStoreMesh(STREAM_DEFAULT,&candidate);
    AcReal3* x_sum   = (AcReal3*)malloc(sizeof(AcReal3)*model.info.int_params[AC_mx]);
    AcReal3* y_sum   = (AcReal3*)malloc(sizeof(AcReal3)*model.info.int_params[AC_my]);

    AcReal epsilon  = pow(10.0,-12.0);
    auto relative_diff = [](const auto a, const auto b)
    {
	    const auto abs_diff = fabs(a-b);
	    return  abs_diff/a;
    };
    auto in_eps_threshold = [&](const auto a, const auto b)
    {
	    return relative_diff(a,b) < epsilon;
    };

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };

    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
	x_sum[i] = (AcReal3){0.,0.,0.};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
		{
			x_sum[i] += (AcReal3){model.vertex_buffer[AX][IDX(i,j,k)],model.vertex_buffer[AY][IDX(i,j,k)],model.vertex_buffer[AZ][IDX(i,j,k)]};
		}
    }
    for(int j = dims.n0.y; j < dims.n1.y; ++j)
    {
	y_sum[j] = (AcReal3){0.,0.,0.};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
		{
			y_sum[j] += (AcReal3){model.vertex_buffer[AX][IDX(i,j,k)],model.vertex_buffer[AY][IDX(i,j,k)],model.vertex_buffer[AZ][IDX(i,j,k)]};
		}
    }
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(int j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[AX][IDX(i,j,k)] -= x_sum[i].x;
		model.vertex_buffer[AY][IDX(i,j,k)] -= x_sum[i].y;
		model.vertex_buffer[AZ][IDX(i,j,k)] -= x_sum[i].z;

		model.vertex_buffer[AX][IDX(i,j,k)] -= y_sum[j].x;
		model.vertex_buffer[AY][IDX(i,j,k)] -= y_sum[j].y;
		model.vertex_buffer[AZ][IDX(i,j,k)] -= y_sum[j].z;
	}
      }
    }
    bool ax_correct = true;
    bool ay_correct = true;
    bool az_correct = true;
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(int j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		ax_correct &= in_eps_threshold(model.vertex_buffer[AX][IDX(i,j,k)],candidate.vertex_buffer[AX][IDX(i,j,k)]);
		ay_correct &= in_eps_threshold(model.vertex_buffer[AY][IDX(i,j,k)],candidate.vertex_buffer[AY][IDX(i,j,k)]);
		az_correct &= in_eps_threshold(model.vertex_buffer[AZ][IDX(i,j,k)],candidate.vertex_buffer[AZ][IDX(i,j,k)]);
	}
      }
    }
    retval = !(ax_correct && ay_correct && az_correct);
    fprintf(stderr,"AX ... %s\n", ax_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"AY ... %s\n", ay_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"AZ ... %s\n", az_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    if (pid == 0)
        fprintf(stderr, "KERNEL_FUSION_TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");
    
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }
    finalized = true;

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);

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
