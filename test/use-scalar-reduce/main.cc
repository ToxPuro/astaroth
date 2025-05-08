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
main(int argc, char* argv[])
{
    atexit(acAbort);

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const int nx = argc > 1 ? atoi(argv[1]) : 2*9;
    const int ny = argc > 2 ? atoi(argv[2]) : 2*11;
    const int nz = argc > 3 ? atoi(argv[3]) : 4*7;

    acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    const int3 decomp = acDecompose(nprocs,info);

    acSetGridMeshDims(nx, ny, nz, &info);
    acSetLocalMeshDims(nx/decomp.x, ny/decomp.y, nz/decomp.z, &info);

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);


    // GPU alloc & compute
    acGridInit(info);
    auto graph = acGetOptimizedDSLTaskGraph(rhs);

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };

    const  AcReal local_max = (pid+1)*2.0;
    const  AcReal max_val   = (nprocs)*2.0;
    model.vertex_buffer[F][IDX(info[AC_nlocal].x/2, info[AC_nlocal].y/2, info[AC_nlocal].z/2)] = local_max;

    acGridSynchronizeStream(STREAM_ALL);
    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT,model);
    acGridSynchronizeStream(STREAM_ALL);

    acGridExecuteTaskGraph(graph,1);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);


    int f_correct = 1;
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		f_correct &= candidate.vertex_buffer[F][IDX(i,j,k)] == max_val;
	}
      }
    }

    int local_max_correct  = local_max == acDeviceGetOutput(acGridGetDevice(),  F_LOCAL_MAX);
    int global_max_correct = max_val   == acDeviceGetOutput(acGridGetDevice(), F_GLOBAL_MAX);

    fprintf(stderr,"LOCAL  MAX %d: %14e,%14e ... %s\n",pid,local_max,acDeviceGetOutput(acGridGetDevice(), F_LOCAL_MAX),local_max_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL!" AC_COL_RESET);
    fprintf(stderr,"GLOBAL MAX %d: %14e,%14e ... %s\n",pid,max_val,acDeviceGetOutput(acGridGetDevice(), F_GLOBAL_MAX),global_max_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL!" AC_COL_RESET);
    fprintf(stderr,"F %d: ... %s\n",pid,f_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    MPI_Allreduce(MPI_IN_PLACE, &f_correct,   1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &local_max_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_max_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);



    const bool success = f_correct && local_max_correct && global_max_correct;
    if (pid == 0)
    {
        fprintf(stderr, "USE_SCALAR_REDUCE_TEST complete: %s\n",
                success ? "No errors found" : "One or more errors found");
    }
    
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);
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
