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
#include "user_constants.h"
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
    acSetGridMeshDims(nx, ny, nz, &info);
    acSetLocalMeshDims(nx, ny, nz, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);


    auto graph = acGetOptimizedDSLTaskGraph(rhs);
    acGridExecuteTaskGraph(graph,1);


    if (pid == 0)
        acHostMeshRandomize(&model);
    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };
    auto test = [&](const bool all_ones_test)
    {
        acHostMeshApplyPeriodicBounds(&model);
    	acGridLoadMesh(STREAM_DEFAULT, model);
    	acGridSynchronizeStream(STREAM_ALL);



    	acGridExecuteTaskGraph(graph,1);
    	acGridSynchronizeStream(STREAM_ALL);
    	acGridStoreMesh(STREAM_DEFAULT,&candidate);
    	acGridSynchronizeStream(STREAM_ALL);

    	acDeviceStoreProfile(acGridGetDevice(), PROF_Z,  &model);
    	acGridSynchronizeStream(STREAM_ALL);
    	const AcReal* z_sum_gpu = model.profile[PROF_Z];

    	AcReal* z_sum       = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal].z);
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    	    z_sum[k] = 0.0;
    		for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	    		{
    	    			auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
				z_sum[k] += val;
			}
		}
	}

    	AcReal epsilon  = 4*pow(10.0,-11.0);
    	auto relative_diff = [](const auto a, const auto b)
    	{
    	        const auto abs_diff = fabs(a-b);
    	        return  abs_diff/a;
    	};
        auto in_eps_threshold = [&](const auto a, const auto b)
        {
                if(a == b) return true;
                return relative_diff(a,b) < epsilon;
        };

    	bool sums_correct = true;
    	for(size_t i = 0; i < dims.m1.z; ++i)
    	{
    	    bool correct =  in_eps_threshold(z_sum[i],z_sum_gpu[i]);
    	    sums_correct &= correct;
    	    if(!correct) fprintf(stderr,"Z SUM WRONG: %ld, %14e, %14e\n",i,z_sum[i],z_sum_gpu[i]);
    	}

    	fprintf(stderr,"Z SUM REDUCTION... %s\n", sums_correct    ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	bool correct = sums_correct;

    	free(z_sum);

    	return !correct;
    };

    for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    	for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    		for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
		{
			model.vertex_buffer[FIELD][IDX(i,j,k)] = 1.0;
		}
	

    fprintf(stderr,"\nTest: all ones\n");
    int retval = test(true);

    const int NUM_RAND_TESTS = argc > 4 ? atoi(argv[4]) : 10;
    for(int test_iteration = 0; test_iteration < NUM_RAND_TESTS; ++test_iteration)
    {
    	constexpr size_t RAND_RANGE = 100;
    	acHostMeshRandomize(&model);
    	fprintf(stderr,"\nTest: Rand\n");
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    		for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    			for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		//Skew slightly positive to not have zero mean --> finite condition number for summing the floating point numbers
    	    		model.vertex_buffer[FIELD][IDX(i,j,k)] -= 0.49;
    	    		model.vertex_buffer[FIELD][IDX(i,j,k)] *= RAND_RANGE;
    	    	}

    	retval |= test(false);
    }
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }
    finalized = true;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);

    if (pid == 0)
        fprintf(stderr, "MPI-PROFILE-REDUCE-TEST complete: %s\n",
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
