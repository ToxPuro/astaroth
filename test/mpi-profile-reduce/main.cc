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

    const int nx = argc > 1 ? atoi(argv[1]) : 16;
    const int ny = argc > 2 ? atoi(argv[2]) : 16;
    const int nz = argc > 3 ? atoi(argv[3]) : 16;
    acSetGridMeshDims(nx, ny, nz, &info);
    const int3 decomp = acDecompose(nprocs,info);
    acSetLocalMeshDims(nx/decomp.x, ny/decomp.y, nz/decomp.z, &info);
    // GPU alloc & compute
    acGridInit(info);

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);

    auto graph = acGetOptimizedDSLTaskGraph(rhs);
    acGridExecuteTaskGraph(graph,1);


    acHostMeshRandomize(&model);
    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };
    auto test = [&](const bool)
    {
    	acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
    	acGridSynchronizeStream(STREAM_ALL);



    	acGridExecuteTaskGraph(graph,1);
    	acGridSynchronizeStream(STREAM_ALL);
    	acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    	acGridSynchronizeStream(STREAM_ALL);

    	acDeviceStoreProfile(acGridGetDevice(), PROF_Z,  &model);
    	acDeviceStoreProfile(acGridGetDevice(), PROF_Y,  &model);
    	acDeviceStoreProfile(acGridGetDevice(), PROF_X,  &model);

    	acDeviceStoreProfile(acGridGetDevice(), PROF_XY,  &model);
    	acDeviceStoreProfile(acGridGetDevice(), PROF_XZ,  &model);

    	acDeviceStoreProfile(acGridGetDevice(), PROF_YX,  &model);
    	acDeviceStoreProfile(acGridGetDevice(), PROF_YZ,  &model);

    	acDeviceStoreProfile(acGridGetDevice(), PROF_ZX,  &model);
    	acDeviceStoreProfile(acGridGetDevice(), PROF_ZY,  &model);
    	acGridSynchronizeStream(STREAM_ALL);
    	const AcReal* z_sum_gpu = model.profile[PROF_Z];
    	const AcReal* x_sum_gpu = model.profile[PROF_X];
    	const AcReal* y_sum_gpu = model.profile[PROF_Y];

    	const AcReal* xy_sum_gpu = model.profile[PROF_XY];
    	const AcReal* xz_sum_gpu = model.profile[PROF_XZ];

    	const AcReal* yx_sum_gpu = model.profile[PROF_YX];
    	const AcReal* yz_sum_gpu = model.profile[PROF_YZ];

    	const AcReal* zx_sum_gpu = model.profile[PROF_ZX];
    	const AcReal* zy_sum_gpu = model.profile[PROF_ZY];

    	AcReal* x_sum       = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal].x);
    	AcReal* z_sum       = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal].z);
    	AcReal* y_sum       = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal].y);

    	AcReal* xy_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].xy);
    	AcReal* xz_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].xz);
    	AcReal* yx_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].xy);
    	AcReal* yz_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].yz);
    	AcReal* zx_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].xz);
    	AcReal* zy_sum      = (AcReal*)malloc(sizeof(AcReal)*model.info[AC_mlocal_products].yz);

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

    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    	    x_sum[i] = 0.0;
    		for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    			for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    	    		x_sum[i] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	}
    	for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    	{
    	    y_sum[j] = 0.0;
    		for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    			for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	    		y_sum[j] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	}

    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    	    		const size_t index = i + model.info[AC_mlocal].x*j;
    	    		xy_sum[index] = 0.0;
    			for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    			{
    	    			xy_sum[index]    += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		}
    		}
    	}
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    		{
    	    	const size_t index = i + model.info[AC_mlocal].x*k;
    	    	xz_sum[index] = 0.0;
    			for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    			{
    	    		xz_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    	}
    		}
    	}
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
    	    	const size_t index = j + model.info[AC_mlocal].y*i;
    	    	yx_sum[index] = 0.0;
    			for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    			{
    	    		yx_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    	}
    		}
    	}
    	for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
    	    	const size_t index = j + model.info[AC_mlocal].y*k;
    	    	yz_sum[index] = 0.0;
    			for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    			{
    	    		yz_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    	}
    		}
    	}
    	for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    	{
    		for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    		{
    	    	const size_t index = k + model.info[AC_mlocal].z*i;
    	    	zx_sum[index] = 0.0;
    			for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    			{
    	    		zx_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    	}
    		}
    	}
    	for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
    	    	const size_t index = k + model.info[AC_mlocal].z*j;
    	    	zy_sum[index] = 0.0;
    			for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    			{
    	    		zy_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    	}
    		}
    	}


	MPI_Allreduce(MPI_IN_PLACE,z_sum,model.info[AC_mlocal].z,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().xy);
	MPI_Allreduce(MPI_IN_PLACE,x_sum,model.info[AC_mlocal].x,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().yz);
	MPI_Allreduce(MPI_IN_PLACE,y_sum,model.info[AC_mlocal].y,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().xz);

	MPI_Allreduce(MPI_IN_PLACE,xy_sum,model.info[AC_mlocal_products].xy,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().z);
	MPI_Allreduce(MPI_IN_PLACE,xz_sum,model.info[AC_mlocal_products].xz,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().y);

	MPI_Allreduce(MPI_IN_PLACE,yx_sum,model.info[AC_mlocal_products].xy,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().z);
	MPI_Allreduce(MPI_IN_PLACE,yz_sum,model.info[AC_mlocal_products].yz,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().x);

	MPI_Allreduce(MPI_IN_PLACE,zx_sum,model.info[AC_mlocal_products].xz,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().y);
	MPI_Allreduce(MPI_IN_PLACE,zy_sum,model.info[AC_mlocal_products].yz,AC_REAL_MPI_TYPE, MPI_SUM, acGridMPISubComms().x);

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
    	for(size_t i = dims.n0.z; i < dims.n1.z; ++i)
    	{
    	    bool correct =  in_eps_threshold(z_sum[i],z_sum_gpu[i]);
    	    sums_correct &= correct;
    	    if(!correct) fprintf(stderr,"Z SUM WRONG %d: %ld, %14e, %14e\n",pid,i,z_sum[i],z_sum_gpu[i]);
    	}
    	bool x_sum_correct = true;
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    	    bool correct =  in_eps_threshold(x_sum[i],x_sum_gpu[i]);
    	    x_sum_correct &= correct;
    	    if(!correct) fprintf(stderr,"X SUM WRONG: %14e, %14e\n",x_sum[i],x_sum_gpu[i]);
    	}

    	bool y_sum_correct = true;
    	for(size_t i = dims.n0.y; i < dims.n1.y; ++i)
    	{
    	    bool correct =  in_eps_threshold(y_sum[i],y_sum_gpu[i]);
    	    y_sum_correct &= correct;
    	    //if(!correct) fprintf(stderr,"Y SUM WRONG: %14e, %14e\n",y_sum[i],y_sum_gpu[i]);
    	}
    	bool xy_sum_correct = true;
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    	    	const size_t index = i + model.info[AC_mlocal].x*j;
		{
    	    		const bool correct = in_eps_threshold(xy_sum[index],xy_sum_gpu[index]);
    	    		xy_sum_correct &= correct;
    	    		if(!correct) fprintf(stderr,"XY SUM WRONG: %14e, %14e, %14e\n",xy_sum[index],xy_sum_gpu[index],relative_diff(xy_sum[index], xy_sum_gpu[index]));
		}
    	    }
    	}
    	bool xz_sum_correct = true;
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    		{
    	    	const size_t index = i + model.info[AC_mlocal].x*k;
    	    	const bool correct = in_eps_threshold(xz_sum[index],xz_sum_gpu[index]);
    	    	xz_sum_correct &= correct;
    	    	//if(!correct) fprintf(stderr,"XZ SUM WRONG: %14e, %14e\n",xz_sum[index],xz_sum_gpu[index]);
    	    }
    	}
    	bool yx_sum_correct = true;
    	for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
    	    	const size_t index = j + model.info[AC_mlocal].y*i;
    	    	const bool correct = in_eps_threshold(yx_sum[index],yx_sum_gpu[index]);
    	    	yx_sum_correct &= correct;
    	    	if(!correct) fprintf(stderr,"YX SUM WRONG: %14e, %14e\n",yx_sum[index],yx_sum_gpu[index]);
    		}
    	}
    	bool yz_sum_correct = true;
    	for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
    		for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    		{
    	    	const size_t index = j + model.info[AC_mlocal].y*k;
    	    	const bool correct = in_eps_threshold(yz_sum[index],yz_sum_gpu[index]);
    	    	yz_sum_correct &= correct;
    	    	//if(!correct) fprintf(stderr,"YZ SUM WRONG: %14e, %14e\n",yz_sum[index],yz_sum_gpu[index]);
    		}
    	}
    	bool zx_sum_correct = true;
    	for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    	{
    		for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    		{
    	    	const size_t index = k + model.info[AC_mlocal].z*i;
    	    	const bool correct = in_eps_threshold(zx_sum[index],zx_sum_gpu[index]);
    	    	zx_sum_correct &= correct;
    	    	//if(!correct) fprintf(stderr,"ZX SUM WRONG: %14e, %14e\n",zx_sum[index],zx_sum_gpu[index]);
    		}
    	}
    	bool zy_sum_correct = true;
    	for(size_t k = dims.n0.z; k < dims.n1.z; ++k)
    	{
    		for(size_t j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
    	    	const size_t index = k + model.info[AC_mlocal].z*j;
    	    	const bool correct = in_eps_threshold(zy_sum[index],zy_sum_gpu[index]);
    	    	zy_sum_correct &= correct;
    	    	if(!correct) fprintf(stderr,"ZY SUM WRONG: %14e, %14e\n",zy_sum[index],zy_sum_gpu[index]);
    		}
    	}

    	fprintf(stderr,"X SUM REDUCTION %d ... %s\n", pid, x_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	fprintf(stderr,"Y SUM REDUCTION %d ... %s\n", pid, y_sum_correct    ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	fprintf(stderr,"Z SUM REDUCTION %d ... %s\n", pid, sums_correct    ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    	fprintf(stderr,"XY SUM REDUCTION %d ... %s\n", pid, xy_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	fprintf(stderr,"XZ SUM REDUCTION %d ... %s\n", pid, xz_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    	fprintf(stderr,"YX SUM REDUCTION %d ... %s\n", pid, yx_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	fprintf(stderr,"YZ SUM REDUCTION %d ... %s\n", pid, yz_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    	fprintf(stderr,"ZX SUM REDUCTION %d ... %s\n", pid, zx_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	fprintf(stderr,"ZY SUM REDUCTION %d ... %s\n", pid, zy_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    	bool correct = sums_correct && x_sum_correct && y_sum_correct
			&& xy_sum_correct && xz_sum_correct
			&& yx_sum_correct && yz_sum_correct
			&& zx_sum_correct && zy_sum_correct
			;
    	free(x_sum);
    	free(y_sum);
    	free(z_sum);
    	free(xy_sum);
    	free(xz_sum);
    	free(yx_sum);
    	free(yz_sum);
    	free(zx_sum);
    	free(zy_sum);
        MPI_Allreduce(MPI_IN_PLACE, &correct,   1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    	return !correct;
    };

    for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    	for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
    		for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
		{
			model.vertex_buffer[FIELD][IDX(i,j,k)] = 1.0;
		}
	

    if(pid == 0) fprintf(stderr,"\nTest: all ones\n");
    int retval = test(true);

    const int NUM_RAND_TESTS = argc > 4 ? atoi(argv[4]) : 10;
    for(int test_iteration = 0; test_iteration < NUM_RAND_TESTS; ++test_iteration)
    {
    	constexpr size_t RAND_RANGE = 100;
    	acHostMeshRandomize(&model);
    	if(pid == 0) fprintf(stderr,"\nTest: Rand\n");
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
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    finalized = true;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);

    if (pid == 0)
    {
        fprintf(stderr, "MPI-PROFILE-REDUCE-TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");
    }

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
