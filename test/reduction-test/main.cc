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

    AcBuffer gpu_field_zyx        =acDeviceTranspose(acGridGetDevice(),STREAM_DEFAULT,ZYX,FIELD);
    AcBuffer gpu_field_xzy        =acDeviceTranspose(acGridGetDevice(),STREAM_DEFAULT,XZY,FIELD);
    AcBuffer gpu_field_yxz        =acDeviceTranspose(acGridGetDevice(),STREAM_DEFAULT,YXZ,FIELD);
    AcBuffer gpu_field_yzx        =acDeviceTranspose(acGridGetDevice(),STREAM_DEFAULT,YZX,FIELD);
    AcBuffer gpu_field_zxy        =acDeviceTranspose(acGridGetDevice(),STREAM_DEFAULT,ZXY,FIELD);

    acGridSynchronizeStream(STREAM_ALL);
    AcBuffer field_zyx        =acBufferCopy(gpu_field_zyx,false);
    AcBuffer field_xzy        =acBufferCopy(gpu_field_xzy,false);
    AcBuffer field_yxz        =acBufferCopy(gpu_field_yxz,false);
    AcBuffer field_yzx        =acBufferCopy(gpu_field_yzx,false);
    AcBuffer field_zxy        =acBufferCopy(gpu_field_zxy,false);


    acGridExecuteTaskGraph(graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridFinalizeReduceLocal(graph);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT,&candidate);


    const auto gpu_max_val = acDeviceGetOutput(acGridGetDevice(), AC_max_val);
    const auto gpu_min_val = acDeviceGetOutput(acGridGetDevice(), AC_min_val);
    const auto gpu_sum_val = acDeviceGetOutput(acGridGetDevice(), AC_sum_val);
    const auto gpu_int_sum = acDeviceGetOutput(acGridGetDevice(), AC_int_sum_val);

    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreProfile(acGridGetDevice(), PROF_X,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_Y,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_Z,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_XY,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_XZ,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_YX,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_YZ,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_ZX,  &model);
    acDeviceStoreProfile(acGridGetDevice(), PROF_ZY,  &model);
    acGridSynchronizeStream(STREAM_ALL);
    const AcReal* x_sum_gpu = model.profile[PROF_X];
    const AcReal* y_sum_gpu = model.profile[PROF_Y];
    const AcReal* z_sum_gpu = model.profile[PROF_Z];
    const AcReal* xy_sum_gpu = model.profile[PROF_XY];
    const AcReal* xz_sum_gpu = model.profile[PROF_XZ];
    const AcReal* yx_sum_gpu = model.profile[PROF_YX];
    const AcReal* yz_sum_gpu = model.profile[PROF_YZ];
    const AcReal* zx_sum_gpu = model.profile[PROF_ZX];
    const AcReal* zy_sum_gpu = model.profile[PROF_ZY];

    AcReal cpu_max_val = -1000000.000000000;
    AcReal cpu_min_val = +1000000.000000000;
    int cpu_int_sum = 0;
    long double long_cpu_sum_val = (long double)0.0;
    bool transpose_correct = true;
    {

    	auto TRANSPOSED_IDX = [](const int z, const int y, const int x)
    	{
    	    const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    	    return x + dims.m1.z*y + dims.m1.y*dims.m1.z*z;
    	};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		auto transposed_val = field_zyx[TRANSPOSED_IDX(i,j,k)];
    	    		auto t_eq = val == transposed_val;
    	    		transpose_correct &= t_eq;
    	    		if(!t_eq)
    	    		{
    	    			printf("WRONG: %14e, %14e at %d,%d,%d\n",val,transposed_val,i,j,k);
    	    		}
    	    	}
    	    }
    	}
    }
    bool xzy_correct = true;
    {

    	auto TRANSPOSED_IDX = [](const int x, const int z, const int y)
    	{
    	    const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    	    return x + dims.m1.x*y + dims.m1.x*dims.m1.z*z;
    	};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		auto transposed_val = field_xzy[TRANSPOSED_IDX(i,j,k)];
    	    		auto t_eq = val == transposed_val;
    	    		xzy_correct &= t_eq;
    	    		if(!t_eq)
    	    		{
    	    			printf("WRONG: %14e, %14e at %d,%d,%d\n",val,transposed_val,i,j,k);
    	    		}
    	    	}
    	    }
    	}
    }
    bool yxz_correct = true;
    {

    	auto TRANSPOSED_IDX = [](const int y, const int x, const int z)
    	{
    	    const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    	    return x + dims.m1.y*y + dims.m1.y*dims.m1.x*z;
    	};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		auto transposed_val = field_yxz[TRANSPOSED_IDX(i,j,k)];
    	    		auto t_eq = val == transposed_val;
    	    		yxz_correct &= t_eq;
    	    		if(!t_eq)
    	    		{
    	    			printf("WRONG: %14e, %14e at %d,%d,%d\n",val,transposed_val,i,j,k);
    	    		}
    	    	}
    	    }
    	}
    }
    bool yzx_correct = true;
    {

    	auto TRANSPOSED_IDX = [](const int z, const int x, const int y)
    	{
    	    const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    	    return x + dims.m1.y*y + dims.m1.y*dims.m1.z*z;
    	};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		auto transposed_val = field_yzx[TRANSPOSED_IDX(i,j,k)];
    	    		auto t_eq = val == transposed_val;
    	    		yzx_correct &= t_eq;
    	    		if(!t_eq)
    	    		{
    	    			printf("WRONG: %14e, %14e at %d,%d,%d\n",val,transposed_val,i,j,k);
    	    		}
    	    	}
    	    }
    	}
    }
    bool zxy_correct = true;
    {

    	auto TRANSPOSED_IDX = [](const int y, const int z, const int x)
    	{
    	    const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    	    return x + dims.m1.z*y + dims.m1.x*dims.m1.z*z;
    	};
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    	{
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
    		{
    			for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	    	{
    	    		auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
    	    		auto transposed_val = field_zxy[TRANSPOSED_IDX(i,j,k)];
    	    		auto t_eq = val == transposed_val;
    	    		zxy_correct &= t_eq;
    	    		if(!t_eq)
    	    		{
    	    			printf("WRONG: %14e, %14e at %d,%d,%d\n",val,transposed_val,i,j,k);
    	    		}
    	    	}
    	    }
    	}
    }
    AcReal* z_sum   = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mz]);
    AcReal* x_sum   = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mx]);
    AcReal* y_sum   = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_my]);
    AcReal* xy_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mxy]);
    AcReal* xz_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mxz]);
    AcReal* yx_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mxy]);
    AcReal* yz_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_myz]);
    AcReal* zx_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_mxz]);
    AcReal* zy_sum  = (AcReal*)malloc(sizeof(AcReal)*model.info.int_params[AC_myz]);
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
	x_sum[i] = 0.0;
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    		for(int j = dims.n0.y; j < dims.n1.y; ++j)
			x_sum[i] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    }
    for(int j = dims.n0.y; j < dims.n1.y; ++j)
    {
	y_sum[j] = 0.0;
    	for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
			y_sum[j] += model.vertex_buffer[FIELD][IDX(i,j,k)];
    }
    for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    {
	z_sum[k] = 0.0;
    	for(int j = dims.n0.y; j < dims.n1.y; ++j)
    	{
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
		{
			auto val   = model.vertex_buffer[FIELD][IDX(i,j,k)];
			cpu_max_val = (val > cpu_max_val)  ? val : cpu_max_val;
			if(val < cpu_min_val)
			{
				cpu_min_val = val;
			}
			long_cpu_sum_val += (long double)val;
			cpu_int_sum += (int)val;
			z_sum[k] += val;
		}
	}
    }
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int j = dims.n0.y; j < dims.n1.y; ++j)
    	{
		const int index = i + model.info[AC_mx]*j;
		xy_sum[index] = 0.0;
    		for(int k = dims.n0.z; k < dims.n1.z;  ++k)
    		{
			xy_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int k = dims.n0.z; k < dims.n1.z; ++k)
    	{
		const int index = i + model.info[AC_mx]*k;
		xz_sum[index] = 0.0;
    		for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
			xz_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
		const int index = j + model.info[AC_my]*i;
		yx_sum[index] = 0.0;
    		for(int k = dims.n0.z; k < dims.n1.z; ++k)
    		{
			yx_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
		const int index = j + model.info[AC_my]*k;
		yz_sum[index] = 0.0;
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
    		{
			yz_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	{
		const int index = k + model.info[AC_mz]*i;
		zx_sum[index] = 0.0;
    		for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    		{
			zx_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
		const int index = k + model.info[AC_mz]*j;
		zy_sum[index] = 0.0;
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
    		{
			zy_sum[index] += model.vertex_buffer[FIELD][IDX(i,j,k)];
		}
    	}
    }
    AcReal cpu_sum_val = (AcReal)long_cpu_sum_val;
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
    bool sums_correct = true;
    for(int i = dims.n0.z; i < dims.n1.z; ++i)
    {
    	bool correct =  in_eps_threshold(z_sum[i],z_sum_gpu[i]);
	sums_correct &= correct;
	if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",z_sum[i],z_sum_gpu[i]);
    }
    bool x_sum_correct = true;
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	bool correct =  in_eps_threshold(x_sum[i],x_sum_gpu[i]);
	x_sum_correct &= correct;
	if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",x_sum[i],x_sum_gpu[i]);
    }
    bool y_sum_correct = true;
    for(int i = dims.n0.y; i < dims.n1.y; ++i)
    {
    	bool correct =  in_eps_threshold(y_sum[i],y_sum_gpu[i]);
	y_sum_correct &= correct;
	if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",y_sum[i],y_sum_gpu[i]);
    }
    bool xy_sum_correct = true;
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int j = dims.n0.y; j < dims.n1.y; ++j)
    	{
		const int index = i + model.info[AC_mx]*j;
		const bool correct = in_eps_threshold(xy_sum[index],xy_sum_gpu[index]);
		xy_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",xy_sum[index],xy_sum_gpu[index]);
	}
    }
    bool xz_sum_correct = true;
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int k = dims.n0.z; k < dims.n1.z; ++k)
    	{
		const int index = i + model.info[AC_mx]*k;
		const bool correct = in_eps_threshold(xz_sum[index],xz_sum_gpu[index]);
		xz_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",xz_sum[index],xz_sum_gpu[index]);
	}
    }
    bool yx_sum_correct = true;
    for(int i = dims.n0.x; i < dims.n1.x; ++i)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
		const int index = j + model.info[AC_my]*i;
		const bool correct = in_eps_threshold(yx_sum[index],yx_sum_gpu[index]);
		yx_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",yx_sum[index],yx_sum_gpu[index]);
    	}
    }
    bool yz_sum_correct = true;
    for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    {
    	for(int k = dims.n0.z; k < dims.n1.z; ++k)
    	{
		const int index = j + model.info[AC_my]*k;
		const bool correct = in_eps_threshold(yz_sum[index],yz_sum_gpu[index]);
		yz_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",yz_sum[index],yz_sum_gpu[index]);
    	}
    }
    bool zx_sum_correct = true;
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int i = dims.n0.x; i < dims.n1.x; ++i)
    	{
		const int index = k + model.info[AC_mz]*i;
		const bool correct = in_eps_threshold(zx_sum[index],zx_sum_gpu[index]);
		zx_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",zx_sum[index],zx_sum_gpu[index]);
    	}
    }
    bool zy_sum_correct = true;
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
		const int index = k + model.info[AC_mz]*j;
		const bool correct = in_eps_threshold(zy_sum[index],zy_sum_gpu[index]);
		zy_sum_correct &= correct;
		if(!correct) fprintf(stderr,"WRONG: %14e, %14e\n",zy_sum[index],zy_sum_gpu[index]);
    	}
    }
    //TP: since each profile is allowed to have epsilon amount of round-off the result using them should be at most epsilon*NUM_PROFILES
    epsilon *= NUM_PROFILES;
    bool remove_mean_correct = true;
    for(int k = dims.n0.z; k < dims.n1.z; ++k)
    {
    	for(int j = dims.n0.y; j < dims.n1.y;  ++j)
    	{
    		for(int i = dims.n0.x; i < dims.n1.x; ++i)
    		{
    	    		auto res = model.vertex_buffer[FIELD][IDX(i,j,k)];
			auto gpu_res =  candidate.vertex_buffer[FIELD][IDX(i,j,k)];
			res -= x_sum[i];
			res -= y_sum[j];
			res -= z_sum[k];

			res -= xy_sum[i + model.info[AC_mx]*j];
			res -= xz_sum[i + model.info[AC_mx]*k];

			res -= yx_sum[j + model.info[AC_my]*i];
			res -= yz_sum[j + model.info[AC_my]*k];

			res -= zx_sum[k + model.info[AC_mz]*i];
			res -= zy_sum[k + model.info[AC_mz]*j];
			const bool correct = in_eps_threshold(res,gpu_res);
			remove_mean_correct &= correct;
			if(!correct)  fprintf(stderr,"WRONG: %14e, %14e, %14e\n",res,gpu_res,relative_diff(res,gpu_res));
    		}
	}
    }

    fprintf(stderr,"MAX REDUCTION... %s %14e %14e\n", cpu_max_val == gpu_max_val ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_max_val,gpu_max_val);
    fprintf(stderr,"MIN REDUCTION... %s %14e %14e\n", cpu_min_val == gpu_min_val ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_min_val,gpu_min_val);
    fprintf(stderr,"SUM REDUCTION... %s %14e %14e\n", in_eps_threshold(cpu_sum_val,gpu_sum_val) ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_sum_val,gpu_sum_val);
    fprintf(stderr,"INT SUM REDUCTION... %s %d %d\n", cpu_int_sum == gpu_int_sum ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET,cpu_int_sum,gpu_int_sum); 
    fprintf(stderr,"X SUM REDUCTION... %s\n", x_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"Y SUM REDUCTION... %s\n", y_sum_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"Z SUM REDUCTION... %s\n", sums_correct    ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"XY SUM REDUCTION... %s\n", xy_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"XZ SUM REDUCTION... %s\n", xz_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"YX SUM REDUCTION... %s\n", yx_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"YZ SUM REDUCTION... %s\n", yz_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"ZX SUM REDUCTION... %s\n", zx_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"ZY SUM REDUCTION... %s\n", zy_sum_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"ZYX TRANSPOSE... %s\n", transpose_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"XZY TRANSPOSE... %s\n", xzy_correct       ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"YXZ TRANSPOSE... %s\n", yxz_correct       ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"YZX TRANSPOSE... %s\n", yzx_correct       ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"ZXY TRANSPOSE... %s\n", zxy_correct       ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"REMOVE MEAN... %s\n", remove_mean_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }
    finalized = true;

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);

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
