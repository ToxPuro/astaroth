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
#include "math_utils.h"
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
    acPushToConfig(info,AC_periodic_grid,(AcBool3){false,false,false});

    const int nx = argc > 1 ? atoi(argv[1]) : 2*9;
    const int ny = argc > 2 ? atoi(argv[2]) : 2*11;
    const int nz = argc > 3 ? atoi(argv[3]) : 4*7;

    acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    const int3 decomp = acDecompose(nprocs,info);

    acSetGridMeshDims(nx*decomp.x, ny*decomp.y, nz*decomp.z, &info);
    acSetLocalMeshDims(nx, ny, nz, &info);
    acHostUpdateParams(&info);

    acGridInit(info);

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    size_t free_mem,total_mem;
    acDeviceMemGetInfo(acGridGetDevice(),&free_mem,&total_mem);
    const size_t total_reals = total_mem/sizeof(AcReal);
    const AcReal number_of_available_vtxbufs = ((AcReal)total_reals)/(info[AC_mlocal].x*info[AC_mlocal].y*info[AC_mlocal].z);
    fprintf(stderr,"Can fit %d vtxbufs\n", int(floor(number_of_available_vtxbufs)));
    fflush(stderr);

    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT,model);
    acGridSynchronizeStream(STREAM_ALL);

    //TP: for benchmarking purposes
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(baseline_y),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_three_rays),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_three_rays_general),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_up_general),1);

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(baseline),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(revise_nine_rays),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(revise_nine_rays_along_ray),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(revise_boundary_ray),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_nine_rays),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_nine_rays_general),1);

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_right_ray_general),1);

    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };

    int f_correct = 1;
    auto reset_f = [&](const Field f)
    {
    	for(size_t i = 0; i < dims.m1.x; ++i)
    	{
    	  for(size_t j = 0; j < dims.m1.y; ++j)
    	  {
    		for(size_t k = 0; k < dims.m1.z;  ++k)
    	    {
    	    	model.vertex_buffer[f][IDX(i,j,k)] = 1.0;
    	    }
    	  }
    	}
    };
    auto check_f = [&](const char* name, const Field f)
    {
      acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
      acGridSynchronizeStream(STREAM_ALL);
      for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
      {
        for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
        {
      	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
          {
          	const bool local_correct = candidate.vertex_buffer[f][IDX(i,j,k)] == model.vertex_buffer[f][IDX(i,j,k)];
          	if(!local_correct) fprintf(stderr,"%s: %s Wrong at %ld,%ld,%ld: %14e,%14e\n",name,field_names[f],i,j,k,model.vertex_buffer[f][IDX(i,j,k)],candidate.vertex_buffer[f][IDX(i,j,k)]);
          	f_correct &= local_correct;
          }
        }
      }
    };

    reset_f(QRAD);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n1.z-1; k >= dims.n0.z;  --k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i,j,k+1)] + 1.0;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_backwards_rays),1);
    check_f("Backwards",QRAD);

    reset_f(QRAD);
    for(size_t i = 0; i < dims.m1.x; ++i)
    {
      for(size_t j = 0; j < dims.m1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i,j,k-1)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_forwards_rays),1);
    check_f("Forwards",QRAD);

    reset_f(QRAD);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i,j-1,k)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_up_rays),1);
    check_f("Up",QRAD);

    reset_f(QRAD);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n1.y-1; j >= dims.n0.y; --j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i,j+1,k)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_down_rays),1);
    check_f("Down",QRAD);

    reset_f(QRAD);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i-1,j,k)] + 1.0;
	}
      }
    }

    /**
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_right_rays),1);
    check_f();

    reset_f();
    for(size_t i = dims.n1.x-1; i >= dims.n0.x; --i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i+1,j,k)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_left_rays),1);
    check_f();
    **/

    reset_f(QRAD);
    for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    {
      for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
      {
        for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
        {
		model.vertex_buffer[QRAD][IDX(i,j,k)] = model.vertex_buffer[QRAD][IDX(i-1,j-1,k-1)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_ppp_rays),1);
    check_f("(1,1,1)",QRAD);

    reset_f(Q_PPP);
    reset_f(Q_MPP);
    reset_f(Q_PMP);
    reset_f(Q_MMP);
    for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
    {
      for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
      {
        for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
        {
		model.vertex_buffer[Q_PPP][IDX(i,j,k)] = model.vertex_buffer[Q_PPP][IDX(i-1,j-1,k-1)] + 1.0;
		model.vertex_buffer[Q_MPP][IDX(i,j,k)] = model.vertex_buffer[Q_MPP][IDX(i+1,j-1,k-1)] + 1.0;
		model.vertex_buffer[Q_PMP][IDX(i,j,k)] = model.vertex_buffer[Q_PMP][IDX(i-1,j+1,k-1)] + 1.0;
		model.vertex_buffer[Q_MMP][IDX(i,j,k)] = model.vertex_buffer[Q_MMP][IDX(i+1,j+1,k-1)] + 1.0;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_nine_rays),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(trace_nine_rays),1);
    for(size_t k = dims.n0.z-1; k < dims.n1.z+1;  ++k)
    {
      for(size_t i = dims.n0.x-1; i < dims.n1.x+1; ++i)
      {
        for(size_t j = dims.n0.y-1; j < dims.n1.y+1; ++j)
        {
		if(
				k >= dims.n0.z && k < dims.n1.z
			&&      i >= dims.n0.x && i < dims.n1.x
			&&      j >= dims.n0.y && j < dims.n1.y
		  )
		{
			continue;
		}
          	const bool local_correct = candidate.vertex_buffer[Q_PPP][IDX(i,j,k)] == model.vertex_buffer[Q_PPP][IDX(i,j,k)];
          	if(!local_correct) fprintf(stderr,"Dirichlet boundary: %s Wrong at %ld,%ld,%ld: %14e,%14e\n",field_names[Q_PPP],i,j,k,model.vertex_buffer[Q_PPP][IDX(i,j,k)],candidate.vertex_buffer[Q_PPP][IDX(i,j,k)]);
          	f_correct &= local_correct;

	}
      }
    }
    check_f("four rays",Q_PPP);
    check_f("four rays",Q_MPP);
    check_f("four rays",Q_PMP);
    check_f("four rays",Q_MMP);

    acHostMeshDestroy(&model);
    finalized = true;

    acGridQuit();
    ac_MPI_Finalize();
    const bool success = f_correct;
    if (pid == 0)
    {
        fprintf(stderr, "VARIABLE_DIMS_TEST complete: %s\n",
                success ? "No errors found" : "One or more errors found");
    }
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
