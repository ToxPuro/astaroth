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

#include <stdlib.h>
double
drand()
{
	return (double)(rand()) / (double)(rand());
}

int
main(int argc, char* argv[])
{
    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.comm->handle = MPI_COMM_WORLD;
    acPushToConfig(info,AC_periodic_grid,(AcBool3){false,false,false});

    const int nx = argc > 1 ? atoi(argv[1]) : 2*9;
    const int ny = argc > 2 ? atoi(argv[2]) : 2*11;
    const int nz = argc > 3 ? atoi(argv[3]) : 4*7;

    acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    const int3 decomp = acDecompose(nprocs,info);
    const int3 pid_3d = acGetPid3D(pid,decomp,info);

    acSetGridMeshDims(nx*decomp.x, ny*decomp.y, nz*decomp.z, &info);
    acSetLocalMeshDims(nx, ny, nz, &info);
    acHostUpdateParams(&info);

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    acGridInit(info);
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT,model);

    acGridSynchronizeStream(STREAM_ALL);
    const auto get_tg = [&](const auto& cs)
    {
	    return acGetOptimizedDSLTaskGraph(cs,true);
    };

    const auto get_tg_fixed_bcs = [&](const auto& cs)
    {
	    return acGetOptimizedDSLTaskGraph(cs,true,fixed_bcs);
    };

    //TP: for benchmarking purposes
    acGridExecuteTaskGraph(get_tg(trace_nine_rays_general),1);
    acGridExecuteTaskGraph(get_tg(ppp_general),1);
    acGridExecuteTaskGraph(get_tg(trace_up_general),1);
    acGridExecuteTaskGraph(get_tg(trace_three_rays_general),1);
    acGridExecuteTaskGraph(get_tg(trace_right_ray_general),1);
    acGridExecuteTaskGraph(get_tg(trace_up_rays),1);
    acGridExecuteTaskGraph(get_tg(trace_right_rays),1);

    acGridExecuteTaskGraph(get_tg(baseline_y),1);
    acGridExecuteTaskGraph(get_tg(trace_three_rays),1);

    acGridExecuteTaskGraph(get_tg(baseline),1);
    acGridExecuteTaskGraph(get_tg(revise_nine_rays),1);
    acGridExecuteTaskGraph(get_tg(revise_nine_rays_along_ray),1);
    acGridExecuteTaskGraph(get_tg(revise_boundary_ray),1);
    acGridExecuteTaskGraph(get_tg(trace_nine_rays),1);
    acGridExecuteTaskGraph(get_tg(trace_nine_rays_general),1);

    acGridExecuteTaskGraph(get_tg(trace_right_ray_general),1);

    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };

    int f_correct = 1;
    auto reset_f_to_val = [&](const Field f,const AcReal val)
    {
    	for(size_t i = 0; i < dims.m1.x; ++i)
    	{
    	  for(size_t j = 0; j < dims.m1.y; ++j)
    	  {
    		for(size_t k = 0; k < dims.m1.z;  ++k)
    	    {
    	    	model.vertex_buffer[f][IDX(i,j,k)] = val;
    	    }
    	  }
    	}
    };
    auto reset_f = [&](const Field f)
    {
	    reset_f_to_val(f,0.0);
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
          	if(!local_correct) fprintf(stderr,"%s: %s Wrong at (%d,%d,%d) %ld,%ld,%ld: %14e,%14e\n", name,field_names[f],pid_3d.x,pid_3d.y,pid_3d.z,i,j,k,model.vertex_buffer[f][IDX(i,j,k)],candidate.vertex_buffer[f][IDX(i,j,k)]);
          	f_correct &= local_correct;
          }
        }
      }
    };

    reset_f_to_val(QRAD,pid_3d.x*nx);
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

    acGridExecuteTaskGraph(get_tg(trace_right_rays),1);
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){1,0,0})
			})
		    ,1
		    );
    const auto fixed_bc_right = get_tg_fixed_bcs(trace_right_rays);
    if(pid_3d.x != 0)
    {
    	acGridExecuteTaskGraph(fixed_bc_right,1); 
    }
    check_f("Right",QRAD);

    reset_f_to_val(QRAD,(decomp.x-1-pid_3d.x)*nx);
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

    acGridExecuteTaskGraph(get_tg(trace_left_rays),1);
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){-1,0,0})
			})
		    ,1
		    );
    const auto fixed_bc_left = get_tg_fixed_bcs(trace_left_rays);
    if(pid_3d.x != decomp.x-1)
    {
    	acGridExecuteTaskGraph(fixed_bc_left,1); 
    }
    check_f("Left",QRAD);

    reset_f_to_val(QRAD,(decomp.z-1-pid_3d.z)*nz);
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
    acGridExecuteTaskGraph(get_tg(trace_backwards_rays),1);
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){0,0,-1})
			})
		    ,1
		    );
    const auto fixed_bc_backwards = get_tg_fixed_bcs(trace_backwards_rays);
    if(pid_3d.z != decomp.z-1)
    {
    	acGridExecuteTaskGraph(fixed_bc_backwards,1); 
    }
    check_f("Backwards",QRAD);

    reset_f_to_val(QRAD,pid_3d.z*nz);
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

    acGridExecuteTaskGraph(get_tg(trace_forwards_rays),1);
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){0,0,1})
			})
		    ,1
		    );
    const auto fixed_bc_forward = get_tg_fixed_bcs(trace_forwards_rays);
    if(pid_3d.z != 0)
    {
    	acGridExecuteTaskGraph(fixed_bc_forward,1); 
    }
    check_f("Forwards",QRAD);


    reset_f_to_val(QRAD,pid_3d.y*ny);
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
    acGridExecuteTaskGraph(get_tg(trace_up_rays),1); 
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){0,1,0})
			})
		    ,1
		    );
    const auto fixed_bc_up = get_tg_fixed_bcs(trace_up_rays);
    if(pid_3d.y != 0)
    {
    	acGridExecuteTaskGraph(fixed_bc_up,1); 
    }
    check_f("Up",QRAD);



    reset_f(QRAD);
    reset_f_to_val(QRAD,(decomp.y-1-pid_3d.y)*ny);
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

    acGridExecuteTaskGraph(get_tg(trace_down_rays),1);
    acGridExecuteTaskGraph(
	    acGridBuildTaskGraph({
	    		acScan({QRAD}, (int3){0,-1,0})
			})
		    ,1
		    );
    const auto fixed_bc_down = get_tg_fixed_bcs(trace_down_rays);
    if(pid_3d.y != decomp.y-1)
    {
    	acGridExecuteTaskGraph(fixed_bc_down,1); 
    }
    check_f("Down",QRAD);




    acHostMeshDestroy(&model);

    acGridQuit();
    ac_MPI_Finalize();
    const bool success = f_correct;
    if (pid == 0)
    {
        fprintf(stderr, "RAY_TEST complete: %s\n",
                success ? "No errors found" : "One or more errors found");
    }
    fflush(stdout);

    return !success;
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
