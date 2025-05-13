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
    const int3 pid3d = acGetPid3D(pid,decomp,info);

    acSetGridMeshDims(nx*decomp.x, ny*decomp.y, nz*decomp.z, &info);
    acSetLocalMeshDims(nx, ny, nz, &info);
    acHostUpdateParams(&info);

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);


    acGridInit(info);
    const auto extended_dims = acGetMeshDims(acGridGetLocalMeshInfo(),F_EXT);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_normal),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [](const int x, const int y, const int z, const Field f)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo(),f);
    };


    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(add_field_normal),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    int f_correct = 1;
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const bool local_correct = candidate.vertex_buffer[F][IDX(i,j,k,F)] == AC_F_INIT + 1.0;
		if(!local_correct) fprintf(stderr,"F Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }

    dims = acGetMeshDims(acGridGetLocalMeshInfo(),F_EXT);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(add_field_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    int f_ext_correct = 1;
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const bool local_correct = candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)] == AC_F_EXT_INIT+1.0;
		if(!local_correct) fprintf(stderr,"F_EXT Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
		f_correct &= local_correct;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(copy_normal_to_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		if(i >= info[AC_extended_halo].x+ dims.n0.x && j >= info[AC_extended_halo].y+ dims.n0.y && k >= info[AC_extended_halo].z + dims.n0.z && i < dims.n1.x-info[AC_extended_halo].x && j < dims.n1.y-info[AC_extended_halo].y && k < dims.n1.z-info[AC_extended_halo].z)
		{
			const bool local_correct = candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)] == AC_F_INIT+1.0;
			if(!local_correct) fprintf(stderr,"Inside F --> F_EXT Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
			f_ext_correct &= local_correct;
		}
		else
		{
			const bool local_correct = candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)] == AC_F_EXT_INIT;
			if(!local_correct) fprintf(stderr,"Out F --> F_EXT Wrong at %ld,%ld,%ld: %14e,%14e\n",i,j,k,AC_F_EXT_INIT,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
			f_ext_correct &= local_correct;
		}
	}
      }
    }
    const int3 launch_dims_int3 = 
		    (int3)
		    {
		    	info[AC_nlocal].x/2,
		    	info[AC_nlocal].y/2,
		    	info[AC_nlocal].z/2
		    };
    const Volume launch_dims = to_volume(
		    launch_dims_int3
		    );

    const Volume launch_start = to_volume(info[AC_nmin]);
    const Volume launch_end = launch_dims + launch_start;
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraphWithBounds(add_field_normal , launch_start, launch_end),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);

    dims = acGetMeshDims(acGridGetLocalMeshInfo(),F);

    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		bool local_correct = candidate.vertex_buffer[F][IDX(i,j,k,F)] == AC_F_INIT+2.0;
		if(i >= launch_end.x || j >= launch_end.y || k >= launch_end.z)
		{
			//TP: this is zero instead of 2 because of the buffer swapping
			local_correct = candidate.vertex_buffer[F][IDX(i,j,k,F)] == 0.0;
		}
		if(!local_correct) fprintf(stderr,"F left side/rigth side Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }

    //const auto extended_dims = acGetMeshDims(acGridGetLocalMeshInfo(),F_EXT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(copy_extended_to_normal),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const bool local_correct = candidate.vertex_buffer[F][IDX(i,j,k,F)] == AC_F_INIT + 1.0;
		if(!local_correct) fprintf(stderr,"F Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_normal),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivx_normal),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.x == 0 && i == dims.n0.x) || (pid3d.x == decomp.x-1 && i == dims.n1.x-1)) ? -1.0 :
 				       ((pid3d.x == 0 && i == dims.n0.x+1) || (pid3d.x == decomp.x-1 && i == dims.n1.x-2)) ? 0.70/3.0 :
 				       ((pid3d.x == 0 && i == dims.n0.x+2) || (pid3d.x == decomp.x-1 && i == dims.n1.x-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F][IDX(i,j,k,F)]-correct) < 1e-12;
		if(!local_correct) fprintf(stderr,"F Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_normal),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivy_normal),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.y == 0 && j == dims.n0.y) || (pid3d.y == decomp.y-1 && j == dims.n1.y-1)) ? -1.0 :
 				       ((pid3d.y == 0 && j == dims.n0.y+1) || (pid3d.y == decomp.y-1 && j == dims.n1.y-2)) ? 0.70/3.0 :
 				       ((pid3d.y == 0 && j == dims.n0.y+2) || (pid3d.y == decomp.y-1 && j == dims.n1.y-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F][IDX(i,j,k,F)]-correct) < 1e-12;
		if(!local_correct) fprintf(stderr,"F Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_normal),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivz_normal),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.z == 0 && k == dims.n0.z)   || (pid3d.z == decomp.z-1 && k == dims.n1.z-1)) ? -1.0 :
 				       ((pid3d.z == 0 && k == dims.n0.z+1) || (pid3d.z == decomp.z-1 && k == dims.n1.z-2)) ? 0.70/3.0 :
 				       ((pid3d.z == 0 && k == dims.n0.z+2) || (pid3d.z == decomp.z-1 && k == dims.n1.z-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F][IDX(i,j,k,F)]-correct) < 1e-12;
		if(!local_correct) fprintf(stderr,"F Wrong at %ld,%ld,%ld: %14e\n",i,j,k,candidate.vertex_buffer[F][IDX(i,j,k,F)]);
		f_correct &= local_correct;
	}
      }
    }

    dims = extended_dims;

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivx_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.x == 0 && i == dims.n0.x)   || (pid3d.x == decomp.x-1 && i == dims.n1.x-1)) ? -1.0 :
 				       ((pid3d.x == 0 && i == dims.n0.x+1) || (pid3d.x == decomp.x-1 && i == dims.n1.x-2)) ? 0.70/3.0 :
 				       ((pid3d.x == 0 && i == dims.n0.x+2) || (pid3d.x == decomp.x-1 && i == dims.n1.x-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]-correct) < 1e-12;
		if(!local_correct) 
		{
			fprintf(stderr,"X DER F_EXT Wrong in %d at %ld,%ld,%ld: %14e\n",pid3d.x,i,j,k,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
			fprintf(stderr,"DIMS %ld,%ld,%ld\n",dims.m1.x,dims.m1.y,dims.m1.z);
			exit(EXIT_FAILURE);

		}
		f_correct &= local_correct;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivy_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.y == 0 && j == dims.n0.y)   || (pid3d.y == decomp.y-1 && j == dims.n1.y-1)) ? -1.0 :
 				       ((pid3d.y == 0 && j == dims.n0.y+1) || (pid3d.y == decomp.y-1 && j == dims.n1.y-2)) ? 0.70/3.0 :
 				       ((pid3d.y == 0 && j == dims.n0.y+2) || (pid3d.y == decomp.y-1 && j == dims.n1.y-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]-correct) < 1e-12;
		if(!local_correct) 
		{
			fprintf(stderr,"Y DER F_EXT Wrong in %d at %ld,%ld,%ld: %14e\n",pid3d.x,i,j,k,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
			fprintf(stderr,"DIMS %ld,%ld,%ld\n",dims.m1.x,dims.m1.y,dims.m1.z);
			exit(EXIT_FAILURE);

		}
		f_correct &= local_correct;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(derivz_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t i = dims.n0.x; i < dims.n1.x; ++i)
    {
      for(size_t j = dims.n0.y; j < dims.n1.y; ++j)
      {
    	for(size_t k = dims.n0.z; k < dims.n1.z;  ++k)
	{
		const AcReal correct = ((pid3d.z == 0 && k == dims.n0.z)   || (pid3d.z == decomp.z-1 && k == dims.n1.z-1)) ? -1.0 :
 				       ((pid3d.z == 0 && k == dims.n0.z+1) || (pid3d.z == decomp.z-1 && k == dims.n1.z-2)) ? 0.70/3.0 :
 				       ((pid3d.z == 0 && k == dims.n0.z+2) || (pid3d.z == decomp.z-1 && k == dims.n1.z-3)) ? -0.10/3.0 :
			0.0;
		const bool local_correct = std::abs(candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]-correct) < 1e-12;
		if(!local_correct) 
		{
			fprintf(stderr,"Z DER F_EXT Wrong in %d at %ld,%ld,%ld: %14e\n",pid3d.x,i,j,k,candidate.vertex_buffer[F_EXT][IDX(i,j,k,F_EXT)]);
			fprintf(stderr,"DIMS %ld,%ld,%ld\n",dims.m1.x,dims.m1.y,dims.m1.z);
			exit(EXIT_FAILURE);

		}
		f_correct &= local_correct;
	}
      }
    }
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(reduce_extended),1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT,&candidate);
    acGridSynchronizeStream(STREAM_ALL);
    const AcReal sum = acDeviceGetOutput(acGridGetDevice(),F_EXT_SUM);
    const AcReal correct_sum = decomp.x*decomp.y*decomp.z*dims.nn.x*dims.nn.y*dims.nn.z*AC_F_EXT_INIT;
    const bool f_ext_reduce_correct = sum == correct_sum;
    if(!f_ext_reduce_correct) fprintf(stderr,"Reduce wrong!: %14e vs. %14e\n",correct_sum,sum);

    fprintf(stderr,"F %d: ... %s\n",pid,f_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"F_EXT %d: ... %s\n",pid,f_ext_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"F_EXT Reduce %d: ... %s\n",pid,f_ext_reduce_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    MPI_Allreduce(MPI_IN_PLACE, &f_correct,   1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &f_ext_correct,   1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    {
        char slice_frame_dir[2048];
        sprintf(slice_frame_dir, "slices");
        if (pid == 0) {
            {
                constexpr size_t cmdlen = 4096;
                static char cmd[cmdlen];
             
                // Improvement suggestion: use create_directory() from <filesystem> (C++17) or mkdir() from
                // <sys/stat.h> (POSIX)
                snprintf(cmd, cmdlen, "mkdir -p %s", "slices");
                system(cmd);
            }    
        }
        MPI_Barrier(acGridMPIComm()); // Ensure directory is created for all procs
        
        char label[80];
        sprintf(label, "step_0");
        acGridWriteSlicesToDiskCollectiveSynchronous(slice_frame_dir, label);
    }

    const bool success = f_correct && f_ext_correct;
    if (pid == 0)
    {
        fprintf(stderr, "VARIABLE_DIMS_TEST complete: %s\n",
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
