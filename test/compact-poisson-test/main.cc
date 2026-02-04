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

    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("sor.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    (2*AC_REAL_PI)/(info[AC_ngrid].x+1),
	    (2*AC_REAL_PI)/(info[AC_ngrid].y+1),
	    (2*AC_REAL_PI)/(info[AC_ngrid].z+1)
    });
    acPushToConfig(info,AC_first_gridpoint,info[AC_ds]);

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_periodic_grid,(AcBool3){false,false,false});
    info.comm->handle = MPI_COMM_WORLD;

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    acSetLocalMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);

    acGridInit(info);
    acDeviceSetInput(acGridGetDevice(),AC_SOR_omega,1.0);
    const auto empty_graph = acGetOptimizedDSLTaskGraph(empty_steps);
    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);

    //TP: this sets for the next graph that the halo exchange is red-black and only for it
    //acDeviceSetInput(acGridGetDevice(),AC_red_black_halo_exchange,AC_RED_BLACK_STATE_RED);
    const auto jacobi_graph = acGetOptimizedDSLTaskGraph(jacobi_step);
    const auto cg_graph = acGetOptimizedDSLTaskGraph(cg_step);
    const auto cg_4th_order_graph = acGetOptimizedDSLTaskGraph(cg_4th_order_step);
    const auto residual_graph = acGetOptimizedDSLTaskGraph(get_residual);
    acGridExecuteTaskGraph(initcond_graph,1);
    const AcReal rhs_l2 = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_rhs2));
    AcReal relative_residual = 10e8;
    const int max_step = 100000;
    int step = 0;
    while(relative_residual > 1e-15 && step < max_step)
    {
    	//acGridExecuteTaskGraph(jacobi_graph,1);
    	acGridExecuteTaskGraph(cg_graph,1);
    	acGridExecuteTaskGraph(cg_4th_order_graph,1);
    	acGridExecuteTaskGraph(residual_graph,1);
	const int N = info[AC_ngrid].x*info[AC_ngrid].y*info[AC_ngrid].z;
	const AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2)/N);
	relative_residual = residual/rhs_l2;
	fprintf(stderr,"Relative residual: %.14e\n",relative_residual);
	++step;
    }
    if(pid == 0) fprintf(stderr,"Final relative residual: %14e\n",relative_residual);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_diff_to_analytical_solution),1);
    const AcReal l2_diff = sqrt(info[AC_ds].x*info[AC_ds].y*info[AC_ds].z*acDeviceGetOutput(acGridGetDevice(),AC_l2_from_analytical_solution));
    const AcReal l2_diff_4th_order = sqrt(info[AC_ds].x*info[AC_ds].y*info[AC_ds].z*acDeviceGetOutput(acGridGetDevice(),AC_4th_order_l2_from_analytical_solution));
    fprintf(stderr,"L2 diff to solution: %.14e\n",l2_diff);
    fprintf(stderr,"L2 4th order diff to solution: %.14e\n",l2_diff_4th_order);
    fprintf(stderr,"h is: %.14e\n",info[AC_ds].x);
    fprintf(stderr,"h^2 is: %.14e\n",info[AC_ds_2].x);
    fprintf(stderr,"h^4 is: %.14e\n",info[AC_ds_4].x);
    AcReal sum = 0.0;
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&model);
    for(int x = 0; x < info[AC_mlocal].x; ++x)
    {
    	for(int y = 0; y < info[AC_mlocal].y; ++y)
    	{
    		for(int z = 0; z < info[AC_mlocal].z; ++z)
    		{
			const AcReal3 pos = (AcReal3){info[AC_ds].x*(x-NGHOST),
					    +info[AC_ds].x*(y-NGHOST),
					    +info[AC_ds].x*(z-NGHOST)} + info[AC_first_gridpoint];
			const AcReal f = -std::sin(pos.x)*std::sin(pos.y)*std::sin(pos.z)*(1.0/3.0);
			const AcReal res = model.vertex_buffer[HEAT_SOLUTION][acVertexBufferIdx(x,y,z,info)];
			const AcReal fourth_order_res = model.vertex_buffer[HEAT_SOLUTION_4TH_ORDER][acVertexBufferIdx(x,y,z,info)];
			const AcReal err = f-res;
			const AcReal fourth_order_err = f-fourth_order_res;
			if(x == 0 && y == 0 && z == 0)
			{
				fprintf(stderr,"Err at 0,0,0: %.14e\n",err);
			}
			if(x == NGHOST-1 && y == NGHOST-1 && z == NGHOST-1)
			{
				fprintf(stderr,"Err at 2,2,2: %.14e\n",err);
				fprintf(stderr,"F at (-1,-1,-1): %.14e\n",f);
			}
			if(x == info[AC_nlocal_max].x && y == info[AC_nlocal_max].x && z == info[AC_nlocal_max].x)
			{
				fprintf(stderr,"Err at 2,2,2: %.14e\n",err);
				fprintf(stderr,"F at (+1,+1,+1): %.14e\n",f);
			}
			if(x == NGHOST && y == NGHOST && z == NGHOST)
			{
				fprintf(stderr,"Err at 3,3,3: %.14e\n",err);
				fprintf(stderr,"4th order Err at 3,3,3: %.14e\n",fourth_order_err);
				fprintf(stderr,"Diff at 3,3,3: %.14e\n",fourth_order_res-res);
			}
			if(x == info[AC_mlocal].x/2 && y == info[AC_mlocal].x/2 && z == info[AC_mlocal].x/2)
			{
				fprintf(stderr,"Err at middle: %.14e\n",err);
				fprintf(stderr,"4th order Err at middle: %.14e\n",fourth_order_err);
				fprintf(stderr,"Diff at middle: %.14e\n",fourth_order_res-res);
			}
			sum += err*err;

		}
	}
    }
    fprintf(stderr,"L2 error including boundaries: %.14e\n",sqrt(info[AC_ds].x*info[AC_ds].x*info[AC_ds].x*sum));
    fprintf(stderr,"At (0,0,0): %.14e\n",model.vertex_buffer[HEAT_SOLUTION][0]);
    acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);
    acGridSynchronizeStream(STREAM_ALL);

    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "SOR_TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return retval == AC_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
