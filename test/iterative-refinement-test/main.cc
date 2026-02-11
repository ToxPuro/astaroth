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
    const auto sor_graph = acGetOptimizedDSLTaskGraph(sor_red_black_step);
    const auto residual_graph = acGetOptimizedDSLTaskGraph(get_residual);
    acGridExecuteTaskGraph(initcond_graph,1);
    const AcReal rhs_l2 = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_rhs2));
    AcReal relative_residual = 10e8;
    const int max_step = 100000;
    int step = 0;
    while(relative_residual > 1e-15 && step < max_step)
    {
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_sg_residual),1);
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_inner_l2_norm),1);
	const AcReal r0 = acDeviceGetOutput(acGridGetDevice(),AC_inner_l2_norm);
	AcReal inner_relative_residual = 1.0;	
	//fprintf(stderr,"R0: %.14e\n",r0);
	while(inner_relative_residual > 1e-5)
	{
    		acGridExecuteTaskGraph(sor_graph,1);
		const AcReal r = acDeviceGetOutput(acGridGetDevice(),AC_inner_l2_norm);
        	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_inner_l2_norm),1);
		inner_relative_residual = r/r0;
		//fprintf(stderr,"Inner relative residual: %.14e\n",inner_relative_residual);
	}
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(update_solution),1);
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
    fprintf(stderr,"L2 diff to solution: %.14e\n",l2_diff);
    fprintf(stderr,"h is: %.14e\n",info[AC_ds].x);
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
