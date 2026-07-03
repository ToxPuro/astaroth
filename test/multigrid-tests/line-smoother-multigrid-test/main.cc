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
#include "math_utils.h"
#include "errchk.h"
#include "user_constants.h"

#include "../../stdlib/geometric_multigrid.h"

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
main(int argc, char* argv[])
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
    acLoadConfig("mg.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].x,
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].y,
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].z
    });

    acPushToConfig(info,AC_first_gridpoint,
    	info[AC_ds]
    );

    //1.2 works better for GMG and 1.8 works better for plain SOR
    acPushToConfig(info,AC_SOR_omega,
		    1.2
		    //1.8
    );

    acPushToConfig(info,AC_periodic_grid,
    (AcBool3){
    	false,false,false
    });

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    info.comm->handle = MPI_COMM_WORLD;

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int nx = argc > 1 ? atoi(argv[1]): 31;
    const int ny = argc > 2 ? atoi(argv[2]): 31;
    const int nz = argc > 3 ? atoi(argv[3]): 31;
    
    //const int nx = 63;
    //const int ny = 63;
    //const int nz = 63;
    acSetGridMeshDims(nx,ny,nz, &info);
    acSetLocalMeshDims(nx,ny,nz, &info);

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }
    gmg_populate_central_coeffients(&info);
    acGridInit(info);
    gmg_setup(&info);

    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);
    const int n_levels = info[AC_gmg_number_of_levels];
    gmg_v_cycle(n_levels,1e-1);
    fprintf(stderr,"Hello\n");
    fflush(stderr);
    const AcReal relative_residual_tolerance = 1e-14;
    //const AcReal relative_residual_tolerance = 1.5e-1;
  
    fprintf(stderr,"GMG\n");
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    	const auto res_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual_norm);
    	acGridExecuteTaskGraph(initcond_graph,1);
    	acGridExecuteTaskGraph(res_graph,1);
    	AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	fprintf(stderr,"Initial Residual: %14e\n",residual);
	const AcReal init_residual = residual;
    	int n_steps = 0;
	AcReal sum_time = 0.0;
    	while(residual > 1e-8)
    	{
	    const AcReal start_time = MPI_Wtime();
            gmg_v_cycle(n_levels,relative_residual_tolerance);
	    const AcReal end_time   = MPI_Wtime();
	    sum_time += end_time-start_time;
    	    acGridExecuteTaskGraph(res_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	    acGridWriteSlicesToDiskCollectiveSynchronous("slices", n_steps, 0.0);
    	    ++n_steps;
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
	fprintf(stderr,"Asymptotic convergence factor: %.14e\n",pow(residual/init_residual,1.0/n_steps));
	fprintf(stderr,"On average a single V cycle took: %.14e seconds\n",sum_time/n_steps);
    }
    const auto smooth_y = acGetOptimizedDSLTaskGraph(line_y_smoother_step);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(compute_y_smooth_rhs),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(line_y_smoother_residual),1);
    const auto rhs_norm = sqrt(acDeviceGetOutput(acGridGetDevice(), AC_rhs2));
    AcReal residual_norm = sqrt(acDeviceGetOutput(acGridGetDevice(), AC_residual2));
    AcReal relative_norm = residual_norm/rhs_norm;
    while(relative_norm > 1e-4)
    {
	    acGridExecuteTaskGraph(smooth_y,1);
            acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(line_y_smoother_residual),1);
    	    residual_norm = sqrt(acDeviceGetOutput(acGridGetDevice(), AC_residual2));
    	    relative_norm = residual_norm/rhs_norm;
	    fprintf(stderr,"Relative norm: %.14e\n",relative_norm);
	    fflush(stderr);
    }

    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;
    if (pid == 0)
        fprintf(stderr, "LINE SMOOTHER TEST complete: %s\n",
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
