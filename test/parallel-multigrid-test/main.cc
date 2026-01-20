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
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
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
    acPushToConfig(info,AC_gmg_number_of_levels,
		    2
    );

    acPushToConfig(info,AC_periodic_grid,
    (AcBool3){
    	false,false,false
    });

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    //acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    //acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_EXTERNAL);
    //acPushToConfig(info,AC_domain_decomposition,(int3){2,1,1});
    info.comm->handle = MPI_COMM_WORLD;

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    
    const int nxgrid = 31;
    const int nygrid = 31;
    const int nzgrid = 31;
    acSetGridMeshDims(nxgrid,nygrid,nzgrid, &info);
    const int3 decomp = acDecompose(nprocs,info);
    int nx = (1+nxgrid)/decomp.x;
    int ny = (1+nygrid)/decomp.y;
    int nz = (1+nzgrid)/decomp.z;
    const int3 pid3d = acGetPid3D(pid, decomp, info);
    if(pid3d.x == decomp.x-1) nx--;
    if(pid3d.y == decomp.y-1) ny--;
    if(pid3d.z == decomp.z-1) nz--;
    acPushToConfig(info,AC_power_of_two_minus_one_grid,true);
    acPushToConfig(info,AC_allow_non_divisible_grid,true);
    acSetLocalMeshDims(nx,ny,nz,&info);

    fprintf(stderr,"%d Local Mesh: (%d,%d,%d)\n"
		    ,pid
		    ,info[AC_nlocal].x
		    ,info[AC_nlocal].y
		    ,info[AC_nlocal].z
	   );

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
    //Test that can build test ComputeSteps
    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_write_del2),1);
    const int n_levels = info[AC_gmg_number_of_levels];
    for(int i = 0; i < n_levels; ++i)
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)i);
    	const Volume full_launch_start = to_volume(info[AC_nmin]);
	const Volume full_launch_dims = to_volume(info[level_dims[i]]);
    	const Volume full_launch_end = full_launch_dims + full_launch_start;
    	const auto mg_residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual,full_launch_start,full_launch_end);

	if(i < 4-1)
	{
    		const Volume launch_start = to_volume(info[AC_nmin]);
		const Volume launch_dims = to_volume(info[level_dims[i+1]]);
    		const Volume launch_end = launch_dims + launch_start;
    		const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end);
    		acGetOptimizedDSLTaskGraph(gmg_restrict_solution, launch_start, launch_end);
    		const auto prolong_graph = acGetOptimizedDSLTaskGraph(gmg_prolong_solution, launch_start, launch_end);
	}
    	const Volume launch_start = to_volume(info[AC_nmin]);
    	const Volume launch_dims = to_volume(info[level_dims[i]]);
    	const Volume launch_end = launch_dims + launch_start;
    	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(randomize_residual_and_solution,launch_start,launch_end),1);
    }


    fprintf(stderr,"1/h2: %.14e\n",1.0/info[AC_inv_ds_2].x);
    fflush(stderr);

    gmg_v_cycle(n_levels,1e-1);
    fprintf(stderr,"GMG\n");
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    	const auto residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual_norm);
    	acGridExecuteTaskGraph(initcond_graph,1);
    	acGridExecuteTaskGraph(residual_graph,1);
    	AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	fprintf(stderr,"Initial Residual: %14e\n",residual);
	const AcReal relative_convergence_rate = 1e-14;
    	int n_steps = 0;
    	while(residual > 1e-8)
    	{
            gmg_v_cycle(n_levels,relative_convergence_rate);
    	    acGridExecuteTaskGraph(residual_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	    acGridWriteSlicesToDiskCollectiveSynchronous("slices", n_steps, 0.0);
    	    ++n_steps;
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
    }
    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;
    if (pid == 0)
        fprintf(stderr, "GMG_TEST complete: %s\n",
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
