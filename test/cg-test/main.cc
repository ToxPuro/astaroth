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
    acLoadConfig("cg.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    (2*AC_REAL_PI)/info[AC_ngrid].x,
	    (2*AC_REAL_PI)/info[AC_ngrid].y,
	    (2*AC_REAL_PI)/info[AC_ngrid].z
    });

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_cg_preconditioned,true);
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
    //Test that can build test ComputeSteps
    const auto empty_graph = acGetOptimizedDSLTaskGraph(empty_steps);
    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);
    const auto cg_init_graph = acGetOptimizedDSLTaskGraph(init_cg);
    const auto sor_graph = acGetOptimizedDSLTaskGraph(sor_red_black_step);
    const auto cg_graph = acGetOptimizedDSLTaskGraph(cg_step);
    const auto residual_graph = acGetOptimizedDSLTaskGraph(get_residual);
    const auto local_residual_graph = acGetOptimizedDSLTaskGraph(get_local_residual,true);
    const int MAX_STEPS = 200;

    const auto get_z = [&]()
    {
      
      acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_cg_z),1);

      //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(init_local_cg),1);
      //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_local_residual,true),1);
      //auto local_residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_local_l2_norm);
      //while(local_residual > 1e-8)
      //{
      // acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(cg_local_step,true),1);
      // acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_local_residual,true),1);
      // local_residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_local_l2_norm);
      // //fprintf(stderr,"Local res: %.14e\n",local_residual);
      //}
    };
    const auto cg_solve = [&]()
    {
       int step  = 0;
       acGridExecuteTaskGraph(cg_init_graph,1);
       acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);

       get_z();
       acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(copy_z_to_p),1);
       {
           //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(copy_z_to_solution),1);
           acGridExecuteTaskGraph(residual_graph,1);
           //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(get_residual_for_z),1);
           const AcReal residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_l2_norm);
           acLogFromRootProc(pid,"Initial residual: %.14e\n",residual);
       }
       AcReal residual = 10e8;
       while(residual > 1e-8 && step++ < MAX_STEPS)
       {
       	//acGridExecuteTaskGraph(cg_graph,1);
	{
          get_z();
	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(cg_compute_alpha_and_advance),1);
          get_z();
	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(cg_compute_beta_and_advance_p),1);
        }
       	acGridExecuteTaskGraph(residual_graph,1);
        residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_l2_norm);
	acLogFromRootProc(pid,"Residual: %.14e\n",residual);
       }
       acLogFromRootProc(pid,"CG took %d steps\n",step);
       acLogFromRootProc(pid,"Final residual: %.14e\n",residual);
    };

    const auto sor_solve = [&]()
    {
	int step  = 0;
       {
           acGridExecuteTaskGraph(residual_graph,1);
           const AcReal residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_l2_norm);
    	   acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &model);
           acLogFromRootProc(pid,"Initial residual: %.14e\n",residual);
       }
       AcReal residual = 10e8;
       while(residual > 1e-8 && step++ < MAX_STEPS)
       {
       	   acGridExecuteTaskGraph(sor_graph,1);
       	   acGridExecuteTaskGraph(residual_graph,1);
           residual = acDeviceGetOutput(acGridGetDevice(),AC_residual_l2_norm);
           //fprintf(stderr,"Residual: %.14e\n",residual);
       }
       acLogFromRootProc(pid,"SOR took %d steps\n",step);
       acLogFromRootProc(pid,"Final residual: %.14e\n",residual);
    };



    acDeviceSetInput(acGridGetDevice(),AC_laplace_sign,-1.0);
    acLogFromRootProc(pid,"CG: \n");
    acGridExecuteTaskGraph(initcond_graph,1);
    cg_solve();

    acLogFromRootProc(pid,"SOR: \n");
    acGridExecuteTaskGraph(initcond_graph,1);
    sor_solve();

    //TP: CG will fail due to not being SPD
    /**
    fprintf(stderr,"Switching sign -1 ---> +1\n");
    acDeviceSetInput(acGridGetDevice(),AC_laplace_sign,1.0);
    fprintf(stderr,"CG: \n");
    acGridExecuteTaskGraph(initcond_graph,1);
    cg_solve();

    fprintf(stderr,"SOR: \n");
    acGridExecuteTaskGraph(initcond_graph,1);
    sor_solve();
    **/

    acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &model);
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
