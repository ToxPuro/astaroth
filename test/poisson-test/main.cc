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
#include "../../stdlib/grid.h"

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
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    acPushToConfig(info,AC_coordinate_system,AC_SPHERICAL_COORDINATES);
    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    info.comm->handle = MPI_COMM_WORLD;
    //info[AC_coordinate_system] = AC_CARTESIAN_COORDINATES;
    //TP: this is said to be a good empirical value based on the eigenspectrum of the 3d laplacian matrix

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    acSetLocalMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    ac_compute_power_law_mapping_x(&info,info[AC_power_law_mapping_exponent]);
    ac_compute_inv_sin_theta(&info);
    ac_compute_inv_r(&info);
    ac_compute_cot_theta(&info);
    acPushToConfig(info,AC_SOR_omega,1.8);

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

    acGridInit(info);
    acPrintMeshInfo(acDeviceGetLocalConfig(acGridGetDevice()));
    fflush(stdout);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond),1);

    const auto jacobi_graph = acGetOptimizedDSLTaskGraph(jacobi_step);
    const auto sor_rb_graph    = acGetOptimizedDSLTaskGraph(sor_red_black_step);
    //const auto sor_graph    = acGetOptimizedDSLTaskGraph(sor_red_black_step);
    const auto residual_graph = acGetOptimizedDSLTaskGraph(get_residual);
    const size_t NUM_SOLVING_STEPS = info[AC_n_solving_steps];
    for (size_t i = 0; i < NUM_SOLVING_STEPS; ++i)
    {
    	//acGridExecuteTaskGraph(jacobi_graph,1);
    	acGridExecuteTaskGraph(sor_rb_graph,1);
	if(i % 10000 == 0)
	{
    		acGridExecuteTaskGraph(residual_graph,1);
		const int N = info[AC_ngrid].x*info[AC_ngrid].y*info[AC_ngrid].z;
		AcReal residual_norm = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2)/N);
		printf("Residual: %zu,%7e\n",i,residual_norm);
	}
    }

    //acGridExecuteTaskGraph(acGetDSLTaskGraph(check_residual),1);
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

    acGridStoreMesh(STREAM_DEFAULT, &candidate);


    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "FIELD_ARR_TEST complete: %s\n",
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
