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
#include "user_constants.h"
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


    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);


    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("integration.conf", &info);
    acHostUpdateParams(&info); 

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    info.comm->handle = MPI_COMM_WORLD;

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    AcReal* data_arr = (AcReal*)malloc(sizeof(AcReal)*10*10*10*10);
    if(data_arr == NULL)
    {
	    fprintf(stderr,"Was not able top allocate!\n");
	    fflush(stderr);
    }
    for(int x = 0; x <= 10; ++x)
    {
      for(int y = 0; y <= 10; ++y)
      {
        for(int z = 0; z <= 10; ++z)
        {
          for(int w = 0; w <= 10; ++w)
          {
		  AcReal pos_x = info[AC_ds].x*x;
		  AcReal pos_y = info[AC_ds].y*y;
		  AcReal pos_z = info[AC_ds].z*z;
		  AcReal pos_w = info[AC_ds_w]*w;
		  data_arr[x + 10*(y + 10*(z + 10*(w)))] = pos_x*pos_y*pos_z*pos_w;
          }
        }
      }
    }
    info[DATA] = data_arr;
    // GPU alloc & compute
    acGridInit(info);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(calc_integral),1);

    const AcReal res = acDeviceGetOutput(acGridGetDevice(),AC_integral_res);
    fprintf(stderr,"Integral is: %.14e\n",res);

    free(data_arr);
    const int retval = AC_SUCCESS;
    if (pid == 0)
        fprintf(stderr, "REDUCTION_TEST complete: %s\n",
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
