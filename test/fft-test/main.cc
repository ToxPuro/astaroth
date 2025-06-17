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
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    (2*AC_REAL_PI)/info[AC_ngrid].x,
	    (2*AC_REAL_PI)/info[AC_ngrid].y,
	    (2*AC_REAL_PI)/info[AC_ngrid].z
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
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    acSetLocalMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);

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
    acGridExecuteTaskGraph(acGetDSLTaskGraph(fft_solve),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond),1);
    acDeviceFFTR2C(acGridGetDevice(),HEAT_INIT,HEAT_COMPLEX);
    AcMeshDims comp_dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridExecuteTaskGraph(acGetDSLTaskGraph(fft_solve),1);
    acDeviceFFTC2R(acGridGetDevice(),HEAT_COMPLEX_SOLUTION,HEAT_SOLUTION);
    acDeviceFFTC2R(acGridGetDevice(),HEAT_COMPLEX,HEAT_FORWARD_AND_BACK);
    acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &model);
    acGridSynchronizeStream(STREAM_ALL);
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };
    fprintf(stderr,"AC_len: %14e,%14e,%14e\n",acDeviceGetLocalConfig(acGridGetDevice())[AC_len].x,acDeviceGetLocalConfig(acGridGetDevice())[AC_len].y,acDeviceGetLocalConfig(acGridGetDevice())[AC_len].z);
    fprintf(stderr,"Magnitude at (%d,%d,%d): %14e\n",NGHOST+1,NGHOST+1,NGHOST+1,model.vertex_buffer[HEAT_FREQUENCY_MAGNITUDE][IDX(NGHOST+1,NGHOST+1,NGHOST+1)]);
    fprintf(stderr,"Magnitude at (%d,%d,%d): %14e\n",NGHOST+31,NGHOST+31,NGHOST+31,model.vertex_buffer[HEAT_FREQUENCY_MAGNITUDE][IDX(NGHOST+31,NGHOST+31,NGHOST+31)]);
    fprintf(stderr,"Magnitude at (%d,%d,%d): %14e\n",NGHOST+16,NGHOST+16,NGHOST+16,model.vertex_buffer[HEAT_FREQUENCY_MAGNITUDE][IDX(NGHOST+16,NGHOST+16,NGHOST+16)]);
    for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    {
    	for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	{
    		for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    		{
			;
			//if(model.vertex_buffer[HEAT_FREQUENCY_MAGNITUDE][IDX(x,y,z)] != 0.0)
			//{
			//}
		}
	}
    }

    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "POISSON_TEST complete: %s\n",
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
