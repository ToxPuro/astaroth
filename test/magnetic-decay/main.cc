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

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define NUM_INTEGRATION_STEPS (100)


#define DER2_3 (1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 (3. / 2.)
#define DER2_0 (-49. / 18.)

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
    int retval = 0;

    MPI_Init(NULL,NULL);

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("PC-AC.conf", &info);
    info.comm = MPI_COMM_WORLD;

#if AC_RUNTIME_COMPILATION
    const char* build_str = "-DUSE_HIP=ON  -DOPTIMIZE_FIELDS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DMAX_THREADS_PER_BLOCK=512 -DBUILD_TESTS=ON -DDSL_MODULE_FILE=solve_two.ac -DDSL_MODULE_DIR=../../test/magnetic-decay/DSL/local";
    acCompile(build_str,"magnetic-decay",info);
    acLoadLibrary();
    acLoadUtils();
#endif

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    std::vector<Field> all_fields;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields.push_back((Field)i);
    }
    const AcReal dt = 0.0001;
    acDeviceSetInput(acGridGetDevice(), AC_dt,dt);
    acDeviceSetInput(acGridGetDevice(), AC_step_num, 0);
    AcTaskGraph* graph = acGetDSLTaskGraph(AC_rhs);

    // dconst arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    {

	for (size_t isubstep = 0; isubstep < 3; ++isubstep)
	{
  		acDeviceSetInput(acGridGetDevice(), AC_step_num, isubstep);
        	acGridExecuteTaskGraph(graph,1);
        	acGridSynchronizeStream(STREAM_ALL);
	}
    }
    	

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acStoreConfig(acDeviceGetLocalConfig(acGridGetDevice()),"run.conf");
    //acHostMeshApplyPeriodicBounds(&candidate);
    //acDeviceSwapBuffer(acGridGetDevice(), UU);
    //acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "2D-TEST complete: %s\n",
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
