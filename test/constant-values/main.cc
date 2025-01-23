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
main(int argc, char* argv[])
{
    atexit(acAbort);
    int retval = 0;

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int nx = argc > 1 ? atoi(argv[1]) : 2*9;
    const int ny = argc > 2 ? atoi(argv[2]) : 2*11;
    const int nz = argc > 3 ? atoi(argv[3]) : 4*7;
    acSetMeshDims(nx, ny, nz, &info);

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
    acDeviceSetInput(acGridGetDevice(), STEP_NUM, FIRST);
    AcTaskGraph* graph = acGetOptimizedDSLTaskGraph(rhs);
    acDeviceSetInput(acGridGetDevice(), STEP_NUM, SECOND);
    graph = acGetOptimizedDSLTaskGraph(rhs);

    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };
    for (size_t k= dims.n0.z; k < dims.nn.z; ++k) {
        for (size_t j = dims.n0.y; j < dims.nn.y; ++j) {
            for (size_t i = dims.n0.x; i < dims.nn.x; ++i) {
    		const int index = IDX(i,j,k);
    		model.vertex_buffer[FIELD][index] = 0.0;
            }
        }
    }

    acGridLoadMesh(STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    {

        acGridExecuteTaskGraph(graph,1);
        acGridSynchronizeStream(STREAM_ALL);
    }
    	

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    //acHostMeshApplyPeriodicBounds(&candidate);
    //acDeviceSwapBuffer(acGridGetDevice(), UU);
    //acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);


    if(pid == 0)
    {
          for (size_t k = dims.n0.z; k < dims.nn.z; ++k) {
              for (size_t j = dims.n0.y; j < dims.nn.y; ++j) {
                  for (size_t i = dims.n0.x; i < dims.nn.x; ++i) {
          		const int index = IDX(i,j,k);
          		model.vertex_buffer[FIELD][index] = 1.0;
			const auto model_val = model.vertex_buffer[FIELD][index];
			const auto gpu_val   = candidate.vertex_buffer[FIELD][index];
			if(model_val != gpu_val) printf("model: %14e|gpu: %14e\t at: %ld,%ld,%ld\n",model_val,gpu_val,i,j,k);
                  }
              }
          }
      acHostMeshApplyPeriodicBounds(&model);


      const AcResult res = acVerifyMesh("constant values", model, candidate);
      if (res != AC_SUCCESS) {
          retval = res;
          WARNCHK_ALWAYS(retval);
      }
      for (size_t k = dims.n0.z; k < dims.n1.z; ++k) {
             for (size_t j = dims.n0.y; j < dims.n1.y; ++j) {
                 for (size_t i = dims.n0.x; i < dims.n1.x; ++i) {
          		const int index = IDX(i,j,k);
			auto val = model.vertex_buffer[FIELD][index]; 
			if(!std::isfinite(val))
				printf("WRONG\n");
		 }
	     }
      }
      fflush(stdout);
    }




    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "CONSTANT VALUES complete: %s\n",
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
