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
#include "user_constants.inc"

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


    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

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

    acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    info.comm->handle = MPI_COMM_WORLD;

    acSetGridMeshDims(8, 8, 8, &info);
    acSetLocalMeshDims(8, 8, 8, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);



    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };
    const AcReal w[7] = 
    {
				0.10539922456186433,
				0.36787944117144233,
				0.7788007830714049,
				1.0,
				0.7788007830714049,
				0.36787944117144233,
				0.10539922456186433
    };
    AcReal coeffs[7][7][7];
    memset(coeffs,0,sizeof(coeffs));
    for(int x = -NGHOST; x <= NGHOST; ++x)
    {
    	for(int y = -NGHOST; y <= NGHOST; ++y)
	{
    		for(int z = -NGHOST; z <= NGHOST; ++z)
		{
			coeffs[x+NGHOST][y+NGHOST][z+NGHOST] = w[x+NGHOST]*w[y+NGHOST]*w[z+NGHOST];
		}
	}
    }
    AcReal coeff_sum = 0.0;
    for(int x = -NGHOST; x <= NGHOST; ++x)
    {
    	for(int y = -NGHOST; y <= NGHOST; ++y)
	{
    		for(int z = -NGHOST; z <= NGHOST; ++z)
		{
			coeff_sum += coeffs[x+NGHOST][y+NGHOST][z+NGHOST];
		}
	}
    }
    for(int x = -NGHOST; x <= NGHOST; ++x)
    {
    	for(int y = -NGHOST; y <= NGHOST; ++y)
	{
    		for(int z = -NGHOST; z <= NGHOST; ++z)
		{
			coeffs[x+NGHOST][y+NGHOST][z+NGHOST] /= coeff_sum;
		}
	}
    }

    auto gaussian_smooth = [&](const int i, const int j, const int k, const Field f)
    {
	    const AcReal* arr = model.vertex_buffer[f];
	    AcReal res = 0.0;
            for(int x = -NGHOST; x <= NGHOST; ++x)
            {
            	for(int y = -NGHOST; y <= NGHOST; ++y)
                {
            		for(int z = -NGHOST; z <= NGHOST; ++z)
                	{
				res += coeffs[x+NGHOST][y+NGHOST][z+NGHOST]*arr[IDX(i+x,j+y,k+z)];
                	}
                }
            }
	    return res;

    };
    const auto test_with_index = [&](const int index)
    {
      if (pid == 0) acHostMeshRandomize(&model);
      acGridLoadMesh(STREAM_DEFAULT, model);
      acGridSynchronizeStream(STREAM_ALL);

      acDeviceSetInput(acGridGetDevice(),AC_field_index,index);
      acHostMeshApplyPeriodicBounds(&model);
      const auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
      for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
      {
      	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
          {
      		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
          	{
          		model.vertex_buffer[F_OUT[index]][IDX(x,y,z)] = gaussian_smooth(x,y,z,F_IN[index]);
          		//model.vertex_buffer[F_OUT[index]][IDX(x,y,z)] = (AcReal)index;
          	}
          }
      }
      acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(graph),1);

      acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
      acGridSynchronizeStream(STREAM_ALL);

      acHostMeshApplyPeriodicBounds(&candidate);
      acHostMeshApplyPeriodicBounds(&model);

      const AcResult res = acVerifyMeshWithMaximumError("inplace_gaussian", model, candidate,10.0);
      if (res == AC_SUCCESS) return true;
      return false;
    };

    int index = 3;
    AcResult res = AC_SUCCESS;
    if(!test_with_index(index))
    {
	    res = AC_FAILURE;
    }

    index = 4;
    if(!test_with_index(index))
    {
	    res = AC_FAILURE;
    }

    //TP: allow ulp error of 10.0 since the coefficients are not exactly the same due to floating-point

    if (res != AC_SUCCESS) {
        retval = res;
        WARNCHK_ALWAYS(retval);
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
        fprintf(stderr, "INPLACE_GAUSSIAN-TEST complete: %s\n",
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

