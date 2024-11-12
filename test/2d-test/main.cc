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
    acSetMeshDims(64, 64, 1, &info);
    //acSetMeshDims(44, 44, 44, &info);

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
    const AcReal dt = 0.000001;
    acDeviceSetInput(acGridGetDevice(), AC_dt,dt);
    AcTaskDefinition periodic_ops[] = {
            acHaloExchange(all_fields),
            acBoundaryCondition(BOUNDARY_XY,BOUNDCOND_PERIODIC,all_fields)
    };
    AcTaskGraph* comm_graph = acGridBuildTaskGraph(periodic_ops);
    AcTaskGraph* graph = acGetDSLTaskGraph(rhs);

    if (pid == 0)
    	acHostMeshRandomize(&model);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridExecuteTaskGraph(comm_graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Periodic boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    // dconst arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    {

        acGridExecuteTaskGraph(graph,1);
        acGridSynchronizeStream(STREAM_ALL);
    }
    	

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(comm_graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    //acHostMeshApplyPeriodicBounds(&candidate);
    //acDeviceSwapBuffer(acGridGetDevice(), UU);
    //acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);

    const int3 nn_min = acGetMinNN(model.info);
    const int nx_min = nn_min.x;
    const int ny_min = nn_min.y;
    const int nz_min = nn_min.z;

    const int3 nn_max = acGetMaxNN(model.info);
    const int nx_max = nn_max.x;
    const int ny_max = nn_max.y;
    const int nz_max = nn_max.z;
    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };

    auto calc_derxx = [&](const int i, const int j, const int k, const AcReal* arr, const AcReal dsx)
    {
	    const AcReal inv_dsx2 = (1.0/dsx)*(1.0/dsx);
	    return
		    arr[IDX(i-3,j,k)]*(inv_dsx2*DER2_3) +
		    arr[IDX(i-2,j,k)]*(inv_dsx2*DER2_2) +
		    arr[IDX(i-1,j,k)]*(inv_dsx2*DER2_1) +
		    arr[IDX(i,j,k)]  *(inv_dsx2*DER2_0) +
		    arr[IDX(i+1,j,k)]*(inv_dsx2*DER2_1) +
		    arr[IDX(i+2,j,k)]*(inv_dsx2*DER2_2) +
		    arr[IDX(i+3,j,k)]*(inv_dsx2*DER2_3);

    };

    auto calc_deryy = [&](const int i, const int j, const int k, const AcReal* arr, const AcReal dsy)
    {
	    const AcReal inv_dsy2 = (1.0/dsy)*(1.0/dsy);
	    return
		    arr[IDX(i,j-3,k)]*(inv_dsy2*DER2_3) +
		    arr[IDX(i,j-2,k)]*(inv_dsy2*DER2_2) +
		    arr[IDX(i,j-1,k)]*(inv_dsy2*DER2_1) +
		    arr[IDX(i,j,k)]  *(inv_dsy2*DER2_0) +
		    arr[IDX(i,j+1,k)]*(inv_dsy2*DER2_1) +
		    arr[IDX(i,j+2,k)]*(inv_dsy2*DER2_2) +
		    arr[IDX(i,j+3,k)]*(inv_dsy2*DER2_3);

    };

    const int3 mm = acGetLocalMM(model.info);
    if(pid == 0)
    {
      AcReal* derxx = (AcReal*)malloc(sizeof(AcReal)*mm.x*mm.y);
      AcReal* deryy = (AcReal*)malloc(sizeof(AcReal)*mm.x*mm.y);
      AcReal* temp  = (AcReal*)malloc(sizeof(AcReal)*mm.x*mm.y);

      for (int step_number = 0; step_number < NUM_INTEGRATION_STEPS; ++step_number) {
          acHostMeshApplyPeriodicBounds(&model);
          for (int k = nz_min; k < nz_max; ++k) {
              for (int j = ny_min; j < ny_max; ++j) {
                  for (int i = nx_min; i < nx_max; ++i) {
          		const int index = IDX(i,j,k);
          		derxx[index] = calc_derxx(i,j,k,model.vertex_buffer[UU],model.info[AC_ds].x);
          		deryy[index] = calc_deryy(i,j,k,model.vertex_buffer[UU],model.info[AC_ds].y);
          		temp[index] = model.vertex_buffer[UU][IDX(i-1,j,k)];
                  }
              }
          }
          for (int k = nz_min; k < nz_max; ++k) {
              for (int j = ny_min; j < ny_max; ++j) {
                  for (int i = nx_min; i < nx_max; ++i) {
          		const int index = IDX(i,j,k);
          		//model.vertex_buffer[UU][index] = model.vertex_buffer[UU][index] + dt*(temp[index]);
          		model.vertex_buffer[UU][index] = model.vertex_buffer[UU][index] + dt*(derxx[index] + deryy[index]);
                  }
              }
          }
      }
      acHostMeshApplyPeriodicBounds(&model);


      const AcResult res = acVerifyMesh("2d-heat", model, candidate);
      if (res != AC_SUCCESS) {
          retval = res;
          WARNCHK_ALWAYS(retval);
      }
      for (int k = nz_min; k < nz_max; ++k) {
             for (int j = ny_min; j < ny_max; ++j) {
                 for (int i = nx_min; i < nx_max; ++i) {
          		const int index = IDX(i,j,k);
			auto val = model.vertex_buffer[UU][index]; 
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
