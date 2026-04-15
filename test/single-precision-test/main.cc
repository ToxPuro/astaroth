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

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("sor.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    (2*AC_REAL_PI)/(info[AC_ngrid].x+1),
	    (2*AC_REAL_PI)/(info[AC_ngrid].y+1),
	    (2*AC_REAL_PI)/(info[AC_ngrid].z+1)
    });
    acPushToConfig(info,AC_first_gridpoint,info[AC_ds]);

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    AcMesh model, candidate;

    acPushToConfig(info,AC_skip_single_gpu_optim,true);
    info.comm->handle = MPI_COMM_WORLD;
    acUpdateDecompositionParams(&info);
    acHostUpdateParams(&info);
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    // GPU alloc & compute
    acGridInit(info);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(rhs),1);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &model);
    acGridSynchronizeStream(STREAM_ALL);
    const auto dims = acGetMeshDims(info);
    int rel = 0;
    AcReal rel_err_sum = 0.0;
    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
			const AcReal dbl_val = model.vertex_buffer[F][acVertexBufferIdx(x,y,z,info)];
			const AcReal sgl_val = model.vertex_buffer[F_SINGLE_PRECISION_RES_IN_DOUBLE][acVertexBufferIdx(x,y,z,info)];
			const AcReal rel_err = std::abs((dbl_val-sgl_val)/dbl_val);
			const bool wrong = (rel_err > 2e-5);
			rel += wrong;
			rel_err_sum += rel_err;
			if(wrong) fprintf(stderr,"Too big difference at %zu,%zu,%zu: %.14e\n",x,y,z,rel_err);
    		}
    	}
    }

    acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);
    int global_rel = 0;
    MPI_Allreduce(&rel,&global_rel,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    AcReal global_rel_err_sum = 0;
    MPI_Allreduce(&rel_err_sum,&global_rel_err_sum,1,AC_REAL_MPI_TYPE,MPI_SUM,MPI_COMM_WORLD);

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
    {
	fprintf(stderr,"Average relative error was: %.14e\n",rel_err_sum/(info[AC_nlocal].x*info[AC_nlocal].y*info[AC_nlocal].z));
        fprintf(stderr, "SINGLE PRECISION TEST complete: %s\n",
                global_rel == 0 ? "No errors found" : "One or more errors found");
    }


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
