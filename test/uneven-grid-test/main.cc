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
    acPushToConfig(info,AC_allow_non_divisible_grid,true);
    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_LINEAR);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_EXTERNAL);
    acPushToConfig(info,AC_domain_decomposition,(int3){1,2,1});

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int nx = argc > 1 ? atoi(argv[1]): 8;
    const int ny = argc > 2 ? atoi(argv[2]): 8;
    const int nz = argc > 3 ? atoi(argv[3]): 8;
    acSetGridMeshDims(15, 15, 15, &info);
    //TP: backwards compatibility
    if(pid == 1)
    {
    	acSetLocalMeshDims(15, 7, 15, &info);
    }
    else
    {
    	acSetLocalMeshDims(15, 8, 15, &info);
    }
    fprintf(stderr,"nlocal: (%d,%d)\n",info[AC_nlocal].x,info[AC_nlocal].y);
    //acSetMeshDims(44, 44, 44, &info);

    AcMesh model, candidate;
    {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    fprintf(stderr,"%d| Before init\n",pid);
    fflush(stderr);
    acGridInit(info);
    fprintf(stderr,"%d| after init\n",pid);
    fflush(stderr);

    std::vector<Field> all_fields;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields.push_back((Field)i);
    }
    const AcReal dt = 0.000001;
    acDeviceSetInput(acGridGetDevice(), AC_dt,dt);
    AcTaskDefinition periodic_ops[] = {
            acHaloExchange(all_fields),
            acBoundaryCondition(BOUNDARY_XYZ,BOUNDCOND_PERIODIC,all_fields)
    };
    AcTaskGraph* comm_graph = acGridBuildTaskGraph(periodic_ops);
    AcTaskGraph* graph = acGetDSLTaskGraph(rhs);

    if (pid == 0)
    	acHostMeshRandomize(&model);
    acDeviceLoadMesh(acGridGetDevice(),STREAM_DEFAULT,model);
    acGridExecuteTaskGraph(comm_graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&model);
    auto dims = acGetMeshDims(info);
    for(int v = 0; v < NUM_VTXBUF_HANDLES; ++v)
    {
	if(pid == 0)
	{
		const int x = 3;
		int y = 3;
		const int z = 0;
		int index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Bot Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Bot+1 Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Bot+2 Val: %.14e\n",model.vertex_buffer[v][index]);
	}
	else if(pid == 1)
	{
		int x = 3;
		int y = 9+NGHOST;
		const int z = 0;
		int index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top+1 Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top+2 Val: %.14e\n",model.vertex_buffer[v][index]);
		y = 9+NGHOST;
		x += 16;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top Right Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top+1 Right Val: %.14e\n",model.vertex_buffer[v][index]);
		++y;
		index = acVertexBufferIdx(x,y,z,info);
		fprintf(stderr,"Top+2 Right Val: %.14e\n",model.vertex_buffer[v][index]);
	}
	/**
    	for(int x = dims.n0.x; x < dims.n1.x; ++x)
    	{
    		for(int y = dims.n0.y; y < dims.n1.y; ++y)
    		{
    			for(int z = dims.n0.z; z < dims.n1.z; ++z)
    			{
				const auto gpu_val = candidate.vertex_buffer[v][index];
				const auto cpu_val = model.vertex_buffer[v][index];
    	    			if(gpu_val != cpu_val)
				{
					fprintf(stderr,"Wrong at (%d,%d,%d): %.14e,%.14e!!\n"
							,x
							,y
							,z
							,gpu_val
							,cpu_val
							);
				}
    			}
    		}
    	}
	**/
    }
    fflush(stdout);

    // dconst arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    {

        acGridExecuteTaskGraph(graph,1);
        acGridSynchronizeStream(STREAM_ALL);
    }
    	

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(comm_graph,1);
    acGridSynchronizeStream(STREAM_ALL);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);
    //acGridStoreMesh(STREAM_DEFAULT, &candidate);
    //acHostMeshApplyPeriodicBounds(&candidate);
    //acDeviceSwapBuffer(acGridGetDevice(), UU);
    //acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);


    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "UNEVEN-GRID-TEST complete: %s\n",
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
