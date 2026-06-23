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
#include "user_constants.inc"

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
const AcReal3
get_wavevector(const int3 index, const AcMeshInfo info)
{
	const int3 global_idx  = (int3)
	{
		index.x - info[AC_nmin].x,
		index.y - info[AC_nmin].y,
		index.z - info[AC_nmin].z
	};
	const auto k_x = info[AC_frequency_spacing].x*((global_idx.x <= info[AC_ngrid].x/2) ? global_idx.x : global_idx.x - info[AC_ngrid].x);
	const auto k_y = info[AC_frequency_spacing].y*((global_idx.y <= info[AC_ngrid].y/2) ? global_idx.y : global_idx.y - info[AC_ngrid].y);
	const auto k_z = info[AC_frequency_spacing].z*((global_idx.z <= info[AC_ngrid].z/2) ? global_idx.z : global_idx.z - info[AC_ngrid].z);
	return (AcReal3){k_x,k_y,k_z};
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
    acPushToConfig(info,AC_periodic_grid,(AcBool3){true,true,false});
    info.comm->handle = MPI_COMM_WORLD;

    const int max_devices = 64;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    const int3 decomp = acDecompose(nprocs,info);
    acSetLocalMeshDims(info[AC_ngrid].x/decomp.x,info[AC_ngrid].y/decomp.y,info[AC_ngrid].z/decomp.z, &info);

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DFFT_ENABLED=ON -DUSE_HEFFTE=ON -DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);

    Field communicated_fields[2];
    communicated_fields[0] = F_X;
    communicated_fields[1] = F_Y;

    acGridInit(info);
    const auto top_z_halo_exchange = 
	    acGridBuildTaskGraph({
			    acHaloExchangeBoundary(communicated_fields,2,BOUNDARY_Z_TOP)
			    });
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(boundconds),1);
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };
    AcMeshDims comp_dims = acGetMeshDims(acGridGetLocalMeshInfo());
    info = acDeviceGetLocalConfig(acGridGetDevice());
    const int3 domain_coordinates = info[AC_domain_coordinates];
    for(size_t x = 0; x < comp_dims.m1.x; ++x)
    {
       for(size_t y = 0; y < comp_dims.m1.y; ++y)
       {
       	for(size_t z = 0; z < comp_dims.m1.z; ++z)
       	{
		model.vertex_buffer[F_X][IDX(x,y,z)] = AcReal(domain_coordinates.x);
		model.vertex_buffer[F_Y][IDX(x,y,z)] = AcReal(domain_coordinates.y);
       	}
       }
    }
    acDeviceLoadMesh(acGridGetDevice(),STREAM_DEFAULT,model);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(top_z_halo_exchange,1);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&model);
    acGridSynchronizeStream(STREAM_ALL);
    bool correct = true;
    AcReal epsilon  =  16*pow(10.0,-12.0);
    auto relative_diff = [](const auto a, const auto b)
    {
            const auto abs_diff = fabs(a-b);
            return  abs_diff/a;
    };
    auto in_eps_threshold = [&](const auto a, const auto b)
    {
       if(a == b) return true;
	    if(a == 0.0)
	    {
		    return fabs(b) < epsilon;
	    }
	    if(b == 0.0)
	    {
		    return fabs(a) < epsilon;
	    }
            return relative_diff(a,b) < epsilon;
    };
    for(size_t x = 0; x < comp_dims.m1.x; ++x)
    {
       for(size_t y = 0; y < comp_dims.m1.y; ++y)
       {
       	for(size_t z = 0; z < comp_dims.m1.z; ++z)
       	{
		const AcReal x_value = model.vertex_buffer[F_X][IDX(x,y,z)];
		const AcReal y_value = model.vertex_buffer[F_Y][IDX(x,y,z)];
		if(z < comp_dims.n1.z) continue;
		AcReal correct_x_val = AcReal(domain_coordinates.x);
		if(x < comp_dims.n0.x) correct_x_val -= 1.0;
		if(x >= comp_dims.n1.x) correct_x_val += 1.0;
		if(correct_x_val < 0) correct_x_val = AcReal(decomp.x)-1;
		if(correct_x_val > AcReal(decomp.x)-1) correct_x_val = 0;
		AcReal correct_y_val = AcReal(domain_coordinates.y);
		if(y < comp_dims.n0.y) correct_y_val -= 1.0;
		if(y >= comp_dims.n1.y) correct_y_val += 1.0;
		if(correct_y_val < 0) correct_y_val = AcReal(decomp.y)-1;
		if(correct_y_val > AcReal(decomp.y)-1) correct_y_val = 0;
		{
			if(!in_eps_threshold(x_value,correct_x_val))
			{
				fprintf(stderr,"X WRONG!: at %zu,%zu,%zu %.14e,%.14e\n",x,y,z,x_value,correct_x_val);
				correct = false;
			}
			if(!in_eps_threshold(y_value,correct_y_val))
			{
				fprintf(stderr,"Y WRONG!: at %zu,%zu,%zu %.14e,%.14e\n",x,y,z,y_value,correct_y_val);
				correct = false;
			}
		}
       	}
       }
    }


    int retval = correct ? AC_SUCCESS : AC_FAILURE;

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "BOUNDARY HALO EXCHANGE TEST complete: %s\n",
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
