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
#include "math_utils.h"
#include "errchk.h"
#include "user_constants.h"

#include "../../stdlib/geometric_multigrid.h"

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
    acLoadConfig("mg.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].x,
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].y,
    	    (2.0*AC_REAL_PI)/info[AC_ngrid].z
    });

    acPushToConfig(info,AC_first_gridpoint,
    	info[AC_ds]
    );

    //1.2 works better for GMG and 1.8 works better for plain SOR
    acPushToConfig(info,AC_SOR_omega,
		    1.2
		    //1.8
    );

    acPushToConfig(info,AC_periodic_grid,
    (AcBool3){
    	false,false,false
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
    const int nx = argc > 1 ? atoi(argv[1]): 31;
    const int ny = argc > 2 ? atoi(argv[2]): 31;
    const int nz = argc > 3 ? atoi(argv[3]): 31;
    
    //const int nx = 63;
    //const int ny = 63;
    //const int nz = 63;
    acSetGridMeshDims(nx,ny,nz, &info);
    acSetLocalMeshDims(nx,ny,nz, &info);

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
    gmg_populate_central_coeffients(&info);
    acGridInit(info);
    gmg_setup(&info);
    //Test that can build test ComputeSteps
    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_write_del2),1);
    for(int i = 0; i < 4; ++i)
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)i);
    	const Volume full_launch_start = to_volume(info[AC_nmin]);
	const Volume full_launch_dims = to_volume(info[level_dims[i]]);
    	const Volume full_launch_end = full_launch_dims + full_launch_start;
    	const auto mg_residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual,full_launch_start,full_launch_end);

	if(i < 4-1)
	{
    		const Volume launch_start = to_volume(info[AC_nmin]);
		const Volume launch_dims = to_volume(info[level_dims[i+1]]);
    		const Volume launch_end = launch_dims + launch_start;
    		const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end);
    		acGetOptimizedDSLTaskGraph(gmg_restrict_solution, launch_start, launch_end);
    		const auto prolong_graph = acGetOptimizedDSLTaskGraph(gmg_prolong_solution, launch_start, launch_end);
	}
    	const Volume launch_start = to_volume(info[AC_nmin]);
    	const Volume launch_dims = to_volume(info[level_dims[i]]);
    	const Volume launch_end = launch_dims + launch_start;
    	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(randomize_residual_and_solution,launch_start,launch_end),1);
    }

    bool restriction_correct = true;
    AcReal epsilon  = pow(10.0,-12.0);
    auto relative_diff = [](const auto a, const auto b)
    {
	    const auto abs_diff = fabs(a-b);
	    return  abs_diff/a;
    };
    auto in_eps_threshold = [&](const auto a, const auto b)
    {
	    if(a != 0.0 && b == 0.0) return false;
	    return relative_diff(a,b) < epsilon;
    };
    int3 n_local = info[level_dims[0]];
    for(int x = NGHOST; x < n_local.x+2*NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+2*NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+2*NGHOST;++z)
    		{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[0]);
    	    		model.vertex_buffer[GMG_SOLUTIONS[0]][index] = 0.0;

		}
	}
    }
    n_local = info[level_dims[1]];
    for(int x = NGHOST; x < n_local.x+2*NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+2*NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+2*NGHOST;++z)
    		{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
    	    		model.vertex_buffer[GMG_SOLUTIONS[1]][index] = 0.0;

		}
	}
    }

    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
    	    		model.vertex_buffer[GMG_SOLUTIONS[1]][index] = 1.0;
		}
	}
    }
    acDeviceLoadMesh(acGridGetDevice(),STREAM_DEFAULT,model);
    acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_correction_from_next_level),1);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&model);
    n_local = info[level_dims[0]];
    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[0]);
			if(x == 12 && y == 12 && z == 12)
			{
				fprintf(stderr,"Prolonged values at (%d,%d,%d): %.14e\n",x,y,z,model.vertex_buffer[GMG_SOLUTIONS[0]][index]);
			}
		}
	}
    }
    n_local = info[level_dims[1]];
    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
			//fprintf(stderr,"Coarse values: %.14e\n",model.vertex_buffer[GMG_SOLUTIONS[1]][index]);
		}
	}
    }
    //exit(EXIT_SUCCESS);


    const auto test_restriction = [&]()
    {
    	//Test restriction operator on random data
    	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(randomize_residual_and_solution),1);
    	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)1);
    	const Volume launch_start = to_volume(info[AC_nmin]);
    	const Volume launch_dims = to_volume(info[level_dims[1]]);
    	const Volume launch_end = launch_dims + launch_start;
    	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(randomize_residual_and_solution,launch_start,launch_end),1);
    	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    	gmg_restrict_to_level(1);
    	acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
    	acDeviceSynchronizeStream(acGridGetDevice(),STREAM_ALL);
    	const int3 n_local = info[level_dims[1]];
    	const int3 m_local = n_local + 2*NGHOST;


    	for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    	{
    		for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    		{
    			for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    			{
    	    		const int index = acVertexBufferIdx(x,y,z,info,GMG_RESIDUALS[1]);
    	    		{ 
    	    			int i = 2*x + 1 - NGHOST;
    	    			int j = 2*y + 1 - NGHOST;
    	    			int k = 2*z + 1 - NGHOST;
    	    			AcReal res = 0.0;
    	    			for(int di = -1; di < 2; ++di)
    	    			{
    	    				for(int dj = -1; dj < 2; ++dj)
    	    				{
    	    					for(int dk = -1; dk < 2; ++dk)
    	    					{
    	    						//Facet class 3 has the weight 1
    	    						int weight = 1;
    	    						const int facet_class = abs(di) + abs(dj) + abs(dk);
    	    						if(facet_class == 0)
    	    						{
    	    							weight = 8;
    	    						}
    	    						else if(facet_class == 1)
    	    						{
    	    							weight = 4;
    	    						}
    	    						else if(facet_class == 2)
    	    						{
    	    							weight = 2;
    	    						}
    	    						const int fine_index= acVertexBufferIdx(i+di,j+dj,k+dk,info);
    	    						res += weight*candidate.vertex_buffer[GMG_RESIDUALS[0]][fine_index];
    	    					}
    	    				}
    	    			}
    	    			model.vertex_buffer[GMG_RESIDUALS[1]][index] = res/64.0;
    	    		}
    			}
    		}
    	}
        for(int x = NGHOST; x < n_local.x+NGHOST;++x)
        {
        	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
        	{
        		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
        		{
            		const int index = acVertexBufferIdx(x,y,z,info,GMG_RESIDUALS[1]);
            		const bool local_correct = in_eps_threshold(model.vertex_buffer[GMG_RESIDUALS[1]][index],candidate.vertex_buffer[GMG_RESIDUALS[1]][index]);
            		restriction_correct &= local_correct;
            		if(!local_correct)
            		{
            			printf("Incorrect at (%d,%d,%d): %.14e vs. %.14e\n",x,y,z
            								,model.vertex_buffer[GMG_RESIDUALS[1]][index]
            								,candidate.vertex_buffer[GMG_RESIDUALS[1]][index]
            			      );
            		}
            	}
            }
        }
    };


    fprintf(stderr,"1/h2: %.14e\n",1.0/info[AC_inv_ds_2].x);
    fflush(stderr);
    /**
    fprintf(stderr,"SOR\n");
    {
    	acGridExecuteTaskGraph(initcond_graph,1);
    	AcReal residual = 10e8;
    	int n_steps = 0;
    	while(residual > 1e-8)
    	{
    	    ++n_steps;
    	    const int level = 0;
    	    acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
    	    const auto sor_graph         = acGetOptimizedDSLTaskGraph(gmg_poisson_sor_red_black_step);
    	    const auto mg_residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual);
    	    acGridExecuteTaskGraph(sor_graph,1);
    	    acGridExecuteTaskGraph(mg_residual_graph,1);

    	    acGridExecuteTaskGraph(residual_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
    }
    **/


    const int n_levels = info[AC_gmg_number_of_levels];
    gmg_v_cycle(n_levels,1e-1);
    test_restriction();
    const AcReal relative_residual_tolerance = 1e-14;
    //const AcReal relative_residual_tolerance = 1.5e-1;
  
    fprintf(stderr,"GMG\n");
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)0);
    	const auto res_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual_norm);
    	acGridExecuteTaskGraph(initcond_graph,1);
    	acGridExecuteTaskGraph(res_graph,1);
    	AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	fprintf(stderr,"Initial Residual: %14e\n",residual);
	const AcReal init_residual = residual;
    	int n_steps = 0;
	AcReal sum_time = 0.0;
    	while(residual > 1e-8)
    	{
	    const AcReal start_time = MPI_Wtime();
            gmg_v_cycle(n_levels,relative_residual_tolerance);
	    const AcReal end_time   = MPI_Wtime();
	    sum_time += end_time-start_time;
    	    acGridExecuteTaskGraph(res_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_GMG_residual2[0]));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	    acGridWriteSlicesToDiskCollectiveSynchronous("slices", n_steps, 0.0);
    	    ++n_steps;
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
	fprintf(stderr,"Asymptotic convergence factor: %.14e\n",pow(residual/init_residual,1.0/n_steps));
	fprintf(stderr,"On average a single V cycle took: %.14e seconds\n",sum_time/n_steps);
    }
    int retval = AC_SUCCESS;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;
    if (pid == 0)
        fprintf(stderr, "GMG_TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");
    return EXIT_SUCCESS;


    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
			candidate.vertex_buffer[GMG_RESIDUALS[0]][acVertexBufferIdx(x,y,z,info)] = 1.0;
    		}
    	}
    }
    acDeviceLoadMesh(acGridGetDevice(),STREAM_DEFAULT,candidate);

    {
	const int level = 2;
    	gmg_restrict_to_level(level);
    	acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);
    	auto coarse_dims = acGetMeshDims(acGridGetLocalMeshInfo(),GMG_SOLUTIONS[level]);
    	for(size_t x = coarse_dims.n0.x; x < coarse_dims.n1.x;++x)
    	{
    		for(size_t y = coarse_dims.n0.y; y < coarse_dims.n1.y;++y)
    		{
    			for(size_t z = coarse_dims.n0.z; z < coarse_dims.n1.z;++z)
    			{
    	    			const int index = acVertexBufferIdx(x,y,z,info,GMG_RESIDUALS[level]);
    	    			const AcReal val = candidate.vertex_buffer[GMG_RESIDUALS[level]][index];
    	    			printf("Restriction value: at (%zu,%zu,%zu): %.14e\n",x,y,z,val);
    	    		}
    	    	}
    	}
    }


    bool prolongation_correct = true;
    n_local = info[level_dims[1]];
    const int3 m_local = n_local + 2*NGHOST;
    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
			const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
			candidate.vertex_buffer[GMG_SOLUTIONS[1]][index] = 0.0;
			if(x == 8 && y == 8 && z == 8) candidate.vertex_buffer[GMG_SOLUTIONS[1]][index] = 1.0;
		}
	}
    }
    gmg_store_and_prolong(candidate,1);



    for(int x = -1; x <= 1; ++x)
    {
    	for(int y = -1; y <= 1; ++y)
    	{
    		for(int z = -1; z <= 1; ++z)
    		{
			const int3 index = (int3){15+x,15+y,15+z};
			const int facet_class = std::abs(x) + std::abs(y) + std::abs(z);

			AcReal correct_val{};
			if(facet_class == 0) 
			{
				correct_val = 1.0;
			}
			if(facet_class == 1) 
			{
				correct_val = 0.5;
			}
			if(facet_class == 2) 
			{
				correct_val = 0.25;
			}
			if(facet_class == 3) 
			{
				correct_val = 0.125;
			}
			const AcReal val = candidate.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(index.x,index.y,index.z,info)];
			prolongation_correct &= in_eps_threshold(val,correct_val);
			if(!in_eps_threshold(val,correct_val))
			{
				fprintf(stderr,"At (%d,%d,%d) should be %.14e, was %.14e\n"
						,index.x
						,index.y
						,index.z
						,correct_val,val);
			}
		}
	}

    }
    gmg_store_and_prolong(candidate,1);
    
    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
			const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
			candidate.vertex_buffer[GMG_SOLUTIONS[1]][index] = 1.0;
		}
	}
    }

    gmg_store_and_prolong(candidate,1);

    for(int x = -1; x <= 1; ++x)
    {
    	for(int y = -1; y <= 1; ++y)
    	{
    		for(int z = -1; z <= 1; ++z)
    		{
			const int3 index = (int3){15+x,15+y,15+z};
			const int facet_class = std::abs(x) + std::abs(y) + std::abs(z);

			const AcReal correct_val = 1.0;
			const AcReal val = candidate.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(index.x,index.y,index.z,info)];
			prolongation_correct &= in_eps_threshold(val,correct_val);
			if(!in_eps_threshold(val,correct_val))
			{
				fprintf(stderr,"At (%d,%d,%d) should be %.14e, was %.14e\n"
						,index.x
						,index.y
						,index.z
						,correct_val,val);
			}
		}
	}

    }

    for(int x = NGHOST; x < n_local.x+NGHOST;++x)
    {
    	for(int y = NGHOST; y < n_local.y+NGHOST;++y)
    	{
    		for(int z = NGHOST; z < n_local.z+NGHOST;++z)
    		{
			const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[1]);
			candidate.vertex_buffer[GMG_SOLUTIONS[1]][index] = 0.0;
			if(y == 8 && z == 8) candidate.vertex_buffer[GMG_SOLUTIONS[1]][index] = 1.0;
		}
	}
    }


    gmg_store_and_prolong(candidate,1);

    for(int x = -1; x <= 1; ++x)
    {
    	for(int y = -1; y <= 1; ++y)
    	{
    		for(int z = -1; z <= 1; ++z)
    		{
			const int3 index = (int3){15+x,15+y,15+z};
			const int facet_class = std::abs(x) + std::abs(y) + std::abs(z);

			AcReal correct_val = 1.0;
			if(y == 0 && z == 0) correct_val = 1.0;
			else if(facet_class == 2 && x != 0) correct_val = 0.5;
			else if(facet_class == 2) correct_val = 0.25;
			else if(facet_class == 1) correct_val = 0.5;
			else if(facet_class == 3) correct_val = 0.25;

			const AcReal val = candidate.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(index.x,index.y,index.z,info)];
			prolongation_correct &= in_eps_threshold(val,correct_val);
			if(!in_eps_threshold(val,correct_val))
			{
				fprintf(stderr,"At (%d,%d,%d) should be %.14e, was %.14e\n"
						,index.x
						,index.y
						,index.z
						,correct_val,val);
			}
		}
	}

    }


    /**
    const int level = 2;
    auto coarse_dims = acGetMeshDims(acGridGetLocalMeshInfo(),GMG_SOLUTIONS[level]);
    for(size_t x = coarse_dims.n0.x; x < coarse_dims.n1.x;++x)
    {
       for(size_t y = coarse_dims.n0.y; y < coarse_dims.n1.y;++y)
       {
    	for(size_t z = coarse_dims.n0.z; z < coarse_dims.n1.z;++z)
    	{
        		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[level]);
        		candidate.vertex_buffer[GMG_SOLUTIONS[level]][index] = 0.0;
        		if(x == 3 && y == 3 && z == 3) candidate.vertex_buffer[GMG_SOLUTIONS[level]][index] = 1.0;
        	}
        }
    }
    store_and_prolong(level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(write_del2),1);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_ALL,&candidate);
    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
        		const AcReal val = candidate.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(x,y,z,info)];
        		if(val != 0.0) fprintf(stderr,"Solution after prolongation from level %d at (%zu,%zu,%zu): %.14e\n",level,x,y,z,val);
    		}
    	}
    }

    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
        		const AcReal val = candidate.vertex_buffer[GMG_RESIDUALS[0]][acVertexBufferIdx(x,y,z,info)];
        		if(val != 0.0) fprintf(stderr,"DEL2 after prolongation from level %d at (%zu,%zu,%zu): %.14e\n",level,x,y,z,val);
    		}
    	}
    }
    **/

    /**
    const bool correct = restriction_correct && prolongation_correct;
    int retval = correct ? AC_SUCCESS : AC_FAILURE;
    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;


    return retval == AC_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
    **/
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
