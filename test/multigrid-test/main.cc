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


AcReal gmg_central_coeffs[5]{};
const std::array<AcInt3Param,5> level_dims = 
{
	AC_nlocal_gmg_level_0,
	AC_nlocal_gmg_level_1,
	AC_nlocal_gmg_level_2,
	AC_nlocal_gmg_level_3,
	AC_nlocal_gmg_level_4
};

void
restrict_to_level(const int level)
{
    const auto info = acGridGetLocalMeshInfo();
    int restrict_level = 0;	
    while(restrict_level < level)
    {
	const Volume launch_start = to_volume(info[AC_nmin]);
	const Volume launch_dims = to_volume(info[level_dims[restrict_level+1]]);
	const Volume launch_end = launch_dims + launch_start;
    	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)restrict_level);
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_restrict_residual,launch_start,launch_end),1);
    	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)(restrict_level+1));
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_copy_rhs_to_residual,launch_start,launch_end),1);
    	++restrict_level;
    }
};

void
store_and_prolong(AcMesh mesh, int level)
{
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_boundconds),1);
    const auto info = acGridGetLocalMeshInfo();
    --level;
    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, mesh);
    while(level >= 0)
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	const Volume launch_start = to_volume(info[AC_nmin]);
	const Volume launch_dims = to_volume(info[level_dims[level]]);
	const Volume launch_end = launch_dims + launch_start;
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_prolong_solution,launch_start,launch_end),1);
    	--level;
    }
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &mesh);
};

void
get_galerkin_operator(AcMesh mesh, const int level)
{
    const auto info = acGridGetLocalMeshInfo();
    const AcReal h2_inv = info[AC_inv_ds_2].x/std::pow(4,level);
    const std::array<Stencil,5> galerkin_operator_stencils = 
    {
    	(Stencil)0,
    	stencil_gmg_laplace_level_1,
    	stencil_gmg_laplace_level_2,
    	stencil_gmg_laplace_level_3,
    	stencil_gmg_laplace_level_4
    };
    const std::array<Stencil,5> galerkin_neighbours_operator_stencils = 
    {
    	(Stencil)0,
    	stencil_gmg_laplace_neighbours_level_1,
    	stencil_gmg_laplace_neighbours_level_2,
    	stencil_gmg_laplace_neighbours_level_3,
    	stencil_gmg_laplace_neighbours_level_4
    };
    auto coarse_dims = acGetMeshDims(acGridGetLocalMeshInfo(),GMG_SOLUTIONS[level]);
    const Volume hat_basis_position = to_volume(
    	((to_int3(coarse_dims.nn)/2) + (int3){1,1,1})
    );
    printf("Dims at level: (%zu,%zu,%zu)\n"
    		,coarse_dims.nn.x
    		,coarse_dims.nn.y
    		,coarse_dims.nn.z
          );
    printf("Hat basis position: (%zu,%zu,%zu)\n"
    		,hat_basis_position.x
    		,hat_basis_position.y
    		,hat_basis_position.z
          );
    for(size_t x = 0; x < coarse_dims.m1.x;++x)
    {
       for(size_t y = 0; y < coarse_dims.m1.y;++y)
       {
    	for(size_t z = 0; z < coarse_dims.m1.z;++z)
    	{
        		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[level]);
        		mesh.vertex_buffer[GMG_SOLUTIONS[level]][index] = 0.0;
        		mesh.vertex_buffer[GMG_RESIDUALS[level]][index] = 0.0;
        		if(x == hat_basis_position.x && y == hat_basis_position.y && z == hat_basis_position.z) mesh.vertex_buffer[GMG_SOLUTIONS[level]][index] = 1.0;
        	}
        }
    }
    
    store_and_prolong(mesh,level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(write_del2),1);
    
    acDeviceStoreMesh(acGridGetDevice(),STREAM_ALL,&mesh);
    
    acGridSynchronizeStream(STREAM_ALL);
    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
        			const AcReal val = mesh.vertex_buffer[GMG_RESIDUALS[0]][acVertexBufferIdx(x,y,z,info)];
        			const AcReal sol_val = mesh.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(x,y,z,info)];
        			//if(val != 0.0) fprintf(stderr,"DEL2 after prolongation from level %d at (%zu,%zu,%zu): %.14e\n",level,x,y,z,val);
    		}
    	}
    }
    for(size_t x = dims.n0.x; x < dims.n1.x; ++x)
    {
    	for(size_t y = dims.n0.y; y < dims.n1.y; ++y)
    	{
    		for(size_t z = dims.n0.z; z < dims.n1.z; ++z)
    		{
        			const AcReal val = mesh.vertex_buffer[GMG_RESIDUALS[0]][acVertexBufferIdx(x,y,z,info)];
        			const AcReal sol_val = mesh.vertex_buffer[GMG_SOLUTIONS[0]][acVertexBufferIdx(x,y,z,info)];
        			//if(sol_val != 0.0) fprintf(stderr,"Solution after prolongation from level %d at (%zu,%zu,%zu): %.14e\n",level,x,y,z,sol_val);
    		}
    	}
    }
    for(size_t x = coarse_dims.n0.x; x < coarse_dims.n1.x;++x)
    {
       for(size_t y = coarse_dims.n0.y; y < coarse_dims.n1.y;++y)
       {
    	for(size_t z = coarse_dims.n0.z; z < coarse_dims.n1.z;++z)
    	{
        		const int index = acVertexBufferIdx(x,y,z,info,GMG_SOLUTIONS[level]);
        		const AcReal val  = mesh.vertex_buffer[GMG_SOLUTIONS[level]][index];
			//if(val != 0.0) fprintf(stderr,"Source at (%zu,%zu,%zu): %.14e\n",x,y,z,val);
        	}
        }
    }
    restrict_to_level(level);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_ALL,&mesh);
    acGridSynchronizeStream(STREAM_ALL);
    for(size_t x = coarse_dims.n0.x; x < coarse_dims.n1.x;++x)
    {
    	for(size_t y = coarse_dims.n0.y; y < coarse_dims.n1.y;++y)
    	{
    		for(size_t z = coarse_dims.n0.z; z < coarse_dims.n1.z;++z)
    		{
        			const int index = acVertexBufferIdx(x,y,z,info,GMG_RHS[level]);
        			const AcReal val = mesh.vertex_buffer[GMG_RHS[level]][index];
        			if(val != 0.0)
        			{
        				fprintf(stderr,"Level %d galerkin operator at (%zu,%zu,%zu): %.14e\n",level,x,y,z,val);
        			}
        		}
        	}
    }
    fprintf(stderr,"\n");
    AcReal stencil[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
    //stencil[0][1][1] = 1.0*h2_inv;
    //stencil[2][1][1] = 1.0*h2_inv;

    //stencil[1][0][1] = 1.0*h2_inv;
    //stencil[1][2][1] = 1.0*h2_inv;

    //stencil[1][1][0] = 1.0*h2_inv;
    //stencil[1][1][2] = 1.0*h2_inv;

    //stencil[1][1][1] = -6.0*h2_inv;
    for(int x = -1; x <= 1; ++x)
    {
    	for(int y = -1; y <= 1; ++y)
    	{
    		for(int z = -1; z <= 1; ++z)
    		{
            		const int index = acVertexBufferIdx(hat_basis_position.x + x,hat_basis_position.y + y,hat_basis_position.z + z,info,GMG_RESIDUALS[level]);
            		const AcReal val = mesh.vertex_buffer[GMG_RESIDUALS[level]][index];
    			stencil[z+NGHOST][y+NGHOST][x+NGHOST] = val;
    		}
    	}
    }
    const int central_index = acVertexBufferIdx(hat_basis_position.x,hat_basis_position.y,hat_basis_position.z,info,GMG_RESIDUALS[level]);
    const AcReal central_coeff = mesh.vertex_buffer[GMG_RESIDUALS[level]][central_index];
    //const AcReal central_coeff = -6.0*h2_inv;
    gmg_central_coeffs[level] = central_coeff;
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils[level],stencil);
    stencil[1][1][1] = 0.0;
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_neighbours_operator_stencils[level],stencil);
    acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
    fprintf(stderr,"\n");
};

void
get_galerkin_operators(AcMesh mesh)
{
    mesh.info[AC_GMG_CENTRAL_COEFFS] = &gmg_central_coeffs[0];
    get_galerkin_operator(mesh,1);
    get_galerkin_operator(mesh,2);
    get_galerkin_operator(mesh,3);
    get_galerkin_operator(mesh,4);
    acDeviceLoad(acGridGetDevice(), STREAM_DEFAULT, mesh.info, AC_GMG_CENTRAL_COEFFS);
    acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
}

const int MAX_GMG_LEVEL = 3;
void
gmg_level_step(AcMesh mesh, const int level)
{
  const auto info = acGridGetLocalMeshInfo();
  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);

  const auto sor_graph         = acGetOptimizedDSLTaskGraph(gmg_sor_red_black_step);
  //const auto sor_graph         = acGetOptimizedDSLTaskGraph(sor_red_black_step);
  //const auto sor_graph = acGetOptimizedDSLTaskGraph(jacobi_step);
  ///
  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_residual_scalar),1);
  AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2));
  fprintf(stderr,"Residual coming in: %14e\n",residual);//
  acGridExecuteTaskGraph(sor_graph,1); //Pre-smooth step
				       //
  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_residual_scalar),1);
  residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2));
  fprintf(stderr,"Residual after first smooth: %14e\n",residual);//
  if(level == MAX_GMG_LEVEL)
  {
	acGridExecuteTaskGraph(sor_graph,100);
  }
  else
  {
  	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_residual),1); //Get residual
          const Volume launch_start = to_volume(info[AC_nmin]);
          const Volume launch_dims = to_volume(info[level_dims[level+1]]);
          const Volume launch_end = launch_dims + launch_start;
          const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end); 
          acGridExecuteTaskGraph(restrict_graph,1); //Restrict residual to the next level
  	  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	  gmg_level_step(mesh,level+1);
  	  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_correction_from_next_level),1); //Prolong and add the solution from the next level
  	  acGridExecuteTaskGraph(sor_graph,1); //Post-smooth step
  }
}

void
gmg_v_cycle(AcMesh mesh)
{
	gmg_level_step(mesh,0);
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
    const int mx = 33;
    const int my = 33;
    const int mz = 33;
    
    //const int mx = 65;
    //const int my = 65;
    //const int mz = 65;
    acSetGridMeshDims(mx-2*NGHOST,my-2*NGHOST,mz-2*NGHOST, &info);
    acSetLocalMeshDims(mx-2*NGHOST,my-2*NGHOST,mz-2*NGHOST, &info);

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

    info[AC_GMG_CENTRAL_COEFFS] = &gmg_central_coeffs[0];
    acGridInit(info);
    //Test that can build test ComputeSteps
    const auto empty_graph = acGetOptimizedDSLTaskGraph(empty_steps);
    const auto initcond_graph = acGetOptimizedDSLTaskGraph(initcond);
    const auto residual_graph = acGetOptimizedDSLTaskGraph(get_residual);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(write_del2),1);
    for(int i = 0; i < 4; ++i)
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)i);
    	const Volume full_launch_start = to_volume(info[AC_nmin]);
	const Volume full_launch_dims = to_volume(info[level_dims[i]]);
    	const Volume full_launch_end = full_launch_dims + full_launch_start;
    	const auto sor_graph         = acGetOptimizedDSLTaskGraph(sor_red_black_step,full_launch_start,full_launch_end);
    	const auto gm_sor_graph         = acGetOptimizedDSLTaskGraph(gmg_sor_red_black_step,full_launch_start,full_launch_end);
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
    	restrict_to_level(1);
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
    	    			int i = 2*x;
    	    			int j = 2*y;
    	    			int k = 2*z;
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
    	    			if(x == 1 && y == 1 && z == 1)
    	    			{
    	    				const int fine_index= acVertexBufferIdx(i,j,k,info);
    	    				printf("fine residual: %.14e\n",candidate.vertex_buffer[GMG_RESIDUALS[0]][fine_index]);
    	    				printf("RES: %.14e\n",res/64.0);
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


    get_galerkin_operators(candidate);
    fprintf(stderr,"1/h2: %.14e\n",1.0/info[AC_inv_ds_2].x);
    fflush(stderr);
    //exit(EXIT_SUCCESS);
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
    	    const auto sor_graph         = acGetOptimizedDSLTaskGraph(sor_red_black_step);
    	    const auto mg_residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual);
    	    acGridExecuteTaskGraph(sor_graph,1);
    	    acGridExecuteTaskGraph(mg_residual_graph,1);
    	    if(level < 4-1)
    	    {
    			const Volume launch_start = to_volume(info[AC_nmin]);
    	    	const Volume launch_dims = to_volume(info[level_dims[level+1]]);
    			const Volume launch_end = launch_dims + launch_start;
    			const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end);
    			acGridExecuteTaskGraph(restrict_graph,1); }

    		acGridExecuteTaskGraph(residual_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
    }


    gmg_v_cycle(candidate);
    //test_restriction();
    fprintf(stderr,"GMG\n");
    {
    	acGridExecuteTaskGraph(initcond_graph,1);
    	acGridExecuteTaskGraph(residual_graph,1);
    	AcReal residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2));
    	fprintf(stderr,"Initial Residual: %14e\n",residual);
    	int n_steps = 0;
    	while(residual > 1e-8)
    	{
    	    ++n_steps;
	    printf("\n\nCycle %d\n\n",n_steps);
            gmg_v_cycle(candidate);

    	    acGridExecuteTaskGraph(residual_graph,1);
    	    residual = sqrt(acDeviceGetOutput(acGridGetDevice(),AC_residual2));
    	    fprintf(stderr,"Residual: %14e\n",residual);
    	}
    	fprintf(stderr,"Final residual: %14e\n",residual);
    	fprintf(stderr,"Took %d steps\n",n_steps);
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
    	restrict_to_level(level);
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
    const int3 n_local = info[level_dims[1]];
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
    store_and_prolong(candidate,1);



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
    store_and_prolong(candidate,1);
    
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

    store_and_prolong(candidate,1);

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


    store_and_prolong(candidate,1);

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
