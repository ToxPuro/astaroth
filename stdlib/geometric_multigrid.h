#include "math_utils.h"

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
gmg_restrict_to_level(const int level)
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

	acDeviceLaunchKernel(acGridGetDevice(),STREAM_DEFAULT,gmg_copy_rhs_to_residual_kernel,launch_start,launch_end);
	acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_copy_rhs_to_residual,launch_start,launch_end),1);
    	++restrict_level;
    }
};

void
gmg_store_and_prolong(AcMesh mesh, int level)
{
    //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_boundconds),1);
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
get_galerkin_operator(AcMeshInfo& info, const int level)
{
    AcMesh mesh{};
    acHostMeshCreate(info,&mesh);
    const AcReal h2_inv = info[AC_inv_ds_2].x/std::pow(4,level);
    const std::array<Stencil,5> galerkin_operator_stencils_r1 = 
    {
    	(Stencil)0,
    	stencil_gmg_laplace_level_1_r1,
    	stencil_gmg_laplace_level_2_r1,
    	stencil_gmg_laplace_level_3_r1,
    	stencil_gmg_laplace_level_4_r1
    };
#if STENCIL_ORDER == 4
    const std::array<Stencil,5> galerkin_operator_stencils_r2 = 
    {
    	(Stencil)0,
    	stencil_gmg_laplace_level_1_r2,
    	stencil_gmg_laplace_level_2_r2,
    	stencil_gmg_laplace_level_3_r2,
    	stencil_gmg_laplace_level_4_r2
    };
#endif
#if STENCIL_ORDER == 6
    const std::array<Stencil,5> galerkin_operator_stencils_r3 = 
    {
    	(Stencil)0,
    	stencil_gmg_laplace_level_1_r3,
    	stencil_gmg_laplace_level_2_r3,
    	stencil_gmg_laplace_level_3_r3,
    	stencil_gmg_laplace_level_4_r3
    };
#endif
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
    	((to_int3(coarse_dims.nn)/2) + (int3){NGHOST,NGHOST,NGHOST})
    );
    /**
    fprintf(stderr,"Dims at level %d: (%zu,%zu,%zu)\n"
		,level
    		,coarse_dims.nn.x
    		,coarse_dims.nn.y
    		,coarse_dims.nn.z
          );
    fprintf(stderr,"Hat basis position: (%zu,%zu,%zu)\n"
    		,hat_basis_position.x
    		,hat_basis_position.y
    		,hat_basis_position.z
          );
    **/

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
    gmg_store_and_prolong(mesh,level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_write_del2),1);
    
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&mesh);
    
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
    gmg_restrict_to_level(level);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&mesh);
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
        				//fprintf(stderr,"Level %d galerkin operator at (%d,%d,%d): %.14e\n",level
					//		,int(x)-int(hat_basis_position.x)
					//		,int(y)-int(hat_basis_position.y)
					//		,int(z)-int(hat_basis_position.z)
					//		,val);
        			}
        		}
        	}
    }
    //fprintf(stderr,"\n");
    AcReal stencil[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
    //stencil[0][1][1] = 1.0*h2_inv;
    //stencil[2][1][1] = 1.0*h2_inv;

    //stencil[1][0][1] = 1.0*h2_inv;
    //stencil[1][2][1] = 1.0*h2_inv;

    //stencil[1][1][0] = 1.0*h2_inv;
    //stencil[1][1][2] = 1.0*h2_inv;

    //stencil[1][1][1] = -6.0*h2_inv;
    for(int x = -NGHOST; x <= NGHOST; ++x)
    {
    	for(int y = -NGHOST; y <= NGHOST; ++y)
    	{
    		for(int z = -NGHOST; z <= NGHOST; ++z)
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
    //Here we load to both r1 and (r2/r3) since we are not sure is the user using compact poisson or not
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r1[level],stencil);
#if STENCIL_ORDER == 4
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r2[level],stencil);
#endif
#if STENCIL_ORDER == 6
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r3[level],stencil);
#endif
    stencil[NGHOST][NGHOST][NGHOST] = 0.0;
    //acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_neighbours_operator_stencils[level],stencil);
    acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
    //fprintf(stderr,"\n");
    acHostMeshDestroy(&mesh);
};

void
gmg_populate_central_coeffients(AcMeshInfo* info)
{
    (*info)[AC_GMG_CENTRAL_COEFFS] = &gmg_central_coeffs[0];
}

void
get_galerkin_operators(AcMeshInfo* info)
{
    (*info)[AC_GMG_CENTRAL_COEFFS] = &gmg_central_coeffs[0];
    AcMeshInfo& modifiable_info = *info;
    for(int level = 0; level < (*info)[AC_gmg_number_of_levels]; ++level)
    {
    	get_galerkin_operator(modifiable_info,level);
    }
    acDeviceLoad(acGridGetDevice(), STREAM_DEFAULT, *info, AC_GMG_CENTRAL_COEFFS);
    acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
}

void
gmg_setup(AcMeshInfo* info)
{
	if((*info)[AC_use_coarse_galerkin_operators])
	{
		get_galerkin_operators(info);
	}
}
void
gmg_level_step(const int level, const int number_of_levels)
{
  const auto info = acGridGetLocalMeshInfo();
  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);

  const auto sor_graph         = acGetOptimizedDSLTaskGraph(gmg_optimized_smoother);
  //const auto sor_graph         = acGetOptimizedDSLTaskGraph(gmg_jacobi_smoother);
  //const auto sor_graph         = acGetOptimizedDSLTaskGraph(gmg_poisson_sor_red_black_step);
  //const auto sor_graph         = acGetOptimizedDSLTaskGraph(sor_red_black_step);
  //const auto sor_graph = acGetOptimizedDSLTaskGraph(jacobi_step);
  acGridExecuteTaskGraph(sor_graph,1); //Pre-smooth step
  if(level == number_of_levels-1)
  {
	acGridExecuteTaskGraph(sor_graph,1000);
  }
  else
  {
  	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_residual),1); //Get residual
	  {
    	  	AcTaskDefinition periodic_ops[] = {
    	  	        acHaloExchange({GMG_RESIDUALS[level]}),
    	  	};
	  	const auto halo_exchange = acGridBuildTaskGraph(periodic_ops);
		acGridExecuteTaskGraph(halo_exchange,1);
	  }

          const Volume launch_start = to_volume(info[AC_nmin]);
          const Volume launch_dims = to_volume(info[level_dims[level+1]]);
          const Volume launch_end = launch_dims + launch_start;
	  
          const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end); 
          acGridExecuteTaskGraph(restrict_graph,1); //Restrict residual to the next level
	  gmg_level_step(level+1,number_of_levels);
  	  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	  {
    	  	AcTaskDefinition periodic_ops[] = {
    	  	        acHaloExchange({GMG_SOLUTIONS[level+1]}),
    	  	};
	  	const auto halo_exchange = acGridBuildTaskGraphWithBounds(periodic_ops,launch_start,launch_end);
		acGridExecuteTaskGraph(halo_exchange,1);
	  }
	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_correction_from_next_level),1); //Prolong and add the solution from the next level
  	  acGridExecuteTaskGraph(sor_graph,1); //Post-smooth step
  }
}

void
gmg_v_cycle(const int number_of_levels)
{
	gmg_level_step(0,number_of_levels);
}
