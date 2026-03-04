#include "math_utils.h"

AcReal gmg_central_coeffs[11]{};
const std::array<AcInt3Param,11> level_dims = 
{
	AC_nlocal_gmg_level_0,
	AC_nlocal_gmg_level_1,
	AC_nlocal_gmg_level_2,
	AC_nlocal_gmg_level_3,
	AC_nlocal_gmg_level_4,
	AC_nlocal_gmg_level_5,
	AC_nlocal_gmg_level_6,
	AC_nlocal_gmg_level_7,
	AC_nlocal_gmg_level_8,
	AC_nlocal_gmg_level_9,
	AC_nlocal_gmg_level_10
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
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_copy_rhs_to_residual,launch_start,launch_end),1);
    	++restrict_level;
    }
};

void
gmg_prolong(AcMesh mesh, int level)
{
    //acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_boundconds),1);
    const auto info = acGridGetLocalMeshInfo();
    --level;
    while(level >= 0)
    {
	acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	const Volume launch_start = to_volume(info[AC_nmin]);
	const Volume launch_dims = to_volume(info[level_dims[level]]);
	const Volume launch_end = launch_dims + launch_start;
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_prolong_solution,launch_start,launch_end),1);
    	--level;
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
    const std::array<Stencil,11> galerkin_operator_stencils_r1 = 
    {
    	stencil_gmg_laplace_level_0_r1,
    	stencil_gmg_laplace_level_1_r1,
    	stencil_gmg_laplace_level_2_r1,
    	stencil_gmg_laplace_level_3_r1,
    	stencil_gmg_laplace_level_4_r1,
    	stencil_gmg_laplace_level_5_r1,
    	stencil_gmg_laplace_level_6_r1,
    	stencil_gmg_laplace_level_7_r1,
    	stencil_gmg_laplace_level_8_r1,
    	stencil_gmg_laplace_level_9_r1,
    	stencil_gmg_laplace_level_10_r1
    };

    const std::array<Stencil,11> galerkin_y_line_operators = 
    {
	stencil_gmg_laplace_y_line_l0,
	stencil_gmg_laplace_y_line_l1,
	stencil_gmg_laplace_y_line_l2,
	stencil_gmg_laplace_y_line_l3,
	stencil_gmg_laplace_y_line_l4,
	stencil_gmg_laplace_y_line_l5,
	stencil_gmg_laplace_y_line_l6,
	stencil_gmg_laplace_y_line_l7,
	stencil_gmg_laplace_y_line_l8,
	stencil_gmg_laplace_y_line_l9,
	stencil_gmg_laplace_y_line_l10
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
    acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_write_hat_basis,coarse_dims.n0,coarse_dims.n1),1);
    gmg_prolong(mesh,level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_write_del2),1);
    auto dims = acGetMeshDims(acGridGetLocalMeshInfo());
    gmg_restrict_to_level(level);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_copy_residual_to_tmp,coarse_dims.n0,coarse_dims.n1),1);
    acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&mesh);
    acGridSynchronizeStream(STREAM_ALL);
    AcReal stencil[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
    for(int x = -NGHOST; x <= NGHOST; ++x)
    {
    	for(int y = -NGHOST; y <= NGHOST; ++y)
    	{
    		for(int z = -NGHOST; z <= NGHOST; ++z)
    		{
            		const int index = acVertexBufferIdx(hat_basis_position.x + x,hat_basis_position.y + y,hat_basis_position.z + z,info,GMG_RESIDUALS[level]);
            		const AcReal val = mesh.vertex_buffer[GMG_TMPS[level]][index];
    			stencil[z+NGHOST][y+NGHOST][x+NGHOST] = val;
    		}
    	}
    }
    const int central_index = acVertexBufferIdx(hat_basis_position.x,hat_basis_position.y,hat_basis_position.z,info,GMG_RESIDUALS[level]);
    const AcReal central_coeff = mesh.vertex_buffer[GMG_RESIDUALS[level]][central_index];
    gmg_central_coeffs[level] = central_coeff;
    //Here we load to both r1 and (r2/r3) since we are not sure is the user using compact poisson or not
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r1[level],stencil);
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_y_line_operators[level],stencil);
    acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r1[level],stencil);
    if(level <= 4)
    {
#if STENCIL_ORDER == 4
    	acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r2[level],stencil);
#endif
#if STENCIL_ORDER == 6
    	acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_operator_stencils_r3[level],stencil);
#endif
    }
    stencil[NGHOST][NGHOST][NGHOST] = 0.0;
    //acDeviceLoadStencil(acGridGetDevice(),STREAM_DEFAULT,galerkin_neighbours_operator_stencils[level],stencil);
    acDeviceSynchronizeStream(acGridGetDevice(),STREAM_DEFAULT);
    //fprintf(stderr,"\n");
    acHostMeshDestroy(&mesh);
};

void
gmg_populate_central_coeffients(AcMeshInfo* info)
{
  AcMeshInfo& config = *info;
  int nx = config[AC_nlocal].x;
  int ny = config[AC_nlocal].y;
  int nz = config[AC_nlocal].z;
  int max_level = 0;
  while(nx >= 2*NGHOST+1 && ny >= 2*NGHOST+1 && nz >= 2*NGHOST+1)
  {
  	nx /= 2;
  	ny /= 2;
  	nz /= 2;
  	++max_level;
  }
  if(max_level < config[AC_gmg_number_of_levels])
  {
  	acPushToConfig(config,AC_gmg_number_of_levels,max_level);
  	int rank = 0;
#if AC_MPI_ENABLED
  	if(config.comm != NULL && config.comm->handle!= MPI_COMM_NULL)
  	{
  		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  	}
#endif
  	if(rank == 0)
  	{
  		fprintf(stderr,"Astaroth Warning: Limited maximum gmg levels to %d to avoid impossible coarse grids\n",max_level);
  	}
  }
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

std::array<AcTaskGraph*,11> halo_exchange_residuals{};
std::array<AcTaskGraph*,11> halo_exchange_solutions{};
void
gmg_get_halo_exchange_operators(const AcMeshInfo info)
{
	const int number_of_levels = info[AC_gmg_number_of_levels];
	for(int level = 0; level < number_of_levels; ++level)
	{
		{
          		const Volume launch_start = to_volume(info[AC_nmin]);
          		const Volume launch_dims = to_volume(info[level_dims[level]]);
          		const Volume launch_end = launch_dims + launch_start;
			halo_exchange_residuals[level] = acGridBuildTaskGraph(
    				{
    				       acHaloExchange({GMG_RESIDUALS[level]})
    				}
				,launch_start,launch_end);
		}

		{
          		const Volume launch_start = to_volume(info[AC_nmin]);
          		const Volume launch_dims = to_volume(info[level_dims[level+1]]);
          		const Volume launch_end = launch_dims + launch_start;
			halo_exchange_solutions[level] = acGridBuildTaskGraph(
    					{
    					       acHaloExchange({GMG_SOLUTIONS[level+1]})
    					}
				,launch_start,launch_end);
		}

	}
}

void
gmg_setup(AcMeshInfo* info)
{
	AcMeshInfo& config = *info;

	if(config[AC_use_coarse_galerkin_operators])
	{
		get_galerkin_operators(info);
	}
	gmg_get_halo_exchange_operators(config);
}

void
gmg_iterative_smoother_step(const int level)
{
	const auto init_iterative_smoother = acGetOptimizedDSLTaskGraph(gmg_init_iterative_smoother);
	const auto iterative_smoother_step = acGetOptimizedDSLTaskGraph(gmg_smoother_step);
	const auto iterative_smoother_get_residual = acGetOptimizedDSLTaskGraph(gmg_smoother_residual_norm);
	const auto iterative_smoother_finalize = acGetOptimizedDSLTaskGraph(gmg_smoother_update_solution);
	
	acGridExecuteTaskGraph(init_iterative_smoother,1);
	acGridExecuteTaskGraph(iterative_smoother_get_residual,1);
	const AcReal iterative_smoother_residual0 = acDeviceGetOutput(acGridGetDevice(),AC_smoother_residual_l2_norm[level]);
	AcReal iterative_smoother_relative_residual = 1.0;
	while(iterative_smoother_relative_residual > 1e-12)
	{
		acGridExecuteTaskGraph(iterative_smoother_step,1);
		acGridExecuteTaskGraph(iterative_smoother_get_residual,1);
	    	iterative_smoother_relative_residual = acDeviceGetOutput(acGridGetDevice(),AC_smoother_residual_l2_norm[level])/iterative_smoother_residual0;
	}
	acGridExecuteTaskGraph(iterative_smoother_finalize,1);
}

void
gmg_smoothing_step(const int level)
{
  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
  const auto smoother = acDeviceGetLocalConfig(acGridGetDevice())[AC_GMG_SMOOTHER];
  switch(smoother)
  {
	  case SPAI_SMOOTHER:
	  {
		acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_optimized_smoother),1);
		break;
	  }
	  case Y_LINE_SMOOTHER:
	  {
		for(int color = 0; color < 2; ++color)
		{
  			acDeviceSetInput(acGridGetDevice(),AC_GMG_SMOOTHER_COLOR,color);
			gmg_iterative_smoother_step(level);
		}
		break;
	  }
  }
}
void
gmg_level_step(const int level, const int number_of_levels, const AcReal relative_residual_tolerance)
{
  const auto info = acGridGetLocalMeshInfo();
  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
  
  for(int i = 0; i < info[AC_gmg_pre_smooth_steps]; ++i)
  {
  	gmg_smoothing_step(level); //Pre-smooth step
  }
  //Now using CG on the coarsest level
  //TODO: option to choose the coarse level solver since we might not always be SPD
  if(level == number_of_levels-1)
  {
  	const auto residual_graph = acGetOptimizedDSLTaskGraph(gmg_get_residual_norm);
    	acGridExecuteTaskGraph(residual_graph,1);
	const AcReal residual0_norm = acDeviceGetOutput(acGridGetDevice(), AC_GMG_residual_l2_norm[level]);
	AcReal residual_norm = acDeviceGetOutput(acGridGetDevice(), AC_GMG_residual_l2_norm[level]);
	AcReal relative_residual_norm = residual_norm/residual0_norm;
        const Volume launch_start = to_volume(info[AC_nmin]);
        const Volume launch_dims = to_volume(info[level_dims[level]]);
	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_init_cg_residual,launch_start,launch_dims),1);
	while(relative_residual_norm > relative_residual_tolerance)
	{
		acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_cg_coarsest_level_step,launch_start,launch_dims),1);
    		acGridExecuteTaskGraph(residual_graph,1);
		residual_norm = acDeviceGetOutput(acGridGetDevice(), AC_GMG_residual_l2_norm[level]);
		relative_residual_norm = residual_norm/residual0_norm;
	}
  }
  else
  {
  	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_residual),1); //Get residual
	  acGridExecuteTaskGraph(halo_exchange_residuals[level],1);

          const Volume launch_start = to_volume(info[AC_nmin]);
          const Volume launch_dims = to_volume(info[level_dims[level+1]]);
          const Volume launch_end = launch_dims + launch_start;
	  
          const auto restrict_graph = acGetOptimizedDSLTaskGraph(gmg_restrict_residual, launch_start, launch_end); 
          acGridExecuteTaskGraph(restrict_graph,1); //Restrict residual to the next level
	  gmg_level_step(level+1,number_of_levels,relative_residual_tolerance);
  	  acDeviceSetInput(acGridGetDevice(),AC_GMG_LEVEL,(GMG_LEVEL)level);
	  acGridExecuteTaskGraph(halo_exchange_solutions[level],1);
	  acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(gmg_get_correction_from_next_level),1); //Prolong and add the solution from the next level
  	  for(int i = 0; i < info[AC_gmg_post_smooth_steps]; ++i)
  	  {
  	  	gmg_smoothing_step(level); //Post-smooth step
  	  }
  }
}

void
gmg_v_cycle(const int number_of_levels, const AcReal relative_residual_tolerance)
{
	gmg_level_step(0,number_of_levels,relative_residual_tolerance);
}
void
gmg_setup_parallel_grid_decomposition(AcMeshInfo* dst)
{

    const int nxgrid = (*dst)[AC_ngrid].x;
    const int nygrid = (*dst)[AC_ngrid].y;
    const int nzgrid = (*dst)[AC_ngrid].z;
    acUpdateDecompositionParams(dst);

    const int3 decomp = (*dst)[AC_domain_decomposition];
    int nx = (1+nxgrid)/decomp.x;
    int ny = (1+nygrid)/decomp.y;
    int nz = (1+nzgrid)/decomp.z;
    acSetGridMeshDims(nxgrid,nygrid,nzgrid, dst);
    const int3 pid3d = (*dst)[AC_domain_coordinates];
    if(pid3d.x == decomp.x-1) nx--;
    if(pid3d.y == decomp.y-1) ny--;
    if(pid3d.z == decomp.z-1) nz--;
    acPushToConfig((*dst),AC_power_of_two_minus_one_grid,true);
    acPushToConfig((*dst),AC_allow_non_divisible_grid,true);
    acSetLocalMeshDims(nx,ny,nz,dst);
}
