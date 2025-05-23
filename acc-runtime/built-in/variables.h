//TP: for now TWO_D means XY setup
run_const bool3 AC_dimension_inactive = (bool3){false,false,TWO_D};

run_const real3 AC_ds
run_const real AC_dsmin = AC_dimension_inactive.z ? min(AC_ds.x,AC_ds.y) : min(AC_ds)
run_const real AC_dsmin_2 = AC_dsmin*AC_dsmin

run_const real3 AC_inv_ds = 1.0/AC_ds
run_const real3 AC_inv_ds_2 = AC_inv_ds*AC_inv_ds
run_const real3 AC_inv_ds_3 = AC_inv_ds_2*AC_inv_ds
run_const real3 AC_inv_ds_4 = AC_inv_ds_2*AC_inv_ds_2
run_const real3 AC_inv_ds_5 = AC_inv_ds_3*AC_inv_ds_2
run_const real3 AC_inv_ds_6 = AC_inv_ds_3*AC_inv_ds_3

run_const real3 AC_ds_2 = AC_ds*AC_ds
run_const real3 AC_ds_3 = AC_ds_2*AC_ds
run_const real3 AC_ds_4 = AC_ds_2*AC_ds_2
run_const real3 AC_ds_5 = AC_ds_3*AC_ds_2
run_const real3 AC_ds_6 = AC_ds_3*AC_ds_3

int3 AC_nmin = (int3)
		{
			AC_dimension_inactive.x ? 0 : NGHOST,
			AC_dimension_inactive.y ? 0 : NGHOST,
			AC_dimension_inactive.z ? 0 : NGHOST
		}


run_const AcProcMappingStrategy AC_proc_mapping_strategy = AC_PROC_MAPPING_STRATEGY_MORTON
run_const AcDecomposeStrategy   AC_decompose_strategy    = AC_DECOMPOSE_STRATEGY_MORTON
run_const AcMPICommStrategy     AC_MPI_comm_strategy     = AC_MPI_COMM_STRATEGY_DUP_WORLD

run_const int3  AC_domain_decomposition = ac_get_process_decomposition()

//TP: these could be run_const but gives really bad performance otherwise
int3 AC_ngrid 
int3 AC_mgrid  = AC_ngrid + 2*AC_nmin
int3 AC_nlocal = (int3)
		{
			AC_ngrid.x/AC_domain_decomposition.x,
			AC_ngrid.y/AC_domain_decomposition.y,
			AC_ngrid.z/AC_domain_decomposition.z
		}

int3 AC_mlocal = AC_nlocal + 2*AC_nmin
int3 AC_nlocal_max = AC_nlocal + AC_nmin
int AC_nlocal_max_dim = max(AC_nlocal)
int3 AC_ngrid_max = AC_ngrid + AC_nmin

run_const real3 AC_nlocal_inv = (real3) {1.0/AC_nlocal.x, 1.0/AC_nlocal.y, 1.0/AC_nlocal.z}
run_const real3 AC_ngrid_inv = (real3){1.0/AC_ngrid.x, 1.0/AC_ngrid.y, 1.0/AC_ngrid.z}

AcDimProducts AC_nlocal_products = ac_get_dim_products(AC_nlocal)
run_const AcDimProductsInv AC_nlocal_products_inv = ac_get_dim_products_inv(AC_nlocal_products)

AcDimProducts AC_ngrid_products = ac_get_dim_products(AC_ngrid)
run_const AcDimProductsInv AC_ngrid_products_inv = ac_get_dim_products_inv(AC_ngrid_products)

AcDimProducts AC_mlocal_products = ac_get_dim_products(AC_mlocal)
run_const AcDimProductsInv AC_mlocal_products_inv = ac_get_dim_products_inv(AC_mlocal_products)

AcDimProducts AC_mgrid_products = ac_get_dim_products(AC_mgrid)
run_const AcDimProductsInv AC_mgrid_products_inv = ac_get_dim_products_inv(AC_mgrid_products)

run_const bool  AC_allow_non_periodic_bcs_with_periodic_grid = false
run_const bool3 AC_periodic_grid
run_const bool AC_fully_periodic_grid = AC_periodic_grid.x && AC_periodic_grid.y && AC_periodic_grid.z

run_const real3 AC_len = (AC_ngrid + AC_periodic_grid - 1)*AC_ds
run_const real3 AC_first_gridpoint =  (real3){
					AC_periodic_grid.x*AC_ds.x*0.5,
					AC_periodic_grid.y*AC_ds.y*0.5,
					AC_periodic_grid.z*AC_ds.z*0.5
				       }


#if AC_LAGRANGIAN_GRID 
#if TWO_D == 0
Field3 AC_COORDS
#else
Field2 AC_COORDS
#endif
#endif

run_const bool  AC_host_has_row_memory_order

			

run_const bool  AC_lagrangian_grid = AC_LAGRANGIAN_GRID
int3 AC_thread_block_loop_factors = (int3){1,1,1}
int3 AC_max_tpb_for_reduce_kernels = (int3){-1,8,8}
int3 AC_reduction_tile_dimensions
int3 AC_multigpu_offset = (int3){0,0,0}
int3 AC_domain_coordinates

run_const bool AC_include_3d_halo_corners = false
run_const bool AC_skip_single_gpu_optim = false

run_const AcCoordinateSystem AC_coordinate_system = AC_CARTESIAN_COORDINATES
run_const bool3 AC_nonequidistant_grid = (bool3){false,false,false}

run_const bool AC_sparse_autotuning=false

output real AC_default_real_output

run_const bool AC_shear = false
//Uses GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory
run_const bool AC_use_cuda_aware_mpi = true
input real AC_shear_delta_y
run_const int3 AC_raytracing_block_factors = (int3){1,1,1}
run_const int AC_x_ray_shared_mem_block_size = 32
run_const int AC_z_ray_shared_mem_block_size = 1
//TP: these belong here but at the moment are deprecated
/**
run_const AC_xy_plate_bufsize
run_const AC_xz_plate_bufsize
run_const AC_yz_plate_bufsize
**/
