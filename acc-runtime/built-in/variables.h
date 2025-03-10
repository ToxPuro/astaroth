//TP: for now TWO_D means XY setup
run_const bool3 AC_dimension_inactive (bool3){false,false,TWO_D};

run_const real3 AC_ds
run_const real AC_dsmin = AC_dimension_inactive.z ? min(AC_ds.x,AC_ds.y) : min(AC_ds)

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

//TP: these could be run_const but gives really bad performance otherwise
int3 AC_nmin = (int3)
		{
			AC_dimension_inactive.x ? 0 : NGHOST,
			AC_dimension_inactive.y ? 0 : NGHOST,
			AC_dimension_inactive.z ? 0 : NGHOST
		}

int3 AC_nlocal
int3 AC_mlocal = AC_nlocal + 2*AC_nmin
int3 AC_ngrid 
int3 AC_mgrid  = AC_ngrid + 2*AC_nmin
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

run_const bool3 AC_periodic_grid
run_const bool AC_fully_periodic_grid = AC_periodic_grid.x && AC_periodic_grid.y && AC_periodic_grid.z

run_const real3 AC_len = (AC_ngrid + AC_periodic_grid - 1)*AC_ds

run_const int AC_proc_mapping_strategy
run_const int AC_decompose_strategy
run_const int AC_MPI_comm_strategy
#if AC_LAGRANGIAN_GRID 
#if TWO_D == 0
Field3 AC_COORDS
#else
Field2 AC_COORDS
#endif
#endif

run_const int3  AC_domain_decomposition
run_const bool  AC_host_has_row_memory_order

			

run_const bool  AC_lagrangian_grid = AC_LAGRANGIAN_GRID
int3 AC_thread_block_loop_factors
int3 AC_max_tpb_for_reduce_kernels
int3 AC_reduction_tile_dimensions
int3 AC_multigpu_offset
int3 AC_domain_coordinates

run_const bool AC_include_3d_halo_corners
run_const bool AC_skip_single_gpu_optim

run_const AC_COORDINATE_SYSTEM AC_coordinate_system
run_const bool3 AC_nonequidistant_grid

output real AC_default_real_output
//TP: these belong here but at the moment are deprecated
/**
run_const AC_xy_plate_bufsize
run_const AC_xz_plate_bufsize
run_const AC_yz_plate_bufsize
**/
