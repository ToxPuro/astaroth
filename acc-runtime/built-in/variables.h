run_const real3 AC_ds
run_const real AC_dsmin

run_const real3 AC_inv_ds
run_const real3 AC_inv_ds_2
run_const real3 AC_inv_ds_3
run_const real3 AC_inv_ds_4
run_const real3 AC_inv_ds_5
run_const real3 AC_inv_ds_6
run_const int3 AC_nlocal
run_const int3 AC_mlocal

run_const int3 AC_ngrid
run_const int3 AC_mgrid
run_const int3 AC_nmin
run_const int3 AC_nlocal_max

run_const int3 AC_ngrid_max


run_const AcDimProducts AC_nlocal_products
run_const AcDimProducts AC_ngrid_products
run_const AcDimProducts AC_mlocal_products
run_const AcDimProducts AC_mgrid_products
run_const real3 AC_len

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
run_const bool3 AC_dimension_inactive
run_const bool  AC_lagrangian_grid
int3 AC_thread_block_loop_factors
int3 AC_max_tpb_for_reduce_kernels
int3 AC_reduction_tile_dimensions
int3 AC_multigpu_offset
int3 AC_domain_coordinates

run_const bool AC_include_3d_halo_corners


//TP: these belong here but at the moment are deprecated
/**
run_const AC_xy_plate_bufsize
run_const AC_xz_plate_bufsize
run_const AC_yz_plate_bufsize
**/
