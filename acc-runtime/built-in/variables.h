const int NGHOST_VAL = 3
run_const real AC_dsx
run_const real AC_dsy
#if TWO_D == 0
run_const real AC_dsz
#endif

run_const real AC_inv_dsx
run_const real AC_inv_dsy
#if TWO_D == 0
run_const real AC_inv_dsz
#endif

run_const real AC_inv_dsx_2
run_const real AC_inv_dsy_2
#if TWO_D == 0
run_const real AC_inv_dsz_2
#endif

run_const real AC_inv_dsx_3
run_const real AC_inv_dsy_3
#if TWO_D == 0
run_const real AC_inv_dsz_3
#endif

run_const real AC_inv_dsx_4
run_const real AC_inv_dsy_4
#if TWO_D == 0
run_const real AC_inv_dsz_4
#endif

run_const real AC_inv_dsx_5
run_const real AC_inv_dsy_5
#if TWO_D == 0
run_const real AC_inv_dsz_5
#endif

run_const real AC_inv_dsx_6
run_const real AC_inv_dsy_6
#if TWO_D == 0
run_const real AC_inv_dsz_6
#endif

run_const int AC_mx
run_const int AC_my
#if TWO_D == 0
run_const int AC_mz
#endif

run_const int AC_nx
run_const int AC_ny
#if TWO_D == 0
run_const int AC_nz
#endif

run_const int AC_nxgrid
run_const int AC_nygrid
#if TWO_D == 0
run_const int AC_nzgrid
#endif

run_const int AC_mxgrid
run_const int AC_mygrid
#if TWO_D == 0
run_const int AC_mzgrid
#endif

run_const int AC_nx_min
run_const int AC_ny_min
#if TWO_D == 0
run_const int AC_nz_min
#endif

run_const int AC_nx_max
run_const int AC_ny_max
#if TWO_D == 0
run_const int AC_nz_max
#endif

run_const int AC_nxgrid_max
run_const int AC_nygrid_max
#if TWO_D == 0
run_const int AC_nzgrid_max
#endif

run_const int AC_mxy
run_const int AC_mxygrid
run_const int AC_nxy
run_const int AC_nxygrid
#if TWO_D == 0
run_const int AC_nxyz
run_const int AC_nxyzgrid
#endif


run_const real AC_xlen
run_const real AC_ylen
#if TWO_D == 0
run_const real AC_zlen
#endif


run_const int AC_proc_mapping_strategy
run_const int AC_decompose_strategy
run_const int AC_MPI_comm_strategy
#if AC_LAGRANGIAN_GRID 
Field COORDS_X
Field COORDS_Y
#if TWO_D == 0
Field COORDS_Z
#endif
#endif

run_const int3 AC_domain_decomposition
int3 AC_multigpu_offset
int3 AC_domain_coordinates
