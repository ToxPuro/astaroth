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

run_const int AC_mxy
run_const int AC_nxy
#if TWO_D == 0
run_const int AC_nxyz
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
