real AC_dsx
real AC_dsy
real AC_dsz

real AC_inv_dsx
real AC_inv_dsy
real AC_inv_dsz

int AC_mx
int AC_my
int AC_mz

int AC_nx
int AC_ny
int AC_nz

int AC_nxgrid
int AC_nygrid
int AC_nzgrid

int AC_nx_min
int AC_ny_min
int AC_nz_min

int AC_nx_max
int AC_ny_max
int AC_nz_max

int AC_mxy
int AC_nxy
int AC_nxyz


real AC_xlen
real AC_ylen
real AC_zlen

int3 AC_domain_decomposition
int3 AC_multigpu_offset
int3 AC_domain_coordinates

int AC_proc_mapping_strategy
int AC_decompose_strategy
int AC_MPI_comm_strategy
#if AC_LAGRANGIAN_GRID 
Field COORDS_X
Field COORDS_Y
Field COORDS_Z
#endif
