hostdefine NGHOST (STENCIL_ORDER/2)
real AC_dsx
real AC_dsy
real AC_dsz

real AC_inv_dsx
real AC_inv_dsy
real AC_inv_dsz

int AC_mx
int AC_my
#if TWO_D == 0
int AC_mz
#endif

int AC_nx
int AC_ny
#if TWO_D == 0
int AC_nz
#endif

int AC_nxgrid
int AC_nygrid
#if TWO_D == 0
int AC_nzgrid
#endif

int AC_nx_min
int AC_ny_min
#if TWO_D == 0
int AC_nz_min
#endif

int AC_nx_max
int AC_ny_max
#if TWO_D == 0
int AC_nz_max
#endif

int AC_mxy
int AC_nxy
#if TWO_D == 0
int AC_nxyz
#endif


real AC_xlen
real AC_ylen
#if TWO_D == 0
real AC_zlen
#endif


int AC_proc_mapping_strategy
int AC_decompose_strategy
int AC_MPI_comm_strategy
#if AC_LAGRANGIAN_GRID 
Field COORDS_X
Field COORDS_Y
#if TWO_D == 0
Field COORDS_Z
#endif
#endif

