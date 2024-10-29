// physical grid
run_const real AC_xorig
run_const real AC_yorig
run_const real AC_zorig

run_const real AC_center_x
run_const real AC_center_y
run_const real AC_center_z

gmem real AC_x[mx]
gmem real AC_y[my]
gmem real AC_z[mz]

grid_position() {
//MR: generalize, using x,y,z?
    return real3((globalVertexIdx.x - AC_nx_min) * AC_dsx,
                 (globalVertexIdx.y - AC_ny_min) * AC_dsy,
                 (globalVertexIdx.z - AC_nz_min) * AC_dsz)
}

grid_centre() {
//MR: generalize, using x,y,z?
    return real3(((globalGridN.x-1) * AC_dsx)/2.0,
                 ((globalGridN.y-1) * AC_dsy)/2.0,
                 ((globalGridN.z-1) * AC_dsz)/2.0)
}
