// physical grid
real AC_xlen
real AC_ylen
real AC_zlen
real AC_xorig
real AC_yorig
real AC_zorig

real AC_dsx, AC_dsy, AC_dsz
real AC_dsmin

grid_position() {
    return real3((globalVertexIdx.x - AC_nx_min) * AC_dsx,
                 (globalVertexIdx.y - AC_ny_min) * AC_dsy,
                 (globalVertexIdx.z - AC_nz_min) * AC_dsz)
}

grid_centre() {
    return real3(((globalGridN.x-1) * AC_dsx)/2.0,
                 ((globalGridN.y-1) * AC_dsy)/2.0,
                 ((globalGridN.z-1) * AC_dsz)/2.0)
}
