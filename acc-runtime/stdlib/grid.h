// physical grid
run_const real3 AC_origin

run_const real3 AC_center

gmem real AC_x[AC_mlocal.x]
gmem real AC_y[AC_mlocal.y]
gmem real AC_z[AC_mlocal.z]

grid_position() {
//MR: generalize, using x,y,z?
//TP: implicitly assumes [0,AC_len] domain
    return (globalVertexIdx - AC_nmin)*AC_ds
}

grid_center() {
//MR: generalize, using x,y,z?
//TP: implicitly assumes [0,AC_len] domain
    return 0.5*AC_len;
}
