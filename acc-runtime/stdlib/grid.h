// physical grid
run_const real AC_xorig
run_const real AC_yorig
run_const real AC_zorig

run_const real AC_center_x
run_const real AC_center_y
run_const real AC_center_z

gmem real AC_x[AC_mlocal.x]
gmem real AC_y[AC_mlocal.y]
gmem real AC_z[AC_mlocal.z]


grid_position() {
//MR: generalize, using x,y,z?
    return (globalVertexIdx - AC_nlocal)*AC_ds
}

grid_centre() {
//MR: generalize, using x,y,z?
    return ((AC_ngrid-1)*AC_ds)*0.5;
}
