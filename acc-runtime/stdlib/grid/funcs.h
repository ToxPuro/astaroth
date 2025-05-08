
grid_position() {
//MR: generalize, using x,y,z?
//TP: implicitly assumes equidistant grid
    return ((globalVertexIdx - AC_nmin)*AC_ds) + AC_first_gridpoint
}

grid_position(int3 local_point) {
//TP: implicitly assumes equidistant grid
    global_point = (local_point-AC_nmin) + AC_multigpu_offset
    return global_point*AC_ds + AC_first_gridpoint
}

grid_center() {
//MR: generalize, using x,y,z?
//TP: implicitly assumes equidistant grid
    return (0.5*AC_len) + AC_first_gridpoint;
}
