
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
