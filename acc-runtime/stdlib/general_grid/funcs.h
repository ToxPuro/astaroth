grid_position() {
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[vertexIdx.x],
				AC_y[vertexIdx.y],
				AC_z[vertexIdx.z]
			     )
	}
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[vertexIdx.x],
				AC_theta[vertexIdx.y],
				AC_phi[vertexIdx.z]
			     )
	}
}

grid_xyz() {
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[vertexIdx.x],
				AC_y[vertexIdx.y],
				AC_z[vertexIdx.z]
			     )
	}
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[vertexIdx.x]*AC_sin_theta[vertexIdx.y]*AC_cos_phi[vertexIdx.z],
				AC_r[vertexIdx.x]*AC_sin_theta[vertexIdx.y]*AC_sin_phi[vertexIdx.z],
				AC_r[vertexIdx.x]*AC_cos_theta[vertexIdx.y]
			     )
	}
}



grid_position(int3 local_point) {
    global_point = local_point + AC_multigpu_offset
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[global_point.x],
				AC_y[global_point.y],
				AC_z[global_point.z]
			     )
	}
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[global_point.x],
				AC_theta[global_point.y],
				AC_phi[global_point.z]
			     )
	}
}

grid_center() {
    if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
    {
	    return real3(
			    AC_x[AC_nxgrid.x/2],
			    AC_y[AC_nxgrid.y/2],
			    AC_z[AC_nzgrid.z/2]
			)
    }
    fatal_error_message(false,"grid_center only implemented in Cartesian!\n");
}
