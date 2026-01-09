#ifndef AC_GENERAL_GRID_FUNCS_H

#define AC_GENERAL_GRID_FUNCS_H

#ifdef AC_HOME
#include "$AC_HOME/acc-runtime/stdlib/general_grid/vars.h"
#endif


grid_position() {
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
	    	if(AC_equidistant_in_all_directions)
	    	{
    	    	    return ((globalVertexIdx - AC_nmin)*AC_ds) + AC_first_gridpoint
	    	}
		return 
			real3(
				AC_x[vertexIdx.x],
				AC_y[vertexIdx.y],
				AC_z[vertexIdx.z]
			     )
	}
	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[vertexIdx.x],
				AC_theta[vertexIdx.y],
				AC_phi[vertexIdx.z]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_position!\n");
	}
	return real3(0.0,0.0,0.0)
}

grid_position_extended() {
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[vertexIdx.x],
				AC_y[vertexIdx.y],
				AC_z[vertexIdx.z]
			     )
	}
	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r_extended[vertexIdx.x],
				AC_theta[vertexIdx.y],
				AC_phi[vertexIdx.z]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_position!\n");
	}
	return real3(0.0,0.0,0.0)
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
	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[vertexIdx.x]*AC_sin_theta[vertexIdx.y]*AC_cos_phi[vertexIdx.z],
				AC_r[vertexIdx.x]*AC_sin_theta[vertexIdx.y]*AC_sin_phi[vertexIdx.z],
				AC_r[vertexIdx.x]*AC_cos_theta[vertexIdx.y]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_xyz!\n");
	}
	return real3(0.0,0.0,0.0)
}



grid_position(int3 local_point) {
    	global_point = (local_point-AC_nmin) + AC_multigpu_offset
	if(AC_equidistant_in_all_directions)
	{
    		return global_point*AC_ds + AC_first_gridpoint
	}
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[global_point.x],
				AC_y[global_point.y],
				AC_z[global_point.z]
			     )
	}
    	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[global_point.x],
				AC_theta[global_point.y],
				AC_phi[global_point.z]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_position!\n");
	}
	return real3(0.0,0.0,0.0)
}

grid_position_extended(int3 local_point) {
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
    	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r_extended[global_point.x],
				AC_theta[global_point.y],
				AC_phi[global_point.z]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_position!\n");
	}
	return real3(0.0,0.0,0.0)
}

grid_center() {
    if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
    {
	    if(AC_equidistant_in_all_directions)
	    {
    		return (0.5*AC_len) + AC_first_gridpoint;
	    }
	    return real3(
			    AC_x[AC_ngrid.x/2],
			    AC_y[AC_ngrid.y/2],
			    AC_z[AC_ngrid.z/2]
			)
    }
    fatal_error_message(false,"grid_center only implemented in Cartesian!\n");
}
get_integration_weight()
{
	w = 1.0
	if(AC_nonequidistant_grid.x) w *= AC_mapping_func_derivative_x[vertexIdx.x]
	else w *= AC_ds.x

	if(AC_nonequidistant_grid.y) w *= AC_mapping_func_derivative_y[vertexIdx.y]
	else w *= AC_ds.y

	if(AC_nonequidistant_grid.z) w *= AC_mapping_func_derivative_z[vertexIdx.z]
	else w *= AC_ds.z

	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		r = AC_r[vertexIdx.x]
		w *= r*r
		w *= AC_sin_theta[vertexIdx.y]
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		r = AC_r[vertexIdx.x]
		w *= r
	}
	return w
}

get_integration_weight_extended()
{
	w = 1.0
	if(AC_nonequidistant_grid.x) w *= AC_mapping_func_derivative_x_extended[vertexIdx.x]
	else w *= AC_ds.x

	if(AC_nonequidistant_grid.y) w *= AC_mapping_func_derivative_y_extended[vertexIdx.y]
	else w *= AC_ds.y

	if(AC_nonequidistant_grid.z) w *= AC_mapping_func_derivative_z_extended[vertexIdx.z]
	else w *= AC_ds.z

	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		r = AC_r_extended[vertexIdx.x]
		w *= r*r
		w *= AC_sin_theta[vertexIdx.y]
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		r = AC_r_extended[vertexIdx.x]
		w *= r
	}
	return w
}
#endif
