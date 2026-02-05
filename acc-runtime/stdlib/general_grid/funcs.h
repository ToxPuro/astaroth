#ifndef AC_GENERAL_GRID_FUNCS_H

hostdefine AC_GENERAL_GRID_INCLUDED (1)
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
	if(AC_equidistant_in_all_directions)
	{
    		global_point = (local_point-AC_nmin) + AC_multigpu_offset
    		return global_point*AC_ds + AC_first_gridpoint
	}
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[local_point.x],
				AC_y[local_point.y],
				AC_z[local_point.z]
			     )
	}
    	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r[local_point.x],
				AC_theta[local_point.y],
				AC_phi[local_point.z]
			     )
	}
	else
	{
    		fatal_error_message(false,"Unsupported coordinate system for grid_position!\n");
	}
	return real3(0.0,0.0,0.0)
}

grid_position_extended(int3 local_point) {
	if(AC_coordinate_system == AC_CARTESIAN_COORDINATES)
	{
		return 
			real3(
				AC_x[local_point.x],
				AC_y[local_point.y],
				AC_z[local_point.z]
			     )
	}
    	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		return 
			real3(
				AC_r_extended[local_point.x],
				AC_theta[local_point.y],
				AC_phi[local_point.z]
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
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    return real3(
			    AC_r[AC_ngrid.x/2],
			    AC_theta[AC_ngrid.y/2],
			    AC_phi[AC_ngrid.z/2]
			)
    }
    fatal_error_message(false,"grid_center only implemented in Cartesian and spherical!\n");
    return real3(0.,0.,0.)
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
grid_inv_spacing()
{
	real3 res
	if(AC_nonequidistant_grid.x)
	{
		res.x = AC_INV_MAPPING_FUNC_DER_X
	}
	else
	{
		res.x = AC_inv_ds.x
	}
	if(AC_nonequidistant_grid.y)
	{
		res.y = AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		res.y = AC_inv_ds.y
	}
	if(AC_nonequidistant_grid.z)
	{
		res.z = AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		res.z = AC_inv_ds.z
	}
	return res
}

grid_inv_spacing_extended()
{
	real3 res
	if(AC_nonequidistant_grid.x)
	{
		res.x = AC_INV_MAPPING_FUNC_DER_X_extended
	}
	else
	{
		res.x = AC_inv_ds.x
	}
	if(AC_nonequidistant_grid.y)
	{
		res.y = AC_INV_MAPPING_FUNC_DER_Y_extended
	}
	else
	{
		res.y = AC_inv_ds.y
	}
	if(AC_nonequidistant_grid.z)
	{
		res.z = AC_INV_MAPPING_FUNC_DER_Z_extended
	}
	else
	{
		res.z = AC_inv_ds.z
	}
	return res
}
#endif
