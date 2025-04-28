Stencil sld_diff_left_left
{
	[0][0][-2] = -1,
	[0][0][-1] =   1
}
Stencil sld_diff_left
{
	[0][0][-1] = -1,
	[0][0][0] =   1
}
Stencil sld_add_left
{
	[0][0][-1] =  1,
	[0][0][0] =   1
}
Stencil sld_diff_right
{
	[0][0][0] = -1,
	[0][0][1] =   1
}
Stencil sld_add_right
{
	[0][0][0] =   1,
	[0][0][1] =   1
}

Stencil sld_diff_right_right
{
	[0][0][1] = -1,
	[0][0][2] =   1
}

Stencil sld_diff_down_down
{
	[0][-2][0] = -1,
	[0][-1][0] =   1
}
Stencil sld_diff_down
{
	[0][-1][0] = -1,
	[0][0 ][0] =   1
}
Stencil sld_add_down
{
	[0][-1][0] =   1,
	[0][0 ][0] =   1
}
Stencil sld_diff_up
{
	[0][0][0] = -1,
	[0][1][0] =   1
}
Stencil sld_add_up
{
	[0][0][0] =   1,
	[0][1][0] =   1
}

Stencil sld_diff_up_up
{
	[0][1][0] = -1,
	[0][2][0] =   1
}

Stencil sld_diff_back_back
{
	[-2][0][0] = -1,
	[-1][0][0] =   1
}
Stencil sld_diff_back
{
	[-1][0][0] = -1,
	[0 ][0][0] =   1
}
Stencil sld_add_back
{
	[-1][0][0] =   1,
	[0 ][0][0] =   1
}
Stencil sld_diff_front
{
	[0][0][0] = -1,
	[1][0][0] =   1
}
Stencil sld_add_front
{
	[0][0][0] =   1,
	[1][0][0] =   1
}

Stencil sld_diff_front_front
{
	[1][0][0] = -1,
	[2][0][0] =   1
}
Stencil sld_get_left
{
	[0][0][-1] = 1
}
Stencil sld_get_right
{
	[0][0][1] = 1
}
Stencil sld_get_down
{
	[0][-1][0] = 1
}
Stencil sld_get_up
{
	[0][1][0] = 1
}
Stencil sld_get_back
{
	[-1][0][0] = 1
}
Stencil sld_get_front
{
	[1][0][0] = 1
}

get_x_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return real2(
				interpolate_middle_left(characteristic_speed),
				interpolate_middle_right(characteristic_speed)
		    )
}
get_y_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return real2(
				interpolate_middle_down(characteristic_speed),
				interpolate_middle_up(characteristic_speed)
		    )
}
get_z_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return real2(
				interpolate_middle_back(characteristic_speed),
				interpolate_middle_front(characteristic_speed)
		    )
}
get_left_slope(Field f)
{
	return minmod_alt(sld_diff_left_left(f),sld_diff_left(f))
}
get_x_slope(Field f)
{
	return minmod_alt(sld_diff_left(f),sld_diff_right(f))
}
get_right_slope(Field f)
{
	return minmod_alt(sld_diff_right(f),sld_diff_right_right(f))
}

get_down_slope(Field f)
{
	return minmod_alt(sld_diff_down_down(f),sld_diff_down(f))
}
get_y_slope(Field f)
{
	return minmod_alt(sld_diff_down(f),sld_diff_up(f))
}
get_up_slope(Field f)
{
	return minmod_alt(sld_diff_up(f),sld_diff_up_up(f))
}

get_back_slope(Field f)
{
	return minmod_alt(sld_diff_back_back(f),sld_diff_back(f))
}
get_z_slope(Field f)
{
	return minmod_alt(sld_diff_back(f),sld_diff_front(f))
}
get_front_slope(Field f)
{
	return minmod_alt(sld_diff_front(f),sld_diff_front_front(f))
}
get_x_interface_values(Field f)
{
	left_slope  = get_left_slope(f)
	slope       = get_x_slope(f)
	right_slope = get_right_slope(f)
	return real4(
			sld_get_left(f)+left_slope,
			f-slope,
			f+slope,
			sld_get_right(f)-right_slope
		    )
}
get_y_interface_values(Field f)
{
	down_slope  = get_down_slope(f)
	slope       = get_y_slope(f)
	up_slope    = get_up_slope(f)
	return real4(
			sld_get_down(f)+down_slope,
			f-slope,
			f+slope,
			sld_get_up(f)-up_slope
		    )
}
get_z_interface_values(Field f)
{
	back_slope  = get_back_slope(f)
	slope       = get_y_slope(f)
	front_slope = get_front_slope(f)
	return real4(
			sld_get_back(f)+back_slope,
			f-slope,
			f+slope,
			sld_get_front(f)-front_slope
		    )
}
get_x_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	cs = get_x_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_x_interface_values(f)

	left_diff = sld_diff_left(f)
	left_add  = sld_add_left(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff = sld_diff_right(f)
	right_add  =  sld_add_right(f)
	right_slope_ratio = 0.0
	right_interface_diff = interface_values.w - interface_values.z
	if((right_interface_diff)*right_diff > 0.0)
	{
	    if(abs(right_add/right_diff) > fdiff_limit) right_diff = sign(right_add,right_diff)/fdiff_limit
	    right_slope_ratio = (right_interface_diff)right_diff
	}
	right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	right_flux = 0.5*cs.x*right_Q*(right_interface_diff)
	return real2(left_flux,right_flux)
}
get_y_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	cs = get_y_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_y_interface_values(f)

	left_diff = sld_diff_down(f)
	left_add  = sld_add_down(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff = sld_diff_up(f)
	right_add  =  sld_add_up(f)
	right_slope_ratio = 0.0
	right_interface_diff = interface_values.w - interface_values.z
	if((right_interface_diff)*right_diff > 0.0)
	{
	    if(abs(right_add/right_diff) > fdiff_limit) right_diff = sign(right_add,right_diff)/fdiff_limit
	    right_slope_ratio = (right_interface_diff)right_diff
	}
	right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	right_flux = 0.5*cs.x*right_Q*(right_interface_diff)
	return real2(left_flux,right_flux)
}

get_z_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	cs = get_z_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_z_interface_values(f)

	left_diff = sld_diff_back(f)
	left_add  = sld_add_back(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff = sld_diff_front(f)
	right_add  =  sld_add_front(f)
	right_slope_ratio = 0.0
	right_interface_diff = interface_values.w - interface_values.z
	if((right_interface_diff)*right_diff > 0.0)
	{
	    if(abs(right_add/right_diff) > fdiff_limit) right_diff = sign(right_add,right_diff)/fdiff_limit
	    right_slope_ratio = (right_interface_diff)right_diff
	}
	right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	right_flux = 0.5*cs.x*right_Q*(right_interface_diff)
	return real2(left_flux,right_flux)
}
//TP: works only for equidistant cartesian at the moment
get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	x_fluxes = get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	y_fluxes = get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	z_fluxes = get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	return 
		  (x_fluxes.y - x_fluxes.x)/AC_inv_ds.x
		+ (y_fluxes.y - y_fluxes.x)/AC_inv_ds.y
		+ (z_fluxes.y - z_fluxes.x)/AC_inv_ds.z
}
calculate_characteristic_speed(Field3 uu, real w_uu, real sound_speed, real w_sound) 
{
	return w_uu*norm(uu) + w_sound*sound_speed
}
