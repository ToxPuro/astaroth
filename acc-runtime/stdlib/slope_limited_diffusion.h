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


ExpSum Stencil sld_diff_left_left_exp
{
	[0][0][-2] = -1,
	[0][0][-1] =   1
}

ExpSum Stencil sld_diff_left_exp
{
	[0][0][-1] = -1,
	[0][0][0] =   1
}

ExpSum Stencil sld_add_left_exp
{
	[0][0][-1] =  1,
	[0][0][0] =   1
}
ExpSum Stencil sld_diff_right_exp
{
	[0][0][0] = -1,
	[0][0][1] =   1
}
ExpSum Stencil sld_add_right_exp
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

ExpSum Stencil sld_diff_right_right_exp
{
	[0][0][1] = -1,
	[0][0][2] =   1
}

ExpSum Stencil sld_diff_down_down_exp
{
	[0][-2][0] = -1,
	[0][-1][0] =   1
}
ExpSum Stencil sld_diff_down_exp
{
	[0][-1][0] = -1,
	[0][0 ][0] =   1
}
ExpSum Stencil sld_add_down_exp
{
	[0][-1][0] =   1,
	[0][0 ][0] =   1
}
ExpSum Stencil sld_diff_up_exp
{
	[0][0][0] = -1,
	[0][1][0] =   1
}
ExpSum Stencil sld_add_up_exp
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

ExpSum Stencil sld_diff_up_up_exp
{
	[0][1][0] = -1,
	[0][2][0] =   1
}

ExpSum Stencil sld_diff_back_back_exp
{
	[-2][0][0] = -1,
	[-1][0][0] =   1
}
ExpSum Stencil sld_diff_back_exp
{
	[-1][0][0] = -1,
	[0 ][0][0] =   1
}
ExpSum Stencil sld_add_back_exp
{
	[-1][0][0] =   1,
	[0 ][0][0] =   1
}
ExpSum Stencil sld_diff_front_exp
{
	[0][0][0] = -1,
	[1][0][0] =   1
}
ExpSum Stencil sld_add_front_exp
{
	[0][0][0] =   1,
	[1][0][0] =   1
}

ExpSum Stencil sld_diff_front_front_exp
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
ExpSum Stencil sld_get_left_exp
{
	[0][0][-1] = 1
}
ExpSum Stencil sld_get_right_exp
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
ExpSum Stencil sld_get_down_exp
{
	[0][-1][0] = 1
}
ExpSum Stencil sld_get_up_exp
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
ExpSum Stencil sld_get_back_exp
{
	[0][-1][0] = 1
}
ExpSum Stencil sld_get_front_exp
{
	[0][1][0] = 1
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
get_left_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_left_left_exp(f),sld_diff_left_exp(f))
	return minmod_alt(sld_diff_left_left(f),sld_diff_left(f))
}
get_x_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_left_exp(f),sld_diff_right_exp(f))
	return minmod_alt(sld_diff_left(f),sld_diff_right(f))
}
get_right_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_right_exp(f),sld_diff_right_right_exp(f))
	return minmod_alt(sld_diff_right(f),sld_diff_right_right(f))
}

get_down_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_down_down_exp(f),sld_diff_down_exp(f))
	return minmod_alt(sld_diff_down_down(f),sld_diff_down(f))
}
get_y_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_down_exp(f),sld_diff_up_exp(f))
	return minmod_alt(sld_diff_down(f),sld_diff_up(f))
}
get_up_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_up_exp(f),sld_diff_up_up_exp(f))
	return minmod_alt(sld_diff_up(f),sld_diff_up_up(f))
}

get_back_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_back_back_exp(f),sld_diff_back_exp(f))
	return minmod_alt(sld_diff_back_back(f),sld_diff_back(f))
}
get_z_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_back_exp(f),sld_diff_front_exp(f))
	return minmod_alt(sld_diff_back(f),sld_diff_front(f))
}
get_front_slope(Field f, bool ln_field)
{
	if(ln_field) return minmod_alt(sld_diff_front_exp(f),sld_diff_front_front_exp(f))
	return minmod_alt(sld_diff_front(f),sld_diff_front_front(f))
}
get_x_interface_values(Field f, bool ln_field)
{
	left_slope  = get_left_slope(f,ln_field)
	slope       = get_x_slope(f,ln_field)
	right_slope = get_right_slope(f,ln_field)
	return real4(
			ln_field ? sld_get_left_exp(f)+left_slope : sld_get_left(f) + left_slope,
			ln_field ? exp(f)-slope : f-slope,
			ln_field ? exp(f)+slope : f+slope,
			ln_field ? sld_get_right_exp(f)-right_slope : sld_get_right(f)-right_slope
		    )
}
get_y_interface_values(Field f, bool ln_field)
{
	down_slope  = get_down_slope(f,ln_field)
	slope       = get_y_slope(f,ln_field)
	up_slope    = get_up_slope(f,ln_field)
	return real4(
			ln_field ? sld_get_down_exp(f)+down_slope : sld_get_down(f)+down_slope,
			ln_field ? exp(f)-slope : f-slope,
			ln_field ? exp(f)+slope : f+slope,
			ln_field ? sld_get_up_exp(f)-up_slope : sld_get_up(f)-up_slope
		    )
}
get_z_interface_values(Field f, bool ln_field)
{
	back_slope  = get_back_slope(f,ln_field)
	slope       = get_y_slope(f,ln_field)
	front_slope = get_front_slope(f,ln_field)
	return real4(
			ln_field ? sld_get_back_exp(f)+back_slope : sld_get_back(f)+back_slope,
			ln_field ? exp(f)-slope : f-slope,
			ln_field ? exp(f)+slope : f+slope,
			ln_field ? sld_get_front_exp(f)-front_slope : sld_get_front(f)-front_slope
		    )
}
get_x_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	cs = get_x_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_x_interface_values(f,ln_field)

	left_diff = ln_field ? sld_diff_left_exp(f) : sld_diff_left(f)
	left_add  = ln_field ? sld_add_left_exp(f)      : sld_add_left(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff =  ln_field ? sld_diff_right_exp(f) : sld_diff_right(f)
	right_add  =  ln_field ? sld_add_right_exp(f)  : sld_add_right(f)
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
get_y_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	cs = get_y_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_y_interface_values(f,ln_field)

	left_diff = ln_field ? sld_diff_down_exp(f) : sld_diff_down(f)
	left_add  = ln_field ? sld_add_down_exp(f) : sld_add_down(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff =  ln_field ? sld_diff_up_exp(f) : sld_diff_up(f)
	right_add  =  ln_field ? sld_add_up_exp(f) : sld_add_up(f)
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

get_z_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	cs = get_z_interpolated_characteristic_speeds(characteristic_speed)
	interface_values = get_z_interface_values(f,ln_field)

	left_diff = ln_field ? sld_diff_back_exp(f) : sld_diff_back(f)
	left_add  = ln_field ? sld_add_back_exp(f)  : sld_add_back(f)
	left_slope_ratio = 0.0
	left_interface_diff = interface_values.y - interface_values.x
	if((left_interface_diff)*left_diff > 0.0)
	{
	    if(abs(left_add/left_diff) > fdiff_limit) left_diff = sign(left_add,left_diff)/fdiff_limit
	    left_slope_ratio = (left_interface_diff)left_diff
	}
	left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	left_flux = 0.5*cs.x*left_Q*(left_interface_diff)

	right_diff =  ln_field ? sld_diff_front_exp(f) : sld_diff_front(f)
	right_add  =  ln_field ? sld_add_front_exp(f)  : sld_add_front(f)
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

get_x_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}
get_y_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}
get_z_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}

//TP: works only for equidistant cartesian at the moment
get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	x_fluxes = get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
	y_fluxes = get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
	z_fluxes = get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
	div = 
		  (x_fluxes.y - x_fluxes.x)/AC_inv_ds.x
		+ (y_fluxes.y - y_fluxes.x)/AC_inv_ds.y
		+ (z_fluxes.y - z_fluxes.x)/AC_inv_ds.z
	return div
}

get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_slope_limited_divergence(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}
get_slope_limited_all(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, Field lnrho)
{
	x_fluxes = get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	y_fluxes = get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	z_fluxes = get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	heat = 0.0
		density_m1 = sld_get_left_exp(lnrho)
		density = exp(lnrho)
		density_p1 = sld_get_right_exp(lnrho)

		f_m1 = sld_get_left(f)
		f    = exp(lnrho)
		f_p1 = sld_get_right(f)
		heat += 0.5*(
				  (x_fluxes.x*(density*f-density_m1*f_m1)/AC_inv_ds.x)
				+ (x_fluxes.y*(density_p1*f_p1-density*f)/AC_inv_ds.x)
			    )

		density_m1 = sld_get_down_exp(lnrho)
		density = exp(lnrho)
		density_p1 = sld_get_up_exp(lnrho)

		f_m1 = sld_get_down(f)
		f    = exp(lnrho)
		f_p1 = sld_get_up(f)
		heat += 0.5*(
				  (x_fluxes.x*(density*f-density_m1*f_m1)/AC_inv_ds.x)
				+ (x_fluxes.y*(density_p1*f_p1-density*f)/AC_inv_ds.x)
			    )

		density_m1 = sld_get_back_exp(lnrho)
		density = exp(lnrho)
		density_p1 = sld_get_front_exp(lnrho)

		f_m1 = sld_get_back(f)
		f    = exp(lnrho)
		f_p1 = sld_get_front(f)
		heat += 0.5*(
				  (x_fluxes.x*(density*f-density_m1*f_m1)/AC_inv_ds.x)
				+ (x_fluxes.y*(density_p1*f_p1-density*f)/AC_inv_ds.x)
			    )

	div =
		  (x_fluxes.y - x_fluxes.x)/AC_inv_ds.x
		+ (y_fluxes.y - y_fluxes.x)/AC_inv_ds.y
		+ (z_fluxes.y - z_fluxes.x)/AC_inv_ds.z
	return 
		real5(
				div,
				0.5*(x_fluxes.x + x_fluxes.y),
				0.5*(y_fluxes.x + y_fluxes.y),
				0.5*(z_fluxes.x + z_fluxes.y),
				heat
		     )

}


calculate_characteristic_speed(real w_uu, Field3 uu, real w_sound, real sound_speed, real w_alfven, real3 bb, real lnrho, real mu) 
{ 
	inv_rho = exp(-lnrho)
	inv_mu  = 1.0/mu
	alfven_speed = sqrt(dot(bb,bb)*inv_rho*inv_mu)
	return w_uu*norm(uu) + w_sound*sound_speed + w_alfven*alfven_speed
}

calculate_characteristic_speed(real w_uu, Field3 uu, real w_sound, real sound_speed) 
{ 
	return w_uu*norm(uu) + w_sound*sound_speed
}
