/**
 * This file implements Slope-limited diffusion (SLD) which is a technique which
 * adds enough diffusion to resolve steep slopes while trying to keep the added 
 * diffusion to a minimum.
 * See https://indico.fysik.su.se/event/6870/sessions/1096/attachments/4557/5291/notes.pdf
 * for a good reference.
 * This explanation is not from an expert so take it with a grain of salt, but SLD
 * measures two jumps: one at a cell-faces and one over a cell.
 * Unresolved slopes correspond to jumps where the jump over a cell-face is much larger than a jump over the cell.
 * When such jumps are noticed extra diffusion is added to flatten then.
 */

#include "$AC_HOME/acc-runtime/stdlib/general_grid/vars.h"
struct sld_interface_values
{
	real left_left;
	real left;
	real right;
	real right_right;
}
struct sld_flux
{
	real left;
	real right;
}

struct sld_fluxes
{
	sld_flux x;
	sld_flux y;
	sld_flux z;
} 
#if STENCIL_ORDER == 2
get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	suppress_unused_warning(f)
	suppress_unused_warning(characteristic_speed)
	suppress_unused_warning(fdiff_limit)
	suppress_unused_warning(h_slope_limited)
	suppress_unused_warning(nlf)
	suppress_unused_warning(ln_field)
	fatal_error_message(true,"SLD requires STENCIL_ORDER >= 4")
	return 0.0
}
get_slope_limited_divergence_and_average_fluxes(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	suppress_unused_warning(f)
	suppress_unused_warning(characteristic_speed)
	suppress_unused_warning(fdiff_limit)
	suppress_unused_warning(h_slope_limited)
	suppress_unused_warning(nlf)
	fatal_error_message(true,"SLD requires STENCIL_ORDER >= 4")
	real4 res
	return res
}
get_slope_limited_divergence_and_heat(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, Field lnrho)
{
	suppress_unused_warning(f)
	suppress_unused_warning(characteristic_speed)
	suppress_unused_warning(fdiff_limit)
	suppress_unused_warning(h_slope_limited)
	suppress_unused_warning(nlf)
	suppress_unused_warning(lnrho)
	fatal_error_message(true,"SLD requires STENCIL_ORDER >= 4")
	real2 res
	return res
}
get_slope_limited_all(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, Field lnrho)
{
	suppress_unused_warning(f)
	suppress_unused_warning(characteristic_speed)
	suppress_unused_warning(fdiff_limit)
	suppress_unused_warning(h_slope_limited)
	suppress_unused_warning(nlf)
	suppress_unused_warning(lnrho)
	fatal_error_message(true,"SLD requires STENCIL_ORDER >= 4")
	real5 res
	return res
}
#else
/**
sld_read(Field f, int x_offset, int y_offset, int z_offset)
{
	return f[vertexIdx.x+x_offset][vertexIdx.y+y_offset][vertexIdx.z+z_offset]
}

sld_diff_left_left(Field f)
{
	return sld_read(f,-1,0,0)-sld_read(f,-2,0,0)
}
sld_diff_left(Field f)
{
	return sld_read(f,+0,0,0)-sld_read(f,-1,0,0)
}
sld_add_left(Field f)
{
	return sld_read(f,+0,0,0)+sld_read(f,-1,0,0)
}
sld_diff_right(Field f)
{
	return sld_read(f,+1,0,0)-sld_read(f,-0,0,0)
}
sld_diff_right_right(Field f)
{
	return sld_read(f,+2,0,0)-sld_read(f,+1,0,0)
}
sld_add_right(Field f)
{
	return sld_read(f,+1,0,0)+sld_read(f,-0,0,0)
}

//exp_skip(real x) {return exp(x)}

sld_diff_left_left_exp(Field f)
{
	return exp(sld_read(f,-1,0,0))-exp(sld_read(f,-2,0,0))
}
sld_diff_left_exp(Field f)
{
	return exp(sld_read(f,+0,0,0))-exp(sld_read(f,-1,0,0))
}
sld_add_left_exp(Field f)
{
	return exp(sld_read(f,+0,0,0))+exp(sld_read(f,-1,0,0))
}
sld_diff_right_exp(Field f)
{
	return exp(sld_read(f,+1,0,0))-exp(sld_read(f,-0,0,0))
}
sld_add_right_exp(Field f)
{
	return exp(sld_read(f,+1,0,0))+exp(sld_read(f,-0,0,0))
}
sld_diff_right_right_exp(Field f)
{
	return exp(sld_read(f,+2,0,0))-exp(sld_read(f,+1,0,0))
}

sld_diff_down_down(Field f)
{
	return sld_read(f,0,-1,0)-sld_read(f,0,-2,0)
}
sld_diff_down(Field f)
{
	return sld_read(f,0,+0,0)-sld_read(f,0,-1,0)
}
sld_add_down(Field f)
{
	return sld_read(f,0,+0,0)+sld_read(f,0,-1,0)
}
sld_diff_up(Field f)
{
	return sld_read(f,0,+1,0)-sld_read(f,0,-0,0)
}
sld_diff_up_up(Field f)
{
	return sld_read(f,0,+2,0)-sld_read(f,0,+1,0)
}
sld_add_up(Field f)
{
	return sld_read(f,0,+1,0)+sld_read(f,0,-0,0)
}

sld_diff_down_down_exp(Field f)
{
	return exp(sld_read(f,0,-1,0))-exp(sld_read(f,0,-2,0))
}
sld_diff_down_exp(Field f)
{
	return exp(sld_read(f,0,+0,0))-exp(sld_read(f,0,-1,0))
}
sld_add_down_exp(Field f)
{
	return exp(sld_read(f,0,+0,0))+exp(sld_read(f,0,-1,0))
}
sld_diff_up_exp(Field f)
{
	return exp(sld_read(f,0,+1,0))-exp(sld_read(f,0,-0,0))
}
sld_diff_up_up_exp(Field f)
{
	return exp(sld_read(f,0,+2,0))-exp(sld_read(f,0,+1,0))
}
sld_add_up_exp(Field f)
{
	return exp(sld_read(f,0,+1,0))+exp(sld_read(f,0,-0,0))
}

sld_diff_back_back(Field f)
{
	return sld_read(f,0,0,-1)-sld_read(f,0,0,-2)
}
sld_diff_back(Field f)
{
	return sld_read(f,0,0,+0)-sld_read(f,0,0,-1)
}
sld_add_back(Field f)
{
	return sld_read(f,0,0,+0)+sld_read(f,0,0,-1)
}
sld_diff_front(Field f)
{
	return sld_read(f,0,0,+1)-sld_read(f,0,0,-0)
}
sld_diff_front_front(Field f)
{
	return sld_read(f,0,0,+2)-sld_read(f,0,0,+1)
}
sld_add_front(Field f)
{
	return sld_read(f,0,0,+1)+sld_read(f,0,0,-0)
}

sld_diff_back_back_exp(Field f)
{
	return exp(sld_read(f,0,0,-1))-exp(sld_read(f,0,0,-2))
}
sld_diff_back_exp(Field f)
{
	return exp(sld_read(f,0,0,+0))-exp(sld_read(f,0,0,-1))
}
sld_add_back_exp(Field f)
{
	return exp(sld_read(f,0,0,+0))+exp(sld_read(f,0,0,-1))
}
sld_diff_front_exp(Field f)
{
	return exp(sld_read(f,0,0,+1))-exp(sld_read(f,0,0,-0))
}
sld_diff_front_front_exp(Field f)
{
	return exp(sld_read(f,0,0,+2))-exp(sld_read(f,0,0,+1))
}
sld_add_front_exp(Field f)
{
	return exp(sld_read(f,0,0,+1))+exp(sld_read(f,0,0,-0))
}

sld_get_left(Field f)
{
	return sld_read(f,-1,0,0)
}

sld_get_right(Field f)
{
	return sld_read(f,+1,0,0)
}


sld_get_left_exp(Field f)
{
	return exp(sld_read(f,-1,0,0))
}

sld_get_right_exp(Field f)
{
	return exp(sld_read(f,+1,0,0))
}

sld_get_down(Field f)
{
	return sld_read(f,0,-1,0)
}

sld_get_up(Field f)
{
	return sld_read(f,0,+1,0)
}


sld_get_down_exp(Field f)
{
	return exp(sld_read(f,0,-1,0))
}

sld_get_up_exp(Field f)
{
	return exp(sld_read(f,0,+1,0))
}

sld_get_back(Field f)
{
	return sld_read(f,0,0,-1)
}

sld_get_front(Field f)
{
	return sld_read(f,0,0,+1)
}


sld_get_back_exp(Field f)
{
	return exp(sld_read(f,0,0,-1))
}

sld_get_front_exp(Field f)
{
	return exp(sld_read(f,0,0,+1))
}
**/


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
ExpSum Stencil sld_get_left_left_exp
{
	[0][0][-2] = 1
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
	[-1][0][0] = 1
}
ExpSum Stencil sld_get_front_exp
{
	[1][0][0] = 1
}

get_x_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return sld_flux(
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x-1][vertexIdx.y][vertexIdx.z]),
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x+1][vertexIdx.y][vertexIdx.z])
		    )
}
get_y_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return sld_flux(
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x][vertexIdx.y-1][vertexIdx.z]),
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x][vertexIdx.y+1][vertexIdx.z])
		    )
}
get_z_interpolated_characteristic_speeds(Field characteristic_speed)
{
	return sld_flux(
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z-1]),
			0.5*(characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z]
			+characteristic_speed[vertexIdx.x][vertexIdx.y][vertexIdx.z+1])
		    )
}
get_left_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_left_left_exp(f),sld_diff_left_exp(f))
	return minmod_alt(sld_diff_left_left(f),sld_diff_left(f))
}
get_x_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_left_exp(f),sld_diff_right_exp(f))
	return minmod_alt(sld_diff_left(f),sld_diff_right(f))
}
get_right_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_right_exp(f),sld_diff_right_right_exp(f))
	return minmod_alt(sld_diff_right(f),sld_diff_right_right(f))
}
get_down_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_down_down_exp(f),sld_diff_down_exp(f))
	return minmod_alt(sld_diff_down_down(f),sld_diff_down(f))
}
get_y_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_down_exp(f),sld_diff_up_exp(f))
	return minmod_alt(sld_diff_down(f),sld_diff_up(f))
}
get_up_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_up_exp(f),sld_diff_up_up_exp(f))
	return minmod_alt(sld_diff_up(f),sld_diff_up_up(f))
}
get_back_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_back_back_exp(f),sld_diff_back_exp(f))
	return minmod_alt(sld_diff_back_back(f),sld_diff_back(f))
}
get_z_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_back_exp(f),sld_diff_front_exp(f))
	return minmod_alt(sld_diff_back(f),sld_diff_front(f))
}
get_front_slope(Field f, bool ln_field)
{
	if (ln_field) return minmod_alt(sld_diff_front_exp(f),sld_diff_front_front_exp(f))
	return minmod_alt(sld_diff_front(f),sld_diff_front_front(f))
}
get_x_interface_values(Field f, bool ln_field)
{
	left_slope  = get_left_slope(f,ln_field)
	slope       = get_x_slope(f,ln_field)
	right_slope = get_right_slope(f,ln_field)
	return sld_interface_values(

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
	return sld_interface_values(

			ln_field ? sld_get_down_exp(f)+down_slope : sld_get_down(f)+down_slope,
			ln_field ? exp(f)-slope : f-slope,
			ln_field ? exp(f)+slope : f+slope,
			ln_field ? sld_get_up_exp(f)-up_slope : sld_get_up(f)-up_slope
		    )
}
get_z_interface_values(Field f, bool ln_field)
{
	back_slope  = get_back_slope(f,ln_field)
	slope       = get_z_slope(f,ln_field)
	front_slope = get_front_slope(f,ln_field)
	return sld_interface_values(
			ln_field ? sld_get_back_exp(f)+back_slope : sld_get_back(f)+back_slope,
			ln_field ? exp(f)-slope : f-slope,
			ln_field ? exp(f)+slope : f+slope,
			ln_field ? sld_get_front_exp(f)-front_slope : sld_get_front(f)-front_slope
		    )
}
get_x_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	if(!AC_dimension_inactive.x)
	{
	  cs = get_x_interpolated_characteristic_speeds(characteristic_speed)
	  interface_values = get_x_interface_values(f,ln_field)

	  left_diff = ln_field ? sld_diff_left_exp(f) : sld_diff_left(f)
	  left_add  = ln_field ? sld_add_left_exp(f)      : sld_add_left(f)
	  left_interface_diff = interface_values.left - interface_values.left_left

	  int tmp = (((left_interface_diff)*left_diff) > 0.0)
	  left_slope_ratio = tmp*(left_interface_diff)/left_diff
	  tmp = tmp*(abs(left_add/left_diff) > fdiff_limit)
	  left_diff = (1-tmp)*left_diff + tmp*sign(left_add,left_diff)/fdiff_limit
	  
	  left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	  left_flux = 0.5*cs.left*left_Q*(left_interface_diff)


	  right_diff =  ln_field ? sld_diff_right_exp(f) : sld_diff_right(f)
	  right_add  =  ln_field ? sld_add_right_exp(f)  : sld_add_right(f)
	  right_interface_diff = interface_values.right_right - interface_values.right

	  tmp = (((right_interface_diff)*right_diff) > 0.0)
	  right_slope_ratio = tmp*(right_interface_diff)/right_diff
	  tmp = tmp*(abs(right_add/right_diff) > fdiff_limit)
	  right_diff = (1-tmp)*right_diff + tmp*sign(right_add,right_diff)/fdiff_limit

	  right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	  right_flux = 0.5*cs.right*right_Q*(right_interface_diff)
	  return sld_flux(left_flux,right_flux)
	}
	return sld_flux(0.0,0.0)
}

get_y_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	if(!AC_dimension_inactive.y)
	{
	  cs = get_y_interpolated_characteristic_speeds(characteristic_speed)
	  interface_values = get_y_interface_values(f,ln_field)

	  left_diff = ln_field ? sld_diff_down_exp(f) : sld_diff_down(f)
	  left_add  = ln_field ? sld_add_down_exp(f) : sld_add_down(f)
	  left_interface_diff = interface_values.left - interface_values.left_left

	  int tmp = (((left_interface_diff)*left_diff) > 0.0)
	  left_slope_ratio = tmp*(left_interface_diff)/left_diff
	  tmp = tmp*(abs(left_add/left_diff) > fdiff_limit)
	  left_diff = (1-tmp)*left_diff + tmp*sign(left_add,left_diff)/fdiff_limit

	  left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	  left_flux = 0.5*cs.left*left_Q*(left_interface_diff)

	  right_diff =  ln_field ? sld_diff_up_exp(f) : sld_diff_up(f)
	  right_add  =  ln_field ? sld_add_up_exp(f) : sld_add_up(f)
	  right_interface_diff = interface_values.right_right - interface_values.right

	  tmp = (((right_interface_diff)*right_diff) > 0.0)
	  right_slope_ratio = tmp*(right_interface_diff)/right_diff
	  tmp = tmp*(abs(right_add/right_diff) > fdiff_limit)
	  right_diff = (1-tmp)*right_diff + tmp*sign(right_add,right_diff)/fdiff_limit

	  right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	  right_flux = 0.5*cs.right*right_Q*(right_interface_diff)
	  return sld_flux(left_flux,right_flux)
	}
	return sld_flux(0.0,0.0)
}

get_z_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	if(!AC_dimension_inactive.z)
	{
	  cs = get_z_interpolated_characteristic_speeds(characteristic_speed)
	  interface_values = get_z_interface_values(f,ln_field)

	  left_diff = ln_field ? sld_diff_back_exp(f) : sld_diff_back(f)
	  left_add  = ln_field ? sld_add_back_exp(f)  : sld_add_back(f)
	  left_interface_diff = interface_values.left - interface_values.left_left

	  int tmp = (((left_interface_diff)*left_diff) > 0.0)
	  left_slope_ratio = tmp*(left_interface_diff)/left_diff
	  tmp = tmp*(abs(left_add/left_diff) > fdiff_limit)
	  left_diff = (1-tmp)*left_diff + tmp*sign(left_add,left_diff)/fdiff_limit

	  left_Q = pow(min(1.0,h_slope_limited*left_slope_ratio),nlf)
	  left_flux = 0.5*cs.left*left_Q*(left_interface_diff)

	  right_diff =  ln_field ? sld_diff_front_exp(f) : sld_diff_front(f)
	  right_add  =  ln_field ? sld_add_front_exp(f)  : sld_add_front(f)
	  right_interface_diff = interface_values.right_right - interface_values.right

	  tmp = (((right_interface_diff)*right_diff) > 0.0)
	  right_slope_ratio = tmp*(right_interface_diff)/right_diff
	  tmp = tmp*(abs(right_add/right_diff) > fdiff_limit)
	  right_diff = (1-tmp)*right_diff + tmp*sign(right_add,right_diff)/fdiff_limit

	  right_Q = pow(min(1.0,h_slope_limited*right_slope_ratio),nlf)
	  right_flux = 0.5*cs.right*right_Q*(right_interface_diff)
	  return sld_flux(left_flux,right_flux)
	}
	return sld_flux(0.0,0.0)
}

get_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	return sld_fluxes(
			  get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field),
			  get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field),
			  get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
		         )
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

get_fluxes(Field f,Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return sld_fluxes(
			  get_x_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false),
			  get_y_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false),
			  get_z_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
		         )
}

get_slope_limited_divergence(sld_fluxes fluxes)
{
	real divx
	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		divx = (AC_x12[vertexIdx.x]*AC_x12[vertexIdx.x]*fluxes.x.right - AC_x12[vertexIdx.x-1]*AC_x12[vertexIdx.x-1]*fluxes.x.left)*AC_INV_R*AC_INV_R
	}
	else if (AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		divx = (AC_x12[vertexIdx.x]*fluxes.x.right - AC_x12[vertexIdx.x-1]*fluxes.x.left)*AC_INV_R
	}
	else
	{
		divx = fluxes.x.right - fluxes.x.left
	}

        if(!AC_nonequidistant_grid.x)
	{
          divx *= AC_inv_ds.x
	}
	else
	{
	  dx12 = AC_x12[vertexIdx.x] - AC_x12[vertexIdx.x-1]
	  divx /= dx12
	}

	real divy
	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		divy = (AC_sinth12[vertexIdx.y]*fluxes.y.right - AC_sinth12[vertexIdx.y-1]*fluxes.y.left)
		       *AC_INV_SIN_THETA
	} else
	{
		divy = fluxes.y.right - fluxes.y.left
	}
	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES || AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		divy *= AC_INV_R
	}
        if(!AC_nonequidistant_grid.y)
	{
          divy *= AC_inv_ds.y
	}
	else
	{
	  dy12 = AC_y12[vertexIdx.y] - AC_y12[vertexIdx.y-1]
	  divy /= dy12
	}

	divz = (fluxes.z.right - fluxes.z.left)    // z contribution
	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES) divz *= AC_INV_R*AC_INV_SIN_THETA

        if(!AC_nonequidistant_grid.z)
	{
          divz *= AC_inv_ds.z
	}
	else
	{
	  dz12 = AC_z12[vertexIdx.z] - AC_z12[vertexIdx.z-1]
	  divz /= dz12
	}

	return divx + divy + divz
}

get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	fluxes = get_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
	return get_slope_limited_divergence(fluxes)
}

get_slope_limited_divergence(Field3 f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, bool ln_field)
{
	return
		real3(
			get_slope_limited_divergence(f.x,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field),
			get_slope_limited_divergence(f.y,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field),
			get_slope_limited_divergence(f.z,characteristic_speed,fdiff_limit,h_slope_limited,nlf,ln_field)
		     )
}

get_slope_limited_divergence_and_average_fluxes(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	fluxes = get_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	div = get_slope_limited_divergence(fluxes)
	return 
		real4(
				div,
				0.5*(fluxes.x.left + fluxes.x.right),
				0.5*(fluxes.y.left + fluxes.y.right),
				0.5*(fluxes.z.left + fluxes.z.right)
		     )

}

get_slope_limited_heating(fluxes, Field f, Field lnrho)
{
	//TP: copy-paste TODO: refactor to a function
	density = exp(lnrho)

	density_m1 = sld_get_left_exp(lnrho)
	density_p1 = sld_get_right_exp(lnrho)

	f_m1 = sld_get_left(f)
	f_p1 = sld_get_right(f)
	heat_x = 0.5*(
			  fluxes.x.left*(density*f-density_m1*f_m1)*AC_INV_MAPPING_FUNC_DER_X
			+ fluxes.x.right*(density_p1*f_p1-density*f)*AC_inv_mapping_func_derivative_x[vertexIdx.x+1]
		    )

	density_m1 = sld_get_down_exp(lnrho)
	density_p1 = sld_get_up_exp(lnrho)

	f_m1 = sld_get_down(f)
	f_p1 = sld_get_up(f)
	heat_y = 0.5*(
				  fluxes.y.left*(density*f-density_m1*f_m1)*AC_INV_MAPPING_FUNC_DER_Y
				+ fluxes.y.right*(density_p1*f_p1-density*f)*AC_inv_mapping_func_derivative_y[vertexIdx.y+1]
			    )

	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		heat_y *= AC_INV_R
	}

	density_m1 = sld_get_back_exp(lnrho)
	density_p1 = sld_get_front_exp(lnrho)

	f_m1 = sld_get_back(f)
	f_p1 = sld_get_front(f)
	heat_z = 0.5*(
			       fluxes.z.left*(density*f-density_m1*f_m1)*AC_INV_MAPPING_FUNC_DER_Z
			     + fluxes.z.right*(density_p1*f_p1-density*f)*AC_inv_mapping_func_derivative_z[vertexIdx.z+1]
			    )
	if (AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		heat_z *= AC_INV_R*AC_INV_SIN_THETA
	}
	return heat_x + heat_y + heat_z
}

get_slope_limited_divergence_and_heat(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, Field lnrho)
{
	fluxes = get_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	div = get_slope_limited_divergence(fluxes)
	heat = get_slope_limited_heating(fluxes,f,lnrho)
	return 
		real2(
				div,
				heat
		     )

}
get_slope_limited_all(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf, Field lnrho)
{
	fluxes = get_fluxes(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf)
	div = get_slope_limited_divergence(fluxes)
	heat = get_slope_limited_heating(fluxes,f,lnrho)
	return 
		real5(
				div,
				0.5*(fluxes.x.left + fluxes.x.right),
				0.5*(fluxes.y.left + fluxes.y.right),
				0.5*(fluxes.z.left + fluxes.z.right),
				heat
		     )
}
#endif

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

get_slope_limited_divergence(Field f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_slope_limited_divergence(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}

get_slope_limited_divergence(Field3 f, Field characteristic_speed, real fdiff_limit, real h_slope_limited, real nlf)
{
	return get_slope_limited_divergence(f,characteristic_speed,fdiff_limit,h_slope_limited,nlf,false)
}

