#ifndef AC_POISSON_H
#define AC_POISSON_H

input real AC_SOR_omega = 1.0
run_const bool AC_compact_poisson = false;
#if STENCIL_ORDER == 2
run_const bool AC_poisson_radius_1  = true;
#else
run_const bool AC_poisson_radius_1  = AC_compact_poisson;
#endif
enum SOR_STEP
{
	SOR_RED,
	SOR_BLACK
}
poisson_jacobi_update(real b, Field x_prev, real laplace_sign)
{
	Ax = laplace_sign*laplace_neighbours(x_prev)
	coef = laplace_sign*laplace_central_coeff()
	return (b-Ax)/coef
}

poisson_jacobi_update(real b, potential)
{
	return poisson_jacobi_update(b,potential,1.0)
}

poisson_jacobi_update(real b, Field x_prev, real laplace_sign,real3 inv_spacings_2)
{
	Ax = laplace_sign*laplace_neighbours(x_prev,inv_spacings_2)
	coef = laplace_sign*laplace_central_coeff(inv_spacings_2)
	return (b-Ax)/coef
}

poisson_jacobi_update(real b, potential, real3 inv_spacings_2)
{
	return poisson_jacobi_update(b,potential,1.0,inv_spacings_2)
}


poisson_sor_red_black(int color, real density, Field potential, real omega, real laplace_sign)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update(density,potential,laplace_sign)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}
poisson_sor_red_black(int color, real density, Field potential, real omega)
{
	poisson_sor_red_black(color,density,potential,omega,1.0)
}

poisson_sor_red_black(int color, real density, Field potential, real omega, real laplace_sign, real3 inv_spacings)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update(density,potential,laplace_sign,inv_spacings)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}
poisson_sor_red_black(int color, real density, Field potential, real omega,real3 inv_spacings)
{
	return poisson_sor_red_black(color,density,potential,omega,1.0,inv_spacings)
}




#ifdef AC_GENERAL_DERIVS_H
poisson_jacobi_update_extended(real b , Field x_prev, real laplace_sign)
{
	Ax = laplace_sign*laplace_neighbours_extended(x_prev)
	coef = laplace_sign*laplace_central_coeff_extended()
	return (b-Ax)/coef
}

poisson_jacobi_update_extended(real b , Field x_prev)
{
	return poisson_jacobi_update_extended(b,x_prev,1.0)
}

poisson_jacobi_update_extended(real b , Field x_prev, real laplace_sign, real3 inv_spacings_2)
{
	Ax = laplace_sign*laplace_neighbours_extended(x_prev,inv_spacings_2)
	coef = laplace_sign*laplace_central_coeff_extended(inv_spacings_2)
	return (b-Ax)/coef
}

poisson_jacobi_update_extended(real b , Field x_prev, real3 inv_spacings)
{
	return poisson_jacobi_update_extended(b,x_prev,1.0,inv_spacings_2)
}

poisson_sor_red_black_extended(int color, real density, Field potential, real omega, real laplace_sign)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update_extended(density,potential,laplace_sign)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}
poisson_sor_red_black_extended(int color, real density, Field potential, real omega)
{
	poisson_sor_red_black_extended(color,density,potential,omega,1.0)
}

poisson_sor_red_black_extended(int color, real density, Field potential, real omega, real laplace_sign, real3 inv_spacings)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update_extended(density,potential,laplace_sign,inv_spacings)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}
poisson_sor_red_black_extended(int color, real density, Field potential, real omega, real3 inv_spacings)
{
	poisson_sor_red_black_extended(color,density,potential,omega,1.0,inv_spacings)
}
#endif

#ifdef AC_GENERAL_GRID_VARS_H
//For spherical grids
local_jacobi_spectral_radius_estimate()
{
	inv_dx = AC_INV_MAPPING_FUNC_DER_X
	inv_dy = AC_INV_MAPPING_FUNC_DER_Y
	inv_dz = AC_INV_MAPPING_FUNC_DER_Z

	inv_dx2 = inv_dx*inv_dx
	inv_dy2 = inv_dy*inv_dy
	inv_dz2 = inv_dz*inv_dz

	inv_r = AC_INV_R
	inv_r2 = inv_r*inv_r

	inv_sin_theta = AC_INV_SIN_THETA
	inv_sin_theta2 = inv_sin_theta*inv_sin_theta
	return (2.0/laplace_central_coeff())*
		(
		   cos(AC_REAL_PI/AC_ngrid.x)*inv_dx2
		  +cos(AC_REAL_PI/AC_ngrid.y)*inv_dy2*inv_r2
		  +cos(AC_REAL_PI/AC_ngrid.z)*inv_dz2*inv_r2*inv_sin_theta2
		)
}
initial_sor_omega()
{
	jacobi_radius = local_jacobi_spectral_radius_estimate()
	return (1.0)/(1.0 - 0.5*jacobi_radius)
}
update_sor_omega(Field omega)
{
	jacobi_radius = local_jacobi_spectral_radius_estimate()
	jacobi_radius2 = jacobi_radius*jacobi_radius
	return (1.0)/(1.0 - 0.25*jacobi_radius2*omega)
}
#endif

#endif
