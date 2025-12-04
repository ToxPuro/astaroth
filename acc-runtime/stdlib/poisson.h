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
	poisson_jacobi_update(b,potential,1.0)
}

poisson_jacobi_update(real b, Field x_prev, real laplace_sign,real3 inv_spacings_2)
{
	Ax = laplace_sign*laplace_neighbours(x_prev,inv_spacings_2)
	coef = laplace_sign*laplace_central_coeff(inv_spacings_2)
	return (b-Ax)/coef
}

poisson_jacobi_update(real b, potential, real3 inv_spacings_2)
{
	poisson_jacobi_update(b,potential,1.0,inv_spacings_2)
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
	poisson_sor_red_black(color,density,potential,omega,1.0,inv_spacings)
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
	poisson_jacobi_update_extended(b,x_prev,1.0)
}

poisson_jacobi_update_extended(real b , Field x_prev, real laplace_sign, real3 inv_spacings_2)
{
	Ax = laplace_sign*laplace_neighbours_extended(x_prev,inv_spacings_2)
	coef = laplace_sign*laplace_central_coeff_extended(inv_spacings_2)
	return (b-Ax)/coef
}

poisson_jacobi_update_extended(real b , Field x_prev, real3 inv_spacings)
{
	poisson_jacobi_update_extended(b,x_prev,1.0,inv_spacings_2)
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
