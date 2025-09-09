enum SOR_STEP
{
	SOR_RED,
	SOR_BLACK
}
poisson_jacobi_update(Field density, Field potential)
{
	return (density-laplace_neighbours(potential))/laplace_central_coeff()
}

poisson_jacobi_update_extended(Field density, Field potential)
{
	return (density-laplace_neighbours_extended(potential))/laplace_central_coeff_extended()
}

poisson_sor_red_black(int color, Field density, Field potential, real omega)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update(density,potential)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}

poisson_sor_red_black_extended(int color, Field density, Field potential, real omega)
{
	res = (1-omega)*potential+omega*poisson_jacobi_update_extended(density,potential)
	if((globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) %2 == color)
        {
		write(potential,res)
	}
	else
	{
		write(potential,potential)
	}
}
