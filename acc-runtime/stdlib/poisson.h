poisson_jacobi_update(Field density, Field potential)
{
	return (density-laplace_neighbours(potential))/laplace_central_coeff()
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
