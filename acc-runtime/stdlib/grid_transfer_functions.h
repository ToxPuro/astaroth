#ifndef AC_GRID_TRANSFER_FUNCTIONS_H
#define AC_GRID_TRANSFER_FUNCTIONS_H

/*
 * Meant to be launched on the coarse grid
 */
restrict_full_weighting_1d(Field fine_residual)
{
		i = 2*vertexIdx.x + 1 - NGHOST 
		j = vertexIdx.y 
		k = vertexIdx.z 
		res = 0.0
		int di
		for di in -1:2
		{
			//Facet class 3 has the weight 1
			weight = 1
			facet_class = abs(di)
			if(facet_class == 0)
			{
				weight = 2.0
			}
			else if(facet_class == 1)
			{
				weight = 1.0
			}
			res += weight*fine_residual[i+di][j][k]
		}
		res /= 4.0
		return res
}

/*
 * Meant to be launched on the coarse grid
 */
restrict_full_weighting_3d_even(Field fine_residual)
{
	fine_vertexIdx = localCompdomainVertexIdx*2 + AC_nmin
	interpolation_weights = [1,3,3,1]
	res = 0.0
	for di in 0:3
	{
		for dj in 0:3
		{
			for dk in 0:3
			{
				weight = interpolation_weights[di]*interpolation_weights[dj]*interpolation_weights[dk]
				res    += weight*fine_residual[fine_vertexIdx.x+di][fine_vertexIdx.y+dj][fine_vertexIdx.z+dk]
			}
		}
	}
	res /= 512.0
	return res
}

/*
 * Meant to be launched on the coarse grid
 */
restrict_full_weighting_3d_odd(Field fine_residual)
{
	i = 2*vertexIdx.x + 1 - NGHOST 
	j = 2*vertexIdx.y + 1 - NGHOST 
	k = 2*vertexIdx.z + 1 - NGHOST 
	res = 0.0
	int di
	int dj
	int dk
	for di in -1:2
	{
		for dj in -1:2
		{
			for dk in -1:2
			{
				//Facet class 3 has the weight 1
				weight = 1
				facet_class = abs(di) + abs(dj) + abs(dk)
				if(facet_class == 0)
				{
					weight = 8
				}
				else if(facet_class == 1)
				{
					weight = 4
				}
				else if(facet_class == 2)
				{
					weight = 2
				}
				res += weight*fine_residual[i+di][j+dj][k+dk]
			}
		}
	}
	res /= 64.0
	return res
}

/*
 * Meant to be launched on the coarse grid
 */
restrict_full_weighting(Field fine_residual, int3 fine_dimensions)
{
	if(AC_dimension_inactive == (bool3){false,true,true})
	{
		return restrict_full_weighting_1d(fine_residual)
	}
	//Assumes a cube
	else if(fine_dimensions.x % 2 == 0)
	{
		return restrict_full_weighting_3d_even(fine_residual)
	}
	else
	{
		return restrict_full_weighting_3d_odd(fine_residual)
	}
}

/*
 * Meant to be launched on the fine grid
 */
linear_prolongation(Field coarse_residual)
{
	const int3 shifted_index = vertexIdx - NGHOST + 1
	const bool I_even = (shifted_index.x % 2 == 0)
	const int3 coarse_vertexIdx = (shifted_index/ 2) + NGHOST - 1

	if(I_even) 
	{
		return coarse_residual[coarse_vertexIdx.x][vertexIdx.y][vertexIdx.z]
	}
	else
	{
		return 
			0.5*(  coarse_residual[coarse_vertexIdx.x][vertexIdx.y][vertexIdx.z]
			     + coarse_residual[coarse_vertexIdx.x+1][vertexIdx.y][vertexIdx.z]
			    )

	}
	return 0.0
}

/*
 * Meant to be launched on the fine grid
 */
trilinear_prolongation_even(Field coarse_residual)
{
	coarse_vertexIdx = localCompdomainVertexIdx/2 + AC_nmin
	real x_weights[2]
	real y_weights[2]
	real z_weights[2]
	if(localCompdomainVertexIdx.x % 2 == 0)
	{
		x_weights[0] = 3.0
		x_weights[1] = 1.0
	}
	else
	{
		x_weights[0] = 1.0
		x_weights[1] = 3.0
	}

	if(localCompdomainVertexIdx.y % 2 == 0)
	{
		y_weights[0] = 3.0
		y_weights[1] = 1.0
	}
	else
	{
		y_weights[0] = 1.0
		y_weights[1] = 3.0
	}

	if(localCompdomainVertexIdx.z % 2 == 0)
	{
		z_weights[0] = 3.0
		z_weights[1] = 1.0
	}
	else
	{
		z_weights[0] = 1.0
		z_weights[1] = 3.0
	}
	res = 0.0
	for di in -1:1
	{
		for dj in -1:1
		{
			for dk in -1:1
			{
				weight = x_weights[di+1]*y_weights[dj+1]*z_weights[dk+1]	
				res   += weight*coarse_residual[coarse_vertexIdx.x+di][coarse_vertexIdx.y+dj][coarse_vertexIdx.z+dk]
			}
		}
	}
	res /= 64.0
	return res
}
/*
 * Meant to be launched on the fine grid
 */
trilinear_prolongation_odd(Field coarse_residual)
{
	const int3 shifted_index = localCompdomainVertexIdx + 1
	const bool I_even = (shifted_index.x % 2 == 0)
	const bool J_even = (shifted_index.y % 2 == 0)
	const bool K_even = (shifted_index.z % 2 == 0)

	const int3 coarse_vertexIdx = (shifted_index/ 2) + NGHOST - 1

	if(I_even && J_even && K_even)
	{
		return coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
	}
	if(I_even && J_even && !K_even)
	{
		return 0.5*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
				)
	}
	if(I_even && !J_even && K_even)
	{
		return 0.5*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
				)
	}
	if(!I_even && J_even && K_even)
	{
		return 0.5*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z]
				)
	}
	if(I_even && !J_even && !K_even)
	{
		return 0.25*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z+1]
				)
	}
	if(!I_even && J_even && !K_even)
	{
		return 0.25*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
				)
	}
	if(!I_even && !J_even && K_even)
	{
		return 0.25*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
				)
	}
	if(!I_even && !J_even && !K_even)
	{
		return 0.125*(
								coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y+1][coarse_vertexIdx.z]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y][coarse_vertexIdx.z+1]
								+coarse_residual[coarse_vertexIdx.x][coarse_vertexIdx.y+1][coarse_vertexIdx.z+1]
								+coarse_residual[coarse_vertexIdx.x+1][coarse_vertexIdx.y+1][coarse_vertexIdx.z+1]
				)
	}
	return 0.0
}
/*
 * Meant to be launched on the fine grid
 */
trilinear_prolongation(Field coarse_residual, int3 global_mesh_dims)
{
	if(AC_dimension_inactive == (bool3){false,true,true})
	{
		return linear_prolongation(coarse_residual)
	}
	//Assumes a cube
	else if(global_mesh_dims.x % 2 == 0)
	{
		return trilinear_prolongation_even(coarse_residual)
	}
	else
	{
		return trilinear_prolongation_odd(coarse_residual)
	}
}
#endif
