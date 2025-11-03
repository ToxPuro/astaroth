Stencil interpolate_middle_left
{
	[0][0][-1] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_right
{
	[0][0][1] = 0.5,
	[0][0][0] = 0.5
}
Stencil interpolate_middle_down
{
	[0][-1][0] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_up
{
	[0][1][0] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_back
{
	[-1][0][0] = 0.5,
	[0][0][0] = 0.5
}
Stencil interpolate_middle_front
{
	[1][0][0] = 0.5,
	[0][0][0] = 0.5
}

/*
 * Meant to be launched on the coarse grid
 */
restrict_full_weighting(Field fine_residual, Field coarse_residual)
{
	i = 2*vertexIdx.x
	j = 2*vertexIdx.y
	k = 2*vertexIdx.z
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
	write(coarse_residual,res/64.0);
}

/*
 * Meant to be launched on the fine grid
 */
trilinear_prolongation(Field coarse_residual)
{
	const bool I_even = (vertexIdx.x % 2 == 0)
	const bool J_even = (vertexIdx.y % 2 == 0)
	const bool K_even = (vertexIdx.z % 2 == 0)

	const int3 coarse_vertexIdx = vertexIdx / 2;

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
