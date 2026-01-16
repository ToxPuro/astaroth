/**
 * Copies a Field from the normal grid to the extended grid that has extended_halo around the original grid.
 * Meant to be launched on the extended grid
 * For values outside the original grid the padding value is used which can be unique at each grid point in the extended grid.
 */
copy_to_extended_grid(Field dst, Field src, int3 left_extended_halo, real padding_value)
{
	//Are we inside the subvolume of the extended grid that matches the original grid
	const bool inside_volume = vertexIdx.x >= left_extended_halo.x +NGHOST &&
	                           vertexIdx.y >= left_extended_halo.y +NGHOST &&
	                           vertexIdx.z >= left_extended_halo.z +NGHOST &&
	                           vertexIdx.x <  AC_nlocal_max.x + left_extended_halo.x &&
	                           vertexIdx.y <  AC_nlocal_max.y + left_extended_halo.y &&
	                           vertexIdx.z <  AC_nlocal_max.z + left_extended_halo.z
	//If we are shift the current indexes to work for src otherwise read some dummy values
        const int3 f_indexes = inside_volume ?
				(int3)
				{
					vertexIdx.x-left_extended_halo.x,
					vertexIdx.y-left_extended_halo.y,
					vertexIdx.z-left_extended_halo.z
				}
				:
				(int3)
				{
					NGHOST, NGHOST,NGHOST
				}
	//TP: The field is read with these weird indexes conditional reads based on vertex indexes do not play well with ComputeSteps
	f_val = src[f_indexes.x][f_indexes.y][f_indexes.z]
	if(inside_volume)
	{
		write(dst,f_val)
	}
	else
	{
		write(dst,padding_value)
	}
}

/**
 * Uses the default AC_left_extended_halo param
 */
copy_to_extended_grid(Field dst, Field src, real padding_value)
{
	copy_to_extended_grid(dst,src,AC_left_extended_halo,padding_value)
}

/**
 * Uses the default AC_left_extended_halo param and 0 as a padding value
 */
copy_to_extended_grid(Field dst, Field src)
{
	copy_to_extended_grid(dst,src,AC_left_extended_halo,0.0)
}

/**
 * Copies a Field from the extended grid (that has extended_halo around the original grid) to the normal grid.
 * Meant to be launched on the normal grid.
 */
copy_extended_to_grid(Field dst, Field src, int3 left_extended_halo)
{
	write(dst,src[vertexIdx.x+left_extended_halo.x][vertexIdx.y+left_extended_halo.y][vertexIdx.z+left_extended_halo.z])
}

/**
 * Uses the default AC_left_extended_halo param
 */
copy_extended_to_grid(Field dst, Field src)
{
	copy_extended_to_grid(dst,src,AC_left_extended_halo)
}
