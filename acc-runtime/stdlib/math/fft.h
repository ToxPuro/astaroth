#ifndef AC_MATH_FFT_H
#define AC_MATH_FFT_H

get_wavevector_x()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.x*((global_idx.x <= AC_nlocal.x/2) ? global_idx.x : global_idx.x - AC_nlocal.x)
}
get_wavevector_y()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.y*((global_idx.y <= AC_nlocal.y/2) ? global_idx.y : global_idx.y - AC_nlocal.y)
}
get_wavevector_z()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.z*((global_idx.z <= AC_nlocal.z/2) ? global_idx.z : global_idx.z - AC_nlocal.z)
}
get_wavevector()
{
	return real3
	(
	 	get_wavevector_x(),
	 	get_wavevector_y(),
	 	get_wavevector_z()
	)
}

poisson_fft_solve(ComplexField dst, ComplexField src)
{
	const real3 k = get_wavevector()
	const real k2 = dot(k,k)
	//We do this in the inplace manner to enable more flexible usage
	dst[vertexIdx.x][vertexIdx.y][vertexIdx.z] = 
				(k2 == 0.0) ? complex(0.0,0.0) :
				-src[vertexIdx.x][vertexIdx.y][vertexIdx.z]/k2
}
#endif
