#ifndef AC_MATH_FFT_H
hostdefine AC_MATH_FFT_INCLUDED (1)
#define AC_MATH_FFT_H

get_wavevector_x()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.x*((global_idx.x <= AC_ngrid.x/2) ? global_idx.x : global_idx.x - AC_ngrid.x)
}
get_wavevector_y()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.y*((global_idx.y <= AC_ngrid.y/2) ? global_idx.y : global_idx.y - AC_ngrid.y)
}
get_wavevector_z()
{
	global_idx  = globalVertexIdx - AC_nmin
	return AC_frequency_spacing.z*((global_idx.z <= AC_ngrid.z/2) ? global_idx.z : global_idx.z - AC_ngrid.z)
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

poisson_fft_solve(Field real_dst, Field imag_dst, Field real_src, Field imag_src)
{
        const real3 k = get_wavevector()
        const real k2 = dot(k,k)
        res  = k2 == 0.0 ? complex(0.,0.) : -complex(real_src[vertexIdx.x][vertexIdx.y][vertexIdx.z], imag_src[vertexIdx.x][vertexIdx.y][vertexIdx.z])/k2
        write(real_dst,res.x)
        write(imag_dst,res.y)
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

split_diffusion_update(Field real_dst, Field imag_dst, Field real_src, Field imag_src, real dt, real diffusion_coeff)
{
        const real3 k = get_wavevector()
        const real k2 = dot(k,k)
	const real k2dt = dt*k2
	const real decay = exp(-diffusion_coeff*k2dt)
        res  = decay*complex(real_src[vertexIdx.x][vertexIdx.y][vertexIdx.z], imag_src[vertexIdx.x][vertexIdx.y][vertexIdx.z])
        write(real_dst,res.x)
        write(imag_dst,res.y)
}

Kernel split_diffusion_update_kernel(int real_dst, int imag_dst, int real_src, int imag_src, real dt, real diffusion_coeff)
{
	split_diffusion_update(Field(real_dst),Field(imag_dst),Field(real_src),Field(imag_src),dt,diffusion_coeff)
}
input int  AC_FFT_REAL_SRC
input int  AC_FFT_IMAG_SRC
input int  AC_FFT_REAL_DST
input int  AC_FFT_IMAG_DST
input real AC_FFT_SPLIT_DIFFUSION_UPDATE_DT
input real AC_FFT_SPLIT_DIFFUSION_UPDATE_DIFFUSION_COEFF
BoundConds ac_fft_bcs
{
	periodic(BOUNDARY_XYZ)
}

ComputeSteps AC_fft_split_diffusion_update_step(ac_fft_bcs)
{
	split_diffusion_update_kernel(AC_FFT_REAL_DST,AC_FFT_IMAG_DST,AC_FFT_REAL_SRC,AC_FFT_IMAG_SRC,
				      AC_FFT_SPLIT_DIFFUSION_UPDATE_DT, AC_FFT_SPLIT_DIFFUSION_UPDATE_DIFFUSION_COEFF)
}
#endif
