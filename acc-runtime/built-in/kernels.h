utility Kernel AC_NULL_KERNEL(){}
utility Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_VTXBUF_HANDLES{
		write(Field(field), 0.0)
	}
}
utility Kernel BOUNDCOND_PERIODIC()
{
}


boundary_condition utility Kernel BOUNDCOND_PERIODIC_DEVICE(Field f)
{
    //TP: the boundary_condition qualifier implies that the computational subdomain is skipped
    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src = vertexIdx.x - AC_nmin.x;
    int j_src = vertexIdx.y - AC_nmin.y;
    int k_src = vertexIdx.z - AC_nmin.z;

    // Translate (s.t. the index is always positive)
    i_src += AC_nlocal.x;
    j_src += AC_nlocal.y;
    k_src += AC_nlocal.z;

    //Wrap by using mod operator (a+b)%b --> mod(a,b)
    i_src = (i_src + AC_nlocal.x) % AC_nlocal.x;
    j_src = (j_src + AC_nlocal.x) % AC_nlocal.y;
    k_src = (k_src + AC_nlocal.x) % AC_nlocal.z;

    // Map to mx, my, mz coordinates
    i_src += AC_nmin.x;
    j_src += AC_nmin.y;
    k_src += AC_nmin.z;
    f[vertexIdx.x][vertexIdx.y][vertexIdx.z] = f[i_src][j_src][k_src];
}

utility Kernel AC_VOLUME_COPY(const real[] src,Volume in_offset, Volume in_volume, 
			      real[] out,Volume out_offset,Volume out_volume)
{
    const Volume local_idx = (Volume){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    const Volume in_pos  = local_idx + in_offset;
    const Volume out_pos = local_idx + out_offset;
    const size_t in_idx = in_pos.x +               //
                          in_pos.y * in_volume.x + //
                          in_pos.z * in_volume.x * in_volume.y;
    const size_t out_idx = out_pos.x +                //
                           out_pos.y * out_volume.x + //
                           out_pos.z * out_volume.x * out_volume.y;
    out[out_idx] = src[in_idx];
}

utility Kernel AC_FLUSH_REAL(real[] dst, real val)
{
	dst[vertexIdx.x] = val
}
utility Kernel AC_FLUSH_FLOAT(float[] dst, float val)
{
	dst[vertexIdx.x] = val
}
utility Kernel AC_FLUSH_COMPLEX(complex[] dst, complex val)
{
	dst[vertexIdx.x] = val
}

utility Kernel AC_FLUSH_INT(int[] dst, int val)
{
	dst[vertexIdx.x] = val
}
utility Kernel AC_MULTIPLY_INPLACE(const real val, real[] array)
{
	array[vertexIdx.x] *= val
}

utility Kernel AC_COMPLEX_TO_REAL(const complex[] src, real[] dst)
{
	dst[vertexIdx.x] = src[vertexIdx.x].x
}

utility Kernel AC_REAL_TO_COMPLEX(const real[] src, complex[] dst)
{
	dst[vertexIdx.x].x = src[vertexIdx.x]
	dst[vertexIdx.x].y = 0.0
}

utility Kernel AC_PLANAR_TO_COMPLEX(const real[] real_src, const real[] imag_src, complex[] dst)
{
	dst[vertexIdx.x].x = real_src[vertexIdx.x]
	dst[vertexIdx.x].y = imag_src[vertexIdx.x]
}

utility Kernel AC_COMPLEX_TO_PLANAR(complex[] src, real[] real_dst, real[] imag_dst)
{
	real_dst[vertexIdx.x] = src[vertexIdx.x].x
	imag_dst[vertexIdx.x] = src[vertexIdx.x].y
}

utility Kernel AC_MULTIPLY_INPLACE_COMPLEX(const real val, complex[] dst)
{
	dst[vertexIdx.x].x *= val
	dst[vertexIdx.x].y *= val
}


