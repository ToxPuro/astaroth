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
