utility Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_VTXBUF_HANDLES{
		write(Field(field), 0.0)
	}
}
utility Kernel AC_PERIODIC()
{
}

inline get_normal()
{
           const int z = (vertexIdx.z >= AC_nz_max)
			-(vertexIdx.z < AC_nz_min);
           const int y = (z == 0)*((vertexIdx.y >= AC_ny_max)
			-(vertexIdx.y < AC_ny_min));
           const int x = (z == 0 && y == 0)*((vertexIdx.x >= AC_nx_max)
			-(vertexIdx.x < AC_nx_min));
	   return (int3){x,y,z}
}
inline get_boundary(int3 normal)
{
	const int x =  normal.x == 1  ? AC_nx_max-1
		     : normal.x == -1 ? AC_nx_min
		     : vertexIdx.x;
	const int y =  normal.y == 1  ? AC_ny_max-1
		     : normal.y == -1 ? AC_ny_min
		     : vertexIdx.y;
	const int z =  normal.z == 1  ? AC_nz_max-1
		     : normal.z == -1 ? AC_nz_min
		     : vertexIdx.z;
	return (int3){x,y,z}
}

ac_bc_sym(Field f, int bc_sign)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	for i in 0:NGHOST
	{
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = bc_sign*f[domain.x][domain.y][domain.z];
	}
}
utility Kernel BOUNDCOND_SYMMETRIC_DSL(Field f)
{
	ac_bc_sym(f,1)
}
utility Kernel BOUNDCOND_ANTI_SYMMETRIC_DSL(Field f)
{
	ac_bc_sym(f,-1)
}
utility Kernel BOUNDCOND_A2_DSL(Field f)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	const real boundary_val = f[boundary.x][boundary.y][boundary.z]
	int3 domain = boundary
	int3 ghost  = boundary
	for i in 0:NGHOST
	{
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = 2*boundary_val -f[domain.x][domain.y][domain.z];
	}
}
utility Kernel BOUNDCOND_CONST_DSL(Field f, real const_val)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	for i in 0:NGHOST
	{
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = const_val;
	}
}
inline get_boundcond_spacing(int3 normal)
{
	return normal.x*AC_dsx + normal.y*AC_dsy + normal.z*AC_dsz;
}
utility Kernel BOUNDCOND_PRESCRIBED_DERIVATIVE_DSL(Field f, real prescribed_value)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	const real spacing = get_boundcond_spacing(normal)
	for i in 0:NGHOST
	{
		real distance = 2.0*(i+1)*spacing;
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = f[domain.x][domain.y][domain.z] + distance*prescribed_value;
	}
}
inline get_normal_direction(normal)
{
	return normal.x + normal.y + normal.z;
}
ac_flow_bc(Field f, int flow_direction)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	const real normal_direction = get_normal_direction(normal)
	const real boundary_value  = f[boundary.x][boundary.y][boundary.z]
	const real bc_sign = flow_direction*(boundary_value*normal_direction >= 0.0 ? 1.0 : -1.0)
	for i in 0:NGHOST
	{
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = bc_sign*f[domain.x][domain.y][domain.z]
	}
}
utility Kernel BOUNDCOND_OUTFLOW_DSL(Field f)
{
	ac_flow_bc(f,1)
}
utility Kernel BOUNDCOND_INFLOW_DSL(Field f)
{
	ac_flow_bc(f,-1)
}
