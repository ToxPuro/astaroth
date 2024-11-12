inline get_normal()
{
           const int z = (vertexIdx.z >= AC_nlocal_max.z)
			-(vertexIdx.z < AC_nmin.z);
           const int y = (z == 0)*((vertexIdx.y >= AC_nlocal_max.y)
			-(vertexIdx.y < AC_nmin.y));
           const int x = (z == 0 && y == 0)*((vertexIdx.x >= AC_nlocal_max.x)
			-(vertexIdx.x < AC_nmin.x));
	   return (int3){x,y,z}
}
inline get_boundary(int3 normal)
{
	const int x =  normal.x == 1  ? AC_nlocal_max.x-1
		     : normal.x == -1 ? AC_nmin.x
		     : vertexIdx.x;
	const int y =  normal.y == 1  ? AC_nlocal_max.y-1
		     : normal.y == -1 ? AC_nmin.y
		     : vertexIdx.y;
	const int z =  normal.z == 1  ? AC_nlocal_max.z-1
		     : normal.z == -1 ? AC_nmin.z
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
utility Kernel BOUNDCOND_SYMMETRIC(Field f)
{
	ac_bc_sym(f,1)
}
utility Kernel BOUNDCOND_ANTISYMMETRIC(Field f)
{
	ac_bc_sym(f,-1)
}
utility Kernel BOUNDCOND_A2(Field f)
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
utility Kernel BOUNDCOND_CONST(Field f, real const_val)
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
utility Kernel BOUNDCOND_PRESCRIBED_DERIVATIVE(Field f, real prescribed_value)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	const real spacing = dot(normal,AC_ds)
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
utility Kernel BOUNDCOND_OUTFLOW(Field f)
{
	ac_flow_bc(f,1)
}
utility Kernel BOUNDCOND_INFLOW(Field f)
{
	ac_flow_bc(f,-1)
}
