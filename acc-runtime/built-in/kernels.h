utility Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_VTXBUF_HANDLES{
		write(Field(field), 0.0)
	}
}
utility Kernel AC_PERIODIC()
{
}

get_normal()
{
           const int x = (vertexIdx.x > AC_nx_max)
			-(vertexIdx.x < AC_nx_min);
           const int y = (vertexIdx.y > AC_ny_max)
			-(vertexIdx.y < AC_ny_min);
           const int z = (vertexIdx.z > AC_nz_max)
			-(vertexIdx.z < AC_nz_min);
	   return (int3){x,y,z}
}
get_boundary(int3 normal)
{
	const int x =  normal.x == 1  ? AC_nx_max-1
		     : normal.x == -1 ? AC_nx_min
		     : start.x;
	const int y =  normal.y == 1  ? AC_ny_max-1
		     : normal.y == -1 ? AC_ny_min
		     : start.y;
	const int z =  normal.z == 1  ? AC_nz_max-1
		     : normal.z == -1 ? AC_nz_min
		     : start.z;
	return (int3){x,y,z}
}

ac_sym_bc(Field f, const int sign)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	for i in 0:NGHOST
	{
		domain = domain - normal
		ghost  = ghost  + normal
		f[ghost.x][ghost.y][ghost.z] = sign*f[domain.x][domain.y][domain.z];
	}
}
utility Kernel BOUNDCOND_SYMMETRIC_DSL(Field f)
{
	ac_sym_bc(f,1)
}
utility Kernel BOUNDCOND_ANTI_SYMMETRIC(Field f)
{
	ac_sym_bc(f,-1)
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
get_boundcond_spacing(int3 normal)
{
	return
		normal.x != 0 ? (-(normal.x == -1))*AC_dsx :
		normal.y != 0 ? (-(normal.y == -1))*AC_dsy :
		(-(normal.z == -1))AC_dsz;
}
utility Kernel BOUNDCOND_PRESCRIBED_DERIVATIVE(Field f, real prescribed_value)
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
get_normal_length(normal)
{
	return
		normal.x == 0 ? normal.x :
		normal.y == 0 ? normal.y :
		normal.z;
}
ac_flow_bc(Field f, const int flow_direction)
{
	const int3 normal = get_normal()
	const int3 boundary = get_boundary(normal)
	int3 domain = boundary
	int3 ghost  = boundary
	const real spacing = get_boundcond_spacing(normal)
	const real direction = get_normal_length(normal)
	const real boundary_val = f[boundary.x][boundary.y][boundary.z]
	const real sign = flow_direction*(boundary_value*direction >= 0.0 ? 1.0 : -1.0)
	for i in 0:NGHOST
	{
		f[ghost.x][ghost.y][ghost.z] = sign*f[domain.x][domain.y][domain.z]
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
