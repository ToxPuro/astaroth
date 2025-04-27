

norm(real3 vec)
{
	return sqrt(dot(vec,vec))
}
elemental abs(real x)
{
	return fabs(x)
}
set_min_val(real val, real min_val)
{
	return max(val,min_val)
}
set_max_val(real val, real max_val)
{
	return min(val,max_val)
}
set_zero_below_threshold(real val, real threshold)
{
	return val > threshold ? val : 0.0;
}
epsilon(real x)
{
	return AC_REAL_EPSILON;
}

matmul_transpose(Matrix a, real3 b)
{
	print("Not implemented matmul_tranpose")
	return real3(0.0,0.0,0.0)
}
sign(real val, real set_sign)
{

	return abs(val)*(set_sign < 0.0 ? -1.0 : 1.0);
}

modulo(real a, real p)
{
	return a - floor(a/p)*p
}
nint(real x)
{
	return round(x)
}
//TP: placeholder. Please port from Pencil code general.f90 to DSL :)
inline spline_integral(real[] a,b)
{
	print("NOT IMPLEMENTED spline_integral!\n")
	return a
}
tanh_step_function(real position, real center, real steepness, real L, real R)
{
	return L+(R-L)*0.5*(1.0 + tanh(steepness*(position-center)))	
}
tanh_step_function(real position, real center, real steepness, real3 L, real3 R)
{
	return L+(R-L)*0.5*(1.0 + tanh(steepness*(position-center)))
}


