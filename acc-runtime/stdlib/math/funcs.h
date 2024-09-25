

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
set_zero_below_threshold(real val, real threshold)
{
	return val > threshold ? val : 0.0;
}
epsilon(real x)
{
	return AC_REAL_EPSILON;
}

matmul_transpose(Matrix a, Matrix b)
{
	print("Not implemented matmul_tranpose")
	return real3(0.0,0.0,0.0)
}
sign(real val, real set_sign)
{

	return abs(val)*(set_sign < 0.0 ? -1.0 : 1.0);
}

