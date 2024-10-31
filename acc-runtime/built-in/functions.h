#if TWO_D
Stencil value_stencil
{ 
	[0][0] = 1
}
#else
Stencil value_stencil
{
	[0][0][0] = 1
}
#endif


elemental
value(Field s)
{
	return value_stencil(s)
}
value(Profile profile)
{
	return value_profile(profile)
}

elemental previous(Field s)
{
	return previous_base(s)
}

vecvalue(Field3 v)
{
	return
		real3(
		   value_stencil(v.x),
		   value_stencil(v.y),
		   value_stencil(v.z)
		)
}
vecprevious(Field3 v)
{
	return
		real3(
		   previous_base(v.x),
		   previous_base(v.y),
		   previous_base(v.z)
		)
}

write(Field dst, real src)
{
	write_base(dst,src)
}

write(Field3 dst, real3 src)
{
	write(dst.x, src.x)
	write(dst.y, src.y)
	write(dst.z, src.z)
}
write(Field[] dst, real[] src)
{
	for i in 0:size(dst)
	{
		write(dst[i],src[i])
	}
}
write(Field3[] dst, real3[] src)
{
	for i in 0:size(dst)
	{
		write(dst[i],src[i])
	}
}

vecwrite(Field3 dst, real3 src)
{
	write_base(dst.x, src.x)
	write_base(dst.y, src.y)
	write_base(dst.z, src.z)
}
real3 intrinsic AC_cross
real intrinsic AC_dot


cross(real3 a, real3 b)
{
	return AC_cross(a,b)
}

dot(real3 a, real3 b)
{
	return AC_dot(a, b)
}
dot(real[] a, real[] b)
{
	return AC_dot(a, b)
}
reduce_min(bool condition, real val, param)
{
	reduce_min_real(condition,val,param)
}
reduce_sum(bool condition, real val, param)
{
	reduce_sum_real(condition,val,param)
}
reduce_max(bool condition, real val, param)
{
	reduce_max_real(condition,val,param)
}

reduce_min(bool condition, int val, param)
{
	reduce_min_int(condition,val,param)
}
reduce_sum(bool condition, int val, param)
{
	reduce_sum_int(condition,val,param)
}
reduce_max(bool condition, int val, param)
{
	reduce_max_int(condition,val,param)
}
reduce_sum(bool condition, real val, Profile<X> prof)
{
	reduce_sum_real_x(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<Y> prof)
{
	reduce_sum_real_y(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<Z> prof)
{
	reduce_sum_real_z(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<XY> prof)
{
	reduce_sum_real_xy(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<XZ> prof)
{
	reduce_sum_real_xz(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<YX> prof)
{
	reduce_sum_real_yx(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<YZ> prof)
{
	reduce_sum_real_yz(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<ZX> prof)
{
	reduce_sum_real_zx(condition, val, prof);
}
reduce_sum(bool condition, real val, Profile<ZY> prof)
{
	reduce_sum_real_zy(condition, val, prof);
}
