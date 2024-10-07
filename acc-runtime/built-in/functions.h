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
write(Field dst, Field src)
{
	write_base(dst,value(src))
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
