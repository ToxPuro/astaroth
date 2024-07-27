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
//previous(Field3 s)
//{
//	return 
//		real3(
//				previous(s.x),
//				previous(s.y),
//				previous(s.z)
//		) 
//}

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

elemental 
cross(real3 a, real3 b)
{
	return AC_cross(a,b)
}

elemental 
dot(real3 a, real3 b)
{
	return AC_dot(a, b)
}


