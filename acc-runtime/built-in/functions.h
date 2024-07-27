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


value(Field s)
{
	return value_stencil(s)
}
value(Field3 v)
{
	return real3(value(v.x), value(v.y), value(v.z))
}
vecvalue(Field3 v)
{
	return real3(value(v.x), value(v.y), value(v.z))
}
previous(Field s)
{
	return previous_base(s)
}
previous(Field3 v)
{
	return real3(previous(v.x), previous(v.y), previous(v.z))
}

vecprevious(Field3 v)
{
	return real3(previous(v.x), previous(v.y), previous(v.z))
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
abs(Field3 v) {
    return real3(fabs(value(v.x)), fabs(value(v.y)), fabs(value(v.z)))
}
dot(Field3 field, real3 vec)
{
	return AC_dot(value(field), vec)
}

dot(Field3 a, Field3 b)
{
	return AC_dot(value(a), value(b))
}

dot(real3 vec, Field3 field)
{
	return AC_dot(vec, value(field))
}
dot(real3 a, real3 b)
{
	return AC_dot(a,b)
}
cross(real3 a, real3 b)
{
	return AC_cross(a,b)
}

