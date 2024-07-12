Stencil value_stencil
{
	[0][0][0] = 1
}

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
