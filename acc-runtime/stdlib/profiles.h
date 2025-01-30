struct VecZProfile
{
	Profile<Z> x
	Profile<Z> y
	Profile<Z> z
}

reduce_sum(real3 vec, VecZProfile prof)
{
	reduce_sum(vec.x,prof.x)
	reduce_sum(vec.y,prof.y)
	reduce_sum(vec.z,prof.z)
}
