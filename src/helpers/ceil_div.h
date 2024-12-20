static inline int3
ceil(AcReal3 a)
{
	return (int3){(int)ceil(a.x), (int)ceil(a.y), (int)ceil(a.z)};
}

static inline size3_t
ceil_div(const size3_t& a, const int3& b)
{
	const int3 factors = ceil((AcReal3){(AcReal)a.x, (AcReal)a.y, (AcReal)a.z}/((AcReal3){(AcReal)b.x, (AcReal)b.y, (AcReal)b.z}));
	return (size3_t){(unsigned int)factors.x,(unsigned int)factors.y,(unsigned int)factors.z};
}
static inline int3
ceil_div(const int3& a, const int3& b)
{
	const int3 factors = ceil((AcReal3){(AcReal)a.x, (AcReal)a.y, (AcReal)a.z}/((AcReal3){(AcReal)b.x, (AcReal)b.y, (AcReal)b.z}));
	return (int3){factors.x,factors.y,factors.z};
}
static inline size_t
ceil_div(const size_t& a, const size_t& b)
{
	return (size_t)ceil((AcReal)(1. * a) / b);
}
