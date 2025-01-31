Stencil value_stencil
{
	[0][0][0] = 1
}

elemental
value(Field s) {
	return value_stencil(s)
}
value(Profile<X> profile)
{
	return value_profile_x(profile)
}
value(Profile<Y> profile)
{
	return value_profile_y(profile)
}

value(Profile<Z> profile)
{
	return value_profile_z(profile)
}

value(Profile<XY> profile)
{
	return value_profile_xy(profile)
}

value(Profile<XZ> profile)
{
	return value_profile_xz(profile)
}

value(Profile<YX> profile)
{
	return value_profile_yx(profile)
}

value(Profile<YZ> profile)
{
	return value_profile_yz(profile)
}

value(Profile<ZX> profile)
{
	return value_profile_zx(profile)
}

value(Profile<ZY> profile)
{
	return value_profile_zy(profile)
}


elemental previous(Field s)
{
	return previous_base(s)
}


write(Field dst, real src)
{
	write_base(dst,src)
}

write(Profile<X> dst, real src)
{
	write_profile_x(dst,src)
}

write(Profile<Y> dst, real src)
{
	write_profile_y(dst,src)
}

write(Profile<Z> dst, real src)
{
	write_profile_z(dst,src)
}

write(Profile<XY> dst, real src)
{
	write_profile_xy(dst,src)
}

write(Profile<XZ> dst, real src)
{
	write_profile_xz(dst,src)
}

write(Profile<YX> dst, real src)
{
	write_profile_yx(dst,src)
}

write(Profile<YZ> dst, real src)
{
	write_profile_yz(dst,src)
}

write(Profile<ZX> dst, real src)
{
	write_profile_zx(dst,src)
}

write(Profile<ZY> dst, real src)
{
	write_profile_zy(dst,src)
}

write(Field3 dst, real3 src)
{
	write_base(dst.x, src.x)
	write_base(dst.y, src.y)
	write_base(dst.z, src.z)
}
inline write(Field[] dst, real[] src)
{
	for i in 0:size(dst)
	{
		write(dst[i],src[i])
	}
}
inline write(Field3[] dst, real3[] src)
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

dot(a, b)
{
	return AC_dot(a, b)
}


reduce_min(real val, param)
{
	reduce_min_real(val,param)
}
reduce_sum(real val, param)
{
	reduce_sum_real(val,param)
}
reduce_max(real val, param)
{
	reduce_max_real(val,param)
}

reduce_min(int val, param)
{
	reduce_min_int(val,param)
}
reduce_sum(int val, param)
{
	reduce_sum_int(val,param)
}
reduce_max(int val, param)
{
	reduce_max_int(val,param)
}
reduce_min(float val, param)
{
	reduce_min_float(val,param)
}
reduce_sum(float val, param)
{
	reduce_sum_float(val,param)
}
reduce_max(float val, param)
{
	reduce_max_float(val,param)
}


reduce_sum(real val, Profile<X> prof)
{
	reduce_sum_real_x(val, prof);
}
reduce_sum(real val, Profile<Y> prof)
{
	reduce_sum_real_y(val, prof);
}
reduce_sum(real val, Profile<Z> prof)
{
	reduce_sum_real_z(val, prof);
}
reduce_sum(real val, Profile<XY> prof)
{
	reduce_sum_real_xy(val, prof);
}
reduce_sum(real val, Profile<XZ> prof)
{
	reduce_sum_real_xz(val, prof);
}
reduce_sum(real val, Profile<YX> prof)
{
	reduce_sum_real_yx(val, prof);
}
reduce_sum(real val, Profile<YZ> prof)
{
	reduce_sum_real_yz(val, prof);
}
reduce_sum(real val, Profile<ZX> prof)
{
	reduce_sum_real_zx(val, prof);
}
reduce_sum(real val, Profile<ZY> prof)
{
	reduce_sum_real_zy(val, prof);
}

inline add_arr(real[] a, real[] b)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] + b[i]
	}
	return res;
}

inline add_arr(real3[] a, real3[] b)
{
	real3 res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] + b[i]
	}
	return res;
}

inline div_arr(real[] a, real[] b)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] / b[i]
	}
	return res;
}

inline div_arr(real3[] a, real3[] b)
{
	real3 res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] / b[i]
	}
	return res;
}

inline mult_arr(real[] a, real[] b)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] * b[i]
	}
	return res;
}

inline mult_arr(real[] a, real b)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] * b
	}
	return res;
}

inline mult_arr(real3[] a, real3[] b)
{
	real3 res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] * b[i]
	}
	return res;
}

inline sub_arr(real[] a, real[] b)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] - b[i]
	}
	return res;
}

inline sub_arr(real3[] a, real3[] b)
{
	real3 res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i] - b[i]
	}
	return res;
}

inline dup_arr(real[] a)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i]
	}
	return res;
}

inline dup_arr(real3[] a)
{
	real3 res[size(a)]
	for i in 0:size(a)
	{
		res[i] = a[i]
	}
	return res;
}

inline create_neg_arr(real[] a)
{
	real res[size(a)]
	for i in 0:size(a)
	{
		res[i] = -a[i]
	}
	return res;
}

sum(real3 a)
{
	return a.x + a.y + a.z
}

sum(real2 a)
{
	return a.x + a.y
}

any_AC(bool[] arr, int arr_len)
{
	bool res = false
	for i in 0:arr_len
	{
		res |= arr[i]
	}
	return res
}


//inline any(b) {return b.x || b.y || b.z}
//inline all(b) {b.x && b.y && b.z}

#define any(b) (b.x || b.y || b.z)
#define all(b) (b.x && b.y && b.z)

