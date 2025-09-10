#ifdef AC_HOME
#include "$AC_HOME/acc-runtime/stdlib/general_grid"
#endif
const int AC_n_spherical_harmonics = 6
gmem real AC_PLM[AC_mlocal.y][AC_n_spherical_harmonics][AC_n_spherical_harmonics]
gmem real AC_cos_m_phis[AC_mlocal.z][AC_n_spherical_harmonics]
gmem real AC_sin_m_phis[AC_mlocal.z][AC_n_spherical_harmonics]
global output real AC_upper_positive_MLM[AC_n_spherical_harmonics*AC_n_spherical_harmonics]
global output real AC_upper_negative_MLM[AC_n_spherical_harmonics*AC_n_spherical_harmonics]

global output real AC_lower_positive_MLM[AC_n_spherical_harmonics*AC_n_spherical_harmonics]
global output real AC_lower_negative_MLM[AC_n_spherical_harmonics*AC_n_spherical_harmonics]

positive_real_spherical_harmonic(int m, int l)
{
	return AC_PLM[vertexIdx.y][l][m]*AC_cos_m_phis[vertexIdx.z][m]
}

negative_real_spherical_harmonic(int m, int l)
{
	return AC_PLM[vertexIdx.y][l][m]*AC_sin_m_phis[vertexIdx.z][m]
}
calculate_mlm(Field src)
{
	r = AC_r[vertexIdx.x]
	w = get_integration_weight()
	for l in 0:AC_n_spherical_harmonics
	{
		r_factor = 1.0
		for i in 0:l
		{
			r_factor *= r
		}
		for m in 0:l+1
		{
			rest     = src*r_factor*w
			reduce_sum(positive_real_spherical_harmonic(m,l)*rest,AC_upper_positive_MLM[l + m*AC_n_spherical_harmonics])
			if(m > 0)
			{
				reduce_sum(negative_real_spherical_harmonic(m,l)*rest,AC_upper_negative_MLM[l + m*AC_n_spherical_harmonics])
			}
		}
		//Exterior potential expansion
		r_factor = 1.0
		for i in 0:l+1
		{
			r_factor /= r
		}
		for m in 0:l+1
		{
			rest     = src*r_factor*w
			reduce_sum(positive_real_spherical_harmonic(m,l)*rest,AC_lower_positive_MLM[l + m*AC_n_spherical_harmonics])
			if(m > 0)
			{
				reduce_sum(negative_real_spherical_harmonic(m,l)*rest,AC_lower_negative_MLM[l + m*AC_n_spherical_harmonics])
			}
		}
	}
}

calculate_expansion_with_mlm(Field lower, Field upper)
{
	r = AC_r[vertexIdx.x]

	//Interior potential expansion
	res = 0.0
	for l in 0:AC_n_spherical_harmonics
	{
		r_factor = 1.0/r
		for i in 0:l
		{
			r_factor /= r
		}
		for m in 0:l+1
		{
			res -= r_factor*AC_upper_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l)
			if(m > 0)
			{
				res -= r_factor*AC_upper_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l)
			}
		}
	}
	write(upper,res)

	//Exterior potential expansion
	res = 0.0
	for l in 0:AC_n_spherical_harmonics
	{
		r_factor = 1.0
		for i in 0:l
		{
			r_factor *= r
		}
		for m in 0:l+1
		{
			res -= r_factor*AC_lower_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l)
			if(m > 0)
			{
				res -= r_factor*AC_lower_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l)
			}
		}
	}
	write(lower,res)
}
