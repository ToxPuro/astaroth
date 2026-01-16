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

calculate_mlm(real src)
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
			rest     = src*r_factor*w*sqrt(4*AC_REAL_PI/(2*l+1))
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
			rest     = src*r_factor*w*sqrt(4*AC_REAL_PI/(2*l+1))
			reduce_sum(positive_real_spherical_harmonic(m,l)*rest,AC_lower_positive_MLM[l + m*AC_n_spherical_harmonics])
			if(m > 0)
			{
				reduce_sum(negative_real_spherical_harmonic(m,l)*rest,AC_lower_negative_MLM[l + m*AC_n_spherical_harmonics])
			}
		}
	}
}

calculate_mlm_extended(real src)
{
	r = AC_r_extended[vertexIdx.x]
	w = get_integration_weight_extended()
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
	real r_factor
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

calculate_expansion_with_mlm_extended(Field lower, Field upper)
{
	r = AC_r_extended[vertexIdx.x]

	//Interior potential expansion
	res = 0.0
	real r_factor
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

multipole_expansion_bc_inner(AcBoundary boundary_region, Field f, real G)
{
    const int3 normal = get_normal(boundary_region)
    const int3 boundary = get_boundary(normal)
    int3 domain = boundary
    int3 ghost  = boundary
    for i in 0:NGHOST
    {
            domain = domain - normal
            ghost  = ghost  + normal
    r = grid_position(ghost).x

    
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
            res -= r_factor*AC_lower_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l) *  sqrt(4*AC_REAL_PI/(2*l+1))

            if(m > 0)
            {
                res -= r_factor*AC_lower_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))

            }
        }
    }
    res *= G
    f[ghost.x][ghost.y][ghost.z] = res
    }
}

multipole_expansion_bc_inner_extended(AcBoundary boundary_region, Field f, real G)
{
    const int3 normal = get_normal(boundary_region)
    const int3 boundary = get_boundary(normal)
    int3 domain = boundary
    int3 ghost  = boundary
    for i in 0:NGHOST
    {
            domain = domain - normal
            ghost  = ghost  + normal
    r = grid_position_extended(ghost).x

    
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
            res -= r_factor*AC_lower_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l) *  sqrt(4*AC_REAL_PI/(2*l+1))

            if(m > 0)
            {
                res -= r_factor*AC_lower_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))

            }
        }
    }
    res *= G
    f[ghost.x][ghost.y][ghost.z] = res
    }
}

multipole_expansion_bc_outer(AcBoundary boundary_region, Field f, real G)
{
    const int3 normal = get_normal(boundary_region)
    const int3 boundary = get_boundary(normal)
    int3 domain = boundary
    int3 ghost  = boundary
    for i in 0:NGHOST
    {
        domain = domain - normal
        ghost  = ghost  + normal
        r = grid_position(ghost).x
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
                res -= r_factor*AC_upper_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))
 
                if(m > 0)
                {
                    res -= r_factor*AC_upper_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))

                }
            }
        }
    res *= G
    f[ghost.x][ghost.y][ghost.z] = res
    }
}

multipole_expansion_bc_outer_extended(AcBoundary boundary_region, Field f, real G)
{
    const int3 normal = get_normal(boundary_region)
    const int3 boundary = get_boundary(normal)
    int3 domain = boundary
    int3 ghost  = boundary
    for i in 0:NGHOST
    {
        domain = domain - normal
        ghost  = ghost  + normal
        r = grid_position_extended(ghost).x
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
                res -= r_factor*AC_upper_positive_MLM[l + m*AC_n_spherical_harmonics]*positive_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))
 
                if(m > 0)
                {
                    res -= r_factor*AC_upper_negative_MLM[l + m*AC_n_spherical_harmonics]*negative_real_spherical_harmonic(m,l) * sqrt(4*AC_REAL_PI/(2*l+1))

                }
            }
        }
    res *= G
    f[ghost.x][ghost.y][ghost.z] = res
    }
}
