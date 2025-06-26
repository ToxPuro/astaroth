
#define DER1_3 (1. / 60.)
#define DER1_2 (-3. / 20.)
#define DER1_1 (3. / 4.)
#define DER1_0 (0)

#define DER2_3 (1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 (3. / 2.)
#define DER2_0 (-49. / 18.)


#define DERX_3 (2. / 720.)
#define DERX_2 (-27. / 720.)
#define DERX_1 (270. / 720.)
#define DERX_0 (0)

#define DER6UPWD_3 (  1. / 60.)
#define DER6UPWD_2 ( -6. / 60.)
#define DER6UPWD_1 ( 15. / 60.)
#define DER6UPWD_0 (-20. / 60.)

#define DER6_0 -20.0
#define DER6_1 15.0
#define DER6_2 -6.0
#define DER6_3 1.0

#define DER5_1 2.5
#define DER5_2 2.0
#define DER5_3 0.5

#define DER4_0 (56.0/6.0)
#define DER4_1 (-39.0/6.0)
#define DER4_2 (12.0/6.0)
#define DER4_3 (-1.0)


#define DER3_0 (0)
#define DER3_1 (-13.0/8.0)
#define DER3_2 (1)
#define DER3_3 (-1.0/8.0)


#define DER4i2j_scaling_factor 1/(6.0*180.0)
#define DER4i2j_first 56.0
#define DER4i2j_second -39.0
#define DER4i2j_third 12.0
#define DER4i2j_fourth -1.0

#define DER4i2j_0 -490.0
#define DER4i2j_1 270.0
#define DER4i2j_2 -27.0
#define DER4i2j_3 2.0


#define DERX_3 (2. / 720.)
#define DERX_2 (-27. / 720.)
#define DERX_1 (270. / 720.)
#define DERX_0 (0)

#define DER6UPWD_3 (  1. / 60.)
#define DER6UPWD_2 ( -6. / 60.)
#define DER6UPWD_1 ( 15. / 60.)
#define DER6UPWD_0 (-20. / 60.)

gmem real AC_inv_r[AC_nlocal.x]
gmem real AC_inv_cyl_r[AC_nlocal.x]
gmem real AC_inv_sin_theta[AC_mlocal.y]
gmem real AC_cot_theta[AC_mlocal.y]

gmem real AC_inv_mapping_func_derivative_x[AC_mlocal.x]
gmem real AC_inv_mapping_func_derivative_y[AC_mlocal.y]
gmem real AC_inv_mapping_func_derivative_z[AC_mlocal.z]

gmem real AC_mapping_func_tilde_x[AC_mlocal.x]
gmem real AC_mapping_func_tilde_y[AC_mlocal.y]
gmem real AC_mapping_func_tilde_z[AC_mlocal.z]

#define AC_INV_R         (AC_inv_r[vertexIdx.x-NGHOST])
#define AC_INV_CYL_R     (AC_inv_cyl_r[vertexIdx.x-NGHOST])
#define AC_INV_SIN_THETA (AC_inv_sin_theta[vertexIdx.y])
#define AC_COT           (AC_cot_theta[vertexIdx.y])

#define AC_INV_MAPPING_FUNC_DER_X (AC_inv_mapping_func_derivative_x[vertexIdx.x])
#define AC_INV_MAPPING_FUNC_DER_Y (AC_inv_mapping_func_derivative_y[vertexIdx.y])
#define AC_INV_MAPPING_FUNC_DER_Z (AC_inv_mapping_func_derivative_z[vertexIdx.z])

#define AC_MAPPING_FUNC_TILDE_X  (AC_mapping_func_tilde_x[vertexIdx.x])
#define AC_MAPPING_FUNC_TILDE_Y  (AC_mapping_func_tilde_y[vertexIdx.y])
#define AC_MAPPING_FUNC_TILDE_Z  (AC_mapping_func_tilde_z[vertexIdx.z])

Stencil derx_stencil {
    [0][0][-3] = -DER1_3,
    [0][0][-2] = -DER1_2,
    [0][0][-1] = -DER1_1,
    [0][0][1]  = DER1_1,
    [0][0][2]  = DER1_2,
    [0][0][3]  = DER1_3
}
Stencil derx_2nd_stencil {
    [0][0][-1] = -0.5,
    [0][0][1 ]  = 0.5,
}

Stencil dery_2nd_stencil {
    [0][-1][0] = -0.5,
    [0][1 ][0]  = 0.5,
}
Stencil derz_2nd_stencil {
    [-1][0][0] = -0.5,
    [1 ][0][0]  = 0.5,
}

#define AC_GEN_DERX(NAME,STENCIL_NAME) \
	NAME(Field f) \
	{ \
        	if(AC_dimension_inactive.x) \
		{ \
			return 0.0 \
		} \
		else \
		{ \
			real res = 0.0 \
			if(AC_nonequidistant_grid.x) \
			{ \
				res = STENCIL_NAME(f)*AC_INV_MAPPING_FUNC_DER_X \
			} \
			else \
			{ \
				res= STENCIL_NAME(f)*AC_inv_ds.x \
			} \
			return res \
		} \
	} 

AC_GEN_DERX(derx,derx_stencil)
AC_GEN_DERX(derx_2nd,derx_2nd_stencil)
derx(Profile<X> prof)
{
	if(AC_dimension_inactive.x)
	{
		return 0.0
	}
	else
	{
		real res = 0.0
		if(AC_nonequidistant_grid.x)
		{
			res = derx_stencil(prof)*AC_INV_MAPPING_FUNC_DER_X
		}
		else
		{
			res  = derx_stencil(prof)*AC_inv_ds.x
		}
		return res
	}
}

Stencil dery_stencil {
    [0][-3][0] = -DER1_3,
    [0][-2][0] = -DER1_2,
    [0][-1][0] = -DER1_1,
    [0][1][0]  = DER1_1,
    [0][2][0]  = DER1_2,
    [0][3][0]  = DER1_3
}

#define AC_GEN_DERY(NAME,STENCIL_NAME) \
	NAME(Field f) \
	{ \
        	if(AC_dimension_inactive.y) \
		{ \
			return 0.0 \
		} \
		else {\
			coordinate_factor = 1.0 \
			if(AC_coordinate_system == AC_SPHERICAL_COORDINATES) \
			{ \
				coordinate_factor = AC_INV_R \
			} \
			if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES) \
			{ \
				coordinate_factor = AC_INV_CYL_R \
			} \
			grid_factor = 1.0 \
			if(AC_nonequidistant_grid.y) \
			{ \
				grid_factor = AC_INV_MAPPING_FUNC_DER_Y \
			} \
			else \
			{ \
				grid_factor = AC_inv_ds.y \
			} \
			return STENCIL_NAME(f)*coordinate_factor*grid_factor \
		} \
	}

AC_GEN_DERY(dery,dery_stencil)
AC_GEN_DERY(dery_2nd,dery_2nd_stencil)

dery(Profile<Y> prof)
{
	if(AC_dimension_inactive.y)
	{
		return 0.0
	}
	else
	{
		coordinate_factor = 1.0
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
		{
			coordinate_factor = AC_INV_R
		}
		if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
		{
			coordinate_factor = AC_INV_CYL_R
		}
		grid_factor = 1.0
		if(AC_nonequidistant_grid.y)
		{
			grid_factor = AC_INV_MAPPING_FUNC_DER_Y
		}
		else
		{
			grid_factor = AC_inv_ds.y
		}
		return dery_stencil(prof)*coordinate_factor*grid_factor
	}
}



Stencil derz_stencil {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = -DER1_2,
    [-1][0][0] = -DER1_1,
    [1][0][0]  = DER1_1,
    [2][0][0]  = DER1_2,
    [3][0][0]  = DER1_3
}

#define AC_GEN_DERZ(NAME,STENCIL) \
	NAME(Field f) \
	{ \
		if(AC_dimension_inactive.z) \
		{ \
			return 0.0 \
		} \
		else {\
			coordinate_factor = 1.0 \
			if(AC_coordinate_system == AC_SPHERICAL_COORDINATES) \
			{ \
				coordinate_factor = AC_INV_R*AC_INV_SIN_THETA \
			} \
			grid_factor = 1.0 \
			if(AC_nonequidistant_grid.z) \
			{ \
				grid_factor = AC_INV_MAPPING_FUNC_DER_Z \
			} \
			else \
			{ \
				grid_factor = AC_inv_ds.z \
			} \
			return STENCIL(f)*coordinate_factor*grid_factor \
		} \
	}

AC_GEN_DERZ(derz,derz_stencil)
AC_GEN_DERZ(derz_2nd,derz_2nd_stencil)

derz(Profile<Z> prof)
{
	if(AC_dimension_inactive.z)
	{
		return 0.0
	}
	else
	{
		coordinate_factor = 1.0
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
		{
			coordinate_factor = AC_INV_R*AC_INV_SIN_THETA
		}
		grid_factor = 1.0
		if(AC_nonequidistant_grid.z)
		{
			grid_factor = AC_INV_MAPPING_FUNC_DER_Z
		}
		else
		{
			grid_factor = AC_inv_ds.z
		}
		return derz_stencil(prof)*coordinate_factor*grid_factor
	}
}


Stencil derxx_stencil {
    [0][0][-3] = DER2_3,
    [0][0][-2] = DER2_2,
    [0][0][-1] = DER2_1,
    [0][0][0]  = DER2_0,
    [0][0][1]  = DER2_1,
    [0][0][2]  = DER2_2,
    [0][0][3]  = DER2_3
}

Stencil derxx_neighbours_stencil {
    [0][0][-3] = DER2_3,
    [0][0][-2] = DER2_2,
    [0][0][-1] = DER2_1,
    [0][0][1]  = DER2_1,
    [0][0][2]  = DER2_2,
    [0][0][3]  = DER2_3
}
#define DER2_2nd_1 (1)
#define DER2_2nd_0 (-2)

Stencil derxx_2nd_stencil {
    [0][0][-1]  = DER2_2nd_1,
    [0][0][0 ]  = DER2_2nd_0,
    [0][0][1 ]  = DER2_2nd_1
}

Stencil deryy_2nd_stencil {
    [0][-1][0]  = DER2_2nd_1,
    [0][0 ][0]  = DER2_2nd_0,
    [0][1 ][0]  = DER2_2nd_1
}

Stencil derzz_2nd_stencil {
    [-1][0][0]  = DER2_2nd_1,
    [0 ][0][0]  = DER2_2nd_0,
    [1 ][0][0]  = DER2_2nd_1
}

Stencil derxx_2nd_neighbours_stencil {
    [0][0][-1] = DER2_2nd_1,
    [0][0][1]  = DER2_2nd_1,
}

#define AC_GEN_DERXX(NAME,STENCIL,DERX) \
NAME(Field f) \
{ \
	res = 0.0 \
	if(!AC_dimension_inactive.x) \
	{ \
		if(AC_nonequidistant_grid.x) \
		{ \
			res = STENCIL(f)*(AC_INV_MAPPING_FUNC_DER_X*AC_INV_MAPPING_FUNC_DER_X) + DERX(f)*AC_MAPPING_FUNC_TILDE_X \
		} \
		else \
		{ \
			res = STENCIL(f)*AC_inv_ds_2.x \
		} \
	} \
	return res \
}
AC_GEN_DERXX(derxx,derxx_stencil,derx)
AC_GEN_DERXX(derxx_2nd,derxx_2nd_stencil,derx_2nd)
AC_GEN_DERXX(derxx_neighbours,derxx_neighbours_stencil,derx)
AC_GEN_DERXX(derxx_2nd_neighbours,derxx_2nd_neighbours_stencil,derx_2nd)

derxx_central_coeff()
{
	real res = DER2_0
	if(!AC_nonequidistant_grid.x)
	{
		res *= AC_inv_ds_2.x
	}
	else
	{
		//Tilde factor conveniently vanishes
		 res *= (AC_INV_MAPPING_FUNC_DER_X*AC_INV_MAPPING_FUNC_DER_X) 
	}
	return res
}

derxx_2nd_central_coeff()
{
	real res = DER2_2nd_0
	if(!AC_nonequidistant_grid.x)
	{
		res *= AC_inv_ds_2.x
	}
	else
	{
		//Tilde factor conveniently vanishes
		 res *= (AC_INV_MAPPING_FUNC_DER_X*AC_INV_MAPPING_FUNC_DER_X) 
	}
	return res
}

derxx(Profile<X> prof)
{
	real res = 0.0
	if(!AC_dimension_inactive.x)
	{
		if(!AC_nonequidistant_grid.x)
		{
			res = derxx_stencil(prof)*AC_inv_ds_2.x
		}
		else
		{
			res = derxx_stencil(prof)*(AC_INV_MAPPING_FUNC_DER_X*AC_INV_MAPPING_FUNC_DER_X) + derx(prof)*AC_MAPPING_FUNC_TILDE_X
		}
	}
	return res
}

Stencil deryy_stencil {
    [0][-3][0] = DER2_3,
    [0][-2][0] = DER2_2,
    [0][-1][0] = DER2_1,
    [0][0][0]  = DER2_0,
    [0][1][0]  = DER2_1,
    [0][2][0]  = DER2_2,
    [0][3][0]  = DER2_3
}

Stencil deryy_neighbours_stencil {
    [0][-3][0] = DER2_3,
    [0][-2][0] = DER2_2,
    [0][-1][0] = DER2_1,
    [0][1][0]  = DER2_1,
    [0][2][0]  = DER2_2,
    [0][3][0]  = DER2_3
}

Stencil deryy_2nd_neighbours_stencil {
    [0][-1][0] = DER2_2nd_1,
    [0][1][0]  = DER2_2nd_1,
}

#define AC_GEN_DERYY(NAME,STENCIL,DERY) \
NAME(Field f) \
{ \
	if(AC_dimension_inactive.y) \
	{ \
		return 0.0 \
	} \
	else { \
		coordinate_factor = 1.0 \
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES) \
		{ \
			coordinate_factor = (AC_INV_R*AC_INV_R) \
		} \
		if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES) \
		{ \
			coordinate_factor = (AC_INV_CYL_R*AC_INV_CYL_R) \
		} \
		grid_factor = 1.0 \
		if(AC_nonequidistant_grid.y) \
		{ \
			grid_factor = AC_INV_MAPPING_FUNC_DER_Y*AC_INV_MAPPING_FUNC_DER_Y \
		} \
		else \
		{ \
			grid_factor = AC_inv_ds_2.y \
		} \
		res = STENCIL(f)*coordinate_factor*grid_factor \
		if(AC_nonequidistant_grid.y) \
		{ \
			 res += AC_MAPPING_FUNC_TILDE_Y*DERY(f) \
		} \
		return res \
	} \
}
AC_GEN_DERYY(deryy,deryy_stencil,dery)
AC_GEN_DERYY(deryy_2nd,deryy_2nd_stencil,dery_2nd)
AC_GEN_DERYY(deryy_2nd_neighbours,deryy_2nd_neighbours_stencil,dery_2nd)
AC_GEN_DERYY(deryy_neighbours,deryy_neighbours_stencil,dery)

deryy_2nd_central_coeff()
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = (AC_INV_R*AC_INV_R)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		coordinate_factor = (AC_INV_CYL_R*AC_INV_CYL_R)
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.y)
	{
		grid_factor = AC_INV_MAPPING_FUNC_DER_Y*AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		grid_factor = AC_inv_ds_2.y
	}
	res = coordinate_factor*grid_factor
	//Tilde factor vanishes!
	return DER2_2nd_0*res
}

deryy_central_coeff()
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = (AC_INV_R*AC_INV_R)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		coordinate_factor = (AC_INV_CYL_R*AC_INV_CYL_R)
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.y)
	{
		grid_factor = AC_INV_MAPPING_FUNC_DER_Y*AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		grid_factor = AC_inv_ds_2.y
	}
	res = coordinate_factor*grid_factor
	//Tilde factor vanishes!
	return DER2_0*res
}
deryy(Profile<Y> prof)
{
	if(AC_dimension_inactive.y)
	{
		return 0.0
	}
	else
	{
		coordinate_factor = 1.0
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
		{
			coordinate_factor = (AC_INV_R*AC_INV_R)
		}
		if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
		{
			coordinate_factor = (AC_INV_CYL_R*AC_INV_CYL_R)
		}
		grid_factor = 1.0
		if(AC_nonequidistant_grid.y)
		{
			grid_factor = AC_INV_MAPPING_FUNC_DER_Y*AC_INV_MAPPING_FUNC_DER_Y
		}
		else
		{
			grid_factor = AC_inv_ds_2.y
		}
		res = deryy_stencil(prof)*coordinate_factor*grid_factor
		if(AC_nonequidistant_grid.y)
		{
			 res += AC_MAPPING_FUNC_TILDE_Y*dery(prof)
		}
		return res
	}
}

Stencil derzz_stencil {
    [-3][0][0] = DER2_3,
    [-2][0][0] = DER2_2,
    [-1][0][0] = DER2_1,
    [0][0][0]  = DER2_0,
    [1][0][0]  = DER2_1,
    [2][0][0]  = DER2_2,
    [3][0][0]  = DER2_3
}

Stencil derzz_neighbours_stencil {
    [-3][0][0] = DER2_3,
    [-2][0][0] = DER2_2,
    [-1][0][0] = DER2_1,
    [1][0][0]  = DER2_1,
    [2][0][0]  = DER2_2,
    [3][0][0]  = DER2_3
}

Stencil derzz_2nd_neighbours_stencil {
    [-1][0][0] = DER2_2nd_1,
    [1][0][0]  = DER2_2nd_1,
}
#define AC_GEN_DER_ZZ(NAME,STENCIL,DERZ) \
NAME(Field f) \
{ \
	if(AC_dimension_inactive.z) \
	{ \
		return 0.0 \
	} \
	else {\
		coordinate_factor = 1.0 \
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES) \
		{ \
			coordinate_factor = AC_INV_R*AC_INV_SIN_THETA; \
			coordinate_factor *= coordinate_factor \
		} \
		grid_factor = 1.0 \
		if(AC_nonequidistant_grid.z) \
		{ \
			grid_factor = AC_INV_MAPPING_FUNC_DER_Z*AC_INV_MAPPING_FUNC_DER_Z \
		} \
		else \
		{ \
			grid_factor = AC_inv_ds_2.z \
		} \
		res = STENCIL(f)*coordinate_factor*grid_factor \
		if(AC_nonequidistant_grid.z) \
		{ \
			 res += AC_MAPPING_FUNC_TILDE_Z*DERZ(f) \
		} \
		return res \
	}\
} 

AC_GEN_DER_ZZ(derzz_2nd_neighbours,derzz_2nd_neighbours_stencil,derz_2nd)
AC_GEN_DER_ZZ(derzz_neighbours,derzz_neighbours_stencil,derz)
AC_GEN_DER_ZZ(derzz_2nd,derzz_2nd_stencil,derz_2nd)
AC_GEN_DER_ZZ(derzz,derzz_stencil,derz)

derzz_2nd_central_coeff()
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R*AC_INV_SIN_THETA;
		coordinate_factor *= coordinate_factor
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.z)
	{
		grid_factor = AC_INV_MAPPING_FUNC_DER_Z*AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		grid_factor = AC_inv_ds_2.z
	}
	res = coordinate_factor*grid_factor
	//Tilde factor conveniently vanishes
	return DER2_2nd_0*res
}

derzz_central_coeff()
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R*AC_INV_SIN_THETA;
		coordinate_factor *= coordinate_factor
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.z)
	{
		grid_factor = AC_INV_MAPPING_FUNC_DER_Z*AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		grid_factor = AC_inv_ds_2.z
	}
	res = coordinate_factor*grid_factor
	//Tilde factor conveniently vanishes
	return DER2_0*res
}

derzz(Profile<Z> prof)
{
	if(AC_dimension_inactive.z)
	{
		return 0.0
	}
	else
	{
		coordinate_factor = 1.0
		if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
		{
			coordinate_factor = AC_INV_R*AC_INV_SIN_THETA;
			coordinate_factor *= coordinate_factor
		}
		grid_factor = 1.0
		if(AC_nonequidistant_grid.z)
		{
			grid_factor = AC_INV_MAPPING_FUNC_DER_Z*AC_INV_MAPPING_FUNC_DER_Z
		}
		else
		{
			grid_factor = AC_inv_ds_2.z
		}
		res = derzz_stencil(prof)*coordinate_factor*grid_factor
		if(AC_nonequidistant_grid.z)
		{
			 res += AC_MAPPING_FUNC_TILDE_Z*derz(prof)
		}
		return res
	}
}

Stencil derxy_stencil {
    [0][-3][-3]= DERX_3,
    [0][-2][-2]= DERX_2,
    [0][-1][-1]= DERX_1,
    [0][0][0]  = DERX_0,
    [0][1][1]  = DERX_1,
    [0][2][2]  = DERX_2,
    [0][3][3]  = DERX_3,
    [0][-3][3] = -DERX_3,
    [0][-2][2] = -DERX_2,
    [0][-1][1] = -DERX_1,
    [0][1][-1] = -DERX_1,
    [0][2][-2] = -DERX_2,
    [0][3][-3] = -DERX_3
}
derxy(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_CYL_R
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.x)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_X
	}
	else
	{
		grid_factor *= AC_inv_ds.x
	}
	if(AC_nonequidistant_grid.y)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		grid_factor *= AC_inv_ds.y
	}
	return derxy_stencil(f)*coordinate_factor*grid_factor
}
#define deryx derxy

Stencil derxz_stencil {
    [-3][0][-3]  = DERX_3,
    [-2][0][-2]  = DERX_2,
    [-1][0][-1]  = DERX_1,
    [0][0][0]    = DERX_0,
    [1][0][1]    = DERX_1,
    [2][0][2]    = DERX_2,
    [3][0][3]    = DERX_3,
    [-3][0][3]   = -DERX_3,
    [-2][0][2]   = -DERX_2,
    [-1][0][1]   = -DERX_1,
    [1][0][-1]   = -DERX_1,
    [2][0][-2]   = -DERX_2,
    [3][0][-3]   = -DERX_3
}
derxz(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R*AC_INV_SIN_THETA
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.x)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_X
	}
	else
	{
		grid_factor *= AC_inv_ds.x
	}
	if(AC_nonequidistant_grid.z)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		grid_factor *= AC_inv_ds.z
	}
	return derxz_stencil(f)*coordinate_factor*grid_factor
}

#define derzx derxz

Stencil deryz_stencil {
    [-3][-3][0] = DERX_3,
    [-2][-2][0] = DERX_2,
    [-1][-1][0] = DERX_1,
    [0][0][0]   = DERX_0,
    [1][1][0]   = DERX_1,
    [2][2][0]   = DERX_2,
    [3][3][0]   = DERX_3,
    [-3][3][0]  = -DERX_3,
    [-2][2][0]  = -DERX_2,
    [-1][1][0]  = -DERX_1,
    [1][-1][0]  = -DERX_1,
    [2][-2][0]  = -DERX_2,
    [3][-3][0]  = -DERX_3
}
deryz(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R*AC_INV_R*AC_INV_SIN_THETA
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_CYL_R
	}
	grid_factor = 1.0
	if(AC_nonequidistant_grid.y)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		grid_factor *= AC_inv_ds.y
	}
	if(AC_nonequidistant_grid.z)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		grid_factor *= AC_inv_ds.z
	}
	return deryz_stencil(f)*coordinate_factor*grid_factor
}

#define derzy deryz

Stencil der3x {
    [0][0][-3] = - AC_inv_ds_3.x * DER3_3,
    [0][0][-2] = - AC_inv_ds_3.x * DER3_2,
    [0][0][-1] = - AC_inv_ds_3.x * DER3_1,
    [0][0][1]  = AC_inv_ds_3.x * DER3_1,
    [0][0][2]  = AC_inv_ds_3.x * DER3_2,
    [0][0][3]  = AC_inv_ds_3.x * DER3_3
}

Stencil der3y_stencil {
    [0][0][-3] = - AC_inv_ds_3.y * DER3_3,
    [0][0][-2] = - AC_inv_ds_3.y * DER3_2,
    [0][0][-1] = - AC_inv_ds_3.y * DER3_1,
    [0][0][1]  = AC_inv_ds_3.y * DER3_1,
    [0][0][2]  = AC_inv_ds_3.y * DER3_2,
    [0][0][3]  = AC_inv_ds_3.y * DER3_3
}

der3y(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R*AC_INV_R*AC_INV_R
		return der3y_stencil(f)*(factor)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		factor = AC_INV_CYL_R*AC_INV_CYL_R*AC_INV_CYL_R
		return der3y_stencil(f)*(factor)
	}
	return der3y_stencil(f)
}

Stencil der3z_stencil {
    [0][0][-3] = - AC_inv_ds_3.z * DER3_3,
    [0][0][-2] = - AC_inv_ds_3.z * DER3_2,
    [0][0][-1] = - AC_inv_ds_3.z * DER3_1,
    [0][0][1]  = AC_inv_ds_3.z * DER3_1,
    [0][0][2]  = AC_inv_ds_3.z * DER3_2,
    [0][0][3]  = AC_inv_ds_3.z * DER3_3
}

der3z(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R*AC_INV_SIN_THETA
		factor = factor*factor*factor
		return der3z_stencil(f)*(factor)
	}
	return der3z_stencil(f)
}

Stencil der4x {
    [0][0][-3] = AC_inv_ds_4.x * DER4_3,
    [0][0][-2] = AC_inv_ds_4.x * DER4_2,
    [0][0][-1] = AC_inv_ds_4.x * DER4_1,
    [0][0][0]  = AC_inv_ds_4.x * DER4_0,
    [0][0][1]  = AC_inv_ds_4.x * DER4_1,
    [0][0][2]  = AC_inv_ds_4.x * DER4_2,
    [0][0][3]  = AC_inv_ds_4.x * DER4_3
}
Stencil der4y_stencil {
    [0][-3][0] = AC_inv_ds_4.y * DER4_3,
    [0][-2][0] = AC_inv_ds_4.y * DER4_2,
    [0][-1][0] = AC_inv_ds_4.y * DER4_1,
    [0][0][0]  = AC_inv_ds_4.y * DER4_0,
    [0][1][0]  = AC_inv_ds_4.y * DER4_1,
    [0][2][0]  = AC_inv_ds_4.y * DER4_2,
    [0][3][0]  = AC_inv_ds_4.y * DER4_3
}
der4y(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R
		factor *= factor
		factor *= factor
		return der4y_stencil(f)*(factor)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		factor = AC_INV_CYL_R
		factor *= factor
		factor *= factor
		return der4y_stencil(f)*(factor)
	}
	return der4y_stencil(f)
}
Stencil der4z_stencil {
    [-3][0][0] = AC_inv_ds_4.z * DER4_3,
    [-2][0][0] = AC_inv_ds_4.z * DER4_2,
    [-1][0][0] = AC_inv_ds_4.z * DER4_1,
    [0][0][0]  = AC_inv_ds_4.z * DER4_0,
    [1][0][0]  = AC_inv_ds_4.z * DER4_1,
    [2][0][0]  = AC_inv_ds_4.z * DER4_2,
    [3][0][0]  = AC_inv_ds_4.z * DER4_3
}

der4z(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R*AC_INV_SIN_THETA
		factor *= factor
		factor *= factor
		return der4z_stencil(f)*(factor)
	}
	return der4z_stencil(f)
}

der4x2y(Field f)
{
	print("NOT implemented der4x2y\n")
	return 0.0
}
der4x2z(Field f)
{
	print("NOT implemented der4x2z\n")
	return 0.0
}
der4y2x(Field f)
{
	print("NOT implemented der4y2x\n")
	return 0.0
}
der4y2z(Field f)
{
	print("NOT implemented der4y2z\n")
	return 0.0
}
der4z2x(Field f)
{
	print("NOT implemented der4z2x\n")
	return 0.0
}
der4z2y(Field f)
{
	print("NOT implemented der4z2y\n")
	return 0.0
}
der2i2j2k(Field f)
{
	print("NOT implemented der2i2j2k\n")
	return 0.0
}

Stencil der5x {
    [0][0][-3] = -AC_inv_ds_5.x * DER5_3,
    [0][0][-2] = -AC_inv_ds_5.x * DER5_2,
    [0][0][-1] = -AC_inv_ds_5.x * DER5_1,
    [0][0][1]  = AC_inv_ds_5.x * DER5_1,
    [0][0][2]  = AC_inv_ds_5.x * DER5_2,
    [0][0][3]  = AC_inv_ds_5.x * DER5_3
}
Stencil der5y_stencil {
    [0][-3][0] = -AC_inv_ds_5.y * DER5_3,
    [0][-2][0] = -AC_inv_ds_5.y * DER5_2,
    [0][-1][0] = -AC_inv_ds_5.y * DER5_1,
    [0][1][0]  = AC_inv_ds_5.y * DER5_1,
    [0][2][0]  = AC_inv_ds_5.y * DER5_2,
    [0][3][0]  = AC_inv_ds_5.y * DER5_3
}
der5y(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R
		factor *= factor
		factor *= factor
		factor *= AC_INV_R
		return der5y_stencil(f)*(factor)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		factor = AC_INV_CYL_R
		factor *= factor
		factor *= factor
		factor *= AC_INV_CYL_R
		return der5y_stencil(f)*(factor)
	}
	return der5y_stencil(f)
}
Stencil der5z_stencil {
    [-3][0][0] = -AC_inv_ds_5.z * DER5_3,
    [-2][0][0] = -AC_inv_ds_5.z * DER5_2,
    [-1][0][0] = -AC_inv_ds_5.z * DER5_1,
    [1][0][0]  = AC_inv_ds_5.z * DER5_1,
    [2][0][0]  = AC_inv_ds_5.z * DER5_2,
    [3][0][0]  = AC_inv_ds_5.z * DER5_3
}

der5z(Field f)
{
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor = AC_INV_R*AC_INV_SIN_THETA
		factor *= factor
		factor *= factor
		factor *= AC_INV_R*AC_INV_SIN_THETA
		return der5z_stencil(f)*(factor)
	}
	return der5z_stencil(f)
}


Stencil der5z1y {
[1][1][0] =  DER5_1*DER1_1,
[1][-1][0] =  -DER5_1*DER1_1,
[1][2][0] =  DER5_1*DER1_2,
[1][-2][0] =  -DER5_1*DER1_2,
[1][3][0] =  DER5_1*DER1_3,
[1][-3][0] =  -DER5_1*DER1_3,
[-1][1][0] =  -DER5_1*DER1_1,
[-1][-1][0] =  DER5_1*DER1_1,
[-1][2][0] =  -DER5_1*DER1_2,
[-1][-2][0] =  DER5_1*DER1_2,
[-1][3][0] =  -DER5_1*DER1_3,
[-1][-3][0] =  DER5_1*DER1_3,
[2][1][0] =  DER5_2*DER1_1,
[2][-1][0] =  -DER5_2*DER1_1,
[2][2][0] =  DER5_2*DER1_2,
[2][-2][0] =  -DER5_2*DER1_2,
[2][3][0] =  DER5_2*DER1_3,
[2][-3][0] =  -DER5_2*DER1_3,
[-2][1][0] =  -DER5_2*DER1_1,
[-2][-1][0] =  DER5_2*DER1_1,
[-2][2][0] =  -DER5_2*DER1_2,
[-2][-2][0] =  DER5_2*DER1_2,
[-2][3][0] =  -DER5_2*DER1_3,
[-2][-3][0] =  DER5_2*DER1_3,
[3][1][0] =  DER5_3*DER1_1,
[3][-1][0] =  -DER5_3*DER1_1,
[3][2][0] =  DER5_3*DER1_2,
[3][-2][0] =  -DER5_3*DER1_2,
[3][3][0] =  DER5_3*DER1_3,
[3][-3][0] =  -DER5_3*DER1_3,
[-3][1][0] =  -DER5_3*DER1_1,
[-3][-1][0] =  DER5_3*DER1_1,
[-3][2][0] =  -DER5_3*DER1_2,
[-3][-2][0] =  DER5_3*DER1_2,
[-3][3][0] =  -DER5_3*DER1_3,
[-3][-3][0] =  DER5_3*DER1_3,
}


Stencil der5z1x {
[1][0][1] =  DER5_1*DER1_1,
[1][0][-1] =  -DER5_1*DER1_1,
[1][0][2] =  DER5_1*DER1_2,
[1][0][-2] =  -DER5_1*DER1_2,
[1][0][3] =  DER5_1*DER1_3,
[1][0][-3] =  -DER5_1*DER1_3,
[-1][0][1] =  -DER5_1*DER1_1,
[-1][0][-1] =  DER5_1*DER1_1,
[-1][0][2] =  -DER5_1*DER1_2,
[-1][0][-2] =  DER5_1*DER1_2,
[-1][0][3] =  -DER5_1*DER1_3,
[-1][0][-3] =  DER5_1*DER1_3,
[2][0][1] =  DER5_2*DER1_1,
[2][0][-1] =  -DER5_2*DER1_1,
[2][0][2] =  DER5_2*DER1_2,
[2][0][-2] =  -DER5_2*DER1_2,
[2][0][3] =  DER5_2*DER1_3,
[2][0][-3] =  -DER5_2*DER1_3,
[-2][0][1] =  -DER5_2*DER1_1,
[-2][0][-1] =  DER5_2*DER1_1,
[-2][0][2] =  -DER5_2*DER1_2,
[-2][0][-2] =  DER5_2*DER1_2,
[-2][0][3] =  -DER5_2*DER1_3,
[-2][0][-3] =  DER5_2*DER1_3,
[3][0][1] =  DER5_3*DER1_1,
[3][0][-1] =  -DER5_3*DER1_1,
[3][0][2] =  DER5_3*DER1_2,
[3][0][-2] =  -DER5_3*DER1_2,
[3][0][3] =  DER5_3*DER1_3,
[3][0][-3] =  -DER5_3*DER1_3,
[-3][0][1] =  -DER5_3*DER1_1,
[-3][0][-1] =  DER5_3*DER1_1,
[-3][0][2] =  -DER5_3*DER1_2,
[-3][0][-2] =  DER5_3*DER1_2,
[-3][0][3] =  -DER5_3*DER1_3,
[-3][0][-3] =  DER5_3*DER1_3,
}


Stencil der5y1z {
[1][1][0] =  DER5_1*DER1_1,
[-1][1][0] =  -DER5_1*DER1_1,
[2][1][0] =  DER5_1*DER1_2,
[-2][1][0] =  -DER5_1*DER1_2,
[3][1][0] =  DER5_1*DER1_3,
[-3][1][0] =  -DER5_1*DER1_3,
[1][-1][0] =  -DER5_1*DER1_1,
[-1][-1][0] =  DER5_1*DER1_1,
[2][-1][0] =  -DER5_1*DER1_2,
[-2][-1][0] =  DER5_1*DER1_2,
[3][-1][0] =  -DER5_1*DER1_3,
[-3][-1][0] =  DER5_1*DER1_3,
[1][2][0] =  DER5_2*DER1_1,
[-1][2][0] =  -DER5_2*DER1_1,
[2][2][0] =  DER5_2*DER1_2,
[-2][2][0] =  -DER5_2*DER1_2,
[3][2][0] =  DER5_2*DER1_3,
[-3][2][0] =  -DER5_2*DER1_3,
[1][-2][0] =  -DER5_2*DER1_1,
[-1][-2][0] =  DER5_2*DER1_1,
[2][-2][0] =  -DER5_2*DER1_2,
[-2][-2][0] =  DER5_2*DER1_2,
[3][-2][0] =  -DER5_2*DER1_3,
[-3][-2][0] =  DER5_2*DER1_3,
[1][3][0] =  DER5_3*DER1_1,
[-1][3][0] =  -DER5_3*DER1_1,
[2][3][0] =  DER5_3*DER1_2,
[-2][3][0] =  -DER5_3*DER1_2,
[3][3][0] =  DER5_3*DER1_3,
[-3][3][0] =  -DER5_3*DER1_3,
[1][-3][0] =  -DER5_3*DER1_1,
[-1][-3][0] =  DER5_3*DER1_1,
[2][-3][0] =  -DER5_3*DER1_2,
[-2][-3][0] =  DER5_3*DER1_2,
[3][-3][0] =  -DER5_3*DER1_3,
[-3][-3][0] =  DER5_3*DER1_3,
}


Stencil der5y1x {
[0][1][1] =  DER5_1*DER1_1,
[0][1][-1] =  -DER5_1*DER1_1,
[0][1][2] =  DER5_1*DER1_2,
[0][1][-2] =  -DER5_1*DER1_2,
[0][1][3] =  DER5_1*DER1_3,
[0][1][-3] =  -DER5_1*DER1_3,
[0][-1][1] =  -DER5_1*DER1_1,
[0][-1][-1] =  DER5_1*DER1_1,
[0][-1][2] =  -DER5_1*DER1_2,
[0][-1][-2] =  DER5_1*DER1_2,
[0][-1][3] =  -DER5_1*DER1_3,
[0][-1][-3] =  DER5_1*DER1_3,
[0][2][1] =  DER5_2*DER1_1,
[0][2][-1] =  -DER5_2*DER1_1,
[0][2][2] =  DER5_2*DER1_2,
[0][2][-2] =  -DER5_2*DER1_2,
[0][2][3] =  DER5_2*DER1_3,
[0][2][-3] =  -DER5_2*DER1_3,
[0][-2][1] =  -DER5_2*DER1_1,
[0][-2][-1] =  DER5_2*DER1_1,
[0][-2][2] =  -DER5_2*DER1_2,
[0][-2][-2] =  DER5_2*DER1_2,
[0][-2][3] =  -DER5_2*DER1_3,
[0][-2][-3] =  DER5_2*DER1_3,
[0][3][1] =  DER5_3*DER1_1,
[0][3][-1] =  -DER5_3*DER1_1,
[0][3][2] =  DER5_3*DER1_2,
[0][3][-2] =  -DER5_3*DER1_2,
[0][3][3] =  DER5_3*DER1_3,
[0][3][-3] =  -DER5_3*DER1_3,
[0][-3][1] =  -DER5_3*DER1_1,
[0][-3][-1] =  DER5_3*DER1_1,
[0][-3][2] =  -DER5_3*DER1_2,
[0][-3][-2] =  DER5_3*DER1_2,
[0][-3][3] =  -DER5_3*DER1_3,
[0][-3][-3] =  DER5_3*DER1_3,
}


Stencil der5x1z {
[1][0][1] =  DER5_1*DER1_1,
[-1][0][1] =  -DER5_1*DER1_1,
[2][0][1] =  DER5_1*DER1_2,
[-2][0][1] =  -DER5_1*DER1_2,
[3][0][1] =  DER5_1*DER1_3,
[-3][0][1] =  -DER5_1*DER1_3,
[1][0][-1] =  -DER5_1*DER1_1,
[-1][0][-1] =  DER5_1*DER1_1,
[2][0][-1] =  -DER5_1*DER1_2,
[-2][0][-1] =  DER5_1*DER1_2,
[3][0][-1] =  -DER5_1*DER1_3,
[-3][0][-1] =  DER5_1*DER1_3,
[1][0][2] =  DER5_2*DER1_1,
[-1][0][2] =  -DER5_2*DER1_1,
[2][0][2] =  DER5_2*DER1_2,
[-2][0][2] =  -DER5_2*DER1_2,
[3][0][2] =  DER5_2*DER1_3,
[-3][0][2] =  -DER5_2*DER1_3,
[1][0][-2] =  -DER5_2*DER1_1,
[-1][0][-2] =  DER5_2*DER1_1,
[2][0][-2] =  -DER5_2*DER1_2,
[-2][0][-2] =  DER5_2*DER1_2,
[3][0][-2] =  -DER5_2*DER1_3,
[-3][0][-2] =  DER5_2*DER1_3,
[1][0][3] =  DER5_3*DER1_1,
[-1][0][3] =  -DER5_3*DER1_1,
[2][0][3] =  DER5_3*DER1_2,
[-2][0][3] =  -DER5_3*DER1_2,
[3][0][3] =  DER5_3*DER1_3,
[-3][0][3] =  -DER5_3*DER1_3,
[1][0][-3] =  -DER5_3*DER1_1,
[-1][0][-3] =  DER5_3*DER1_1,
[2][0][-3] =  -DER5_3*DER1_2,
[-2][0][-3] =  DER5_3*DER1_2,
[3][0][-3] =  -DER5_3*DER1_3,
[-3][0][-3] =  DER5_3*DER1_3,
}


Stencil der5x1y {
[0][1][1] =  DER5_1*DER1_1,
[0][-1][1] =  -DER5_1*DER1_1,
[0][2][1] =  DER5_1*DER1_2,
[0][-2][1] =  -DER5_1*DER1_2,
[0][3][1] =  DER5_1*DER1_3,
[0][-3][1] =  -DER5_1*DER1_3,
[0][1][-1] =  -DER5_1*DER1_1,
[0][-1][-1] =  DER5_1*DER1_1,
[0][2][-1] =  -DER5_1*DER1_2,
[0][-2][-1] =  DER5_1*DER1_2,
[0][3][-1] =  -DER5_1*DER1_3,
[0][-3][-1] =  DER5_1*DER1_3,
[0][1][2] =  DER5_2*DER1_1,
[0][-1][2] =  -DER5_2*DER1_1,
[0][2][2] =  DER5_2*DER1_2,
[0][-2][2] =  -DER5_2*DER1_2,
[0][3][2] =  DER5_2*DER1_3,
[0][-3][2] =  -DER5_2*DER1_3,
[0][1][-2] =  -DER5_2*DER1_1,
[0][-1][-2] =  DER5_2*DER1_1,
[0][2][-2] =  -DER5_2*DER1_2,
[0][-2][-2] =  DER5_2*DER1_2,
[0][3][-2] =  -DER5_2*DER1_3,
[0][-3][-2] =  DER5_2*DER1_3,
[0][1][3] =  DER5_3*DER1_1,
[0][-1][3] =  -DER5_3*DER1_1,
[0][2][3] =  DER5_3*DER1_2,
[0][-2][3] =  -DER5_3*DER1_2,
[0][3][3] =  DER5_3*DER1_3,
[0][-3][3] =  -DER5_3*DER1_3,
[0][1][-3] =  -DER5_3*DER1_1,
[0][-1][-3] =  DER5_3*DER1_1,
[0][2][-3] =  -DER5_3*DER1_2,
[0][-2][-3] =  DER5_3*DER1_2,
[0][3][-3] =  -DER5_3*DER1_3,
[0][-3][-3] =  DER5_3*DER1_3,
}





//TP: corresponds to der6_main
Stencil der6x_stencil {
    [0][0][-3] = DER6_3,
    [0][0][-2] = DER6_2,
    [0][0][-1] = DER6_1,
    [0][0][0]  = DER6_0,
    [0][0][1]  = DER6_1,
    [0][0][2]  = DER6_2,
    [0][0][3]  = DER6_3
}
der6x(Field f)
{
	return der6x_stencil(f)*AC_inv_ds_6.x
}
Stencil der6y_stencil {
    [0][-3][0] = DER6_3,
    [0][-2][0] = DER6_2,
    [0][-1][0] = DER6_1,
    [0][0][0]  = DER6_0,
    [0][1][0]  = DER6_1,
    [0][2][0]  = DER6_2,
    [0][3][0]  = DER6_3
}
Stencil der6z_stencil {
    [-3][0][0] = DER6_3,
    [-2][0][0] = DER6_2,
    [-1][0][0] = DER6_1,
    [0][0][0]  = DER6_0,
    [1][0][0]  = DER6_1,
    [2][0][0]  = DER6_2,
    [3][0][0]  = DER6_3
}
der6y(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor_1 = AC_INV_R
		factor_2 = factor_1*factor_1
		factor_4 = factor_2*factor_2
		coordinate_factor = factor_4*factor_2
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		factor_1 = AC_INV_CYL_R
		factor_2 = factor_1*factor_1
		factor_4 = factor_2*factor_2
		coordinate_factor = factor_4*factor_2
	}
	return der6y_stencil(f)*coordinate_factor*AC_inv_ds_6.y
}

der6z(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		factor_1 = AC_INV_R*AC_INV_SIN_THETA
		factor_2 = factor_1*factor_1
		factor_4 = factor_2*factor_2
		coordinate_factor = factor_4*factor_2

	}
	return der6z_stencil(f)*coordinate_factor*AC_inv_ds_6.z
}

der6x_upwd(Field f)
{
	grid_factor = (1.0/60.0)
	if(AC_nonequidistant_grid.x)
	{
		grid_factor = AC_INV_MAPPING_FUNC_DER_X
	}
	else
	{
		grid_factor *= AC_inv_ds.x
	}
	return der6x_stencil(f)*grid_factor
}
der6y_upwd(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R

	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_CYL_R

	}
	grid_factor = (1.0/60.0)
	if(AC_nonequidistant_grid.y)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Y
	}
	else
	{
		grid_factor *= AC_inv_ds.y
	}
	return der6y_stencil(f)*coordinate_factor*grid_factor
}

der6z_upwd(Field f)
{
	coordinate_factor = 1.0
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		coordinate_factor = AC_INV_R*AC_INV_SIN_THETA
	}
	grid_factor = (1.0/60.0)
	if(AC_nonequidistant_grid.z)
	{
		grid_factor *= AC_INV_MAPPING_FUNC_DER_Z
	}
	else
	{
		grid_factor *= AC_inv_ds.z
	}
	return der6z_stencil(f)*coordinate_factor*grid_factor
}


der6x_ignore_spacing(Field f)
{
	return der6x_stencil(f)
}
der6y_ignore_spacing(Field f)
{
	return der6y_stencil(f)
}
der6z_ignore_spacing(Field f)
{
	return der6z_stencil(f)
}
#define DER_UPWD_3 (+2.0/6.0)
#define DER_UPWD_2 (-9.0/6.0)
#define DER_UPWD_1 (+18.0/6.0)
#define DER_UPWD_0 (+18.0/6.0)

Stencil derx_upwind_left
{
    [0][0][-3] = -DER_UPWD_3,
    [0][0][-2] = -DER_UPWD_2,
    [0][0][-1] = -DER_UPWD_1,
    [0][0][ 0] = -DER_UPWD_0 
}

Stencil derx_upwind_right
{
    [0][0][ 3] = +DER_UPWD_3,
    [0][0][ 2] = +DER_UPWD_2,
    [0][0][ 1] = +DER_UPWD_1,
    [0][0][ 0] = +DER_UPWD_0 
}

Stencil dery_upwind_down
{
    [0][-3][0] = -DER_UPWD_3,
    [0][-2][0] = -DER_UPWD_2,
    [0][-1][0] = -DER_UPWD_1,
    [0][ 0][0] = -DER_UPWD_0 
}

Stencil dery_upwind_up
{
    [0][ 3][0] = +DER_UPWD_3,
    [0][ 2][0] = +DER_UPWD_2,
    [0][ 1][0] = +DER_UPWD_1,
    [0][ 0][0] = +DER_UPWD_0 
}

Stencil derz_upwind_back
{
    [-3][0][0] = -DER_UPWD_3,
    [-2][0][0] = -DER_UPWD_2,
    [-1][0][0] = -DER_UPWD_1,
    [ 0][0][0] = -DER_UPWD_0 
}

Stencil derz_upwind_front
{
    [ 3][0][0] = +DER_UPWD_3,
    [ 2][0][0] = +DER_UPWD_2,
    [ 1][0][0] = +DER_UPWD_1,
    [ 0][0][0] = +DER_UPWD_0 
}

derx_upwind(Field f, real vec)
{
	//TP: to get both stencil correctly generated we compute both always
	left = derx_upwind_left(f)
	right = derx_upwind_right(f)
	if(vec.x > 0.0) left
	return right
}

dery_upwind(Field f, real vec)
{
	//TP: to get both stencil correctly generated we compute both always
	down = dery_upwind_down(f)
	up   = dery_upwind_up(f)
	if(vec.y > 0.0) down 
	return up
}

derz_upwind(Field f, real vec)
{
	//TP: to get both stencil correctly generated we compute both always
	back  = derz_upwind_back(f)
	front = derz_upwind_front(f)
	if(vec.z > 0.0) back
	return front
}

//derx(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.x*f[vertexIdx.x-3][vertexIdx.y][vertexIdx.z];
//	res += DER1_3*+AC_inv_ds.x*f[vertexIdx.x+3][vertexIdx.y][vertexIdx.z];
//	res += DER1_2*-AC_inv_ds.x*f[vertexIdx.x-2][vertexIdx.y][vertexIdx.z];
//	res += DER1_2*+AC_inv_ds.x*f[vertexIdx.x+2][vertexIdx.y][vertexIdx.z];
//	res += DER1_1*-AC_inv_ds.x*f[vertexIdx.x-1][vertexIdx.y][vertexIdx.z];
//	res += DER1_1*+AC_inv_ds.x*f[vertexIdx.x+1][vertexIdx.y][vertexIdx.z];
//	return res;
//}
//
//derxx(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.x*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.x*f[vertexIdx.x-3][vertexIdx.y][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.x*f[vertexIdx.x+3][vertexIdx.y][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.x*f[vertexIdx.x-2][vertexIdx.y][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.x*f[vertexIdx.x+2][vertexIdx.y][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.x*f[vertexIdx.x-1][vertexIdx.y][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.x*f[vertexIdx.x+1][vertexIdx.y][vertexIdx.z];
//	return res;
//}
//dery(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-3][vertexIdx.z];
//	res += DER1_3*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+3][vertexIdx.z];
//	res += DER1_2*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-2][vertexIdx.z];
//	res += DER1_2*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+2][vertexIdx.z];
//	res += DER1_1*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-1][vertexIdx.z];
//	res += DER1_1*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+1][vertexIdx.z];
//	return res;
//}
//
//deryy(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res +=  DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-3][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+3][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-2][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+2][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-1][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+1][vertexIdx.z];
//	return res;
//}
//derz(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-3];
//	res += DER1_3*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+3];
//	res += DER1_2*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-2];
//	res += DER1_2*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+2];
//	res += DER1_1*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-1];
//	res += DER1_1*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+1];
//	return res;
//}
//derzz(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res +=  DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-3];
//	res += DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+3];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-2];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+2];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-1];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+1];
//	return res;
//}



