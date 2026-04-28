#ifndef AC_GENERAL_OPERATORS_H
#define AC_GENERAL_OPERATORS_H
/**
 * Calculates u·(∇f),
 * where f and u are vector fields
 * and m = ∇f.
 * f is needed for corrections for curvilinear coordinates.
 */
u_dot_grad(Field3 f, Matrix m,real3 u){
	suppress_unused_warning(f)
	real3 res = m*u
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		res.x = res.x - AC_INV_R*(u.y*f.y+u.z*f.z)
		res.y = res.y + AC_INV_R*(u.y*f.x-u.z*f.z*AC_COT)
		res.z = res.z + AC_INV_R*(u.z*f.x+u.z*f.y*AC_COT)
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res.x = res.x - AC_INV_CYL_R*(u.y*f.y)
		res.y = res.y + AC_INV_CYL_R*(u.y*f.x)
	}
	return res;
}

/**
 * Calculates u·(∇f),
 * where f and u are vector fields and where m = ∇f.
 * Works only in Cartesian.
 */
u_dot_grad(Matrix m,real3 v){
  if(AC_coordinate_system == AC_SPHERICAL_COORDINATES || AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
  	fatal_error_message(true,"u_dot_grad requires the vector field as an extra first parameter in spherical coordinates!!\n");
  }
  return real3(dot(v,m.col(0)),dot(v,m.col(1)),dot(v,m.col(2)))
}


/**
 * Calculates u·(∇f),
 * with a special implementation based on the advection type
 * where u is a vector field and f is a scalar field and gradf = ∇f.
 * At the moment only normal advection implemented.
 */
u_dot_grad_alt(Field f,real3 gradf,real3 u,int advec_type){
	suppress_unused_warning(f)
	if(advec_type == 0)
	{
		return dot(u,gradf)
	}
	fatal_error_message(true,"u_dot_grad_alt: Upwinding and Kurganov-Tadmor not yet implemented")
	return 0.0
}

/**
 * Calculates u·(∇M),
 * where u is a vector field, M is 3x3 tensor field and c = ∇M.
 * (TODO: remove k since not used)
 */
u_dot_grad_mat(k,AcTensor c,real3 uu)
{
	suppress_unused_warning(k)
        Matrix res
        res[0][0] = c[0][0][0]*uu.x + c[0][0][1]*uu.y + c[0][0][2]*uu.z
        res[0][1] = c[0][1][0]*uu.x + c[0][1][1]*uu.y + c[0][1][2]*uu.z
        res[0][2] = c[0][2][0]*uu.x + c[0][2][1]*uu.y + c[0][2][2]*uu.z

        res[1][0] = c[1][0][0]*uu.x + c[1][0][1]*uu.y + c[1][0][2]*uu.z
        res[1][1] = c[1][1][0]*uu.x + c[1][1][1]*uu.y + c[1][1][2]*uu.z
        res[1][2] = c[1][2][0]*uu.x + c[1][2][1]*uu.y + c[1][2][2]*uu.z

        res[2][0] = c[2][0][0]*uu.x + c[2][0][1]*uu.y + c[2][0][2]*uu.z
        res[2][1] = c[2][1][0]*uu.x + c[2][1][1]*uu.y + c[2][1][2]*uu.z
        res[2][2] = c[2][2][0]*uu.x + c[2][2][1]*uu.y + c[2][2][2]*uu.z
        return res
}

/**
 * Calculates u·(∇M),
 * where u is a vector field, M is 3x3 tensor field and c = ∇M with an upwind correction.
 * Not implemented
 */
u_dot_grad_mat_upwd(k,c,uu)
{
	fatal_error_message(true,"u_dot_grad_mat_upwd: Not implemented")
	Matrix res
	return res
}

/**
 * Calculates ∇f
 * ,where f is a scalar field
 */
gradient(Field s) {
    return real3(derx(s), dery(s), derz(s))
}
/**
 * Calculates ∇v
 * ,where v is a vector field
 */
gradient_tensor(Field3 v) {
	return Matrix(
			gradient(v.x),
			gradient(v.y),
			gradient(v.z)
		     )
}
gradient_tensor(Field3 v, Field s)
{
     grads = gradient(s)
     vij   = gradient_tensor(v)
     Matrix gij
     gij[0][0] = vij[0][0]*s + u.x*grads.x
     gij[0][1] = vij[0][1]*s + u.x*grads.y
     gij[0][2] = vij[0][2]*s + u.x*grads.z

     gij[1][0] = vij[1][0]*s + u.y*grads.x
     gij[1][1] = vij[1][1]*s + u.y*grads.y
     gij[1][2] = vij[1][2]*s + u.y*grads.z

     gij[2][0] = vij[2][0]*s + u.z*grads.x
     gij[2][1] = vij[2][1]*s + u.z*grads.y
     gij[2][2] = vij[2][2]*s + u.z*grads.z

     return gij
}


/**
 * Calculates c(∇^k)v, to achieve the effect of upwinding.
 * ,where v is a vector field.
 * c and k depend on the order of stencil used for the advection term
 * where upwinding is to be applied.
 */
elemental gradient_upwd(Field s) {
    return real3(derx_upwd(s), dery_upwd(s), derz_upwd(s))
}



/**
 * Calculates g,
 * where g_i = (∂^2/∂_i)s
 */
elemental gradient2(Field s) {
    return real3(derxx(s), deryy(s), derzz(s))
}

/**
elemental gradient3(Field s) {
    return real3(derx(s), dery(s), derz(s))
}

elemental gradient4(Field s) {
    return real3(derx(s), dery(s), derz(s))
}
**/

/**
 * Calculates g,
 * where g_i = (∂^5/∂_i)s
 */
elemental gradient5(Field s) {
    return real3(der5x(s), der5y(s), der5z(s))
}


/**
 * Calculates g,
 * where g_i = c(∂^k/∂_i)s and c and k are chosen to
 * achieve the effect of upwinding for the stencil employed for the advective term.
 */
elemental gradient6_upwd(s) {
    return real3(derx_upwd(s), dery_upwd(s), derz_upwd(s))
}


/**
 * Calculates ∇·v
 */
divergence(Field3 v) {
    g = derx(v.x) + dery(v.y) + derz(v.z)


    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	   g += AC_INV_R*(v.x*2.0 + AC_COT*v.y);
    }
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    g +=AC_INV_CYL_R*v.x
    }
    return g
}

/**
 * Calculates ∇^2·v
 */
divergence_2nd(Field3 v) {
    g = derx_2nd(v.x) + dery_2nd(v.y) + derz_2nd(v.z)


    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	   g += AC_INV_R*(v.x*2.0 + AC_COT*v.y);
    }
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    g +=AC_INV_CYL_R*v.x
    }
    return g
}
/**
 * Calculates (∇^2·v),
 * where m = ∇(v).
 * Only works in Cartesian
 */
divergence(Matrix m)
{
	b = m[0][0] + m[1][1] + m[2][2];
        if(AC_coordinate_system == AC_SPHERICAL_COORDINATES || AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
        {
        	fatal_error_message(true,"divergence requires the vector field itself to perform corrections for curvilinear coordinates!\ņ");
        }
	return b;
}

/**
 * Calculates (∇^2·a),
 * where m = ∇(v).
 */
divergence(Matrix m, real3 a)
{
	suppress_unused_warning(a)
	b = m[0][0] + m[1][1] + m[2][2];
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		b += AC_INV_R*(a.x*2.0 + AC_COT*a.y)
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
	    b += AC_INV_CYL_R*a.x
	}
		
	return b;
}

/**
 * Computes ∇ × v
 */
curl(Field3 v) {
    g = real3(dery(v.z) - derz(v.y), derz(v.x) - derx(v.z), derx(v.y) - dery(v.x))
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    g.x += v.z*AC_INV_R*AC_COT
	    g.y -= v.z*AC_INV_R
	    g.z += v.y*AC_INV_R
    }
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    g.z += v.y*AC_INV_CYL_R
    }
    return g
}

/**
 * Computes ∇ × v,
 * where m = ∇v.
 * Works only in Cartesian, see the other version for more general.
 */
curl(Matrix m) {
  if(AC_coordinate_system != AC_CARTESIAN_COORDINATES)
  {
	  fatal_error_message(true,"curl with Matrix only is incorrect for non-cartesian coordinates!\n")
  }
  return real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
}

/**
 * Computes ∇ × v,
 * where m = ∇v.
 */
curl(Matrix m, real3 v) {
  suppress_unused_warning(v)
  g = real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
  if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
  {
          g.x += v.z*AC_INV_R*AC_COT
          g.y -= v.z*AC_INV_R
          g.z += v.y*AC_INV_R
  }
  if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
          g.z += v.y*AC_INV_CYL_R
  }
  return g
}

/**
 * Computes ∇ × v,
 * where m = ∇v.
 * Works only in Cartesian.
 * (TODO: why is this needed isn't the normal curl already covariant?)
 */
covariant_curl(Matrix m, real3 v)
{
  return real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
}


/**
 * Computes the Hessian of v,
 * where v is a scalar field.
 */
hessian(Field v)
{
	return Matrix(
			real3(derxx(v), derxy(v), derxz(v)),
			real3(derxy(v), deryy(v), deryz(v)),
			real3(derxz(v), deryz(v), derzz(v))
		     )
}

/**
 * Computes a 3x3x3 tensor T,
 * where T_ijk = (∂^2/∂_j∂_k)v_i and v is a vector field.
 */
del2fi_dxjk(Field3 v)
{
	return Tensor(
			hessian(v.x),
			hessian(v.y),
			hessian(v.z)
		     )
}
/**
 * Computes a 3x3x3 a contravariant tensor T,
 * where T_ijk = (∇j∇k)v_i,
 * m = ∇v (not contravariant) and v is a vector field.
 */
get_d2A(Field3 v, Matrix m)
{
	Tensor res = del2fi_dxjk(v)
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		res[1][0][0] -= m[0][1]*AC_INV_R
		res[1][0][1] -= m[1][1]*AC_INV_R
		res[1][0][2] -= m[2][1]*AC_INV_R

		res[2][0][0] -= m[0][2]*AC_INV_R
		res[2][0][1] -= m[1][2]*AC_INV_R
		res[2][0][2] -= m[2][2]*AC_INV_R

		res[2][1][0] -= m[0][2]*AC_INV_R*AC_COT
		res[2][1][1] -= m[1][2]*AC_INV_R*AC_COT
		res[2][1][2] -= m[2][2]*AC_INV_R*AC_COT
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res[2-1][1-1][0] -= m[0][1]*AC_INV_CYL_R
		res[2-1][1-1][1] -= m[1][1]*AC_INV_CYL_R
		res[2-1][1-1][2] -= m[2][1]*AC_INV_CYL_R
	}

	return res
}
/**
 * Computes a 3x3x3 a contravariant tensor T,
 * where T_ijk = (∇j∇k)v_i and v is a vector field.
 */
get_d2A(Field3 v)
{
	Tensor res = del2fi_dxjk(v)
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		res[2-1][1-1][0] -= dery(v.x)*AC_INV_R
		res[2-1][1-1][1] -= dery(v.y)*AC_INV_R
		res[2-1][1-1][2] -= dery(v.z)*AC_INV_R

		res[3-1][1-1][0] -= derz(v.x)*AC_INV_R
		res[3-1][1-1][1] -= derz(v.y)*AC_INV_R
		res[3-1][1-1][2] -= derz(v.z)*AC_INV_R

		res[3-1][2-1][0] -= derz(v.x)*AC_INV_R*AC_COT
		res[3-1][2-1][1] -= derz(v.y)*AC_INV_R*AC_COT
		res[3-1][2-1][2] -= derz(v.z)*AC_INV_R*AC_COT
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res[2-1][1-1][0] -= dery(v.x)*AC_INV_CYL_R
		res[2-1][1-1][1] -= dery(v.y)*AC_INV_CYL_R
		res[2-1][1-1][2] -= dery(v.z)*AC_INV_CYL_R
	}

	return res
}

/**
 * Computes the jacobian of b,
 * where b = ∇ × v
 */
bij(Field3 v)
{
    	d2A = get_d2A(v)

	Matrix res;
	res[0][0] = d2A[1][0][2] - d2A[2][0][1] 
	res[0][1] = d2A[1][1][2] - d2A[2][1][1] 
	res[0][2] = d2A[1][2][2] - d2A[2][2][1] 

	res[1][0] = d2A[2][0][0] - d2A[0][0][2] 
	res[1][1] = d2A[2][1][0] - d2A[0][1][2] 
	res[1][2] = d2A[2][2][0] - d2A[0][2][2] 

	res[2][0] = d2A[0][0][1] - d2A[1][0][0] 
	res[2][1] = d2A[0][1][1] - d2A[1][1][0] 
	res[2][2] = d2A[0][2][1] - d2A[1][2][0] 

	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
 	  res[3-1][2-1]=res[3-1][2-1]+dery(v.y)*AC_INV_R
          res[2-1][3-1]=res[2-1][3-1]-derz(v.z)*AC_INV_R
          res[1-1][3-1]=res[1-1][3-1]+derz(v.z)*AC_INV_R*AC_COT
          res[3-1][1-1]=res[3-1][1-1]+derx(v.y)*AC_INV_R-v.y*(AC_INV_R*AC_INV_R)
          res[2-1][1-1]=res[2-1][1-1]-derx(v.z)*AC_INV_R+v.z*(AC_INV_R*AC_INV_R)
          res[1-1][2-1]=res[1-1][2-1]+dery(v.z)*AC_INV_R*AC_COT-v.z*(AC_INV_R*AC_INV_R)*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res[3-1][2-1]=res[3-1][2-1]+ dery(v.y)*AC_INV_CYL_R
          	res[3-1][1-1]=res[3-1][1-1]+ derx(v.y)*AC_INV_CYL_R-v.y*(AC_INV_CYL_R*AC_INV_CYL_R)
	}


	return res;
}


/**
 * Computes (∇^4)s,
 * where s is a scalar field. 
 */
elemental del4(Field s) {
  return der4x(s) + der4y(s) + der4z(s)
}

/**
 * Computes (∇^6)s,
 * where s is a scalar field. 
 */
elemental del6(Field s) {
  return der6x(s) + der6y(s) + der6z(s)
}


//TP: old analytical expressions
//    easier and most likely more performant to simply calculate the exponentials together with the stencil ops
/**
//TP: these are mainly for testing
//    if one would actually want production performance for equidistant cartesian with this
//    then one would combine all the coefficients off the different derivative
//    stencils into one stencil and use that
//    expression generated by Sympy
der6x_exp(Field f) {
	return exp(f)*
		(
		 	+derx(f)*derx(f)*derx(f)*derx(f)*derx(f)*derx(f)
			+10*der3x(f)*der3x(f)
			+15*derxx(f)*derxx(f)*derxx(f)
			+6*derx(f)*der5x(f)
			+15*derx(f)*derx(f)*der4x(f)
			+15*derx(f)*derx(f)*derx(f)*derx(f)*derxx(f)
			+15*derxx(f)*der4x(f)
			+20*derx(f)*derx(f)*derx(f)*der3x(f)
			+45*derx(f)*derx(f)*derxx(f)*derxx(f)
			+60*derx(f)*derxx(f)*der3x(f)
			+der6x(f)
		)

}
der6y_exp(Field f) {
	return exp(f)*
		(
		 	+dery(f)*dery(f)*dery(f)*dery(f)*dery(f)*dery(f)
			+10*der3y(f)*der3y(f)
			+15*deryy(f)*deryy(f)*deryy(f)
			+6*dery(f)*der5y(f)
			+15*dery(f)*dery(f)*der4y(f)
			+15*dery(f)*dery(f)*dery(f)*dery(f)*deryy(f)
			+15*deryy(f)*der4y(f)
			+20*dery(f)*dery(f)*dery(f)*der3y(f)
			+45*dery(f)*dery(f)*deryy(f)*deryy(f)
			+60*dery(f)*deryy(f)*der3y(f)
			+der6y(f)
		)

}
der6z_exp(Field f) {
	return exp(f)*
		(
		 	+derz(f)*derz(f)*derz(f)*derz(f)*derz(f)*derz(f)
			+10*der3z(f)*der3z(f)
			+15*derzz(f)*derzz(f)*derzz(f)
			+6*derz(f)*der5z(f)
			+15*derz(f)*derz(f)*der4z(f)
			+15*derz(f)*derz(f)*derz(f)*derz(f)*derzz(f)
			+15*derzz(f)*der4z(f)
			+20*derz(f)*derz(f)*derz(f)*der3z(f)
			+45*derz(f)*derz(f)*derzz(f)*derzz(f)
			+60*derz(f)*derzz(f)*der3z(f)
			+der6z(f)
		)

}
**/

/**
 * Computes (∇^6)exp(s),
 * where s is a scalar field. 
 */
elemental del6_exp(Field f)
{
	return der6x_exp(f) + der6y_exp(f) + der6z_exp(f)
}

/**
 * Computes (∇^6)(s),
 * where s is a scalar field and one of the directions is masked out
 */
del6_masked(Field s, int mask)
{
	x = mask == 1 ? 0.0 : der6x(s)
	y = mask == 2 ? 0.0 : der6y(s)
	z = mask == 3 ? 0.0 : der6z(s)
	return x + y + z
}

/**
 * Computes c(∇^k)(s),
 * where c and k are chosen to achieve upwinding and depend on the order of the advective term
 * s is a scalar field and one of the directions is masked out
 */
del_upwd_masked(real3 velo, Field s, int mask)
{
        x = mask == 1 ? 0.0 : abs(velo.x)*derx_upwd(s)
        y = mask == 2 ? 0.0 : abs(velo.y)*dery_upwd(s)
        z = mask == 3 ? 0.0 : abs(velo.z)*derz_upwd(s)
        return x + y + z
}

/**
 * Computes (∇^6)(s) in a 'strict' manner.
 * Not implemented
 */
elemental del6_strict(Field s) {
	suppress_unused_warning(s)
	fatal_error_message(true,"NOT IMPLEMENTED del6_strict\n")
	return 0.
}

/**
 * Calculates u·(∇f) in a upwinded manner,
 * where f is a scalar field and u is a vector field
 */
elemental ugrad_upw(Field field, real3 velo){

        return dot(velo,gradient(field)) - dot(abs(velo),gradient_upwd(field))
}

/**
 * Calculates u·(∇f) in a upwinded manner,
 * where f and u are vector fields.
 */
elemental ugrad_upw(Field3 field, real3 velo){

        return real3( dot(velo,gradient(field.x)) - dot(abs(velo),gradient_upwd(field.x)),
		      dot(velo,gradient(field.y)) - dot(abs(velo),gradient_upwd(field.y)),
		      dot(velo,gradient(field.z)) - dot(abs(velo),gradient_upwd(field.z)))
}

/**
 * Calculates the upwind correction to u·(∇f),
 * where u is a vector and f a  scalar field.
 */
del_upwd(real3 velo,Field field)
{

	real3 res = abs(velo)*gradient_upwd(field)
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		res.y *= AC_INV_R
		res.z *= AC_INV_R*AC_INV_SIN_THETA
	}
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res.y *= AC_INV_CYL_R
	}
	return sum(res)
}

/**
 * Calculates the upwind correction to u·(∇f),
 * where u and f are vector fields.
 */
del_upwd(real3 velo, Field3 field) {
        return real3( dot(abs(velo),gradient_upwd(field.x)),
                      dot(abs(velo),gradient_upwd(field.y)),
                      dot(abs(velo),gradient_upwd(field.z)))
}

/**
 * Calculates laplacian of s.
 */
laplace(Field s) {
    del2f = derxx(s) + deryy(s) + derzz(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx(s)*AC_INV_CYL_R
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx(s)*AC_INV_R
	    del2f += dery(s)*AC_INV_R*AC_COT
    }
    return del2f
}

anisotropic_laplace(Field s, real3 coeffs)
{
	fatal_error_message(AC_coordinate_system != AC_CARTESIAN_COORDINATES,"Anisotropic laplace only implemented in Cartesian!\n");
	return coeffs.x*derxx(s) + coeffs.y*deryy(s) + coeffs.z*derzz(s)
}

anisotropic_laplace_central_coeff(real3 coeffs) {
    return coeffs.x*derxx_central_coeff() + coeffs.y*deryy_central_coeff() + coeffs.z*derzz_central_coeff()
}
/**
 * Calculates laplacian of s given the squares of the inverse spacings.
 * Most likely only correct in Cartesian.
 */
laplace(Field s, real3 inv_spacing_2) {
    del2f = derxx(s,inv_spacing_2.x) + deryy(s,inv_spacing_2.y) + derzz(s,inv_spacing_2.z)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx(s)*AC_INV_CYL_R
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx(s)*AC_INV_R
	    del2f += dery(s)*AC_INV_R*AC_COT
    }
    return del2f
}
/**
 * Calculates laplacian of s with 2nd order discretization.
 */
laplace_2nd(Field s) {
    del2f = derxx_2nd(s) + deryy_2nd(s) + derzz_2nd(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
            del2f += derx_2nd(s)*AC_INV_CYL_R
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
            del2f += 2*derx_2nd(s)*AC_INV_R
            del2f += dery_2nd(s)*AC_INV_R*AC_COT
    }
    return del2f
}

/**
 * Calculates laplacian of s without the central point included.
 * (Useful for Jacobi and SOR)
 */
laplace_neighbours(Field s) {
    del2f = derxx_neighbours(s) + deryy_neighbours(s) + derzz_neighbours(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx(s)*AC_INV_CYL_R
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx(s)*AC_INV_R
	    del2f += dery(s)*AC_INV_R*AC_COT
    }
    return del2f
}

/**
 * Calculates laplacian of s in 2nd order without the central point included.
 * (Useful for Jacobi and SOR)
 */
laplace_2nd_neighbours(Field s) {
    del2f = derxx_2nd_neighbours(s) + deryy_2nd_neighbours(s) + derzz_2nd_neighbours(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx_2nd(s)*AC_INV_CYL_R
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx_2nd(s)*AC_INV_R
	    del2f += dery_2nd(s)*AC_INV_R*AC_COT
    }
    return del2f
}

/**
 * Calculates laplacian of s in 2nd order without the central point included.
 * (Useful for Jacobi and SOR)
 * Uses the given spacings.
 */
laplace_2nd_neighbours(Field s, real3 inv_spacings_2) {
    del2f = derxx_2nd_neighbours(s,inv_spacings_2.x) + deryy_2nd_neighbours(s,inv_spacings_2.y) + derzz_2nd_neighbours(s,inv_spacings_2.z)
    return del2f
}

/**
 * Calculates laplacian of s without the central point included.
 * (Useful for Jacobi and SOR)
 * Uses the given spacings.
 */
laplace_neighbours(Field s, real3 inv_spacings_2) {
    del2f = derxx_neighbours(s,inv_spacings_2.x) + deryy_neighbours(s,inv_spacings_2.y) + derzz_neighbours(s,inv_spacings_2.z)
    return del2f
}

/**
 * Gives the coefficient for the central point of the Laplacian.
 */
laplace_central_coeff() {
    return derxx_central_coeff() + deryy_central_coeff() + derzz_central_coeff()
}

/**
 * Gives the coefficient for the central point of the Laplacian as discretized in 2nd order.
 */
laplace_2nd_central_coeff() {
    return derxx_2nd_central_coeff() + deryy_2nd_central_coeff() + derzz_2nd_central_coeff()
}

/**
 * Gives the coefficient for the central point of the Laplacian.
 */
laplace_central_coeff(real3 inv_spacings_2) {
    return derxx_central_coeff(inv_spacings_2.x) + deryy_central_coeff(inv_spacings_2.y) + derzz_central_coeff(inv_spacings_2.z)
}

/**
 * Gives the coefficient for the central point of the Laplacian as discretized in 2nd order.
 */
laplace_2nd_central_coeff(real3 inv_spacings_2) {
    return derxx_2nd_central_coeff(inv_spacings_2.x) + deryy_2nd_central_coeff(inv_spacings_2.y) + derzz_2nd_central_coeff(inv_spacings_2.z)
}

/**
 * Gives the coefficient for the central point of the Laplacian.
 */
laplace_central_coeff_extended() {
    return derxx_central_coeff_extended() + deryy_central_coeff_extended() + derzz_central_coeff_extended()
}

/**
 * Gives the coefficient for the central point of the Laplacian.
 */
laplace_central_coeff_extended(real3 inv_spacings_2) {
    return derxx_central_coeff_extended(inv_spacings_2.x) + deryy_central_coeff_extended(inv_spacings_2.y) + derzz_central_coeff_extended(inv_spacings_2.z)
}


/**
 * Calculates the vector Laplacian of s
 */
laplace(Field3 s) {
	d2A = get_d2A(s)
	del2f = real3(
			d2A[0][0][0] + d2A[1][1][0] + d2A[2][2][0],
			d2A[0][0][1] + d2A[1][1][1] + d2A[2][2][1],
			d2A[0][0][2] + d2A[1][1][2] + d2A[2][2][2]
		     )
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		del2f.x -= (2*dery(s.y)+s.x)*AC_INV_CYL_R*AC_INV_CYL_R
		del2f.y += (2*dery(s.x)-s.y)*AC_INV_CYL_R*AC_INV_CYL_R
	}
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		del2f.x += AC_INV_R*(
				2*(
					derx(s.x)-dery(s.y)-derz(s.z)
					-AC_INV_R*(s.x+AC_COT*s.y)
				)
				+AC_COT*dery(s.x)
				)	
		del2f.y += AC_INV_R*(
				2*(
					derx(s.y)-AC_COT*derz(s.z)+dery(s.x)
				)
				+AC_COT*dery(s.y)-AC_INV_R*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)*s.y
				)
		del2f.z += AC_INV_R*(
				2*(
					derx(s.z)+derz(s.x)+AC_COT*derz(s.y)
				  )
				+AC_COT*dery(s.z)-AC_INV_R*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)*s.z
				)
	}
	return del2f
}

/**
 * Computes the traceless rate-of-strain tensor,
 * where uij is the Jacobian of the the velocity
 * and divu is the divergence of it.
 * Works only in Cartesian.
 */
traceless_strain(Matrix uij,divu)
{
  if(AC_coordinate_system == AC_SPHERICAL_COORDINATES || AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
  	fatal_error_message(true,"traceless_strain needs the velocity for curvilinear coordinates");
  }
  Matrix sij
  for row in 0:3{
    sij[row][row] = uij[row][row] - (1.0/3.0)*divu
    for col in row+1:3{
      sij[col][row] = 0.5*(uij[col][row]+uij[row][col])
      sij[row][col] = sij[col][row]
    }
  }
  return sij
}

traceless_strain(Matrix uij)
{
  return traceless_strain(uij,uij[0][0] + uij[1][1] + uij[2][2])
}

/**
 * Computes 3x3 tensor of 5th order derivatives of v.
 */
gij5(v) {
    return Matrix(real3( der5x(v.x), der5y(v.x), der5z(v.x) ), 
		  real3( der5x(v.y), der5y(v.y), der5z(v.y) ), 
		  real3( der5x(v.z), der5y(v.z), der5z(v.z) ))
}
/**
 * Computes the traceless rate-of-strain tensor,
 * where uu is the velocity, uij is the Jacobian of it,
 * divu is the divergence.
 */
traceless_strain(Matrix uij,divu,uu)
{
  suppress_unused_warning(uu)
  Matrix sij
  for row in 0:3{
    sij[row][row] = uij[row][row] - (1.0/3.0)*divu
    for col in row+1:3{
      sij[col][row] = 0.5*(uij[col][row]+uij[row][col])
      sij[row][col] = sij[col][row]
    }
  }
  if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
  {
      sij[1-1][2-1]=sij[1-1][2-1]-0.5*AC_INV_R*uu.y
      sij[1-1][3-1]=sij[1-1][3-1]-0.5*AC_INV_R*uu.z
      sij[2-1][1-1]=sij[1-1][2-1]
      sij[2-1][2-1]=sij[2-1][2-1]+AC_INV_R*uu.x
      sij[2-1][3-1]=sij[2-1][3-1]-0.5*AC_INV_R*AC_COT*uu.z
      sij[3-1][1-1]=sij[1-1][3-1]
      sij[3-1][2-1]=sij[2-1][3-1]
      sij[3-1][3-1]=sij[3-1][3-1]+AC_INV_R*uu.x+AC_COT*AC_INV_R*uu.y

  }
  if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
      sij[1-1][2-1]=sij[1-1][2-1]-0.5*AC_INV_CYL_R*uu.y
      sij[2-1][2-1]=sij[2-1][2-1]+0.5*AC_INV_CYL_R*uu.x
      sij[2-1][1-1]=sij[1-1][2-1]
  }
  return sij
}

/**
 * Computes the rate-of-strain tensor,
 * where v is the velocity.
 */
traceless_rateof_strain(Field3 v) {
    Matrix S

    S[0][0] = (2.0 / 3.0) * derx(v.x) - (1.0 / 3.0) * (dery(v.y) + derz(v.z))
    S[0][1] = (1.0 / 2.0) * (dery(v.x) + derx(v.y))
    S[0][2] = (1.0 / 2.0) * (derz(v.x) + derx(v.z))

    S[1][0] = S[0][1]
    S[1][1] = (2.0 / 3.0) * dery(v.y) - (1.0 / 3.0) * (derx(v.x) + derz(v.z))
    S[1][2] = (1.0 / 2.0) * (derz(v.y) + dery(v.z))

    S[2][0] = S[0][2]
    S[2][1] = S[1][2]
    S[2][2] = (2.0 / 3.0) * derz(v.z) - (1.0 / 3.0) * (derx(v.x) + dery(v.y))

    return S
}

/**
 * Computes ∇(∇·v),
 * and m = ∇(v)
 */
gradient_of_divergence(Field3 v,Matrix m) {
    d2A = get_d2A(v,m)
    graddiv =
    real3(
        d2A[0][0][0] + d2A[1][0][1] + d2A[2][0][2],
        d2A[0][1][0] + d2A[1][1][1] + d2A[2][1][2],
        d2A[0][2][0] + d2A[1][2][1] + d2A[2][2][2]
    )
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
        graddiv.x += m[0][0]*AC_INV_R*2+m[1][0]*AC_INV_R*AC_COT-v.y*(AC_INV_R*AC_INV_R)*AC_COT-v.x*(AC_INV_R*AC_INV_R)*2
        graddiv.y += m[0][1]*AC_INV_R*2+m[1][1]*AC_INV_R*AC_COT-v.y*(AC_INV_R*AC_INV_R)*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)
        graddiv.z += m[0][2]*AC_INV_R*2+m[1][2]*AC_INV_R*AC_COT
    }
    return graddiv
}

/**
 * Computes ∇(∇·v),
 */
gradient_of_divergence(Field3 v) {

    d2A = get_d2A(v)
    graddiv =
    real3(
        d2A[0][0][0] + d2A[1][0][1] + d2A[2][0][2],
        d2A[0][1][0] + d2A[1][1][1] + d2A[2][1][2],
        d2A[0][2][0] + d2A[1][2][1] + d2A[2][2][2]
    )
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
        graddiv.x += derx(v.x)*AC_INV_R*2+derx(v.y)*AC_INV_R*AC_COT-v.y*(AC_INV_R*AC_INV_R)*AC_COT-v.x*(AC_INV_R*AC_INV_R)*2
        graddiv.y += dery(v.x)*AC_INV_R*2+dery(v.y)*AC_INV_R*AC_COT-v.y*(AC_INV_R*AC_INV_R)*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)
        graddiv.z += derz(v.x)*AC_INV_R*2+derz(v.y)*AC_INV_R*AC_COT
    }
    return graddiv
}
/**
 * ∑a²_ij i.e. the Frobenius norm squared of the matrix
 */
contract(Matrix mat) {
    return dot(mat.row(0), mat.row(0)) +
           dot(mat.row(1), mat.row(1)) +
           dot(mat.row(2), mat.row(2))
}

/**
 * Computes A_ij.B_ij
 */
contract(Matrix a, Matrix b) {
    return dot(a.row(0), b.row(0)) +
           dot(a.row(1), b.row(1)) +
           dot(a.row(2), b.row(2))
}

/**
 * Calculates the Euclidean norm of the vector v squared
 */
norm2(real3 v) {
    return ( dot(v,v) )
}

/**
 * Calculates the Euclidean norm of the vector v
 */
length(v) {
    return sqrt( norm2(v) )
}
/**
 * Computes M where
 * M_ij = (∂^2/∂_i)v_j,
 * and v is a vector field.
 */
d2fi_dxj(Field3 v)
{
	return Matrix(
			gradient2(v.x),
			gradient2(v.y),
			gradient2(v.z)
		     )
}

/**
 * Computes v·(∇f),
 * where v is a vector and f is a scalar field.
 */
del6fj(Field f, real3 vec)
{
	return vec.x*der6x(f) + vec.y*der6y(f) + vec.z*der6z(f)
}

/**
 * Computes ∇ × (∇ × v)
 */
curlcurl(Field3 v)
{
	return gradient_of_divergence(v) - laplace(v)
}

#ifdef AC_GENERAL_DERIVS_H
/**
 * Computes the Laplacian of s on the extended grid
 */
laplace_extended(Field s) {
    del2f = derxx_extended(s) + deryy_extended(s) + derzz_extended(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx_extended(s)*AC_INV_CYL_R_extended
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx_extended(s)*AC_INV_R_extended
	    del2f += dery_extended(s)*AC_INV_R_extended*AC_COT_extended
    }
    return del2f
}

/**
 * Computes the Laplacian of s (without taking the central point into account) on the extended grid
 */
laplace_neighbours_extended(Field s) {
    del2f = derxx_neighbours_extended(s) + deryy_neighbours_extended(s) + derzz_neighbours_extended(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx_extended(s)*AC_INV_CYL_R_extended
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx_extended(s)*AC_INV_R_extended
	    del2f += dery_extended(s)*AC_INV_R_extended*AC_COT_extended
    }
    return del2f
}

/**
 * Computes the Laplacian of s (without taking the central point into account) on the extended grid
 */
laplace_neighbours_extended(Field s, real3 inv_spacings_2) {
    del2f = derxx_neighbours_extended(s,inv_spacings_2.x) + deryy_neighbours_extended(s,inv_spacings_2.y) + derzz_neighbours_extended(s,inv_spacings_2.z)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx_extended(s)*AC_INV_CYL_R_extended
    }
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx_extended(s)*AC_INV_R_extended
	    del2f += dery_extended(s)*AC_INV_R_extended*AC_COT_extended
    }
    return del2f
}
#endif
/**
 * Calculates biharmonic operator delta(delta(s))
 */
biharmonic(Field s) {
    if(AC_coordinate_system != AC_CARTESIAN_COORDINATES)
    {
	    fatal_error_message(true,"Biharmonic operator is only supported in Cartesian coordinates at the moment!\n")
    }
    return
	         der4x(s) + der4y(s) + der4z(s)
	    + 2*(der2x2y(s) + der2y2z(s) + der2x2z(s))
}

#endif
