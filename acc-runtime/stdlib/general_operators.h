u_dot_grad(Field3 f, Matrix m,real3 v){
	real3 res = m*v
	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		res.x = res.x - AC_INV_R*(v.y*f.y+v.z*f.z)
		res.y = res.y + AC_INV_R*(v.y*f.x-v.z*f.z*AC_COT)
		res.z = res.z + AC_INV_R*(v.z*f.x+v.z*f.y*AC_COT)
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		res.x = res.x - AC_INV_CYL_R*(v.y*f.y)
		res.y = res.y + AC_INV_CYL_R*(v.y*f.x)
	}
	return res;
}

u_dot_grad_scl_alt(a,b,c,d){
	fatal_error_message(true,"u_dot_grad_scl_alt: Not implemented")
	return 0.0
}
u_dot_grad_mat(k,c,uu)
{
	fatal_error_message(true,"u_dot_grad_mat: Not implemented")
	Matrix res
	return res
}
u_dot_grad_mat_upwd(k,c,uu)
{
	fatal_error_message(true,"u_dot_grad_mat_upwd: Not implemented")
	Matrix res
	return res
}

gradient(Field s) {
    return real3(derx(s), dery(s), derz(s))
}
gradient_tensor(Field3 v) {
	return Matrix(
			gradient(v.x),
			gradient(v.y),
			gradient(v.z)
		     )
}

elemental gradient_upwd(Field s) {
    return real3(der6x_upwd(s), der6y_upwd(s), der6z_upwd(s))
}



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

elemental gradient5(Field s) {
    return real3(der5x(s), der5y(s), der5z(s))
}


elemental gradient6_upwd(s) {
    return real3(der6x_upwd(s), der6y_upwd(s), der6z_upwd(s))
}


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
divergence(Matrix m)
{
	b = m[0][0] + m[1][1] + m[2][2];
	return b;
}
divergence(Matrix m, real3 a)
{
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

curl(Matrix m) {
  if(AC_coordinate_system != AC_CARTESIAN_COORDINATES)
  {
	  print("curl with Matrix only is incorrect for non-cartesian coordinates!\n")
  }
  return real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
}

curl(Matrix m, real3 v) {
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

covariant_curl(Matrix m, real3 v)
{
  return real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
}


hessian(Field v)
{
	return Matrix(
			real3(derxx(v), derxy(v), derxz(v)),
			real3(derxy(v), deryy(v), deryz(v)),
			real3(derxz(v), deryz(v), derzz(v))
		     )
}

del2fi_dxjk(Field3 v)
{
	return Tensor(
			hessian(v.x),
			hessian(v.y),
			hessian(v.z)
		     )
}
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


elemental del4(Field s) {
  return der4x(s) + der4y(s) + der4z(s)
}

elemental del6(Field s) {
  return der6x(s) + der6y(s) + der6z(s)
}

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
elemental del6_exp(Field f)
{
	return der6x_exp(f) + der6y_exp(f) + der6z_exp(f)
}

del6_masked(Field s, int mask)
{
	x = mask == 1 ? 0.0 : der6x(s)
	y = mask == 2 ? 0.0 : der6y(s)
	z = mask == 3 ? 0.0 : der6z(s)
	return x + y + z
}

del6_upwd_masked(real3 velo, Field s, int mask)
{
        x = mask == 1 ? 0.0 : abs(velo.x*der6x_upwd(s))
        y = mask == 2 ? 0.0 : abs(velo.y*der6y_upwd(s))
        z = mask == 3 ? 0.0 : abs(velo.z*der6z_upwd(s))
        return x + y + z
}

elemental del6_strict(Field s) {
	print("NOT IMPLEMENTED del6_strict\n")
	return 0.
}

elemental ugrad_upw(Field field, real3 velo){

        return dot(velo,gradient(field)) - dot(abs(velo),gradient_upwd(field))
}

elemental ugrad_upw(Field3 field, real3 velo){

        return real3( dot(velo,gradient(field.x)) - dot(abs(velo),gradient_upwd(field.x)),
		      dot(velo,gradient(field.y)) - dot(abs(velo),gradient_upwd(field.y)),
		      dot(velo,gradient(field.z)) - dot(abs(velo),gradient_upwd(field.z)))
}

del6_upwd(real3 velo,Field field)
{

        return dot(abs(velo),gradient_upwd(field))
}

del6_upwd(real3 velo, Field3 field) {
        return real3( dot(abs(velo),gradient_upwd(field.x)),
                      dot(abs(velo),gradient_upwd(field.y)),
                      dot(abs(velo),gradient_upwd(field.z)))
}


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

laplace_central_coeff() {
    return derxx_central_coeff() + deryy_central_coeff() + derzz_central_coeff()
}

laplace_2nd_central_coeff() {
    return derxx_2nd_central_coeff() + deryy_2nd_central_coeff() + derzz_2nd_central_coeff()
}

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

traceless_strain(Matrix uij,divu)
{
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

gij5(v) {
    return Matrix(real3( der5x(v.x), der5y(v.x), der5z(v.x) ), 
		  real3( der5x(v.y), der5y(v.y), der5z(v.y) ), 
		  real3( der5x(v.z), der5y(v.z), der5z(v.z) ))
}
traceless_strain(Matrix uij,divu,uu)
{
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

contract(Matrix mat) {
    return dot(mat.row(0), mat.row(0)) +
           dot(mat.row(1), mat.row(1)) +
           dot(mat.row(2), mat.row(2))
}
contract(Matrix a, Matrix b) {
    return dot(a.row(0), b.row(0)) +
           dot(a.row(1), b.row(1)) +
           dot(a.row(2), b.row(2))
}

norm2(real3 v) {
    return ( dot(v,v) )
}

length(v) {
    return sqrt( norm2(v) )
}
d2fi_dxj(Field3 v)
{
	return Matrix(
			gradient2(v.x),
			gradient2(v.y),
			gradient2(v.z)
		     )
}

del6fj(Field f, real3 vec)
{
	return vec.x*der6x(f) + vec.y*der6y(f) + vec.z*der6z(f)
}
curlcurl(Field3 v)
{
	return gradient_of_divergence(v) - laplace(v)
}
