u_dot_grad_vec(Matrix m,real3 v){
  //!!!return real3(dot(v,m.row(0)),dot(v,m.col(1)),dot(v,m.col(2)))
  if(AC_coordinate_system != AC_CARTEESIAN_COORDINATES)
  {
	  print("NOT IMPLEMENTED u_dot_grad_vec for non-carteesian\n")
  }
  return real3(dot(v,m.col(0)),dot(v,m.col(1)),dot(v,m.col(2)))
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
	   return g + AC_INV_R*(v.x*2 + AC_COT*v.y);
    }
    else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    return g+AC_INV_CYL_R*v.x
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
		return b +2*AC_INV_R*(a.x + AC_COT*a.y)
	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
	    return b+AC_INV_CYL_R*a.x
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
    else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    g.z += v.y*AC_INV_CYL_R
    }
    return g
}

curl(Matrix m) {
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
  else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
          g.z += v.y*AC_INV_CYL_R
  }
  return g
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

bij(Field3 v)
{
    	d2A = del2fi_dxjk(v)
    	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    	{
    		d2A[0][0][0] -= AC_INV_R*dery(v.x)
    		d2A[0][0][1] -= AC_INV_R*dery(v.y)
    		d2A[0][0][2] -= AC_INV_R*dery(v.z)

    		d2A[1][0][0] -= AC_INV_R*derz(v.x)
    		d2A[1][0][1] -= AC_INV_R*derz(v.y)
    		d2A[1][0][2] -= AC_INV_R*derz(v.z)

    		d2A[2][1][0] -= AC_INV_R*derz(v.x)*AC_COT
    		d2A[2][1][1] -= AC_INV_R*derz(v.y)*AC_COT
    		d2A[2][1][2] -= AC_INV_R*derz(v.z)*AC_COT
    	}
    	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    	{
    	        d2A[1][0][0] -= dery(v.x)*AC_INV_CYL_R
    	        d2A[1][0][1] -= dery(v.y)*AC_INV_CYL_R
    	        d2A[1][0][2] -= dery(v.z)*AC_INV_CYL_R
    	}

	Matrix res;
	res[0][0] = d2A[1][0][2] - d2A[2][0][1] 
	res[0][1] = d2A[1][1][2] - d2A[2][1][1] 
	res[0][2] = d2A[1][2][2] - d2A[2][2][1] 

	res[1][0] = d2A[2][0][1] - d2A[1][0][2] 
	res[1][1] = d2A[2][1][1] - d2A[1][1][2] 
	res[1][2] = d2A[2][2][1] - d2A[1][2][2] 

	res[2][0] = d2A[0][0][1] - d2A[1][0][0] 
	res[2][1] = d2A[0][1][1] - d2A[1][1][0] 
	res[2][2] = d2A[0][2][1] - d2A[1][2][0] 

	if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
          res[3-1][2-1]+=dery(v.x)*AC_INV_R
          res[2-1][3-1]+=-derz(v.z)*AC_INV_R
          res[1-1][3-1]+=derz(v.z)*AC_INV_R*AC_COT
          res[3-1][1-1]+=dery(v.x)*AC_INV_R           -v.y*(AC_INV_R*AC_INV_R)
          res[2-1][1-1]+=-derz(v.x)*AC_INV_R           +v.z*(AC_INV_R*AC_INV_R)
          res[1-1][2-1]+=derz(v.y)*AC_INV_R*AC_COT -v.z*(AC_INV_R*AC_INV_R)*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)

	}
	else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
	  
          res[3-1][2-1]=res[3-1][2-1] + dery(v.y)*AC_INV_CYL_R
          res[3-1][1-1]=res[3-1][1-1] + dery(v.x)*AC_INV_CYL_R-v.y*AC_INV_CYL_R*AC_INV_CYL_R
	}


	return res;
}


elemental del4(Field s) {
  return der4x(s) + der4y(s) + der4z(s)
}

elemental del6(Field s) {
  return der6x(s) + der6y(s) + der6z(s)
}

elemental del6_exp(Field s) {
  return exp(s)*(der6x(s) + der6y(s) + der6z(s))
}

del6_masked(Field s, int mask)
{
	x = mask == 1 ? 0.0 : der6x(s)
	y = mask == 2 ? 0.0 : der6y(s)
	z = mask == 3 ? 0.0 : der6z(s)
	return x + y + z
}

del6_upwd_masked(Field s, int mask)
{
	x = mask == 1 ? 0.0 : der6x_upwd(s)
	y = mask == 2 ? 0.0 : der6y_upwd(s)
	z = mask == 3 ? 0.0 : der6z_upwd(s)
	return x + y + z
}

elemental del6_strict(Field s) {
	print("NOT IMPLEMENTED del6_strict\n")
	return 0.
}

elemental ugrad_upw(Field field, Field3 velo){

        return dot(velo,gradient(field)) - dot(abs(velo),gradient_upwd(field))
        //if (msk>0) then
        //ugradf = ugradf+del6f_upwind
        //else
        //ugradf = ugradf-del6f_upwind
        //endif
}

elemental del6_upwd(Field s) {
  return der6x_upwd(s) + der6y_upwd(s) + der6z_upwd(s)
}

laplace(Field s) {
    del2f = derxx(s) + deryy(s) + derzz(s)
    if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
	    del2f += derx(s)*AC_INV_CYL_R
    }
    else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
	    del2f += 2*derx(s)*AC_INV_R
	    del2f += 2*dery(s)*AC_INV_R*AC_COT
    }
    return del2f
}

laplace(Field3 s) {
	del2f = real3(
			laplace(s.x),
			laplace(s.y),
			laplace(s.z)
		     )
	if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
	{
		del2f.x -= (2*dery(s.y)+s.x)*AC_INV_CYL_R*AC_INV_CYL_R
		del2f.y += (2*dery(s.x)-s.y)*AC_INV_CYL_R*AC_INV_CYL_R
	}
	else if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
	{
		del2f.x += del2f.x + AC_INV_R*(
				2*(
					derx(s.x)-dery(s.y)-dery(s.z)
					-AC_INV_R*(s.x+AC_COT*s.y)
				)
				+AC_COT*dery(s.x)
				)	
		del2f.y += AC_INV_R*(
				2*(
					derx(s.y)-AC_COT+derz(s.z)+dery(s.x)
				)
				+AC_COT*dery(s.y)-AC_INV_R*AC_INV_SIN_THETA*AC_INV_SIN_THETA*s.y
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
  else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
  {
      sij[1-1][2-1]=sij[1-1][2-1]-0.5*AC_INV_CYL_R*uu.y
      sij[2-1][2-1]=sij[2-1][2-1]+0.5*AC_INV_CYL_R*uu.x
      sij[2-1][1-1]=sij[1-1][2-1]
  }
  return sij
}


traceless_rateof_strain(v) {
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
    d2A = del2fi_dxjk(v)
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
        d2A[0][0][0] -= AC_INV_R*dery(v.x)
        d2A[0][0][1] -= AC_INV_R*dery(v.y)
        d2A[0][0][2] -= AC_INV_R*dery(v.z)

        d2A[1][0][0] -= AC_INV_R*derz(v.x)
        d2A[1][0][1] -= AC_INV_R*derz(v.y)
        d2A[1][0][2] -= AC_INV_R*derz(v.z)

        d2A[2][1][0] -= AC_INV_R*derz(v.x)*AC_COT
        d2A[2][1][1] -= AC_INV_R*derz(v.y)*AC_COT
        d2A[2][1][2] -= AC_INV_R*derz(v.z)*AC_COT
    }
    else if(AC_coordinate_system == AC_CYLINDRICAL_COORDINATES)
    {
            d2A[1][0][0] -= dery(v.x)*AC_INV_CYL_R
            d2A[1][0][1] -= dery(v.y)*AC_INV_CYL_R
            d2A[1][0][2] -= dery(v.z)*AC_INV_CYL_R
    }

    graddiv =
    real3(
        d2A[0][0][0] + d2A[1][0][1] + d2A[2][0][2],
        d2A[0][1][0] + d2A[1][1][1] + d2A[2][1][2],
        d2A[0][2][0] + d2A[1][2][1] + d2A[2][2][2]
    )
    if(AC_coordinate_system == AC_SPHERICAL_COORDINATES)
    {
        graddiv.x += derx(v.x)*AC_INV_R*2+dery(v.x)*AC_INV_R*AC_COT-v.y*(AC_INV_R*AC_INV_R)*AC_COT-v.x*(AC_INV_R*AC_INV_R)*2
        graddiv.y += derx(v.y)*AC_INV_R*2+dery(v.y)*AC_INV_R*AC_COT-v.y*(AC_INV_R)*AC_COT-v.y*(AC_INV_R*AC_INV_R)*(AC_INV_SIN_THETA*AC_INV_SIN_THETA)
        graddiv.z += derx(v.z)*AC_INV_R*2+dery(v.z)*AC_INV_R*AC_COT
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
gaussian_smooth(Field f)
{
	print("NOT IMPLEMENTED del6_strict\n")
	return 0.
}
curlcurl(Field3 v)
{
	return gradient_of_divergence(v) - laplace(v)
}
