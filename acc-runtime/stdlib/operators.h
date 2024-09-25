u_dot_grad_vec(Matrix m,real3 v){
  //!!!return real3(dot(v,m.row(0)),dot(v,m.col(1)),dot(v,m.col(2)))
  return real3(dot(v,m.col(0)),dot(v,m.col(1)),dot(v,m.col(2)))
}


elemental gradient(Field s) {
    return real3(derx(s), dery(s), derz(s))
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
    return derx(v.x) + dery(v.y) + derz(v.z)
}
divergence(Matrix m)
{
	return m[0][0] + m[1][1] + m[2][2]
}

curl(Field3 v) {
    return real3(dery(v.z) - derz(v.y), derz(v.x) - derx(v.z), derx(v.y) - dery(v.x))
}
bij(Field3 v)
{
	Matrix res;
	print("Not implemented bij\n")
	return res;
}

curl(Matrix m) {
  return real3(m[2][1]-m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1])
}




elemental del4(Field s) {
  return der4x(s) + der4y(s) + der4z(s)
}

elemental del6(Field s) {
  return der6x(s) + der6y(s) + der6z(s)
}
del6_masked(Field s, int mask)
{
	x = mask == 1 ? 0.0 : der6x(s)
	y = mask == 2 ? 0.0 : der6y(s)
	z = mask == 3 ? 0.0 : der6z(s)
	return x + y + z
}
elemental del6_strict(Field s) {
	print("NOT IMPLEMENTED del6_strict\n")
	return 0.0
}


elemental del6_upwd(Field s) {
  return der6x_upwd(s) + der6y_upwd(s) + der6z_upwd(s)
}

elemental laplace(Field s) {
    return derxx(s) + deryy(s) + derzz(s)
}


traceless_strain(uij,divu)
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

gradient_of_divergence(v) {
    return real3(
        derxx(v.x) + derxy(v.y) + derxz(v.z),
        derxy(v.x) + deryy(v.y) + deryz(v.z),
        derxz(v.x) + deryz(v.y) + derzz(v.z)
    )
}

contract(Matrix mat) {
    return dot(mat[0], mat[0]) +
           dot(mat[1], mat[1]) +
           dot(mat[2], mat[2])
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
del6fj(Field f, real3 vec)
{
	return vec.x*der6x(f) + vec.y*der6y(f) + vec.z*der6z(f)
}
