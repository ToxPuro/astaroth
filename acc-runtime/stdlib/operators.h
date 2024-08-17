#include "../stdlib/derivs.h"

u_dot_grad_vec(Matrix m,real3 v){
  //!!!return real3(dot(v,m.row(0)),dot(v,m.col(1)),dot(v,m.col(2)))
  return real3(AC_dot(v,m.col(0)),AC_dot(v,m.col(1)),AC_dot(v,m.col(2)))
}
curl_from_matrix(Matrix m) {
  return real3(m.data[2][1]-m.data[1][2], m.data[0][2] - m.data[2][0], m.data[1][0] - m.data[0][1])
}

gradient(s) {
    return real3(derx(s), dery(s), derz(s))
}

gradient6_upwd(s) {
    return real3(der6x_upwd(s), der6y_upwd(s), der6z_upwd(s))
}

gradients_upwd(v) {
    return Matrix(gradient6_upwd(v.x), gradient6_upwd(v.y), gradient6_upwd(v.z))
}

gradients(Field3 v) {
    return Matrix(gradient(v.x), gradient(v.y), gradient(v.z))
}

divergence(v) {
    return derx(v.x) + dery(v.y) + derz(v.z)
}

curl(v) {
    return real3(dery(v.z) - derz(v.y), derz(v.x) - derx(v.z), derx(v.y) - dery(v.x))
}
del4(s) {
  return der4x(s) + der4y(s) + der4z(s)
}

del6(s) {
  return der6x(s) + der6y(s) + der6z(s)
}

del6v(v) {
  return real3(del6(v.x), del6(v.y), del6(v.z))
}

del6_upwd(s) {
  return der6x_upwd(s) + der6y_upwd(s) + der6z_upwd(s)
}

laplace(s) {
    return derxx(s) + deryy(s) + derzz(s)
}

veclaplace(Field3 v) {
    return real3(laplace(v.x), laplace(v.y), laplace(v.z))
}

traceless_strain(uij,divu)
{
  Matrix sij
  for row in 0:3{
    sij.data[row][row] = uij.data[row][row] - (1.0/3.0)*divu
    for col in row+1:3{
      sij.data[col][row] = 0.5*(uij.data[col][row]+uij.data[row][col])
      sij.data[row][col] = sij.data[col][row]
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

    S.data[0][0] = (2.0 / 3.0) * derx(v.x) - (1.0 / 3.0) * (dery(v.y) + derz(v.z))
    S.data[0][1] = (1.0 / 2.0) * (dery(v.x) + derx(v.y))
    S.data[0][2] = (1.0 / 2.0) * (derz(v.x) + derx(v.z))

    S.data[1][0] = S.data[0][1]
    S.data[1][1] = (2.0 / 3.0) * dery(v.y) - (1.0 / 3.0) * (derx(v.x) + derz(v.z))
    S.data[1][2] = (1.0 / 2.0) * (derz(v.y) + dery(v.z))

    S.data[2][0] = S.data[0][2]
    S.data[2][1] = S.data[1][2]
    S.data[2][2] = (2.0 / 3.0) * derz(v.z) - (1.0 / 3.0) * (derx(v.x) + dery(v.y))

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
