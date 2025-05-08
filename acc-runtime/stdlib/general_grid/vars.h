
run_const real3 AC_origin =  0.5*AC_len
run_const real3 AC_center

gmem real AC_x[AC_mlocal.x]
gmem real AC_y[AC_mlocal.y]
gmem real AC_z[AC_mlocal.z]

gmem real AC_r[AC_mlocal.x]
gmem real AC_theta[AC_mlocal.y]
gmem real AC_phi[AC_mlocal.z]

gmem real AC_sin_theta[AC_mlocal.y]
gmem real AC_cos_theta[AC_mlocal.y]

gmem real AC_sin_phi[AC_mlocal.z]
gmem real AC_cos_phi[AC_mlocal.z]

gmem real AC_mapping_func_derivative_x[AC_mlocal.x]
gmem real AC_mapping_func_derivative_y[AC_mlocal.y]
gmem real AC_mapping_func_derivative_z[AC_mlocal.z]
