real intrinsic sin(real) 
real intrinsic cos(real) 
real intrinsic tan(real) 
real intrinsic atan(real)
real intrinsic atan2(real)

real intrinsic sinh(real) 
real intrinsic cosh(real) 
real intrinsic tanh(real)

real intrinsic exp(real)
real intrinsic log(real)

real intrinsic sqrt(real) 
real intrinsic pow(real) 
real intrinsic fabs(real)
real intrinsic floor(real)
real intrinsic round(real)


intrinsic min
intrinsic max

intrinsic broadcast_scalar
intrinsic broadcast_scalar_2d
intrinsic broadcast_scalar_3d
intrinsic broadcast_scalar_4d
intrinsic broadcast_scalar_to_vec

intrinsic ceil
real intrinsic rand_uniform
real intrinsic atomicAdd
intrinsic output_value

intrinsic multm2_sym
intrinsic diagonal

AcDimProducts intrinsic ac_get_dim_products(int3)
AcDimProductsInv intrinsic ac_get_dim_products_inv(AcDimProducts)

intrinsic write_base
intrinsic write_profile_x
intrinsic write_profile_y
intrinsic write_profile_z
intrinsic write_profile_xy
intrinsic write_profile_xz
intrinsic write_profile_yx
intrinsic write_profile_yz
intrinsic write_profile_zx
intrinsic write_profile_zy

intrinsic value_profile_x(Profile<X>)
intrinsic value_profile_y(Profile<Y>)
intrinsic value_profile_z(Profile<Z>)
intrinsic value_profile_xy(Profile<XY>)
intrinsic value_profile_xz(Profile<XZ>)
intrinsic value_profile_yx(Profile<YZ>)
intrinsic value_profile_yz(Profile<YZ>)
intrinsic value_profile_zx(Profile<ZX>)
intrinsic value_profile_zy(Profile<ZY>)

intrinsic reduce_min_real
intrinsic reduce_max_real
intrinsic reduce_sum_real

intrinsic reduce_sum_real_x
intrinsic reduce_sum_real_y
intrinsic reduce_sum_real_z
intrinsic reduce_sum_real_xy
intrinsic reduce_sum_real_xz
intrinsic reduce_sum_real_yx
intrinsic reduce_sum_real_yz
intrinsic reduce_sum_real_zx
intrinsic reduce_sum_real_zy

intrinsic reduce_min_int
intrinsic reduce_max_int
intrinsic reduce_sum_int

intrinsic reduce_min_float
intrinsic reduce_max_float
intrinsic reduce_sum_float
intrinsic matmul_arr

real intrinsic previous_base(Field)
intrinsic print
intrinsic fprintf
intrinsic len
intrinsic size
