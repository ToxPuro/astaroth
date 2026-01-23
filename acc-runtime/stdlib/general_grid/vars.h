#ifndef AC_GENERAL_GRID_VARS_H

#define AC_GENERAL_GRID_VARS_H

hostdefine AC_GENERAL_GRID_VARS_INCLUDED (1)
run_const real3 AC_origin =  0.5*AC_len
run_const real3 AC_center

gmem real AC_x[AC_mlocal.x]
gmem real AC_y[AC_mlocal.y]
gmem real AC_z[AC_mlocal.z]

gmem real AC_r[AC_mlocal.x]
gmem real AC_theta[AC_mlocal.y]
gmem real AC_phi[AC_mlocal.z]

gmem real AC_r_extended[AC_extended_mlocal.x]

gmem real AC_sin_theta[AC_mlocal.y]
gmem real AC_cos_theta[AC_mlocal.y]

gmem real AC_sin_phi[AC_mlocal.z]
gmem real AC_cos_phi[AC_mlocal.z]

gmem real AC_mapping_func_derivative_x[AC_mlocal.x]
gmem real AC_mapping_func_derivative_y[AC_mlocal.y]
gmem real AC_mapping_func_derivative_z[AC_mlocal.z]

gmem real AC_mapping_func_derivative_x_extended[AC_extended_mlocal.x]
gmem real AC_mapping_func_derivative_y_extended[AC_extended_mlocal.y]
gmem real AC_mapping_func_derivative_z_extended[AC_extended_mlocal.z]

gmem real AC_inv_r[AC_nlocal.x]
gmem real AC_inv_cyl_r[AC_nlocal.x]
gmem real AC_inv_sin_theta[AC_mlocal.y]
gmem real AC_cot_theta[AC_mlocal.y]

gmem real AC_inv_r_extended[AC_extended_nlocal.x]
gmem real AC_inv_cyl_r_extended[AC_extended_nlocal.x]
gmem real AC_inv_sin_theta_extended[AC_extended_mlocal.y]
gmem real AC_cot_theta_extended[AC_extended_mlocal.y]

gmem real AC_inv_mapping_func_derivative_x[AC_mlocal.x]
gmem real AC_inv_mapping_func_derivative_y[AC_mlocal.y]
gmem real AC_inv_mapping_func_derivative_z[AC_mlocal.z]

gmem real AC_inv_mapping_func_derivative_x_extended[AC_extended_mlocal.x]
gmem real AC_inv_mapping_func_derivative_y_extended[AC_extended_mlocal.y]
gmem real AC_inv_mapping_func_derivative_z_extended[AC_extended_mlocal.z]

gmem real AC_mapping_func_tilde_x[AC_mlocal.x]
gmem real AC_mapping_func_tilde_y[AC_mlocal.y]
gmem real AC_mapping_func_tilde_z[AC_mlocal.z]

gmem real AC_mapping_func_tilde_x_extended[AC_extended_mlocal.x]
gmem real AC_mapping_func_tilde_y_extended[AC_extended_mlocal.y]
gmem real AC_mapping_func_tilde_z_extended[AC_extended_mlocal.z]

gmem real AC_x12[AC_mlocal.x]
gmem real AC_y12[AC_mlocal.y]
gmem real AC_sinth12[AC_mlocal.y]
gmem real AC_z12[AC_mlocal.z]

#define AC_INV_R         (AC_inv_r[vertexIdx.x-NGHOST])
#define AC_INV_CYL_R     (AC_inv_cyl_r[vertexIdx.x-NGHOST])
#define AC_INV_SIN_THETA (AC_inv_sin_theta[vertexIdx.y])
#define AC_COT           (AC_cot_theta[vertexIdx.y])

#define AC_INV_R_extended AC_inv_r_extended[vertexIdx.x-NGHOST]
#define AC_INV_CYL_R_extended AC_inv_cyl_r_extended[vertexIdx.x-NGHOST]
#define AC_INV_SIN_THETA_extended AC_inv_sin_theta_extended[vertexIdx.y]
#define AC_COT_extended AC_cot_theta_extended[vertexIdx.y]

#define AC_INV_MAPPING_FUNC_DER_X (AC_inv_mapping_func_derivative_x[vertexIdx.x])
#define AC_INV_MAPPING_FUNC_DER_Y (AC_inv_mapping_func_derivative_y[vertexIdx.y])
#define AC_INV_MAPPING_FUNC_DER_Z (AC_inv_mapping_func_derivative_z[vertexIdx.z])

#define AC_INV_MAPPING_FUNC_DER_X_extended (AC_inv_mapping_func_derivative_x_extended[vertexIdx.x])
#define AC_INV_MAPPING_FUNC_DER_Y_extended (AC_inv_mapping_func_derivative_y_extended[vertexIdx.y])
#define AC_INV_MAPPING_FUNC_DER_Z_extended (AC_inv_mapping_func_derivative_z_extended[vertexIdx.z])

#define AC_MAPPING_FUNC_TILDE_X  (AC_mapping_func_tilde_x[vertexIdx.x])
#define AC_MAPPING_FUNC_TILDE_Y  (AC_mapping_func_tilde_y[vertexIdx.y])
#define AC_MAPPING_FUNC_TILDE_Z  (AC_mapping_func_tilde_z[vertexIdx.z])

#define AC_MAPPING_FUNC_TILDE_X_extended  (AC_mapping_func_tilde_x_extended[vertexIdx.x])
#define AC_MAPPING_FUNC_TILDE_Y_extended  (AC_mapping_func_tilde_y_extended[vertexIdx.y])
#define AC_MAPPING_FUNC_TILDE_Z_extended  (AC_mapping_func_tilde_z_extended[vertexIdx.z])

#endif
