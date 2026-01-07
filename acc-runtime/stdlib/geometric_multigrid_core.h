#include "$AC_HOME/acc-runtime/stdlib/grid_transfer_functions.h"
int3 level_divisor = (int3)
{
	AC_dimension_inactive.x ? 1 : 2,
	AC_dimension_inactive.y ? 1 : 2,
	AC_dimension_inactive.z ? 1 : 2
}

int3 AC_nlocal_gmg_level_0 = AC_nlocal
int3 AC_nlocal_gmg_level_1 =
			     (int3){
			       AC_nlocal_gmg_level_0.x % 2 == 0 ? (AC_nlocal_gmg_level_0.x-2)/level_divisor.x :
				       				  (AC_nlocal_gmg_level_0.x)/level_divisor.x,
			       AC_nlocal_gmg_level_0.y % 2 == 0 ? (AC_nlocal_gmg_level_0.y-2)/level_divisor.y :
				       				  (AC_nlocal_gmg_level_0.y)/level_divisor.y,
			       AC_nlocal_gmg_level_0.z % 2 == 0 ? (AC_nlocal_gmg_level_0.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_0.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_2 =
			     (int3){
			       AC_nlocal_gmg_level_1.x % 2 == 0 ? (AC_nlocal_gmg_level_1.x-2)/level_divisor.x :
				       		       	  (AC_nlocal_gmg_level_1.x)/level_divisor.x,
			       AC_nlocal_gmg_level_1.y % 2 == 0 ? (AC_nlocal_gmg_level_1.y-2)/level_divisor.y :
				       		       	  (AC_nlocal_gmg_level_1.y)/level_divisor.y,
			       AC_nlocal_gmg_level_1.z % 2 == 0 ? (AC_nlocal_gmg_level_1.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_1.z)/level_divisor.z
			     }
int3 AC_nlocal_gmg_level_3 =
			     (int3){
			       AC_nlocal_gmg_level_2.x % 2 == 0 ? (AC_nlocal_gmg_level_2.x-2)/level_divisor.x :
				       		       	  (AC_nlocal_gmg_level_2.x)/level_divisor.x,
			       AC_nlocal_gmg_level_2.y % 2 == 0 ? (AC_nlocal_gmg_level_2.y-2)/level_divisor.y :
				       		       	  (AC_nlocal_gmg_level_2.y)/level_divisor.y,
			       AC_nlocal_gmg_level_2.z % 2 == 0 ? (AC_nlocal_gmg_level_2.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_2.z)/level_divisor.z
			     }
int3 AC_nlocal_gmg_level_4 =
			     (int3){
			       AC_nlocal_gmg_level_3.x % 2 == 0 ? (AC_nlocal_gmg_level_3.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_3.x)/level_divisor.x,
			       AC_nlocal_gmg_level_3.y % 2 == 0 ? (AC_nlocal_gmg_level_3.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_3.y)/level_divisor.y,
			       AC_nlocal_gmg_level_3.z % 2 == 0 ? (AC_nlocal_gmg_level_3.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_3.z)/level_divisor.z
			     }

int3 AC_mlocal_gmg_level_0 = AC_nlocal_gmg_level_0 + 2*AC_nmin
int3 AC_mlocal_gmg_level_1 = AC_nlocal_gmg_level_1 + 2*AC_nmin
int3 AC_mlocal_gmg_level_2 = AC_nlocal_gmg_level_2 + 2*AC_nmin
int3 AC_mlocal_gmg_level_3 = AC_nlocal_gmg_level_3 + 2*AC_nmin
int3 AC_mlocal_gmg_level_4 = AC_nlocal_gmg_level_4 + 2*AC_nmin

dims(AC_mlocal_gmg_level_0) Field GMG_SOLUTION_0
dims(AC_mlocal_gmg_level_1) Field GMG_SOLUTION_1 
dims(AC_mlocal_gmg_level_2) Field GMG_SOLUTION_2
dims(AC_mlocal_gmg_level_3) Field GMG_SOLUTION_3
dims(AC_mlocal_gmg_level_4) Field GMG_SOLUTION_4

auxiliary dims(AC_mlocal_gmg_level_0) Field GMG_INITIAL_RHS
auxiliary dims(AC_mlocal_gmg_level_1) Field GMG_RHS_1
auxiliary dims(AC_mlocal_gmg_level_2) Field GMG_RHS_2
auxiliary dims(AC_mlocal_gmg_level_3) Field GMG_RHS_3
auxiliary dims(AC_mlocal_gmg_level_4) Field GMG_RHS_4
//Need to be communicated if use other smoothers than gauss-seidel or jacobi
communicated auxiliary dims(AC_mlocal_gmg_level_0) Field GMG_RESIDUAL_0
communicated auxiliary dims(AC_mlocal_gmg_level_1) Field GMG_RESIDUAL_1 
communicated auxiliary dims(AC_mlocal_gmg_level_2) Field GMG_RESIDUAL_2
communicated auxiliary dims(AC_mlocal_gmg_level_3) Field GMG_RESIDUAL_3
communicated auxiliary dims(AC_mlocal_gmg_level_4) Field GMG_RESIDUAL_4

dconst real AC_GMG_CENTRAL_COEFFS[5]

const Field GMG_SOLUTIONS =
	[
		GMG_SOLUTION_0,
		GMG_SOLUTION_1,
		GMG_SOLUTION_2,
		GMG_SOLUTION_3,
		GMG_SOLUTION_4
	]

const Field GMG_RHS =
	[
		GMG_INITIAL_RHS,
		GMG_RHS_1,
		GMG_RHS_2,
		GMG_RHS_3,
		GMG_RHS_4
	]

const Field GMG_RESIDUALS =
	[
		GMG_RESIDUAL_0,
		GMG_RESIDUAL_1,
		GMG_RESIDUAL_2,
		GMG_RESIDUAL_3,
		GMG_RESIDUAL_4
	]

enum GMG_LEVEL
{
	GMG_LEVEL_0,
	GMG_LEVEL_1,
	GMG_LEVEL_2,
	GMG_LEVEL_3,
	GMG_LEVEL_4
}


input GMG_LEVEL AC_GMG_LEVEL

Kernel gmg_restrict_residual_kernel(GMG_LEVEL level)
{
	restrict_full_weighting(GMG_RESIDUALS[level],GMG_RHS[level+1])
	//The residual is most likely small so zero is a meaningful starting value
	write(GMG_SOLUTIONS[level+1],0.0)
}

Kernel gmg_get_correction_from_next_level_kernel(GMG_LEVEL level)
{
	e = trilinear_prolongation(GMG_SOLUTIONS[level+1])
	write(GMG_SOLUTIONS[level],GMG_SOLUTIONS[level]+e)
}

ComputeSteps
gmg_restrict_residual(gmg_boundconds)
{
	gmg_restrict_residual_kernel(AC_GMG_LEVEL)
}

ComputeSteps
gmg_get_correction_from_next_level(gmg_boundconds)
{
	gmg_get_correction_from_next_level_kernel(AC_GMG_LEVEL)
}
