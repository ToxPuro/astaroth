/**
 * This file is for a geometric multigrid (GMG) implementation of the Poisson equation
 * -Δu = f, in Cartesian coordinates.
 *  For a good introduction to multigrid methods see the book Multigrid by Trottenberg.
 */
#include "$AC_HOME/acc-runtime/stdlib/grid_transfer_functions.h"
#define GMG_PRECISION
#define GMG_OUTPUT_PRECISION real
//#define GMG_PRECISION single_precision
//#define GMG_OUTPUT_PRECISION float

int3 level_divisor = (int3)
{
	AC_dimension_inactive.x ? 1 : 2,
	AC_dimension_inactive.y ? 1 : 2,
	AC_dimension_inactive.z ? 1 : 2
}

bool AC_power_of_two_minus_one_grid = false;
const int AC_gmg_maximum_level = 10;

int3 AC_ngrid_gmg_level_0 = AC_ngrid
int3 AC_ngrid_gmg_level_1 = 
				(int3)
				{
					AC_ngrid_gmg_level_0.x/level_divisor.x,
					AC_ngrid_gmg_level_0.y/level_divisor.y,
					AC_ngrid_gmg_level_0.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_2 = 
				(int3)
				{
					AC_ngrid_gmg_level_1.x/level_divisor.x,
					AC_ngrid_gmg_level_1.y/level_divisor.y,
					AC_ngrid_gmg_level_1.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_3 = 
				(int3)
				{
					AC_ngrid_gmg_level_2.x/level_divisor.x,
					AC_ngrid_gmg_level_2.y/level_divisor.y,
					AC_ngrid_gmg_level_2.z/level_divisor.z
				}

int3 AC_ngrid_gmg_level_4 = 
				(int3)
				{
					AC_ngrid_gmg_level_3.x/level_divisor.x,
					AC_ngrid_gmg_level_3.y/level_divisor.y,
					AC_ngrid_gmg_level_3.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_5 = 
				(int3)
				{
					AC_ngrid_gmg_level_4.x/level_divisor.x,
					AC_ngrid_gmg_level_4.y/level_divisor.y,
					AC_ngrid_gmg_level_4.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_6 = 
				(int3)
				{
					AC_ngrid_gmg_level_5.x/level_divisor.x,
					AC_ngrid_gmg_level_5.y/level_divisor.y,
					AC_ngrid_gmg_level_5.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_7 = 
				(int3)
				{
					AC_ngrid_gmg_level_6.x/level_divisor.x,
					AC_ngrid_gmg_level_6.y/level_divisor.y,
					AC_ngrid_gmg_level_6.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_8 = 
				(int3)
				{
					AC_ngrid_gmg_level_7.x/level_divisor.x,
					AC_ngrid_gmg_level_7.y/level_divisor.y,
					AC_ngrid_gmg_level_7.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_9 = 
				(int3)
				{
					AC_ngrid_gmg_level_8.x/level_divisor.x,
					AC_ngrid_gmg_level_8.y/level_divisor.y,
					AC_ngrid_gmg_level_8.z/level_divisor.z
				}
int3 AC_ngrid_gmg_level_10 = 
				(int3)
				{
					AC_ngrid_gmg_level_9.x/level_divisor.x,
					AC_ngrid_gmg_level_9.y/level_divisor.y,
					AC_ngrid_gmg_level_9.z/level_divisor.z
				}

int3 AC_nlocal_gmg_level_0 = AC_nlocal

int3 AC_nlocal_gmg_level_1 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_0/2 : (int3){
			       AC_nlocal_gmg_level_0.x % 2 == 0 ? (AC_nlocal_gmg_level_0.x-2)/level_divisor.x :
				       				  (AC_nlocal_gmg_level_0.x)/level_divisor.x,
			       AC_nlocal_gmg_level_0.y % 2 == 0 ? (AC_nlocal_gmg_level_0.y-2)/level_divisor.y :
				       				  (AC_nlocal_gmg_level_0.y)/level_divisor.y,
			       AC_nlocal_gmg_level_0.z % 2 == 0 ? (AC_nlocal_gmg_level_0.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_0.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_2 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_1/2 : (int3){
			       AC_nlocal_gmg_level_1.x % 2 == 0 ? (AC_nlocal_gmg_level_1.x-2)/level_divisor.x :
				       		       	  (AC_nlocal_gmg_level_1.x)/level_divisor.x,
			       AC_nlocal_gmg_level_1.y % 2 == 0 ? (AC_nlocal_gmg_level_1.y-2)/level_divisor.y :
				       		       	  (AC_nlocal_gmg_level_1.y)/level_divisor.y,
			       AC_nlocal_gmg_level_1.z % 2 == 0 ? (AC_nlocal_gmg_level_1.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_1.z)/level_divisor.z
			     }
int3 AC_nlocal_gmg_level_3 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_2/2 : (int3){
			       AC_nlocal_gmg_level_2.x % 2 == 0 ? (AC_nlocal_gmg_level_2.x-2)/level_divisor.x :
				       		       	  (AC_nlocal_gmg_level_2.x)/level_divisor.x,
			       AC_nlocal_gmg_level_2.y % 2 == 0 ? (AC_nlocal_gmg_level_2.y-2)/level_divisor.y :
				       		       	  (AC_nlocal_gmg_level_2.y)/level_divisor.y,
			       AC_nlocal_gmg_level_2.z % 2 == 0 ? (AC_nlocal_gmg_level_2.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_2.z)/level_divisor.z
			     }
int3 AC_nlocal_gmg_level_4 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_3/2 : (int3){
			       AC_nlocal_gmg_level_3.x % 2 == 0 ? (AC_nlocal_gmg_level_3.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_3.x)/level_divisor.x,
			       AC_nlocal_gmg_level_3.y % 2 == 0 ? (AC_nlocal_gmg_level_3.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_3.y)/level_divisor.y,
			       AC_nlocal_gmg_level_3.z % 2 == 0 ? (AC_nlocal_gmg_level_3.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_3.z)/level_divisor.z
			     }
int3 AC_nlocal_gmg_level_5 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_4/2 : (int3){
			       AC_nlocal_gmg_level_4.x % 2 == 0 ? (AC_nlocal_gmg_level_4.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_4.x)/level_divisor.x,
			       AC_nlocal_gmg_level_4.y % 2 == 0 ? (AC_nlocal_gmg_level_4.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_4.y)/level_divisor.y,
			       AC_nlocal_gmg_level_4.z % 2 == 0 ? (AC_nlocal_gmg_level_4.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_4.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_6 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_5/2 : (int3){
			       AC_nlocal_gmg_level_5.x % 2 == 0 ? (AC_nlocal_gmg_level_5.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_5.x)/level_divisor.x,
			       AC_nlocal_gmg_level_5.y % 2 == 0 ? (AC_nlocal_gmg_level_5.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_5.y)/level_divisor.y,
			       AC_nlocal_gmg_level_5.z % 2 == 0 ? (AC_nlocal_gmg_level_5.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_5.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_7 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_6/2 : (int3){
			       AC_nlocal_gmg_level_6.x % 2 == 0 ? (AC_nlocal_gmg_level_6.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_6.x)/level_divisor.x,
			       AC_nlocal_gmg_level_6.y % 2 == 0 ? (AC_nlocal_gmg_level_6.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_6.y)/level_divisor.y,
			       AC_nlocal_gmg_level_6.z % 2 == 0 ? (AC_nlocal_gmg_level_6.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_6.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_8 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_7/2 : (int3){
			       AC_nlocal_gmg_level_7.x % 2 == 0 ? (AC_nlocal_gmg_level_7.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_7.x)/level_divisor.x,
			       AC_nlocal_gmg_level_7.y % 2 == 0 ? (AC_nlocal_gmg_level_7.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_7.y)/level_divisor.y,
			       AC_nlocal_gmg_level_7.z % 2 == 0 ? (AC_nlocal_gmg_level_7.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_7.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_9 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_8/2 : (int3){
			       AC_nlocal_gmg_level_8.x % 2 == 0 ? (AC_nlocal_gmg_level_8.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_8.x)/level_divisor.x,
			       AC_nlocal_gmg_level_8.y % 2 == 0 ? (AC_nlocal_gmg_level_8.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_8.y)/level_divisor.y,
			       AC_nlocal_gmg_level_8.z % 2 == 0 ? (AC_nlocal_gmg_level_8.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_8.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_10 =
			     AC_power_of_two_minus_one_grid ? AC_nlocal_gmg_level_9/2 : (int3){
			       AC_nlocal_gmg_level_9.x % 2 == 0 ? (AC_nlocal_gmg_level_9.x-2)/level_divisor.x :
				       		      	  (AC_nlocal_gmg_level_0.x)/level_divisor.x,
			       AC_nlocal_gmg_level_9.y % 2 == 0 ? (AC_nlocal_gmg_level_9.y-2)/level_divisor.y :
				       		      	  (AC_nlocal_gmg_level_9.y)/level_divisor.y,
			       AC_nlocal_gmg_level_9.z % 2 == 0 ? (AC_nlocal_gmg_level_9.z-2)/level_divisor.z :
				       				  (AC_nlocal_gmg_level_9.z)/level_divisor.z
			     }

int3 AC_nlocal_gmg_level_final = AC_gmg_number_of_levels == 1 ? AC_nlocal_gmg_level_0 :
				 AC_gmg_number_of_levels == 2 ? AC_nlocal_gmg_level_1 :
				 AC_gmg_number_of_levels == 3 ? AC_nlocal_gmg_level_2 :
				 AC_gmg_number_of_levels == 4 ? AC_nlocal_gmg_level_3 :
				 AC_gmg_number_of_levels == 5 ? AC_nlocal_gmg_level_4 : 
				 AC_gmg_number_of_levels == 6 ? AC_nlocal_gmg_level_5 : 
				 AC_gmg_number_of_levels == 7 ? AC_nlocal_gmg_level_6 : 
				 AC_gmg_number_of_levels == 8 ? AC_nlocal_gmg_level_7 : 
				 AC_gmg_number_of_levels == 9 ? AC_nlocal_gmg_level_8 : 
				 AC_gmg_number_of_levels == 10 ? AC_nlocal_gmg_level_9 : 
				 AC_gmg_number_of_levels == 11 ? AC_nlocal_gmg_level_10 : AC_nlocal_gmg_level_4

get_gmg_nlocal(int level)
{
	if(level == 1)
	{
		return AC_nlocal_gmg_level_1
	}
	if(level == 2)
	{
		return AC_nlocal_gmg_level_2
	}
	if(level == 3)
	{
		return AC_nlocal_gmg_level_3
	}
	if(level == 4)
	{
		return AC_nlocal_gmg_level_4
	}
	if(level == 5)
	{
		return AC_nlocal_gmg_level_5
	}
	if(level == 6)
	{
		return AC_nlocal_gmg_level_6
	}
	if(level == 7)
	{
		return AC_nlocal_gmg_level_7
	}
	if(level == 8)
	{
		return AC_nlocal_gmg_level_8
	}
	if(level == 9)
	{
		return AC_nlocal_gmg_level_9
	}
	if(level == 10)
	{
		return AC_nlocal_gmg_level_10
	}
	return AC_nlocal_gmg_level_0
}
get_global_gmg_level_dims(int level)
{
	if(level == 1)
	{
		return AC_ngrid_gmg_level_1
	}
	if(level == 2)
	{
		return AC_ngrid_gmg_level_2
	}
	if(level == 3)
	{
		return AC_ngrid_gmg_level_3
	}
	if(level == 4)
	{
		return AC_ngrid_gmg_level_4
	}
	if(level == 5)
	{
		return AC_ngrid_gmg_level_5
	}
	if(level == 6)
	{
		return AC_ngrid_gmg_level_6
	}
	if(level == 7)
	{
		return AC_ngrid_gmg_level_7
	}
	if(level == 8)
	{
		return AC_ngrid_gmg_level_8
	}
	if(level == 9)
	{
		return AC_ngrid_gmg_level_9
	}
	if(level == 10)
	{
		return AC_ngrid_gmg_level_10
	}
	return AC_ngrid_gmg_level_0
}

int3 AC_mlocal_gmg_level_0  = AC_nlocal_gmg_level_0 + 2*AC_nmin
int3 AC_mlocal_gmg_level_1  = AC_nlocal_gmg_level_1 + 2*AC_nmin
int3 AC_mlocal_gmg_level_2  = AC_nlocal_gmg_level_2 + 2*AC_nmin
int3 AC_mlocal_gmg_level_3  = AC_nlocal_gmg_level_3 + 2*AC_nmin
int3 AC_mlocal_gmg_level_4  = AC_nlocal_gmg_level_4 + 2*AC_nmin
int3 AC_mlocal_gmg_level_5  = AC_nlocal_gmg_level_5 + 2*AC_nmin
int3 AC_mlocal_gmg_level_6  = AC_nlocal_gmg_level_6 + 2*AC_nmin
int3 AC_mlocal_gmg_level_7  = AC_nlocal_gmg_level_7 + 2*AC_nmin
int3 AC_mlocal_gmg_level_8  = AC_nlocal_gmg_level_8 + 2*AC_nmin
int3 AC_mlocal_gmg_level_9  = AC_nlocal_gmg_level_9 + 2*AC_nmin
int3 AC_mlocal_gmg_level_10 = AC_nlocal_gmg_level_10 + 2*AC_nmin
int3 AC_mlocal_gmg_level_final = AC_nlocal_gmg_level_final + 2*AC_nmin

dims(AC_mlocal_gmg_level_0)  Field GMG_SOLUTION_0
GMG_PRECISION dims(AC_mlocal_gmg_level_1)  Field GMG_SOLUTION_1 
GMG_PRECISION dims(AC_mlocal_gmg_level_2)  Field GMG_SOLUTION_2
GMG_PRECISION dims(AC_mlocal_gmg_level_3)  Field GMG_SOLUTION_3
GMG_PRECISION dims(AC_mlocal_gmg_level_4)  Field GMG_SOLUTION_4
GMG_PRECISION dims(AC_mlocal_gmg_level_5)  Field GMG_SOLUTION_5
GMG_PRECISION dims(AC_mlocal_gmg_level_6)  Field GMG_SOLUTION_6
GMG_PRECISION dims(AC_mlocal_gmg_level_7)  Field GMG_SOLUTION_7
GMG_PRECISION dims(AC_mlocal_gmg_level_8)  Field GMG_SOLUTION_8
GMG_PRECISION dims(AC_mlocal_gmg_level_9)  Field GMG_SOLUTION_9
GMG_PRECISION dims(AC_mlocal_gmg_level_10) Field GMG_SOLUTION_10

communicated auxiliary dims(AC_mlocal_gmg_level_0) Field GMG_INITIAL_RHS
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_1) Field GMG_RHS_1
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_2) Field GMG_RHS_2
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_3) Field GMG_RHS_3
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_4) Field GMG_RHS_4
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_5) Field GMG_RHS_5
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_6) Field GMG_RHS_6
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_7) Field GMG_RHS_7
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_8) Field GMG_RHS_8
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_9) Field GMG_RHS_9
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_10) Field GMG_RHS_10
//Need to be communicated if use other smoothers than gauss-seidel or jacobi
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_0) Field GMG_RESIDUAL_0
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_1) Field GMG_RESIDUAL_1 
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_2) Field GMG_RESIDUAL_2
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_3) Field GMG_RESIDUAL_3
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_4) Field GMG_RESIDUAL_4
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_5) Field GMG_RESIDUAL_5
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_6) Field GMG_RESIDUAL_6
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_7) Field GMG_RESIDUAL_7
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_8) Field GMG_RESIDUAL_8
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_9) Field GMG_RESIDUAL_9
GMG_PRECISION communicated auxiliary dims(AC_mlocal_gmg_level_10) Field GMG_RESIDUAL_10

auxiliary dims(AC_mlocal_gmg_level_1) Field GMG_TMP_1 
auxiliary dims(AC_mlocal_gmg_level_2) Field GMG_TMP_2
auxiliary dims(AC_mlocal_gmg_level_3) Field GMG_TMP_3
auxiliary dims(AC_mlocal_gmg_level_4) Field GMG_TMP_4
auxiliary dims(AC_mlocal_gmg_level_5) Field GMG_TMP_5
auxiliary dims(AC_mlocal_gmg_level_6) Field GMG_TMP_6
auxiliary dims(AC_mlocal_gmg_level_7) Field GMG_TMP_7
auxiliary dims(AC_mlocal_gmg_level_8) Field GMG_TMP_8
auxiliary dims(AC_mlocal_gmg_level_9) Field GMG_TMP_9
auxiliary dims(AC_mlocal_gmg_level_10) Field GMG_TMP_10

dconst real AC_GMG_CENTRAL_COEFFS[11]

const Field GMG_SOLUTIONS =
	[
		GMG_SOLUTION_0,
		GMG_SOLUTION_1,
		GMG_SOLUTION_2,
		GMG_SOLUTION_3,
		GMG_SOLUTION_4,
		GMG_SOLUTION_5,
		GMG_SOLUTION_6,
		GMG_SOLUTION_7,
		GMG_SOLUTION_8,
		GMG_SOLUTION_9,
		GMG_SOLUTION_10
	]

const Field GMG_RHS =
	[
		GMG_INITIAL_RHS,
		GMG_RHS_1,
		GMG_RHS_2,
		GMG_RHS_3,
		GMG_RHS_4,
		GMG_RHS_5,
		GMG_RHS_6,
		GMG_RHS_7,
		GMG_RHS_8,
		GMG_RHS_9,
		GMG_RHS_10
	]

const Field GMG_RESIDUALS =
	[
		GMG_RESIDUAL_0,
		GMG_RESIDUAL_1,
		GMG_RESIDUAL_2,
		GMG_RESIDUAL_3,
		GMG_RESIDUAL_4,
		GMG_RESIDUAL_5,
		GMG_RESIDUAL_6,
		GMG_RESIDUAL_7,
		GMG_RESIDUAL_8,
		GMG_RESIDUAL_9,
		GMG_RESIDUAL_10
	]

const Field GMG_TMPS =
	[
		GMG_TMP_1,
		GMG_TMP_1,
		GMG_TMP_2,
		GMG_TMP_3,
		GMG_TMP_4,
		GMG_TMP_5,
		GMG_TMP_6,
		GMG_TMP_7,
		GMG_TMP_8,
		GMG_TMP_9,
		GMG_TMP_10
	]

enum GMG_LEVEL
{
	GMG_LEVEL_0,
	GMG_LEVEL_1,
	GMG_LEVEL_2,
	GMG_LEVEL_3,
	GMG_LEVEL_4,
	GMG_LEVEL_5,
	GMG_LEVEL_6,
	GMG_LEVEL_7,
	GMG_LEVEL_8,
	GMG_LEVEL_9,
	GMG_LEVEL_10
}


input GMG_LEVEL AC_GMG_LEVEL

Kernel gmg_restrict_residual_kernel(GMG_LEVEL level)
{
	if(level < AC_gmg_maximum_level) 
	{
		res = restrict_full_weighting(GMG_RESIDUALS[level],get_global_gmg_level_dims(level))
		write(GMG_RHS[level+1],res)
		//The residual is most likely small so zero is a meaningful starting value
		write(GMG_SOLUTIONS[level+1],0.0)
	}
}

Kernel gmg_get_correction_from_next_level_kernel(GMG_LEVEL level)
{
	if(level < AC_gmg_maximum_level)
	{
		e = trilinear_prolongation(GMG_SOLUTIONS[level+1],get_global_gmg_level_dims(level))
		write(GMG_SOLUTIONS[level],GMG_SOLUTIONS[level]+e)
	}
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
Kernel
gmg_write_hat_basis_kernel(GMG_LEVEL level)
{
	coarse_dims = get_gmg_nlocal(level)
	hat_basis_position = (coarse_dims/2) + AC_nmin
	if(vertexIdx == hat_basis_position)
	{
		write(GMG_SOLUTIONS[level],1.0)
	}
	else
	{
		write(GMG_SOLUTIONS[level],0.0)
	}
	write(GMG_RESIDUALS[level],0.0)
}
ComputeSteps
gmg_write_hat_basis(gmg_boundconds)
{
	gmg_write_hat_basis_kernel(AC_GMG_LEVEL)
}

Kernel
gmg_copy_residual_to_tmp_kernel(GMG_LEVEL level)
{
	write(GMG_TMPS[level],GMG_RESIDUALS[level])
}

ComputeSteps
gmg_copy_residual_to_tmp(gmg_boundconds)
{
	gmg_copy_residual_to_tmp_kernel(AC_GMG_LEVEL)
}
