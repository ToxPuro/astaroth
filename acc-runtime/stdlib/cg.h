/**
 * This file implements the CG method for solving sparse symmetric positive definite linear systems.
 * See Saad's Iterative Methods for Sparse Linear Systems Chapter 6. for a good reference.
 * The method is good for our use cases because of its low memory requirements.
 */
global output real CG_rTz
global output real CG_pTAp
global output real CG_rp1Tzp1


run_const bool AC_cg_preconditioned = false

cg_compute_inner_products(real Ap,real p,real r, real z)
{
	reduce_sum(r*z,CG_rTz)	
	reduce_sum(p*Ap,CG_pTAp)	
}

cg_compute_inner_products(real Ap,real p,real r)
{
	cg_compute_inner_products(Ap,p,r,r)
}


cg_advance_solution(Field x,real Ap,real p,Field r)
{
	alpha = CG_rTz/CG_pTAp
	write(x,x + alpha*p)
	rp1 = r - alpha*Ap
	write(r,rp1)
	if(!AC_cg_preconditioned)
	{
		reduce_sum(rp1*rp1,CG_rp1Tzp1)	
	}
	
}

cg_compute_rp1Tzp1(real rp1, real zp1)
{
	if(AC_cg_preconditioned)
	{
		reduce_sum(rp1*zp1,CG_rp1Tzp1)	
	}
}

cg_next_direction(Field p, real z)
{
	beta = CG_rp1Tzp1/CG_rTz
	write(p, z + beta*p)
}

output real CG_local_rTz
output real CG_local_pTAp
output real CG_local_rp1Tzp1

cg_compute_inner_products_local(real Ap,real p,real r, real z)
{
	reduce_sum(r*z, CG_local_rTz)	
	reduce_sum(p*Ap,CG_local_pTAp)	
}

cg_compute_inner_products_local(real Ap,real p,real r)
{
	cg_compute_inner_products_local(Ap,p,r,r)
}


cg_advance_solution_local(Field x,real Ap,real p,Field r)
{
	alpha = CG_local_rTz/CG_local_pTAp
	write(x,x + alpha*p)
	rp1 = r - alpha*Ap
	write(r,rp1)
	if(!AC_cg_preconditioned)
	{
		reduce_sum(rp1*rp1,CG_local_rp1Tzp1)	
	}
	
}

cg_compute_rp1Tzp1_local(real rp1, real zp1)
{
	if(AC_cg_preconditioned)
	{
		reduce_sum(rp1*zp1,CG_local_rp1Tzp1)	
	}
}

cg_next_direction_local(Field p, real z)
{
	beta = CG_local_rp1Tzp1/CG_local_rTz
	write(p, z + beta*p)
}
