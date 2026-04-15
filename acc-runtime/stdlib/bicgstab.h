/**
 * This file implements the BICGSTAB method for solving sparse linear systems.
 * See Saad's Iterative Methods for Sparse Linear Systems Chapter 7. for a good reference.
 * The method is good for our use cases because of its low memory requirements.
 */
global output real BICGSTAB_r0Tv
global output real BICGSTAB_rho_prev
global output real BICGSTAB_rho_next
global output real BICGSTAB_tTs
global output real BICGSTAB_tTt
Field BICGSTAB_S
Field BICGSTAB_H
Field BICGSTAB_R
Field BICGSTAB_R0
Field BICGSTAB_T
Field BICGSTAB_P
Field BICGSTAB_V
Field BICGSTAB_Y
Field BICGSTAB_Z

bicgstab_compute_v_and_r0Tv(real v)
{
	write(BICGSTAB_V,v)
	reduce_sum(BICGSTAB_R0*v,BICGSTAB_r0Tv)
}

bicgstab_compute_h_and_s(Field x)
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	write(BICGSTAB_S,BICGSTAB_R - alpha*BICGSTAB_V)
	write(BICGSTAB_H,x + alpha*BICGSTAB_P)
}

bicgstab_compute_h_and_s_preconditioned(Field x)
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	write(BICGSTAB_S,BICGSTAB_R - alpha*BICGSTAB_V)
	write(BICGSTAB_H,x + alpha*BICGSTAB_Y)
}

bicgstab_compute_t(real As)
{
	t = As
	write(BICGSTAB_T,As)
	reduce_sum(t*t,BICGSTAB_tTt)
	reduce_sum(t*BICGSTAB_S,BICGSTAB_tTs)
}

bicgstab_compute_next_solution(Field x)
{
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	write(x,BICGSTAB_H + omega*BICGSTAB_S)
	write(BICGSTAB_R,BICGSTAB_S - omega*BICGSTAB_T)
}

bicgstab_compute_next_solution_preconditioned(Field x)
{
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	write(x,BICGSTAB_H + omega*BICGSTAB_Z)
	write(BICGSTAB_R,BICGSTAB_S - omega*BICGSTAB_T)
}
bicgstab_compute_rho_next() 
{
	reduce_sum(BICGSTAB_R0*BICGSTAB_R,BICGSTAB_rho_next)
}

bicgstab_compute_rho_prev() 
{
	reduce_sum(BICGSTAB_R0*BICGSTAB_R,BICGSTAB_rho_prev)
}

bicgstab_update_p() 
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	beta = (BICGSTAB_rho_next/BICGSTAB_rho_prev)*(alpha/omega)
	write(BICGSTAB_P,BICGSTAB_R + beta*(BICGSTAB_P - omega*BICGSTAB_V))
}

dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_S
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_H
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_R
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_R0
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_T
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_P
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_V
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_Y
dims(AC_extended_mlocal) Field EXTENDED_BICGSTAB_Z

bicgstab_compute_v_and_r0Tv_extended(real v)
{
	write(EXTENDED_BICGSTAB_V,v)
	reduce_sum(EXTENDED_BICGSTAB_R0*v,BICGSTAB_r0Tv)
}

bicgstab_compute_h_and_s_extended(Field x)
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	write(EXTENDED_BICGSTAB_S,EXTENDED_BICGSTAB_R - alpha*EXTENDED_BICGSTAB_V)
	write(EXTENDED_BICGSTAB_H,x + alpha*EXTENDED_BICGSTAB_P)
}

bicgstab_compute_h_and_s_preconditioned_extended(Field x)
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	write(EXTENDED_BICGSTAB_S,EXTENDED_BICGSTAB_R - alpha*EXTENDED_BICGSTAB_V)
	write(EXTENDED_BICGSTAB_H,x + alpha*EXTENDED_BICGSTAB_Y)
}

bicgstab_compute_t_extended(real As)
{
	t = As
	write(EXTENDED_BICGSTAB_T,As)
	reduce_sum(t*t,BICGSTAB_tTt)
	reduce_sum(t*EXTENDED_BICGSTAB_S,BICGSTAB_tTs)
}

bicgstab_compute_next_solution_extended(Field x)
{
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	write(x,EXTENDED_BICGSTAB_H + omega*EXTENDED_BICGSTAB_S)
	write(EXTENDED_BICGSTAB_R,EXTENDED_BICGSTAB_S - omega*EXTENDED_BICGSTAB_T)
}

bicgstab_compute_next_solution_preconditioned_extended(Field x)
{
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	write(x,EXTENDED_BICGSTAB_H + omega*EXTENDED_BICGSTAB_Z)
	write(EXTENDED_BICGSTAB_R,EXTENDED_BICGSTAB_S - omega*EXTENDED_BICGSTAB_T)
}
bicgstab_compute_rho_next_extended() 
{
	reduce_sum(EXTENDED_BICGSTAB_R0*EXTENDED_BICGSTAB_R,BICGSTAB_rho_next)
}

bicgstab_compute_rho_prev_extended() 
{
	reduce_sum(EXTENDED_BICGSTAB_R0*EXTENDED_BICGSTAB_R,BICGSTAB_rho_prev)
}

bicgstab_update_p_extended() 
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	beta = (BICGSTAB_rho_next/BICGSTAB_rho_prev)*(alpha/omega)
	write(EXTENDED_BICGSTAB_P,EXTENDED_BICGSTAB_R + beta*(EXTENDED_BICGSTAB_P - omega*EXTENDED_BICGSTAB_V))
}

