global output real BICGSTAB_r0Tv
global output real BICGSTAB_rho_prev
global output real BICGSTAB_rho_current
global output real BICGSTAB_tTs
global output real BICGSTAB_tTt
Field BICGSTAB_S
Field BICGSTAB_H
Field BICGSTAB_R
Field BICGSTAB_T
Field BICGSTAB_P
Field BICGSTAB_V

bicgstab_compute_r0Tv(real v, real r0)
{
	write(BICGSTAB_V,v)
	reduce_sum(r0*v,BICGSTAB_r0Tv)
}

bicgstab_compute_h_and_s(Field x, real v)
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	write(BICGSTAB_S,BICGSTAB_R - alpha*v)
	write(BICGTAB_H,x + alpha*BICGSTAB_P)
}

bicgstab_compute_t(real As)
{
	t = As
	write(BICGSTAB_T,As)
	reduce_sum(BICGSTAB_tTt,t*t)
	reduce_sum(BICGSTAB_tTs,t*s)
}

bicgstab_compute_next_solution(Field x)
{
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	write(x,BICGSTAB_H + omega*BICGSTAB_S)
	write(BICGSTAB_R,BICGSTAB_S - omega*BICGSTAB_T)
}
bicgstab_compute_rho_next(real r0) 
{
	reduce_sum(r0*BICGSTAB_R,BICGSTAB_rho_next)
}

bicgstab_compute_rho_prev(real r0) 
{
	reduce_sum(r0*BICGSTAB_R,BICGSTAB_rho_prev)
}

bicgstab_update_p() 
{
	alpha = BICGSTAB_rho_prev/BICGSTAB_r0Tv
	omega = BICGSTAB_tTs/BICGSTAB_tTt
	beta = (BICGSTAB_rho_next/BICGSTAB_rho_prev)*(alpha/omega)
	write(BICGSTAP_P,BICGSTAB_R + beta*(BICGSTAB_P - omega*BICGSTAB_V)
}

