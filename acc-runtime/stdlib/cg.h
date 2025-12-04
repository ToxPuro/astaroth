global output real CG_rTr
global output real CG_pTAp
global output real CG_rp1Trp1

cg_compute_inner_products(real Ap,real p,real r)
{
	reduce_sum(r*r,CG_rTr)	
	reduce_sum(p*Ap,CG_pTAp)	
}

cg_advance_solution(Field x,real Ap,real p,Field r)
{
	alpha = CG_rTr/CG_pTAp
	write(x,x + alpha*p)
	rp1 = r - alpha*Ap
	write(r,rp1)
	reduce_sum(rp1*rp1,CG_rp1Trp1)	
	
}

cg_next_direction(Field p, Field r)
{
	beta = CG_rp1Trp1/CG_rTr
	write(p, r + beta*p)
}
