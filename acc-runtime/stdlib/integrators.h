#define RK_ORDER (4)

#if RK_ORDER == 1
    // Euler
    const real rk_alpha = [0.0, 0.0, 0.0, 0.0]
    const real rk_beta  = [0.0, 1.0, 0.0, 0.0]
#elif RK_ORDER == 2
    const real rk_alpha = [0.0,     0.0, -1.0/2.0, 0.0]
    const real rk_beta  = [0.0, 1.0/2.0,      1.0, 0.0]
#elif RK_ORDER == 3
    const real rk_alpha =[ 0., -5./9., -153./128. ]
    const real rk_beta  =[ 1./3., 15./ 16., 8./15.]
#elif RK_ORDER == 4
    const real rk_beta= [1153189308089./22510343858157.,
                    1772645290293./4653164025191.,
                   -1672844663538./4480602732383.,
                    2114624349019./3568978502595.,
                    5198255086312./14908931495163.]
    const real bhat= [   1016888040809./7410784769900.,
                   11231460423587./58533540763752.,
                   -1563879915014./6823010717585.,
                     606302364029./971179775848.,
                    1097981568119./3980877426909.]
    const real rk_alpha=[ 970286171893./4311952581923.,
                    6584761158862./12103376702013.,
                    2251764453980./15575788980749.,
                   26877169314380./34165994151039., 0.0]
#endif

#if RK_ORDER == 3
rk3(real s0, real s1, real roc, int step_num, real dt) {
    /*
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below
    if AC_step_number > 0 {
        return s1 + rk_beta[AC_step_number] * ((rk_alpha[AC_step_number] / rk_beta[AC_step_number - 1]) * (s1 - s0) + roc * AC_dt)
    } else {
        return s1 + rk_beta[AC_step_number] * roc * AC_dt
    }
    */
    // Workaround
    return s1 + rk_beta[step_num + 1] * ((rk_alpha[step_num] / rk_beta[step_num]) * (s1 - s0) + roc * dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3(real3 f,real3 w,real3 roc,int step_num,real dt){
  return real3( rk3(f.x,w.x,roc.x,step_num,dt),
                rk3(f.y,w.y,roc.y,step_num,dt),
                rk3(f.z,w.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3(Field field, real roc, int step_num, real dt) {
	return rk3(previous(field), value(field), roc, step_num, dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3(Field3 field, real3 roc, int step_num, real dt) {
	return rk3(previous(field), value(field), roc, step_num, dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(real w, real roc, int step_num, real dt) {
    // return rk_alpha[AC_step_number] * w + roc * AC_dt
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below

    //if step_num > 0 {
    //    return rk_alpha[step_num] * w + roc * AC_dt
    //} else {
    //    return roc * AC_dt
    //}
    return rk_alpha[step_num] * w + roc * dt
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(real3 w,real3 roc,int step_num,real dt){
  return real3( rk3_intermediate(w.x,roc.x,step_num,dt),
                rk3_intermediate(w.y,roc.y,step_num,dt),
                rk3_intermediate(w.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(Field field, real roc, int step_num, real dt)
{
	return rk3_intermediate(previous(field), roc, step_num, dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(Field3 field, real3 roc, int step_num, real dt)
{
	return rk3_intermediate(previous(field), roc, step_num, dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(real f, real w, int step_num) {
    return f + rk_beta[step_num] * w
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(real3 f,real3 w,int step_num){
  return real3( rk3_final(f.x,w.x,step_num),
                rk3_final(f.y,w.y,step_num),
                rk3_final(f.z,w.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(Field field, int step_num) {
    return previous(field) + rk_beta[step_num] * value(field)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(Field3 field, int step_num) {
  return real3( rk3_final(field.x,step_num),
                rk3_final(field.y,step_num),
                rk3_final(field.z,step_num)
              )
}
#endif
/*--------------------------------------------------------------------------------------------------------------------------*/
euler(real f, real update, real dt_in)
{
	return f + update*dt_in
}
/*--------------------------------------------------------------------------------------------------------------------------*/
#if RK_ORDER == 4
/*--------------------------------------------------------------------------------------------------------------------*/
rk4_alpha(Field f_alpha, real roc, int step_num, real dt) {
    // explicit runge-kutta 4th vs 3rd order 3 register 5-step scheme
    return f_alpha + rk_alpha[step_num]*roc*dt
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk4_beta(Field f_beta, real roc, int step_num, real dt) {
    // explicit runge-kutta 4th vs 3rd order 3 register 5-step scheme
    return f_beta + rk_beta[step_num]*roc*dt
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk4_alpha(Field3 f_alpha, real3 roc, int step_num, real dt) {
	return real3(
			rk4_alpha(f_alpha.x,roc.x,step_num,dt),
			rk4_alpha(f_alpha.y,roc.y,step_num,dt),
			rk4_alpha(f_alpha.z,roc.z,step_num,dt)
		    )
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk4_beta(Field3 f_beta, real3 roc, int step_num, real dt) {
	return real3(
			rk4_beta(f_beta.x,roc.x,step_num,dt),
			rk4_beta(f_beta.y,roc.y,step_num,dt),
			rk4_beta(f_beta.z,roc.z,step_num,dt)
		    )
}
rk4_error(real df, int step_num,real dt)
{
	return dt*(rk_beta[step_num] - bhat[step_num])*df
}
rk4_error(real3 df, int step_num, real dt)
{
	return 
		real3(
				rk4_error(df.x,step_num,dt),
				rk4_error(df.y,step_num,dt),
				rk4_error(df.z,step_num,dt)
		     )
}
/*--------------------------------------------------------------------------------------------------------------------*/
#endif

