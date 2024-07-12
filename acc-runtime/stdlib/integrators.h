#define RK_ORDER (3)

rk3(real s0, real s1, real roc, int step_num, real dt) {
#if RK_ORDER == 1
    // Euler
    real alpha= 0.0, 0.0, 0.0, 0.0
    real beta = 0.0, 1.0, 0.0, 0.0
#elif RK_ORDER == 2
    real alpha= 0.0,     0.0, -1.0/2.0, 0.0
    real beta = 0.0, 1.0/2.0,      1.0, 0.0
#elif RK_ORDER == 3
    real alpha = 0., -5./9., -153./128.
    real beta = 1./3., 15./ 16., 8./15.
#endif
    /*
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below
    if AC_step_number > 0 {
        return s1 + beta[AC_step_number] * ((alpha[AC_step_number] / beta[AC_step_number - 1]) * (s1 - s0) + roc * AC_dt)
    } else {
        return s1 + beta[AC_step_number] * roc * AC_dt
    }
    */
    // Workaround
    return s1 + beta[step_num + 1] * ((alpha[step_num] / beta[step_num]) * (s1 - s0) + roc * dt)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3(real3 f,real3 w,real3 roc,int step_num,real dt){
  return real3( rk3(f.x,w.x,roc.x,step_num,dt),
                rk3(f.y,w.y,roc.y,step_num,dt),
                rk3(f.z,w.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(real w, real roc, int step_num, real dt) {
    real alpha = 0., -5./9., -153./128.

    // return alpha[AC_step_number] * w + roc * AC_dt
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below

    //if step_num > 0 {
    //    return alpha[step_num] * w + roc * AC_dt
    //} else {
    //    return roc * AC_dt
    //}
    return alpha[step_num] * w + roc * dt
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(real3 w,real3 roc,int step_num,real dt){
  return real3( rk3_intermediate(w.x,roc.x,step_num,dt),
                rk3_intermediate(w.y,roc.y,step_num,dt),
                rk3_intermediate(w.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(real f, real w, int step_num) {
    real beta = 1./3., 15./16., 8./15.
    return f + beta[step_num] * w
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(real3 f,real3 w,int step_num){
  return real3( rk3_final(f.x,w.x,step_num),
                rk3_final(f.y,w.y,step_num),
                rk3_final(f.z,w.z,step_num)
              )
}
euler(real f, real update, real dt_in)
{
	return f + update*dt_in
}

