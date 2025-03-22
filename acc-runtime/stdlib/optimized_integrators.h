const real rk1_alpha = [0.0]
const real rk1_beta  = [1.0]

const real rk2_alpha = [0.0,      -1/2.0]
const real rk2_beta  = [1.0/2.0,   1.0]

const real rk3_alpha =[ 0.,   -5./9.,   -153./128. ]
const real rk3_beta  =[ 1./3., 15./ 16., 8./15.    ]

rk3(Field f, real roc, int step_num, real dt) {
    /*
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below
    if AC_step_number > 0 {
        return s1 + rk_beta[AC_step_number] * ((rk_alpha[AC_step_number] / rk_beta[AC_step_number - 1]) * (s1 - s0) + roc * AC_dt)
    } else {
        return s1 + rk_beta[AC_step_number] * roc * AC_dt
    }
    */
    // Workaround
    const real s1 = previous(f)
    const real s0 = value(f)
    return s1 + rk3_beta[step_num + 1] * ((rk3_alpha[step_num] / rk3_beta[step_num]) * (s1 - s0) + roc * dt)
} /*--------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3(Field3 field, real3 roc, int step_num, real dt) {
	return real3(
			rk3(field.x,roc.x,step_num,dt),
			rk3(field.y,roc.y,step_num,dt),
			rk3(field.z,roc.z,step_num,dt)
		    )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk1_intermediate(Field f, real roc, int step_num, real dt) {
    //TP: rk1_alpha is always zero
    //return rk1_alpha[step_num] * previous(f) + roc * dt
    suppress_unused_warning(step_num)
    return roc*dt;
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk1_intermediate(Field3 f, real3 roc, int step_num, real dt)
{
  return real3( rk1_intermediate(f.x,roc.x,step_num,dt),
                rk1_intermediate(f.y,roc.y,step_num,dt),
                rk1_intermediate(f.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk1_final(Field f,int step_num) {
    return previous(f) + rk1_beta[step_num] * value(f)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk1_final(Field3 f, int step_num){
  return real3( rk1_final(f.x,step_num),
                rk1_final(f.y,step_num),
                rk1_final(f.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk1_final(real f, real w, int step_num) {
    return f + rk1_beta[step_num] * w
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk1_final(real3 f,real3 w,int step_num){
  return real3( rk1_final(f.x,w.x,step_num),
                rk1_final(f.y,w.y,step_num),
                rk1_final(f.z,w.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk2_intermediate(Field f, real roc, int step_num, real dt) {
    previous_value = step_num > 0 ? previous(f) : 0.0;
    return rk2_alpha[step_num] * previous_value + roc * dt
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk2_intermediate(Field3 f, real3 roc, int step_num, real dt)
{
  return real3( rk2_intermediate(f.x,roc.x,step_num,dt),
                rk2_intermediate(f.y,roc.y,step_num,dt),
                rk2_intermediate(f.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk2_final(Field f,int step_num) {
    return previous(f) + rk2_beta[step_num] * value(f)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk2_final(Field3 f, int step_num){
  return real3( rk2_final(f.x,step_num),
                rk2_final(f.y,step_num),
                rk2_final(f.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk2_final(real f, real w, int step_num) {
    return f + rk3_beta[step_num] * w
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk2_final(real3 f,real3 w,int step_num){
  return real3( rk3_final(f.x,w.x,step_num),
                rk3_final(f.y,w.y,step_num),
                rk3_final(f.z,w.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(Field f, real roc, int step_num, real dt) {
    previous_value = step_num > 0 ? previous(f) : 0.0;
    return rk3_alpha[step_num] * previous_value + roc * dt
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(Field3 f, real3 roc, int step_num, real dt)
{
  return real3( rk3_intermediate(f.x,roc.x,step_num,dt),
                rk3_intermediate(f.y,roc.y,step_num,dt),
                rk3_intermediate(f.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(Field f,int step_num) {
    return previous(f) + rk3_beta[step_num] * value(f)
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(Field3 f, int step_num){
  return real3( rk3_final(f.x,step_num),
                rk3_final(f.y,step_num),
                rk3_final(f.z,step_num)
              )
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk3_final(real f, real w, int step_num) {
    return f + rk3_beta[step_num] * w
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_final(real3 f,real3 w,int step_num){
  return real3( rk3_final(f.x,w.x,step_num),
                rk3_final(f.y,w.y,step_num),
                rk3_final(f.z,w.z,step_num)
              )
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
    return rk3_alpha[step_num] * w + roc * dt
}
/*--------------------------------------------------------------------------------------------------------------------------*/
rk3_intermediate(real3 w,real3 roc,int step_num,real dt){
  return real3( rk3_intermediate(w.x,roc.x,step_num,dt),
                rk3_intermediate(w.y,roc.y,step_num,dt),
                rk3_intermediate(w.z,roc.z,step_num,dt)
              )
}
/*--------------------------------------------------------------------------------------------------------------------------*/

const real rk4f_beta= [1153189308089./22510343858157.,
                1772645290293./4653164025191.,
               -1672844663538./4480602732383.,
                2114624349019./3568978502595.,
                5198255086312./14908931495163.]
const real rk4f_bhat= [   1016888040809./7410784769900.,
               11231460423587./58533540763752.,
               -1563879915014./6823010717585.,
                 606302364029./971179775848.,
                1097981568119./3980877426909.]
const real rk4f_alpha=[ 970286171893./4311952581923.,
                6584761158862./12103376702013.,
                2251764453980./15575788980749.,
               26877169314380./34165994151039., 0.0]

rk4_alpha(Field f_alpha, real roc, int step_num, real dt) {
    // explicit runge-kutta 4th vs 3rd order 3 register 5-step scheme
    return f_alpha + rk4f_alpha[step_num]*roc*dt
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk4_beta(Field f_beta, real roc, int step_num, real dt) {
    // explicit runge-kutta 4th vs 3rd order 3 register 5-step scheme
    return f_beta + rk4f_beta[step_num]*roc*dt
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
	return dt*(rk4f_beta[step_num] - rk4f_bhat[step_num])*df
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
run_const int AC_rk_order
/*--------------------------------------------------------------------------------------------------------------------*/
rk_intermediate(Field f, real df, int step_num, real dt)
{
	if(AC_rk_order == 1)
		return rk1_intermediate(f,df,step_num,dt)
	else if(AC_rk_order == 2)
		return rk2_intermediate(f,df,step_num,dt)
	else if(AC_rk_order == 3)
		return rk3_intermediate(f,df,step_num,dt)
	else
		return 0.0;
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_intermediate(Field3 f, real3 df, int step_num, real dt)
{
	return real3(
			rk_intermediate(f.x,df.x,step_num,dt),
			rk_intermediate(f.y,df.y,step_num,dt),
			rk_intermediate(f.z,df.z,step_num,dt)
			)
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final(Field f, int step_num)
{
	if(AC_rk_order == 1)
		return rk1_final(f,step_num)
	else if(AC_rk_order == 2)
		return rk2_final(f,step_num)
	else if(AC_rk_order == 3)
		return rk3_final(f,step_num)
	else 
		return 0.0;
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final(real f, real w, int step_num)
{
	if(AC_rk_order == 1)
		return rk1_final(f,w,step_num)
	else if(AC_rk_order == 2)
		return rk2_final(f,w,step_num)
	else if(AC_rk_order == 3)
		return rk3_final(f,w,step_num)
	else 
		return 0.0;
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final(Field3 f, int step_num)
{
	if(AC_rk_order == 1)
		return rk1_final(f,step_num)
	else if(AC_rk_order == 2)
		return rk2_final(f,step_num)
	else if(AC_rk_order == 3)
		return rk3_final(f,step_num)
	else 
		return real3(0.0,0.0,0.0);
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final(real3 f, real3 w, int step_num)
{
	if(AC_rk_order == 1)
		return rk1_final(f,w,step_num)
	else if(AC_rk_order == 2)
		return rk2_final(f,w,step_num)
	else if(AC_rk_order == 3)
		return rk3_final(f,w,step_num)
	else 
		return real3(0.0,0.0,0.0);
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_number_of_substeps()
{
	if(AC_rk_order == 4)
	{
		return 5;
	}
	return AC_rk_order;
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_intermediate_update(Field f, real df, int step_num, real dt)
{
		write(f,rk_intermediate(f,df,step_num,dt))
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_intermediate_update(Field3 f, real3 df, int step_num, real dt)
{
		write(f,rk_intermediate(f,df,step_num,dt))
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final_update(Field f, int step_num)
{
	write(f,rk_final(f,step_num))
}
/*--------------------------------------------------------------------------------------------------------------------*/
rk_final_update(Field3 f, int step_num)
{
	write(f,rk_final(f,step_num))
}
/*--------------------------------------------------------------------------------------------------------------------*/
