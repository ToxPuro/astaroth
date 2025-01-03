for (int i=0;i<n_pars_equationofstate;i++)p_par_equationofstate[i]=NULL;
pushpars2c$equationofstate_(p_par_equationofstate);
 PCLoad(config,AC_cs20,cs20); // [1-1] cs20 real
 PCLoad(config,AC_gamma,gamma); // [2-1] gamma real
 PCLoad(config,AC_cv,cv); // [3-1] cv real
 PCLoad(config,AC_cp,cp); // [4-1] cp real
 PCLoad(config,AC_lnrho0,lnrho0); // [5-1] lnrho0 real
 PCLoad(config,AC_lnTT0,lnTT0); // [6-1] lnTT0 real
 PCLoad(config,AC_gamma_m1,gamma_m1); // [7-1] gamma_m1 real
 PCLoad(config,AC_gamma1,gamma1); // [8-1] gamma1 real
 PCLoad(config,AC_cv1,cv1); // [9-1] cv1 real
 PCLoad(config,AC_cs2bot,cs2bot); // [10-1] cs2bot real
 PCLoad(config,AC_cs2top,cs2top); // [11-1] cs2top real
 PCLoad(config,AC_leos_isothermal,leos_isothermal); // [12-1] leos_isothermal int
