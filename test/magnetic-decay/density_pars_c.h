for (int i=0;i<n_pars_density;i++)p_par_density[i]=NULL;
pushpars2c$density_(p_par_density);
 PCLoad(config,AC_ldiff_shock,ldiff_shock); // [1-1] ldiff_shock int
 PCLoad(config,AC_diffrho_shock,diffrho_shock); // [2-1] diffrho_shock real
 PCLoad(config,AC_ldiff_hyper3lnrho,ldiff_hyper3lnrho); // [3-1] ldiff_hyper3lnrho int
 PCLoad(config,AC_diffrho_hyper3,diffrho_hyper3); // [4-1] diffrho_hyper3 real
 PCLoad(config,AC_lupw_lnrho,lupw_lnrho); // [5-1] lupw_lnrho int
