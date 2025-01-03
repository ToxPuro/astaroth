for (int i=0;i<n_pars_viscosity;i++)p_par_viscosity[i]=NULL;
pushpars2c$viscosity_(p_par_viscosity);
 PCLoad(config,AC_nu,nu); // [1-1] nu real
 PCLoad(config,AC_zeta,zeta); // [2-1] zeta real
 PCLoad(config,AC_nu_hyper3,nu_hyper3); // [3-1] nu_hyper3 real
 PCLoad(config,AC_nu_shock,nu_shock); // [4-1] nu_shock real
 PCLoad(config,AC_lvisc_nu_const,lvisc_nu_const); // [5-1] lvisc_nu_const int
 PCLoad(config,AC_lvisc_hyper3_nu_const,lvisc_hyper3_nu_const); // [6-1] lvisc_hyper3_nu_const int
 PCLoad(config,AC_lvisc_nu_shock,lvisc_nu_shock); // [7-1] lvisc_nu_shock int
 PCLoad(config,AC_nu_hyper2,nu_hyper2); // [8-1] nu_hyper2 real
