for (int i=0;i<n_pars_magnetic;i++)p_par_magnetic[i]=NULL;
pushpars2c$magnetic_(p_par_magnetic);
 PCLoad(config,AC_eta,eta); // [1-1] eta real
 PCLoad(config,AC_eta_hyper2,eta_hyper2); // [2-1] eta_hyper2 real
 PCLoad(config,AC_eta_hyper3,eta_hyper3); // [3-1] eta_hyper3 real
 PCLoad(config,AC_lresi_eta_const,lresi_eta_const); // [4-1] lresi_eta_const int
 PCLoad(config,AC_lresi_hyper2,lresi_hyper2); // [5-1] lresi_hyper2 int
 PCLoad(config,AC_lresi_hyper3,lresi_hyper3); // [6-1] lresi_hyper3 int
 PCLoad(config,AC_lupw_aa,lupw_aa); // [7-1] lupw_aa int
 PCLoad(config,AC_llorentzforce,llorentzforce); // [8-1] llorentzforce int
 PCLoad(config,AC_linduction,linduction); // [9-1] linduction int
