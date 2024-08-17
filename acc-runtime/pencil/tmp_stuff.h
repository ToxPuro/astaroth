//TP: some temporary work for convection that Matthias can look into if he wants
//TP: handwritten temp implementation of bc_ss_flux 
Kernel entropy_prescribed_heat_flux(int3 normal, real F_boundary)
{
    int3 boundary = get_boundary_from_normal(normal)

    real rho_xy = exp(RHO[boundary.x][boundary.y][boundary.z]);
    real cs2_xy = SS[boundary.x][boundary.y][boundary.z];
    cs2_xy= AC_cs20 *exp(AC_gamma_m1 * (RHO[boundary.x][boundary.y][boundary.z] - AC_lnrho0) + AC_cv1 * cs2_xy);
    //Const chi
    real tmp = F_boundary/(rho_xy * AC_chi * cs2_xy)
    //Kramers
    //real tmp            = F_boundary * pow(rho_xy, 2.0 * AC_nkramers) * pow(AC_cp * AC_gamma_m1, 6.5 * AC_nkramers) /
    //             (AC_hcond0_kramers * pow(cs2_xy, 6.5 * AC_nkramers + 1.0));
    real d = get_d_from_normal(normal)
    int3 domain = boundary
    int3 ghost  = boundary
    for  i  in 0:NGHOST {
        domain = domain - normal
        ghost  = ghost + normal
        real distance = 2 * (i + 1) * d
        real rho_diff = RHO[domain.x][domain.y][domain.z] - RHO[ghost.x][ghost.y][ghost.z]
        SS[ghost.x][ghost.y][ghost.z] = SS[domain.x][domain.y][domain.z] + AC_cp * (AC_cp - AC_cv) * (rho_diff + distance * tmp)
    }
}
//TP: temp handwritten implementation of conv-slab with kramers and possibly something else non-standard
Kernel twopass_solve_intermediate(int step_num, real dt){
  UU = vecvalue(F_UU)
  UIJ = gradients(F_UU)
  DIVU = divergence(F_UU)
  SIJ = traceless_strain(UIJ,DIVU)
  SIJ2 = multm2_sym(SIJ)
  UGU = UIJ*UU  - gradients_upwd(F_UU) * vecvalue_abs(F_UU)
  DEL2U = veclaplace(F_UU)
  GRADDIVU = divergence(F_UU)
  LNRHO = value(RHO)
  RHO1 = exp(-LNRHO)
  GLNRHO = gradient(RHO)
  //gradient6 is for upwinding
  UGLNRHO = dot(GLNRHO,UU) - dot(vecvalue_abs(F_UU), gradient6_upwd(RHO))
  DEL2LNRHO = laplace(RHO)
  SGLNRHO = SIJ*GLNRHO
  GSS = gradient(SS)
  DEL2SS = laplace(SS)
  cs2 = AC_cs20*exp(AC_cv1*value(SS)+AC_gamma_m1*(LNRHO-AC_lnrho0))
  LNTT = AC_lnTT0+AC_cv1*value(SS)+AC_gamma_m1*(LNRHO-AC_lnrho0)
  TT = exp(LNTT)
  TT1 = exp(-LNTT)
  GLNTT = AC_gamma_m1*GLNRHO+AC_cv1*GSS
  DEL2LNTT = AC_gamma_m1*DEL2LNRHO+AC_cv1*DEL2SS
  //gradient6 is for upwinding
  UGSS = dot(GSS,UU) - dot(vecvalue_abs(F_UU), gradient6_upwd(SS))
  FPRES = -cs2*(GLNRHO+GLNTT)*AC_gamma1
  FVISC = AC_nu*(DEL2U+2.0*SGLNRHO + (1.0/3.0)*gradient_of_divergence(F_UU))
  VISC_HEAT = 2.0*AC_nu*SIJ2

  DF_UU = -UGU+FVISC+FPRES
  DF_LNRHO = -DIVU-UGLNRHO


  K_kramers = AC_hcond0_kramers*pow(RHO1,2*AC_nkramers)*pow(TT,6.5*AC_nkramers)
  Krho1 =  K_kramers*RHO1
  G2 = dot(-2.0 * AC_nkramers*GLNRHO + (6.5*AC_nkramers+1)*GLNTT,GLNTT)
  thdiff = Krho1*(DEL2LNTT+G2)

  ztop = AC_zorig + AC_zlen
  pos = grid_position_linear()
  z = pos.z
  prof = exp(-0.5 * pow(((ztop-z)/AC_wcool),2.0))
  heat = - AC_cool*prof*(cs2-AC_cs2cool)/AC_cs2cool

  DF_SS = -UGSS
  DF_SS = DF_SS+TT1*VISC_HEAT
  DF_SS = DF_SS + thdiff
  DF_SS = DF_SS + TT1*RHO1*heat

  DF_UU.z = DF_UU.z + AC_gravz
  W_UU = rk3_intermediate_vector(vecprevious(F_UU),DF_UU,step_num, dt)
  W_SS = rk3_intermediate(previous(SS),DF_SS,step_num, dt)
  W_LNRHO = rk3_intermediate(previous(RHO),DF_LNRHO,step_num, dt)
  vecwrite(F_UU,W_UU)
  write(SS,W_SS)
  write(RHO, W_LNRHO)
}

