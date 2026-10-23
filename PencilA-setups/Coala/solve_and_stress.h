Kernel solve_and_stress(){
  real ac_transformed_pencil_acc
  real ac_transformed_pencil_ssat
  real ac_transformed_pencil_ttc
  real ac_transformed_pencil_ywater
  real ac_transformed_pencil_lambda
  real ac_transformed_pencil_chem_conc[AC_nchemspec__mod__cparam]
  real ac_transformed_pencil_nucl_rmin
  real ac_transformed_pencil_nucl_rate
  real ac_transformed_pencil_conc_satm
  real ac_transformed_pencil_ff_cond
  real ac_transformed_pencil_latent_heat
  real ac_transformed_pencil_lnrho
  real ac_transformed_pencil_rho
  real ac_transformed_pencil_rho1
  real3 ac_transformed_pencil_glnrho
  real3 ac_transformed_pencil_grho
  real ac_transformed_pencil_uglnrho
  real ac_transformed_pencil_ugrho
  real ac_transformed_pencil_glnrho2
  real ac_transformed_pencil_del2lnrho
  real ac_transformed_pencil_del2rho
  real ac_transformed_pencil_del6lnrho
  real ac_transformed_pencil_del6rho
  Matrix ac_transformed_pencil_hlnrho
  real3 ac_transformed_pencil_sglnrho
  real3 ac_transformed_pencil_uij5glnrho
  real ac_transformed_pencil_transprho
  real ac_transformed_pencil_ekin
  real ac_transformed_pencil_uuadvec_glnrho
  real ac_transformed_pencil_uuadvec_grho
  real ac_transformed_pencil_rhos1
  real3 ac_transformed_pencil_glnrhos
  real ac_transformed_pencil_totenergy_rel
  real ac_transformed_pencil_divss
  real ac_transformed_pencil_rhod[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_udropav
  real ac_transformed_pencil_rhodsum
  real3 ac_transformed_pencil_glnrhodsum
  real3 ac_transformed_pencil_uud[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_divud[AC_ndustspec__mod__cparam]
  Matrix ac_transformed_pencil_sdij[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_ma2
  real3 ac_transformed_pencil_fpres
  real ac_transformed_pencil_tcond
  real3 ac_transformed_pencil_sglntt
  real ac_transformed_pencil_uglntt
  real ac_transformed_pencil_advec_cs2
  real ac_transformed_pencil_ss
  real3 ac_transformed_pencil_gss
  real ac_transformed_pencil_ee
  real ac_transformed_pencil_pp
  real ac_transformed_pencil_lntt
  real ac_transformed_pencil_cs2
  real ac_transformed_pencil_cp
  real ac_transformed_pencil_cp1
  real ac_transformed_pencil_cp1tilde
  real3 ac_transformed_pencil_glntt
  real ac_transformed_pencil_tt
  real ac_transformed_pencil_tt1
  real3 ac_transformed_pencil_gtt
  real ac_transformed_pencil_yh
  Matrix ac_transformed_pencil_hss
  Matrix ac_transformed_pencil_hlntt
  real ac_transformed_pencil_del2ss
  real ac_transformed_pencil_del6ss
  real ac_transformed_pencil_del2lntt
  real ac_transformed_pencil_cv
  real ac_transformed_pencil_cv1
  real ac_transformed_pencil_del6lntt
  real ac_transformed_pencil_gamma
  real ac_transformed_pencil_del2tt
  real ac_transformed_pencil_del6tt
  real3 ac_transformed_pencil_glnmumol
  real ac_transformed_pencil_ppvap
  real ac_transformed_pencil_csvap2
  real ac_transformed_pencil_ttb
  real ac_transformed_pencil_rho_anel
  real ac_transformed_pencil_eth
  real3 ac_transformed_pencil_geth
  real ac_transformed_pencil_del2eth
  Matrix ac_transformed_pencil_heth
  real ac_transformed_pencil_eths
  real3 ac_transformed_pencil_geths
  real3 ac_transformed_pencil_rho1gpp
  real3 ac_transformed_pencil_fcont[AC_n_forcing_cont_max__mod__cparam]
  real3 ac_transformed_pencil_gg
  real ac_transformed_pencil_x_mn
  real ac_transformed_pencil_y_mn
  real ac_transformed_pencil_z_mn
  real ac_transformed_pencil_r_mn
  real ac_transformed_pencil_r_mn1
  real ac_transformed_pencil_phix
  real ac_transformed_pencil_phiy
  real ac_transformed_pencil_pomx
  real ac_transformed_pencil_pomy
  real ac_transformed_pencil_rcyl_mn
  real ac_transformed_pencil_rcyl_mn1
  real ac_transformed_pencil_phi_mn
  real3 ac_transformed_pencil_evr
  real3 ac_transformed_pencil_rr
  real3 ac_transformed_pencil_evth
  real ac_transformed_pencil_divu
  real3 ac_transformed_pencil_oo
  real ac_transformed_pencil_o2
  real ac_transformed_pencil_ou
  real ac_transformed_pencil_oxu2
  real3 ac_transformed_pencil_oxu
  real ac_transformed_pencil_u2
  Matrix ac_transformed_pencil_uij
  real3 ac_transformed_pencil_uu
  real3 ac_transformed_pencil_curlo
  Matrix ac_transformed_pencil_sij
  real ac_transformed_pencil_sij2
  Matrix ac_transformed_pencil_uij5
  real3 ac_transformed_pencil_ugu
  real ac_transformed_pencil_ugu2
  Matrix ac_transformed_pencil_oij
  Matrix ac_transformed_pencil_d2uidxj
  Tensor ac_transformed_pencil_uijk
  real3 ac_transformed_pencil_ogu
  real ac_transformed_pencil_u3u21
  real ac_transformed_pencil_u1u32
  real ac_transformed_pencil_u2u13
  real3 ac_transformed_pencil_del2u
  real3 ac_transformed_pencil_del4u
  real3 ac_transformed_pencil_del6u
  real ac_transformed_pencil_u2u31
  real ac_transformed_pencil_u3u12
  real ac_transformed_pencil_u1u23
  real3 ac_transformed_pencil_graddivu
  real3 ac_transformed_pencil_del6u_bulk
  real3 ac_transformed_pencil_grad5divu
  real3 ac_transformed_pencil_rhougu
  real3 ac_transformed_pencil_der6u
  real3 ac_transformed_pencil_transpurho
  real ac_transformed_pencil_divu0
  Matrix ac_transformed_pencil_u0ij
  real3 ac_transformed_pencil_uu0
  real3 ac_transformed_pencil_uu_advec
  real3 ac_transformed_pencil_uuadvec_guu
  real3 ac_transformed_pencil_del6u_strict
  real3 ac_transformed_pencil_del4graddivu
  real3 ac_transformed_pencil_uu_sph
  Matrix ac_transformed_pencil_der6u_res
  real ac_transformed_pencil_lorentz
  real ac_transformed_pencil_hless
  real ac_transformed_pencil_advec_uu
  real ac_transformed_pencil_heat
  real ac_transformed_pencil_cool
  real ac_transformed_pencil_heatcool
  real3 ac_transformed_pencil_aa
  real ac_transformed_pencil_a2
  Matrix ac_transformed_pencil_aij
  real3 ac_transformed_pencil_bb
  real3 ac_transformed_pencil_bbb
  real ac_transformed_pencil_ab
  real ac_transformed_pencil_ua
  real3 ac_transformed_pencil_exa
  real3 ac_transformed_pencil_exatotal
  real ac_transformed_pencil_aps
  real ac_transformed_pencil_b2
  real ac_transformed_pencil_b21
  real ac_transformed_pencil_bf2
  Matrix ac_transformed_pencil_bij
  real3 ac_transformed_pencil_del2a
  real3 ac_transformed_pencil_graddiva
  real3 ac_transformed_pencil_jj
  real3 ac_transformed_pencil_jj_ohm
  real3 ac_transformed_pencil_curlb
  real3 ac_transformed_pencil_e3xa
  real3 ac_transformed_pencil_el
  real ac_transformed_pencil_e2
  Matrix ac_transformed_pencil_bijtilde
  Matrix ac_transformed_pencil_bij_cov_corr
  real ac_transformed_pencil_j2
  real ac_transformed_pencil_jb
  real ac_transformed_pencil_va2
  real3 ac_transformed_pencil_jxb
  real3 ac_transformed_pencil_jxbr
  real ac_transformed_pencil_jxbr2
  real ac_transformed_pencil_ub
  real ac_transformed_pencil_uj
  real ac_transformed_pencil_ob
  real3 ac_transformed_pencil_uxb
  real3 ac_transformed_pencil_uxbb
  real ac_transformed_pencil_uxb2
  real3 ac_transformed_pencil_uxj
  real ac_transformed_pencil_chibp
  real ac_transformed_pencil_beta
  real ac_transformed_pencil_beta1
  real3 ac_transformed_pencil_uga
  real3 ac_transformed_pencil_uuadvec_gaa
  real ac_transformed_pencil_djuidjbi
  real ac_transformed_pencil_jo
  real ac_transformed_pencil_stokesi
  real ac_transformed_pencil_stokesq
  real ac_transformed_pencil_stokesu
  real ac_transformed_pencil_stokesq1
  real ac_transformed_pencil_stokesu1
  real ac_transformed_pencil_ujxb
  real3 ac_transformed_pencil_oxuxb
  real3 ac_transformed_pencil_jxbxb
  real3 ac_transformed_pencil_jxbrxb
  real3 ac_transformed_pencil_gb22
  real3 ac_transformed_pencil_ugb
  real ac_transformed_pencil_ugb22
  real3 ac_transformed_pencil_bgu
  real3 ac_transformed_pencil_bgb
  real3 ac_transformed_pencil_bgbp
  real ac_transformed_pencil_ubgbp
  real3 ac_transformed_pencil_bdivu
  real3 ac_transformed_pencil_glnrhoxb
  real3 ac_transformed_pencil_del4a
  real3 ac_transformed_pencil_del6a
  real3 ac_transformed_pencil_oxj
  real ac_transformed_pencil_diva
  Matrix ac_transformed_pencil_jij
  real ac_transformed_pencil_sj
  real ac_transformed_pencil_ss12
  real ac_transformed_pencil_d6ab
  real ac_transformed_pencil_etava
  real ac_transformed_pencil_etaj
  real ac_transformed_pencil_etaj2
  real ac_transformed_pencil_etajrho
  real ac_transformed_pencil_cosjb
  real ac_transformed_pencil_jparallel
  real ac_transformed_pencil_jperp
  real ac_transformed_pencil_cosub
  real3 ac_transformed_pencil_bunit
  real3 ac_transformed_pencil_hjj
  real ac_transformed_pencil_hj2
  real ac_transformed_pencil_hjb
  real ac_transformed_pencil_coshjb
  real ac_transformed_pencil_hjparallel
  real ac_transformed_pencil_hjperp
  real ac_transformed_pencil_nu_ni1
  real ac_transformed_pencil_gamma_a2
  real ac_transformed_pencil_clight2
  real3 ac_transformed_pencil_gva
  real3 ac_transformed_pencil_vmagfric
  real3 ac_transformed_pencil_bb_sph
  real ac_transformed_pencil_advec_va2
  real ac_transformed_pencil_lam
  real3 ac_transformed_pencil_glam
  real3 ac_transformed_pencil_mf_emf
  real ac_transformed_pencil_mf_emfdotb
  real3 ac_transformed_pencil_uun
  real ac_transformed_pencil_divun
  Matrix ac_transformed_pencil_snij
  real ac_transformed_pencil_rhop
  real3 ac_transformed_pencil_grhop
  real ac_transformed_pencil_peh
  real ac_transformed_pencil_tauascalar
  real ac_transformed_pencil_condensationrate
  real ac_transformed_pencil_watermixingratio
  real ac_transformed_pencil_part_heatcap
  real3 ac_transformed_pencil_gcc[AC_0]
  real ac_transformed_pencil_sgs_heat
  real ac_transformed_pencil_shock
  real3 ac_transformed_pencil_gshock
  real ac_transformed_pencil_shock_perp
  real3 ac_transformed_pencil_gshock_perp
  real3 ac_transformed_pencil_fvisc
  real ac_transformed_pencil_diffus_total
  real ac_transformed_pencil_diffus_total2
  real ac_transformed_pencil_diffus_total3
  real ac_transformed_pencil_visc_heat
  real ac_transformed_pencil_nu
  real3 ac_transformed_pencil_gradnu
  real ac_transformed_pencil_nu_smag
  real3 ac_transformed_pencil_gnu_smag
  real ac_transformed_pencil_stress_ij[6]
  real3 ac_transformed_pencil_gphi
  real pij[6]
  real kij[6]
  real e_t[6]
  real e_x[6]
  real sij_re[6]
  real sij_im[6]
  real delij[6]
  real3 e1, e2, kvec
  int i
  int j
  int p
  int q
  int ik
  int ikx
  int iky
  int ikz
  int stat
  int ij
  int pq
  int ip
  int jq
  int jstress_ij
  real fact
  real delkt
  real om2_min
  real kmin
  real ksqr
  real one_over_k2
  real k1
  real k2
  real k3
  real k1sqr
  real k2sqr
  real k3sqr
  real ksqrt
  real hhtre
  real hhtim
  real hhxre
  real hhxim
  real coefare
  real coefaim
  real ggtre
  real ggtim
  real ggxre
  real ggxim
  real coefbre
  real coefbim
  real e_ij_t
  real e_ij_x
  real cosot
  real sinot
  real sinot_minus
  real om12
  real om
  real om1
  real om2
  real dt1
  real ett
  real etx
  real ext
  real exx
  real discrim2
  real om_rat_lam
  real om_rat_mat
  real om_rat_matt
  real om_rat_tot1
  real ds_t_re
  real ds_t_im
  real ds_x_re
  real ds_x_im
  complex coefa
  complex coefb
  complex om_cmplx
  complex hcomplex_new
  complex gcomplex_new
  complex discrim
  complex det1
  complex lam1
  complex lam2
  complex explam1t
  complex explam2t
  complex cosoth
  complex cosotg
  complex sinoth
  complex sinotg
  bool lsign_om2
  delkt=AC_delk__mod__special
  if (AC_ldelkt__mod__special) {
    if(AC_enum_idelkt__mod__special == AC_enum_jump_string__mod__cparam) {
      if (AC_t__mod__cdata>AC_tdelk__mod__special) {
        delkt=0.
      }
    }
    else if(AC_enum_idelkt__mod__special == AC_enum_exponential_string__mod__cparam) {
      if (AC_t__mod__cdata>AC_tdelk__mod__special) {
        delkt=exp(-(AC_t__mod__cdata-AC_tdelk__mod__special)/AC_tau_delk__mod__special)
      }
    }
    else {
    }
  }
  if (AC_lgpu__mod__cparam  &&  ! AC_lread_scl_factor_file__mod__cdata) {
    if (AC_lreheating_gw__mod__special) {
      scale_factor__mod__special=0.25*((AC_t__mod__cdata+1.)*(AC_t__mod__cdata+1.))
    }
    else if (AC_lmatter_gw__mod__special) {
      scale_factor__mod__special=(AC_t__mod__cdata*AC_t__mod__cdata)/AC_t_equality__mod__special
    }
    else if (AC_ldark_energy_gw__mod__special) {
      scale_factor__mod__special=(AC_t_acceleration__mod__special*AC_t_acceleration__mod__special*AC_t_acceleration__mod__special)/(AC_t__mod__cdata*AC_t_equality__mod__special)
    }
    else if (AC_lscalar__mod__special) {
      scale_factor__mod__special=exp(AC_f_ode__mod__cdata[AC_iinfl_lna__mod__special-1])
    }
    else {
      if (AC_t__mod__cdata+AC_tshift__mod__special==0.) {
        scale_factor__mod__special=1.
      }
      else {
        scale_factor__mod__special=pow((AC_t__mod__cdata+AC_tshift__mod__special),AC_nscale_factor_conformal__mod__special)
      }
    }
  }
  if (AC_lhorndeski__mod__special || AC_lhorndeski_xi__mod__special) {
    if(enum_ihorndeski_time__mod__special == AC_enum_const_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special
      horndeski_alpm_eff__mod__special=AC_horndeski_alpm__mod__special
    }
    else if(enum_ihorndeski_time__mod__special == enum_tanh_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special*tanh(1.-pow((scale_factor__mod__special/AC_scale_factor0__mod__special),AC_horndeski_alpt_exp__mod__special))
    }
    else if(enum_ihorndeski_time__mod__special == enum_exp_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special*exp(-pow((scale_factor__mod__special/AC_scale_factor0__mod__special),AC_horndeski_alpt_exp__mod__special))
    }
    else if(enum_ihorndeski_time__mod__special == enum_scale_factor_power_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special
      horndeski_alpm_eff__mod__special=AC_horndeski_alpm__mod__special*pow((scale_factor__mod__special*AC_a_ini__mod__special/AC_scale_factor0__mod__special),AC_horndeski_alpm_exp__mod__special)
    }
    else if(enum_ihorndeski_time__mod__special == enum_matter_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special
      if (AC_lread_scl_factor_file__mod__cdata && AC_lread_scl_factor_file_exists__mod__special) {
        om_rat_matt=pow((scale_factor__mod__special*AC_a_ini__mod__special/AC_scale_factor0__mod__special),(-3))*AC_omm0__mod__special
        om_rat_tot1=((AC_a_ini__mod__special*AC_h0__mod__special*scale_factor__mod__special/AC_hp_target__mod__cdata/AC_hp_ini__mod__special)*(AC_a_ini__mod__special*AC_h0__mod__special*scale_factor__mod__special/AC_hp_target__mod__cdata/AC_hp_ini__mod__special))
        horndeski_alpm_eff__mod__special=AC_horndeski_alpm__mod__special*(1-om_rat_matt*om_rat_tot1)/(1-AC_omm0__mod__special)
      }
      else {
      }
    }
    else if(enum_ihorndeski_time__mod__special == enum_dark_energy_string__mod__cparam) {
      horndeski_alpt_eff__mod__special=AC_horndeski_alpt__mod__special
      if (AC_lread_scl_factor_file__mod__cdata && AC_lread_scl_factor_file_exists__mod__special) {
        om_rat_tot1=((AC_a_ini__mod__special*AC_h0__mod__special*scale_factor__mod__special/AC_hp_target__mod__cdata/AC_hp_ini__mod__special)*(AC_a_ini__mod__special*AC_h0__mod__special*scale_factor__mod__special/AC_hp_target__mod__cdata/AC_hp_ini__mod__special))
        horndeski_alpm_eff__mod__special=AC_horndeski_alpm__mod__special*om_rat_tot1
      }
      else {
      }
    }
    else {
    }
    if (AC_lread_scl_factor_file__mod__cdata && AC_lread_scl_factor_file_exists__mod__special) {
      if (AC_lhorndeski__mod__special) {
        horndeski_alpm_eff__mod__special=horndeski_alpm_eff__mod__special*AC_hp_target__mod__cdata
        horndeski_alpm_eff2__mod__special=horndeski_alpm_eff__mod__special*AC_hp_target__mod__cdata
      }
      else {
        horndeski_alpm_eff2__mod__special=(1+0.5*horndeski_alpm_eff__mod__special)*(AC_hp_target__mod__cdata*AC_hp_target__mod__cdata)
        horndeski_alpm_eff2__mod__special=horndeski_alpm_eff2__mod__special*0.5*horndeski_alpm_eff__mod__special
        horndeski_alpm_eff3__mod__special=0.5*AC_horndeski_alpm_prime__mod__special*AC_hp_target__mod__cdata
        horndeski_alpm_eff__mod__special=1.+0.5*horndeski_alpm_eff__mod__special
      }
    }
    else {
      if (AC_lhorndeski__mod__special) {
        horndeski_alpm_eff__mod__special=horndeski_alpm_eff__mod__special/scale_factor__mod__special
        horndeski_alpm_eff2__mod__special=horndeski_alpm_eff__mod__special/scale_factor__mod__special
      }
      else {
        horndeski_alpm_eff2__mod__special=(1+0.5*horndeski_alpm_eff__mod__special)/(scale_factor__mod__special*scale_factor__mod__special)
        horndeski_alpm_eff2__mod__special=horndeski_alpm_eff2__mod__special*0.5*horndeski_alpm_eff__mod__special
        horndeski_alpm_eff3__mod__special=0.5*AC_horndeski_alpm_prime__mod__special/scale_factor__mod__special
        horndeski_alpm_eff__mod__special=1.+0.5*horndeski_alpm_eff__mod__special
      }
    }
  }
  if (AC_lread_scl_factor_file__mod__cdata && AC_lread_scl_factor_file_exists__mod__special) {
    appa_om__mod__special=AC_appa_target__mod__cdata
  }
  if (AC_lhorndeski_xi__mod__special) {
    appa_om__mod__special=appa_om__mod__special*horndeski_alpm_eff__mod__special+horndeski_alpm_eff2__mod__special
    appa_om__mod__special=appa_om__mod__special+horndeski_alpm_eff3__mod__special
  }
  s_t_re=0.
  s_t_im=0.
  s_x_re=0.
  s_x_im=0.
  k1=AC_kx_fft__mod__fourier[ikx+AC_ipx__mod__cdata*AC_nx__mod__cparam-1]
  k2=AC_ky_fft__mod__fourier[iky+AC_ipy__mod__cdata*AC_ny__mod__cparam-1]
  k3=AC_kz_fft__mod__fourier[ikz+AC_ipz__mod__cdata*AC_nz__mod__cparam-1]
  k1sqr=(k1*k1)
  k2sqr=(k2*k2)
  k3sqr=(k3*k3)
  ksqr=k1sqr+k2sqr+k3sqr
  ksqrt = sqrt(ksqr)
  if (AC_lroot__mod__cdata && ikx==1 && iky==1 && ikz==1) {
    e1.x = 0.
    e1.y = 0.
    e1.z = 0.
    e2.x = 0.
    e2.y = 0.
    e2.z = 0.
    pij=0.
    kij=0.
    om=0.
    om2=0.
  }
  else {
    one_over_k2=1./ksqr
    if (AC_linflation__mod__special) {
      om2=4.*ksqr-2./(AC_t__mod__cdata*AC_t__mod__cdata)
      lsign_om2=(om2 >= 0.)
      om=sqrt(abs(om2))
    }
    else if (AC_lreheating_gw__mod__special) {
      om2=ksqr-2./((AC_t__mod__cdata+1.)*(AC_t__mod__cdata+1.))
      lsign_om2=(om2 >= 0.)
      om=sqrt(abs(om2))
    }
    else if (AC_lscalar__mod__special) {
      om2=ksqr-0.0
      lsign_om2=(om2 >= 0.)
      om=sqrt(abs(om2))
    }
    else if (AC_lmatter_gw__mod__special  ||  AC_ldark_energy_gw__mod__special) {
      om2=ksqr-2./(AC_t__mod__cdata*AC_t__mod__cdata)
      lsign_om2=(om2 >= 0.)
      om=sqrt(abs(om2))
    }
    else {
      if (delkt!=0.  ||  AC_lhorndeski__mod__special) {
        if (AC_lhorndeski__mod__special) {
          om2=(1.+horndeski_alpt_eff__mod__special)*ksqr+(delkt*delkt)-horndeski_alpm_eff2__mod__special-appa_om__mod__special
          om_cmplx=sqrt(cmplx(om2,0.))
          om=AC_impossible__mod__cparam
        }
        else if (AC_lhorndeski_xi__mod__special) {
          om2=(1.+horndeski_alpt_eff__mod__special)*ksqr+(delkt*delkt)-appa_om__mod__special
        }
        else {
          om2=ksqr+(delkt*delkt)-appa_om__mod__special
          om=sqrt(om2)
        }
      }
      else {
        om2=ksqr-appa_om__mod__special
        om=sqrt(om2)
      }
      lsign_om2=true
    }
    if(abs(k1)<abs(k2)) {
      if(abs(k1)<abs(k3)) {
        e1=real3(0., -k3, +k2)
        e2=real3(k2sqr+k3sqr, -k2*k1, -k3*k1)
      }
      else {
        e1=real3(k2, -k1, 0.)
        e2=real3(k1*k3, k2*k3, -(k1sqr+k2sqr))
      }
    }
    else {
      if(abs(k2)<abs(k3)) {
        e1=real3(-k3, 0., +k1)
        e2=real3(+k1*k2, -(k1sqr+k3sqr), +k3*k2)
      }
      else {
        e1=real3(k2, -k1, 0.)
        e2=real3(k1*k3, k2*k3, -(k1sqr+k2sqr))
      }
    }
    e1=e1/sqrt((e1.x*e1.x)+(e1.y*e1.y)+(e1.z*e1.z))
    e2=e2/sqrt((e2.x*e2.x)+(e2.y*e2.y)+(e2.z*e2.z))
    pij[1-1]=1.-k1sqr*one_over_k2
    pij[2-1]=1.-k2sqr*one_over_k2
    pij[3-1]=1.-k3sqr*one_over_k2
    pij[4-1]=-k1*k2*one_over_k2
    pij[5-1]=-k2*k3*one_over_k2
    pij[6-1]=-k3*k1*one_over_k2
    if (AC_llighthill__mod__special) {
      kij[1-1]=-k1sqr
      kij[2-1]=-k2sqr
      kij[3-1]=-k3sqr
      kij[4-1]=-k1*k2
      kij[5-1]=-k2*k3
      kij[6-1]=-k3*k1
    }
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  e_t[ij-1]=e1.x*e1.x-e2.x*e2.x
  e_x[ij-1]=e1.x*e2.x+e2.x*e1.x
  ij=AC_ij_table__mod__special[2-1][1-1]
  e_t[ij-1]=e1.y*e1.x-e2.y*e2.x
  e_x[ij-1]=e1.y*e2.x+e2.y*e1.x
  ij=AC_ij_table__mod__special[3-1][1-1]
  e_t[ij-1]=e1.z*e1.x-e2.z*e2.x
  e_x[ij-1]=e1.z*e2.x+e2.z*e1.x
  ij=AC_ij_table__mod__special[1-1][2-1]
  e_t[ij-1]=e1.x*e1.y-e2.x*e2.y
  e_x[ij-1]=e1.x*e2.y+e2.x*e1.y
  ij=AC_ij_table__mod__special[2-1][2-1]
  e_t[ij-1]=e1.y*e1.y-e2.y*e2.y
  e_x[ij-1]=e1.y*e2.y+e2.y*e1.y
  ij=AC_ij_table__mod__special[3-1][2-1]
  e_t[ij-1]=e1.z*e1.y-e2.z*e2.y
  e_x[ij-1]=e1.z*e2.y+e2.z*e1.y
  ij=AC_ij_table__mod__special[1-1][3-1]
  e_t[ij-1]=e1.x*e1.z-e2.x*e2.z
  e_x[ij-1]=e1.x*e2.z+e2.x*e1.z
  ij=AC_ij_table__mod__special[2-1][3-1]
  e_t[ij-1]=e1.y*e1.z-e2.y*e2.z
  e_x[ij-1]=e1.y*e2.z+e2.y*e1.z
  ij=AC_ij_table__mod__special[3-1][3-1]
  e_t[ij-1]=e1.z*e1.z-e2.z*e2.z
  e_x[ij-1]=e1.z*e2.z+e2.z*e1.z
  if (AC_lswitch_sign_e_x__mod__special) {
    if (k3<0.) {
      e_x=-e_x
    }
    else if (k3==0.) {
      if (k2<0.) {
        e_x=-e_x
      }
      else if (k2==0.) {
        if (k1<0.) {
          e_x=-e_x
        }
      }
    }
  }
  sij_re=0.
  sij_im=0.
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[1-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[1-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[1-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[1-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[1-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[1-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[1-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[1-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[1-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][2-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[2-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[2-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][2-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[2-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[1-1][1-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[1-1][2-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][3-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[1-1][3-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[2-1][1-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[2-1][2-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[2-1][3-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[2-1][3-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[1-1][1-1]
  ip=AC_ij_table__mod__special[3-1][1-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[2-1][1-1]
  ip=AC_ij_table__mod__special[3-1][2-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[3-1][1-1]
  ip=AC_ij_table__mod__special[3-1][3-1]
  jq=AC_ij_table__mod__special[3-1][1-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[1-1][2-1]
  ip=AC_ij_table__mod__special[3-1][1-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[2-1][2-1]
  ip=AC_ij_table__mod__special[3-1][2-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[3-1][2-1]
  ip=AC_ij_table__mod__special[3-1][3-1]
  jq=AC_ij_table__mod__special[3-1][2-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[1-1][3-1]
  ip=AC_ij_table__mod__special[3-1][1-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[2-1][3-1]
  ip=AC_ij_table__mod__special[3-1][2-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[3-1][3-1]
  pq=AC_ij_table__mod__special[3-1][3-1]
  ip=AC_ij_table__mod__special[3-1][3-1]
  jq=AC_ij_table__mod__special[3-1][3-1]
  sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_re__mod__special[pq-1]
  sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_tpq_im__mod__special[pq-1]
  if (AC_lnonlinear_source__mod__special && AC_lnonlinear_tpq_trans__mod__special) {
    sij_re[ij-1]=sij_re[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_re__mod__special[pq-1]
    sij_im[ij-1]=sij_im[ij-1]+(pij[ip-1]*pij[jq-1]-0.5*pij[ij-1]*pij[pq-1])*AC_nonlinear_tpq_im__mod__special[pq-1]
  }
  ij=AC_ij_table__mod__special[1-1][1-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[2-1][1-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[3-1][1-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[1-1][2-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[2-1][2-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[3-1][2-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[1-1][3-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[2-1][3-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  ij=AC_ij_table__mod__special[3-1][3-1]
  s_t_re=s_t_re+0.5*e_t[ij-1]*sij_re[ij-1]
  s_t_im=s_t_im+0.5*e_t[ij-1]*sij_im[ij-1]
  s_x_re=s_x_re+0.5*e_x[ij-1]*sij_re[ij-1]
  s_x_im=s_x_im+0.5*e_x[ij-1]*sij_im[ij-1]
  if (AC_llighthill__mod__special) {
    ij=AC_ij_table__mod__special[1-1][1-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[2-1][1-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[3-1][1-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[1-1][2-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[2-1][2-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[3-1][2-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[1-1][3-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[2-1][3-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
    ij=AC_ij_table__mod__special[3-1][3-1]
    s_t_re=s_t_re+kij[ij-1]*AC_tpq_re__mod__special[ij-1]
    s_t_im=s_t_im+kij[ij-1]*AC_tpq_im__mod__special[ij-1]
  }
  if (AC_lnophase_in_stress__mod__special) {
    if (AC_lconstmod_in_stress__mod__special) {
      s_t_re=exp(-ksqr/(AC_k_in_stress__mod__special*AC_k_in_stress__mod__special))
      s_x_re=exp(-ksqr/(AC_k_in_stress__mod__special*AC_k_in_stress__mod__special))
    }
    else {
      if (ksqr==0.) {
        s_t_re=0.
        s_x_re=0.
      }
      else {
        s_t_re=sqrt((s_t_re*s_t_re)+(s_t_im*s_t_im))
        s_x_re=sqrt((s_x_re*s_x_re)+(s_x_im*s_x_im))
      }
    }
    s_t_im=0.
    s_x_im=0.
    if (AC_llinphase_in_stress__mod__special) {
      s_t_re=s_t_re*cos(AC_slope_linphase_in_stress__mod__special*AC_t__mod__cdata)
      s_t_im=s_t_re*sin(AC_slope_linphase_in_stress__mod__special*AC_t__mod__cdata)
      s_x_re=s_x_re*cos(AC_slope_linphase_in_stress__mod__special*AC_t__mod__cdata)
      s_x_im=s_x_re*sin(AC_slope_linphase_in_stress__mod__special*AC_t__mod__cdata)
    }
  }
  hhtre=value(Field(AC_ihht__mod__cdata-1))
  hhxre=value(Field(AC_ihhx__mod__cdata-1))
  hhtim=value(Field(AC_ihhtim__mod__cdata-1))
  hhxim=value(Field(AC_ihhxim__mod__cdata-1))
  ggtre=value(Field(AC_iggt__mod__cdata-1))
  ggxre=value(Field(AC_iggx__mod__cdata-1))
  ggtim=value(Field(AC_iggtim__mod__cdata-1))
  ggxim=value(Field(AC_iggxim__mod__cdata-1))
  if (om2>om2_min) {
    om12=1./om2
    if (AC_lhorndeski__mod__special) {
      discrim2=(horndeski_alpm_eff__mod__special*horndeski_alpm_eff__mod__special)-4.*om2
      if (discrim2==0.) {
        discrim2=AC_tini__mod__cparam
      }
      discrim=sqrt(cmplx(discrim2,0.))
      lam1=0.5*(-horndeski_alpm_eff__mod__special+discrim)
      lam2=0.5*(-horndeski_alpm_eff__mod__special-discrim)
      explam1t=exp(lam1*AC_dt__mod__cdata)
      explam2t=exp(lam2*AC_dt__mod__cdata)
      det1=1./discrim
      cosoth=det1*(lam1*explam2t-lam2*explam1t)
      cosotg=det1*(lam1*explam1t-lam2*explam2t)
      sinoth=-det1*(     explam2t-     explam1t)*om_cmplx
      sinotg=+det1*(     explam2t-     explam1t)/om_cmplx*lam1*lam2
    }
    else {
      if (lsign_om2) {
        cosot=cos(om*AC_dt__mod__cdata)
        sinot=sin(om*AC_dt__mod__cdata)
        sinot_minus=-sinot
      }
      else {
        cosot=cosh(om*AC_dt__mod__cdata)
        sinot=sinh(om*AC_dt__mod__cdata)
        sinot_minus=+sinot
      }
    }
    if (AC_lhorndeski__mod__special) {
      coefa=cmplx(hhtre-om12*s_t_re,hhtim-om12*s_t_im)
      coefb=cmplx(ggtre                         ,ggtim    )/om_cmplx
      hcomplex_new= cosoth*coefa+sinoth*coefb+om12*cmplx(s_t_re,s_t_im)
      gcomplex_new=(sinotg*coefa+cosotg*coefb)*om_cmplx
      DF_HHT= hcomplex_new
      DF_HHTIM=aimag(hcomplex_new)
      DF_GGT= gcomplex_new
      DF_GGTIM=aimag(gcomplex_new)
    }
    else {
      om1=1./om
      coefare=(hhtre-om12*s_t_re)
      coefaim=(hhtim-om12*s_t_im)
      coefbre=ggtre*om1
      coefbim=ggtim*om1
      DF_HHT=coefare*cosot+coefbre*sinot+om12*s_t_re
      DF_HHTIM=coefaim*cosot+coefbim*sinot+om12*s_t_im
      DF_GGT=coefbre*cosot*om+coefare*om*sinot_minus
      DF_GGTIM=coefbim*cosot*om+coefaim*om*sinot_minus
      if (AC_itorder_gw__mod__special==2) {
        if (AC_dt__mod__cdata==0.) {
          dt1=0.
        }
        else {
          dt1=1./AC_dt__mod__cdata
        }
        ds_t_re=s_t_re-value(Field(AC_istresst__mod__cdata-1))
        ds_t_im=s_t_im-value(Field(AC_istresstim__mod__cdata-1))
        DF_HHT=value(Field(AC_ihht__mod__cdata-1))  +ds_t_re*om12*(1.-om1*dt1*sinot)
        DF_HHTIM=value(Field(AC_ihhtim__mod__cdata-1))  +ds_t_im*om12*(1.-om1*dt1*sinot)
        DF_GGT=value(Field(AC_iggt__mod__cdata-1))  +ds_t_re*om12*dt1*(1.-cosot)
        DF_GGTIM=value(Field(AC_iggtim__mod__cdata-1))  +ds_t_im*om12*dt1*(1.-cosot)
      }
    }
    if (AC_lhorndeski__mod__special) {
      coefa=cmplx(hhxre-om12*s_x_re,hhxim-om12*s_x_im)
      coefb=cmplx(ggxre                         ,ggxim    )/om_cmplx
      hcomplex_new= cosoth*coefa+sinoth*coefb+om12*cmplx(s_x_re,s_x_im)
      gcomplex_new=(sinotg*coefa+cosotg*coefb)*om_cmplx
      DF_HHX= hcomplex_new
      DF_HHXIM=aimag(hcomplex_new)
      DF_GGX= gcomplex_new
      DF_GGXIM=aimag(gcomplex_new)
    }
    else {
      coefare=(hhxre-om12*s_x_re)
      coefaim=(hhxim-om12*s_x_im)
      coefbre=ggxre*om1
      coefbim=ggxim*om1
      DF_HHX=coefare*cosot+coefbre*sinot+om12*s_x_re
      DF_HHXIM=coefaim*cosot+coefbim*sinot+om12*s_x_im
      DF_GGX=coefbre*cosot*om+coefare*om*sinot_minus
      DF_GGXIM=coefbim*cosot*om+coefaim*om*sinot_minus
      if (AC_itorder_gw__mod__special==2) {
        ds_x_re=s_x_re-value(Field(AC_istressx__mod__cdata-1))
        ds_x_im=s_x_im-value(Field(AC_istressxim__mod__cdata-1))
        DF_HHX=value(Field(AC_ihhx__mod__cdata-1))  +ds_x_re*om12*(AC_dt__mod__cdata-om1*sinot)
        DF_HHXIM=value(Field(AC_ihhxim__mod__cdata-1))  +ds_x_im*om12*(AC_dt__mod__cdata-om1*sinot)
        DF_GGX=value(Field(AC_iggx__mod__cdata-1))  +ds_x_re*om12*(1.-cosot)
        DF_GGXIM=value(Field(AC_iggxim__mod__cdata-1))  +ds_x_im*om12*(1.-cosot)
      }
    }
  }
  else {
    DF_HHT = 0.
    DF_HHTIM = 0.
    DF_GGT = 0.
    DF_GGTIM = 0.
    DF_HHX = 0.
    DF_HHXIM = 0.
    DF_GGX = 0.
    DF_GGXIM = 0.
  }
  DF_STRESST=s_t_re
  DF_STRESSTIM=s_t_im
  DF_STRESSX=s_x_re
  DF_STRESSXIM=s_x_im
}
