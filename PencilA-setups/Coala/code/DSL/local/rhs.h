
  #include "static_var_declares.h"
  #include "../df_declares.h"
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
  real ac_transformed_pencil_ext_force[4]
  real3 ac_transformed_pencil_glnnd[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_gmi[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_gmd[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_gnd[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_grhod[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_ad[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_md[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_mi[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_nd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_rhod[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_rhod1[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_epsd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_udgmi[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_udgmd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_udglnnd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_udgnd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_glnnd2[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_sdglnnd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del2nd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del2rhod[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del6nd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del2md[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del2mi[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_del6lnnd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_gndglnrho[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_glnndglnrho[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_udrop[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_udropgnd[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_fcloud
  real ac_transformed_pencil_ccondens
  real ac_transformed_pencil_ppwater
  real ac_transformed_pencil_ppsat
  real ac_transformed_pencil_ppsf[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_mu1
  real3 ac_transformed_pencil_udropav
  real3 ac_transformed_pencil_glnrhod[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_rhodsum
  real ac_transformed_pencil_rhodsum1
  real3 ac_transformed_pencil_grhodsum
  real3 ac_transformed_pencil_glnrhodsum
  real3 ac_transformed_pencil_old_uud[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_divud[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_ood[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_od2[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_oud[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_ud2[AC_ndustspec__mod__cparam]
  Matrix ac_transformed_pencil_udij[AC_ndustspec__mod__cparam]
  Matrix ac_transformed_pencil_sdij[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_udgud[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_uud[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_del2ud[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_del6ud[AC_ndustspec__mod__cparam]
  real3 ac_transformed_pencil_graddivud[AC_ndustspec__mod__cparam]
  real ac_transformed_pencil_advec_uud[AC_ndustspec__mod__cparam]
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
  real ac_transformed_pencil_cv1
  real ac_transformed_pencil_cp1
  real ac_transformed_pencil_cp1tilde
  real3 ac_transformed_pencil_glntt
  real ac_transformed_pencil_tt
  real ac_transformed_pencil_tt1
  real ac_transformed_pencil_cp
  real ac_transformed_pencil_cv
  real3 ac_transformed_pencil_gtt
  real3 ac_transformed_pencil_glnmu
  real3 ac_transformed_pencil_gmu1
  real ac_transformed_pencil_yh
  Matrix ac_transformed_pencil_hss
  Matrix ac_transformed_pencil_hlntt
  real ac_transformed_pencil_del2ss
  real ac_transformed_pencil_del6ss
  real ac_transformed_pencil_del2tt
  real ac_transformed_pencil_del2lntt
  real ac_transformed_pencil_del6tt
  real ac_transformed_pencil_del6lntt
  real3 ac_transformed_pencil_glnmumol
  real ac_transformed_pencil_csvap2
  real ac_transformed_pencil_rho_anel
  real3 ac_transformed_pencil_rho1gpp
  real3 ac_transformed_pencil_fcont[AC_n_forcing_cont_max__mod__cparam]
  real3 ac_transformed_pencil_curlfcont[AC_n_forcing_cont_max__mod__cparam]
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
  real ac_transformed_pencil_lorentz_gamma2
  real ac_transformed_pencil_lorentz_gamma
  real ac_transformed_pencil_ss_rel2
  real3 ac_transformed_pencil_ss_rel
  Matrix ac_transformed_pencil_ss_rel_ij
  real ac_transformed_pencil_ss_rel_factor
  real ac_transformed_pencil_divss_rel
  real ac_transformed_pencil_heat
  real ac_transformed_pencil_cool
  real ac_transformed_pencil_heatcool
  real3 ac_transformed_pencil_bb
  real3 ac_transformed_pencil_bbb
  Matrix ac_transformed_pencil_bij
  real3 ac_transformed_pencil_jxbr
  real ac_transformed_pencil_ss12
  real ac_transformed_pencil_b2
  real3 ac_transformed_pencil_uxb
  real3 ac_transformed_pencil_jj
  real3 ac_transformed_pencil_aa
  real ac_transformed_pencil_diva
  real3 ac_transformed_pencil_del2a
  Matrix ac_transformed_pencil_aij
  real3 ac_transformed_pencil_bunit
  real ac_transformed_pencil_va2
  real ac_transformed_pencil_j2
  real3 ac_transformed_pencil_el
  real ac_transformed_pencil_e2
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
  real ac_transformed_pencil_cc[1]
  real ac_transformed_pencil_cc1[1]
  real ac_transformed_pencil_gcc[3][0]
  real ac_transformed_pencil_sgs_heat
  real ac_transformed_pencil_shock
  real3 ac_transformed_pencil_gshock
  real ac_transformed_pencil_shock_perp
  real3 ac_transformed_pencil_gshock_perp
  real3 ac_transformed_pencil_fvisc
  real ac_transformed_pencil_diffus_total
  real ac_transformed_pencil_visc_heat
  real ac_transformed_pencil_nu
  real3 ac_transformed_pencil_gradnu
  real ac_transformed_pencil_nu_smag
  real3 df_iuu_pencil
  bool lcommunicate
  real tmp_15_20_21_55
  int i_15_20_21_55
  real tmp_11_15_20_21_55
  int i_11_15_20_21_55
  int j_11_15_20_21_55
  int i_18_20_21_55
  real tmp_16_18_20_21_55
  int i_16_18_20_21_55
  int j_16_18_20_21_55
  real advec_hypermesh_rho_19_20_21_55
  int j_28_55
  real lorentz_gamma_inv2_28_55 = 1.
  Matrix tmp_pencil_3x3_38_55
  int i_38_55
  int j_38_55
  int k_38_55
  int ju_38_55
  real tmp_42_55
  real t_tmp_42_55
  real3 tmp_pencil_3_42_55
  real ttt_42_55[AC_ndustspec__mod__cparam]
  real nd_rho_42_55[AC_ndustspec__mod__cparam]
  real coags_42_55[AC_ndustspec__mod__cparam]
  const aa0_42_55 = 6.107799961
  const aa1_42_55 = 4.436518521e-1
  const aa2_42_55 = 1.428945805e-2
  const aa3_42_55 = 2.650648471e-4
  const aa4_42_55 = 3.031240396e-6
  const aa5_42_55 = 2.034080948e-8
  const aa6_42_55 = 6.136820929e-11
  int i_42_55
  int k_42_55
  real tt_41_42_55
  real kn_41_42_55
  real cor_factor_41_42_55
  real d_coeff_41_42_55
  real di_41_42_55
  real dk_41_42_55
  real dik_41_42_55
  real kbc_41_42_55
  real vmean_i_41_42_55
  real vmean_k_41_42_55
  real vmean_ik_41_42_55
  real gamma_i_41_42_55
  real gamma_k_41_42_55
  real omega_i_41_42_55
  real omega_k_41_42_55
  real sigma_ik_41_42_55
  real deltavd_41_42_55
  real deltavd_therm_41_42_55
  real deltavd_turbu_41_42_55
  real fact_41_42_55
  real deltavd_drift2_41_42_55
  real deltavd_drift2a_41_42_55
  real deltavd_drift2b_41_42_55
  real ust_41_42_55
  real mu_air_41_42_55
  real rho_air_41_42_55
  real rik_41_42_55
  int i_41_42_55
  int j_41_42_55
  int l_41_42_55
  int k_41_42_55
  int lgh_41_42_55
  int lgh_40_41_42_55
  real deltavd_therm_40_41_42_55
  real deltavd_turbu_40_41_42_55
  real fact_40_41_42_55
  real deltavd_drift2_40_41_42_55
  real deltavd_drift2a_40_41_42_55
  real deltavd_drift2b_40_41_42_55
  real ust_40_41_42_55
  real mu_air_40_41_42_55
  real rho_air_40_41_42_55
  real rik_40_41_42_55
  const yr_39_40_41_42_55 = 31556926.
  const u_39_40_41_42_55 = 1660538921e-33
  const mu_gas_39_40_41_42_55 = 23e-1
  const mh_39_40_41_42_55 = 100749e-5*1660538921e-33
  real nh_39_40_41_42_55
  real cs_39_40_41_42_55
  real re_39_40_41_42_55
  real t_eta_39_40_41_42_55
  real ts_i_39_40_41_42_55
  real ts_j_39_40_41_42_55
  real ts_1_39_40_41_42_55
  real st_1_39_40_41_42_55
  real st_2_39_40_41_42_55
  real x_st_39_40_41_42_55
  real beta_st_39_40_41_42_55
  real res_39_40_41_42_55
  real t_dyn_39_40_41_42_55
  real fdiff_77_161
  real density_rhs_77_161
  real density_rhs_60_77_161
  real density_hydro_rhs_60_77_161
  real u_dot_ext_force_60_77_161
  real prefactor_60_77_161 = 1.
  real prefactor2_60_77_161 = 1.
  real lorentz_gamma_inv2_60_77_161 = 1.
  real3 tmpv_60_77_161
  real w_eos_60_77_161
  real gamma_r_60_77_161
  real int_source_60_77_161
  int i_60_77_161
  int i_59_60_77_161
  real2 fran_64_77_161
  real tmp_64_77_161
  real dlnrhodt_64_77_161
  real pdamp_64_77_161
  real fprofile_64_77_161
  real radius2_64_77_161
  real gamma_64_77_161
  real gamma1_64_77_161
  real step_vector_return_value_62_64_77_161
  real step_vector_return_value_63_64_77_161
  real tmp_66_77_161
  real tmp_65_66_77_161
  real tmp_69_77_161
  real gamma_69_77_161
  real tmp_71_77_161
  int j_71_77_161
  real advec_hypermesh_rho_70_71_77_161
  real f_target_76_77_161
  int j_78_161
  int k_98_161
  int iix_98_161
  real deltaud2_90_98_161
  real rep_90_98_161
  real csrho_90_98_161
  real pifactor1_90_98_161
  real pifactor2_90_98_161
  real3 aa_sfta_91_98_161
  real3 fviscd_92_98_161
  real3 tmp_92_98_161
  real3 tmp2_92_98_161
  real tausg1_92_98_161
  real mudrhod1_92_98_161
  real c2_92_98_161
  real s2_92_98_161
  int j_92_98_161
  real3 fviscd_93_98_161
  real3 tmp_93_98_161
  real3 tmp2_93_98_161
  real tausg1_93_98_161
  real mudrhod1_93_98_161
  real c2_93_98_161
  real s2_93_98_161
  int j_93_98_161
  real3 f_target_97_98_161
  int j_97_98_161
  int ju_97_98_161
  real mfluxcond_145_161
  real fdiffd_145_161
  real gshockgnd_145_161
  real imr_145_161
  real tmp1_145_161
  real tmp2_145_161
  real diffus_diffnd_145_161
  real diffus_diffnd3_145_161
  real advec_hypermesh_nd_145_161
  int kk_vec_145_161
  real dndr_tmp_145_161[AC_ndustspec__mod__cparam]
  real dndr_145_161[AC_ndustspec__mod__cparam]
  real nd_substep_145_161[AC_ndustspec__mod__cparam]
  real nd_substep_0_145_161[AC_ndustspec__mod__cparam]
  real k1_145_161[AC_ndustspec__mod__cparam]
  real k2_145_161[AC_ndustspec__mod__cparam]
  real k3_145_161[AC_ndustspec__mod__cparam]
  real k4_145_161[AC_ndustspec__mod__cparam]
  int k_145_161
  int i_145_161
  int j_145_161
  real ff_tmp_103_145_161[AC_ndustspec__mod__cparam]
  real nd_new_103_145_161[AC_ndustspec__mod__cparam]
  int k_103_145_161
  int j_103_145_161
  int jj_103_145_161
  int kk1_103_145_161
  int kk2_103_145_161
  real gs_103_145_161
  int k_102_103_145_161
  const i1_102_103_145_161 = 1
  const i2_102_103_145_161 = min(ndustspec,2)
  const i3_102_103_145_161 = min(ndustspec,3)
  const ii1_102_103_145_161 = ndustspec
  const ii2_102_103_145_161 = max(ndustspec-1,1)
  const ii3_102_103_145_161 = max(ndustspec-2,1)
  real rr1_102_103_145_161 = 0.
  real rr2_102_103_145_161 = 0.
  real rr3_102_103_145_161 = 0.
  int ndust_2nd_species_102_103_145_161
  real ff_tmp_104_145_161[AC_ndustspec__mod__cparam]
  real nd_new_104_145_161[AC_ndustspec__mod__cparam]
  int k_104_145_161
  int j_104_145_161
  int jj_104_145_161
  int kk1_104_145_161
  int kk2_104_145_161
  real gs_104_145_161
  int k_102_104_145_161
  const i1_102_104_145_161 = 1
  const i2_102_104_145_161 = min(ndustspec,2)
  const i3_102_104_145_161 = min(ndustspec,3)
  const ii1_102_104_145_161 = ndustspec
  const ii2_102_104_145_161 = max(ndustspec-1,1)
  const ii3_102_104_145_161 = max(ndustspec-2,1)
  real rr1_102_104_145_161 = 0.
  real rr2_102_104_145_161 = 0.
  real rr3_102_104_145_161 = 0.
  int ndust_2nd_species_102_104_145_161
  real ff_tmp_105_145_161[AC_ndustspec__mod__cparam]
  real nd_new_105_145_161[AC_ndustspec__mod__cparam]
  int k_105_145_161
  int j_105_145_161
  int jj_105_145_161
  int kk1_105_145_161
  int kk2_105_145_161
  real gs_105_145_161
  int k_102_105_145_161
  const i1_102_105_145_161 = 1
  const i2_102_105_145_161 = min(ndustspec,2)
  const i3_102_105_145_161 = min(ndustspec,3)
  const ii1_102_105_145_161 = ndustspec
  const ii2_102_105_145_161 = max(ndustspec-1,1)
  const ii3_102_105_145_161 = max(ndustspec-2,1)
  real rr1_102_105_145_161 = 0.
  real rr2_102_105_145_161 = 0.
  real rr3_102_105_145_161 = 0.
  int ndust_2nd_species_102_105_145_161
  real ff_tmp_106_145_161[AC_ndustspec__mod__cparam]
  real nd_new_106_145_161[AC_ndustspec__mod__cparam]
  int k_106_145_161
  int j_106_145_161
  int jj_106_145_161
  int kk1_106_145_161
  int kk2_106_145_161
  real gs_106_145_161
  int k_102_106_145_161
  const i1_102_106_145_161 = 1
  const i2_102_106_145_161 = min(ndustspec,2)
  const i3_102_106_145_161 = min(ndustspec,3)
  const ii1_102_106_145_161 = ndustspec
  const ii2_102_106_145_161 = max(ndustspec-1,1)
  const ii3_102_106_145_161 = max(ndustspec-2,1)
  real rr1_102_106_145_161 = 0.
  real rr2_102_106_145_161 = 0.
  real rr3_102_106_145_161 = 0.
  int ndust_2nd_species_102_106_145_161
  real ff_tmp_107_145_161[AC_ndustspec__mod__cparam]
  real nd_new_107_145_161[AC_ndustspec__mod__cparam]
  int k_107_145_161
  int j_107_145_161
  int jj_107_145_161
  int kk1_107_145_161
  int kk2_107_145_161
  real gs_107_145_161
  int k_102_107_145_161
  const i1_102_107_145_161 = 1
  const i2_102_107_145_161 = min(ndustspec,2)
  const i3_102_107_145_161 = min(ndustspec,3)
  const ii1_102_107_145_161 = ndustspec
  const ii2_102_107_145_161 = max(ndustspec-1,1)
  const ii3_102_107_145_161 = max(ndustspec-2,1)
  real rr1_102_107_145_161 = 0.
  real rr2_102_107_145_161 = 0.
  real rr3_102_107_145_161 = 0.
  int ndust_2nd_species_102_107_145_161
  real tt_108_145_161
  real kn_108_145_161
  real cor_factor_108_145_161
  real d_coeff_108_145_161
  real di_108_145_161
  real dk_108_145_161
  real dik_108_145_161
  real kbc_108_145_161
  real vmean_i_108_145_161
  real vmean_k_108_145_161
  real vmean_ik_108_145_161
  real gamma_i_108_145_161
  real gamma_k_108_145_161
  real omega_i_108_145_161
  real omega_k_108_145_161
  real sigma_ik_108_145_161
  real deltavd_108_145_161
  real deltavd_therm_108_145_161
  real deltavd_turbu_108_145_161
  real fact_108_145_161
  real deltavd_drift2_108_145_161
  real deltavd_drift2a_108_145_161
  real deltavd_drift2b_108_145_161
  real ust_108_145_161
  real mu_air_108_145_161
  real rho_air_108_145_161
  real rik_108_145_161
  int i_108_145_161
  int j_108_145_161
  int l_108_145_161
  int k_108_145_161
  int lgh_108_145_161
  int lgh_40_108_145_161
  real deltavd_therm_40_108_145_161
  real deltavd_turbu_40_108_145_161
  real fact_40_108_145_161
  real deltavd_drift2_40_108_145_161
  real deltavd_drift2a_40_108_145_161
  real deltavd_drift2b_40_108_145_161
  real ust_40_108_145_161
  real mu_air_40_108_145_161
  real rho_air_40_108_145_161
  real rik_40_108_145_161
  const yr_39_40_108_145_161 = 31556926.
  const u_39_40_108_145_161 = 1660538921e-33
  const mu_gas_39_40_108_145_161 = 23e-1
  const mh_39_40_108_145_161 = 100749e-5*1660538921e-33
  real nh_39_40_108_145_161
  real cs_39_40_108_145_161
  real re_39_40_108_145_161
  real t_eta_39_40_108_145_161
  real ts_i_39_40_108_145_161
  real ts_j_39_40_108_145_161
  real ts_1_39_40_108_145_161
  real st_1_39_40_108_145_161
  real st_2_39_40_108_145_161
  real x_st_39_40_108_145_161
  real beta_st_39_40_108_145_161
  real res_39_40_108_145_161
  real t_dyn_39_40_108_145_161
  real dndfac_109_145_161
  real dndfaci_109_145_161
  real dndfacj_109_145_161
  real tmp_109_145_161
  real momcons_term_x_109_145_161
  real momcons_term_y_109_145_161
  real momcons_term_z_109_145_161
  int i_109_145_161
  int j_109_145_161
  int k_109_145_161
  int l_109_145_161
  int lgh_109_145_161
  bool lmdvar_noevolve_109_145_161
  real cc_tmp_114_117_145_161
  real dmdfac_114_117_145_161
  int k_114_117_145_161
  int l_114_117_145_161
  int lgh_114_117_145_161
  real supsatratio1_113_114_117_145_161
  real pp_113_114_117_145_161
  real ppmon_113_114_117_145_161
  real ppsat_113_114_117_145_161
  real vth_113_114_117_145_161
  real mu_113_114_117_145_161
  real mfluxcondp_116_117_145_161
  real mfluxcondm_116_117_145_161
  real cc_tmp_116_117_145_161
  real coefkp_116_117_145_161
  real coefkm_116_117_145_161
  real coefk0_116_117_145_161
  real dampfact_116_117_145_161
  int k_116_117_145_161
  real supsatratio1_115_116_117_145_161
  real pp_115_116_117_145_161
  real ppmon_115_116_117_145_161
  real ppsat_115_116_117_145_161
  real vth_115_116_117_145_161
  real mu_115_116_117_145_161
  real d2fdx_120_145_161
  real d2fdy_120_145_161
  real d2fdz_120_145_161
  real tmp_120_145_161
  int j_121_145_161
  real d2fdx_122_145_161
  real d2fdy_122_145_161
  real d2fdz_122_145_161
  real tmp_122_145_161
  real f_target_126_145_161
  int i_144_145_161
  int j_144_145_161
  int l_144_145_161
  int lgh_144_145_161
  real updated_nd_144_145_161[AC_ndustspec__mod__cparam]
  real updated_rho_144_145_161[AC_ndustspec__mod__cparam]
  real new_rhod_144_145_161[AC_ndustspec__mod__cparam]
  real new_nd_144_145_161[AC_ndustspec__mod__cparam]
  real deltav_144_145_161[ndustspec][ndustspec]
  int lgh_128_144_145_161
  real deltavd_therm_128_144_145_161
  real deltavd_turbu_128_144_145_161
  real fact_128_144_145_161
  real deltavd_drift2_128_144_145_161
  real deltavd_drift2a_128_144_145_161
  real deltavd_drift2b_128_144_145_161
  real ust_128_144_145_161
  real mu_air_128_144_145_161
  real rho_air_128_144_145_161
  real rik_128_144_145_161
  const yr_39_128_144_145_161 = 31556926.
  const u_39_128_144_145_161 = 1660538921e-33
  const mu_gas_39_128_144_145_161 = 23e-1
  const mh_39_128_144_145_161 = 100749e-5*1660538921e-33
  real nh_39_128_144_145_161
  real cs_39_128_144_145_161
  real re_39_128_144_145_161
  real t_eta_39_128_144_145_161
  real ts_i_39_128_144_145_161
  real ts_j_39_128_144_145_161
  real ts_1_39_128_144_145_161
  real st_1_39_128_144_145_161
  real st_2_39_128_144_145_161
  real x_st_39_128_144_145_161
  real beta_st_39_128_144_145_161
  real res_39_128_144_145_161
  real t_dyn_39_128_144_145_161
  int j_142_143_144_145_161
  int iprogress_142_143_144_145_161
  int nsub_142_143_144_145_161
  int ndt_142_143_144_145_161
  int tot_nsub_142_143_144_145_161
  int tot_ndt_142_143_144_145_161
  real gij_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real gijnew_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real coeff_cfl_142_143_144_145_161
  real eps_gij_142_143_144_145_161
  real massbins_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real dtcflsub_141_142_143_144_145_161
  real dtsub_141_142_143_144_145_161
  real dtlast_141_142_143_144_145_161
  real dt_141_142_143_144_145_161
  real gijsub_in_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real gijsub_out_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real flux_131_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real tabdtcfl_131_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_131_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_131_141_142_143_144_145_161
  int j_131_141_142_143_144_145_161
  int lp_129_131_141_142_143_144_145_161
  int l_129_131_141_142_143_144_145_161
  int i_130_131_141_142_143_144_145_161
  int j_130_131_141_142_143_144_145_161
  int k_130_131_141_142_143_144_145_161
  real cfl_return_value_131_141_142_143_144_145_161
  real gij_1_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real gij_2_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_1_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_2_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  int j_137_141_142_143_144_145_161
  int i_137_141_142_143_144_145_161
  real flux_134_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_134_137_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_134_137_141_142_143_144_145_161
  int j_134_137_141_142_143_144_145_161
  int lp_132_134_137_141_142_143_144_145_161
  int l_132_134_137_141_142_143_144_145_161
  int i_133_134_137_141_142_143_144_145_161
  int j_133_134_137_141_142_143_144_145_161
  int k_133_134_137_141_142_143_144_145_161
  real flux_135_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_135_137_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_135_137_141_142_143_144_145_161
  int j_135_137_141_142_143_144_145_161
  int lp_132_135_137_141_142_143_144_145_161
  int l_132_135_137_141_142_143_144_145_161
  int i_133_135_137_141_142_143_144_145_161
  int j_133_135_137_141_142_143_144_145_161
  int k_133_135_137_141_142_143_144_145_161
  real flux_136_137_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_136_137_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_136_137_141_142_143_144_145_161
  int j_136_137_141_142_143_144_145_161
  int lp_132_136_137_141_142_143_144_145_161
  int l_132_136_137_141_142_143_144_145_161
  int i_133_136_137_141_142_143_144_145_161
  int j_133_136_137_141_142_143_144_145_161
  int k_133_136_137_141_142_143_144_145_161
  real flux_138_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real tabdtcfl_138_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_138_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_138_141_142_143_144_145_161
  int j_138_141_142_143_144_145_161
  int lp_129_138_141_142_143_144_145_161
  int l_129_138_141_142_143_144_145_161
  int i_130_138_141_142_143_144_145_161
  int j_130_138_141_142_143_144_145_161
  int k_130_138_141_142_143_144_145_161
  real cfl_return_value_138_141_142_143_144_145_161
  real gij_1_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real gij_2_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_1_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_2_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  int j_139_141_142_143_144_145_161
  int i_139_141_142_143_144_145_161
  real flux_134_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_134_139_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_134_139_141_142_143_144_145_161
  int j_134_139_141_142_143_144_145_161
  int lp_132_134_139_141_142_143_144_145_161
  int l_132_134_139_141_142_143_144_145_161
  int i_133_134_139_141_142_143_144_145_161
  int j_133_134_139_141_142_143_144_145_161
  int k_133_134_139_141_142_143_144_145_161
  real flux_135_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_135_139_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_135_139_141_142_143_144_145_161
  int j_135_139_141_142_143_144_145_161
  int lp_132_135_139_141_142_143_144_145_161
  int l_132_135_139_141_142_143_144_145_161
  int i_133_135_139_141_142_143_144_145_161
  int j_133_135_139_141_142_143_144_145_161
  int k_133_135_139_141_142_143_144_145_161
  real flux_136_139_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_136_139_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_136_139_141_142_143_144_145_161
  int j_136_139_141_142_143_144_145_161
  int lp_132_136_139_141_142_143_144_145_161
  int l_132_136_139_141_142_143_144_145_161
  int i_133_136_139_141_142_143_144_145_161
  int j_133_136_139_141_142_143_144_145_161
  int k_133_136_139_141_142_143_144_145_161
  real gij_1_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real gij_2_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_1_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real l_k0_2_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  int j_140_141_142_143_144_145_161
  int i_140_141_142_143_144_145_161
  real flux_134_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_134_140_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_134_140_141_142_143_144_145_161
  int j_134_140_141_142_143_144_145_161
  int lp_132_134_140_141_142_143_144_145_161
  int l_132_134_140_141_142_143_144_145_161
  int i_133_134_140_141_142_143_144_145_161
  int j_133_134_140_141_142_143_144_145_161
  int k_133_134_140_141_142_143_144_145_161
  real flux_135_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_135_140_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_135_140_141_142_143_144_145_161
  int j_135_140_141_142_143_144_145_161
  int lp_132_135_140_141_142_143_144_145_161
  int l_132_135_140_141_142_143_144_145_161
  int i_133_135_140_141_142_143_144_145_161
  int j_133_135_140_141_142_143_144_145_161
  int k_133_135_140_141_142_143_144_145_161
  real flux_136_140_141_142_143_144_145_161[AC_ndustspec__mod__cparam]
  real arr_gij_dv_136_140_141_142_143_144_145_161[ndustspec][ndustspec]
  real hj_136_140_141_142_143_144_145_161
  int j_136_140_141_142_143_144_145_161
  int lp_132_136_140_141_142_143_144_145_161
  int l_132_136_140_141_142_143_144_145_161
  int i_133_136_140_141_142_143_144_145_161
  int j_133_136_140_141_142_143_144_145_161
  int k_133_136_140_141_142_143_144_145_161
  real dt1_max_loc
  real dt1_advec_167
  real dt1_diffus_167
  real dt1_src_167
  int isum_169
  if(lgpu) {
    sqrt_ascale__mod__cdata=0.
    hubble__mod__cdata=0.
  }
  if (lanelastic) {
    df_iuu_pencil = DF_UVEC
    DF_UVEC=0.0
  }
  if (AC_lupdate_courant_dt__mod__cdata && (!AC_ldt_paronly__mod__cdata)) {
    advec_cs2__mod__cdata=0.
    maxadvec__mod__cdata=0.
    if (lenergy || ldensity || lmagnetic || lradiation || lneutralvelocity || lcosmicray ||   (ltestfield_z && AC_iuutest__mod__cdata>0)) {
      advec2__mod__cdata=0.
    }
    if (ldensity || lviscosity || lmagnetic || lenergy || ldustvelocity || ldustdensity) {
      advec2_hypermesh__mod__cdata=0.0
    }
    maxdiffus__mod__cdata=0.
    maxdiffus2__mod__cdata=0.
    maxdiffus3__mod__cdata=0.
    maxsrc__mod__cdata=0.
  }
  if (lgpu  ||  ! AC_lcartesian_coords__mod__cdata  ||  !all(AC_lequidist__mod__cdata)) {
    if (AC_lspherical_coords__mod__cdata) {
      dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
      dline_1__mod__cdata.y = AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
      dline_1__mod__cdata.z = AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_sin1th__mod__cdata[AC_m__mod__cdata-1] * AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
      if (lcoarse_mn__mod__cdata) {
      }
    }
    else if (AC_lcylindrical_coords__mod__cdata)   {
      dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
      dline_1__mod__cdata.y = AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
      dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
    }
    else if (AC_lcartesian_coords__mod__cdata)   {
      if (AC_lequidist__mod__cdata.x) {
        dline_1__mod__cdata.x = AC_dx1_scalar__mod__cdata
      }
      else {
        dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
      }
      if (AC_lequidist__mod__cdata.y) {
        dline_1__mod__cdata.y = AC_dy1_scalar__mod__cdata
      }
      else {
        dline_1__mod__cdata.y = AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
      }
      if (AC_lequidist__mod__cdata.z) {
        dline_1__mod__cdata.z = AC_dz1_scalar__mod__cdata
      }
      else {
        dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
      }
    }
    else if (AC_lpipe_coords__mod__cdata)   {
      dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
      dline_1__mod__cdata.y = AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
      dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
    }
    dxmax_pencil__mod__cdata = 0.
    if (nxgrid!= 1) {
      dxmax_pencil__mod__cdata =     1.0/ dline_1__mod__cdata.x
    }
    if (nygrid!= 1) {
      dxmax_pencil__mod__cdata = max(1.0/ dline_1__mod__cdata.y,dxmax_pencil__mod__cdata)
    }
    if (nzgrid!= 1) {
      dxmax_pencil__mod__cdata = max(1.0/ dline_1__mod__cdata.z,dxmax_pencil__mod__cdata)
    }
    dxmin_pencil__mod__cdata = 0.
    if (nxgrid!= 1) {
      dxmin_pencil__mod__cdata =     1.0/ dline_1__mod__cdata.x
    }
    if (nygrid!= 1) {
      dxmin_pencil__mod__cdata = min(1.0/ dline_1__mod__cdata.y,dxmin_pencil__mod__cdata)
    }
    if (nzgrid!= 1) {
      dxmin_pencil__mod__cdata = min(1.0/ dline_1__mod__cdata.z,dxmin_pencil__mod__cdata)
    }
    if (AC_lmaximal_cdtv__mod__cdata) {
      dxyz_2__mod__cdata = max((dline_1__mod__cdata.x*dline_1__mod__cdata.x),(dline_1__mod__cdata.y*dline_1__mod__cdata.y),(dline_1__mod__cdata.z*dline_1__mod__cdata.z))
      dxyz_4__mod__cdata = max((dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x),(dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y),(dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z))
      dxyz_6__mod__cdata = max((dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x),(dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y),(dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z))
    }
    else {
      dxyz_2__mod__cdata =( dline_1__mod__cdata.x* dline_1__mod__cdata.x) +( dline_1__mod__cdata.y* dline_1__mod__cdata.y) +( dline_1__mod__cdata.z* dline_1__mod__cdata.z)
      dxyz_4__mod__cdata =( dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x) +( dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y) +( dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z)
      dxyz_6__mod__cdata =( dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x* dline_1__mod__cdata.x) +( dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y* dline_1__mod__cdata.y) +( dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z* dline_1__mod__cdata.z)
    }
    dvol__mod__cdata = AC_dvol_x__mod__cdata[vertexIdx.x]*AC_dvol_y__mod__cdata[AC_m__mod__cdata-1]*AC_dvol_z__mod__cdata[AC_n__mod__cdata-1]
  }
  if (AC_lcartesian_coords__mod__cdata) {
    if (AC_lpencil__mod__cdata[i_x_mn-1]) {
      ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]
    }
    if (AC_lpencil__mod__cdata[i_y_mn-1]) {
      ac_transformed_pencil_y_mn    = AC_y__mod__cdata[AC_m__mod__cdata-1]
    }
    if (AC_lpencil__mod__cdata[i_z_mn-1]) {
      ac_transformed_pencil_z_mn    = AC_z__mod__cdata[AC_n__mod__cdata-1]
    }
    if (AC_lpencil__mod__cdata[i_r_mn-1]) {
      ac_transformed_pencil_r_mn    = sqrt((AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x])+(AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_y__mod__cdata[AC_m__mod__cdata-1])+(AC_z__mod__cdata[AC_n__mod__cdata-1]*AC_z__mod__cdata[AC_n__mod__cdata-1]))
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn-1]) {
      ac_transformed_pencil_rcyl_mn = sqrt((AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x])+(AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_y__mod__cdata[AC_m__mod__cdata-1]))
    }
    if (AC_lpencil__mod__cdata[i_phi_mn-1]) {
      if (AC_y__mod__cdata[AC_m__mod__cdata-1]==0) {
        ac_transformed_pencil_phi_mn  = 0.
      }
      else {
        ac_transformed_pencil_phi_mn  = atan2(AC_y__mod__cdata[AC_m__mod__cdata-1],AC_x__mod__cdata[vertexIdx.x])
      }
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn1-1]) {
      ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,tini)
    }
    if (AC_lpencil__mod__cdata[i_r_mn1-1]) {
      ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,tini)
    }
    if (AC_lpencil__mod__cdata[i_pomx-1]) {
      ac_transformed_pencil_pomx    = AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
    }
    if (AC_lpencil__mod__cdata[i_pomy-1]) {
      ac_transformed_pencil_pomy    = AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
    }
    if (AC_lpencil__mod__cdata[i_phix-1]) {
      ac_transformed_pencil_phix    =-AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
    }
    if (AC_lpencil__mod__cdata[i_phiy-1]) {
      ac_transformed_pencil_phiy    = AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
    }
  }
  else if (AC_lcylindrical_coords__mod__cdata) {
    if (AC_lpencil__mod__cdata[i_x_mn-1]) {
      ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]*cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_y_mn-1]) {
      ac_transformed_pencil_y_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_z_mn-1]) {
      ac_transformed_pencil_z_mn    = AC_z__mod__cdata[AC_n__mod__cdata-1]
    }
    if (AC_lpencil__mod__cdata[i_r_mn-1]) {
      ac_transformed_pencil_r_mn    = sqrt((AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x])+(AC_z__mod__cdata[AC_n__mod__cdata-1]*AC_z__mod__cdata[AC_n__mod__cdata-1]))
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn-1]) {
      ac_transformed_pencil_rcyl_mn = AC_x__mod__cdata[vertexIdx.x]
    }
    if (AC_lpencil__mod__cdata[i_phi_mn-1]) {
      ac_transformed_pencil_phi_mn  = AC_y__mod__cdata[AC_m__mod__cdata-1]
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn1-1]) {
      ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,tini)
    }
    if (AC_lpencil__mod__cdata[i_r_mn1-1]) {
      ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,tini)
    }
    if (AC_lpencil__mod__cdata[i_pomx-1]) {
      ac_transformed_pencil_pomx    = 1.
    }
    if (AC_lpencil__mod__cdata[i_pomy-1]) {
      ac_transformed_pencil_pomy    = 0.
    }
    if (AC_lpencil__mod__cdata[i_phix-1]) {
      ac_transformed_pencil_phix    = 0.
    }
    if (AC_lpencil__mod__cdata[i_phiy-1]) {
      ac_transformed_pencil_phiy    = 1.
    }
  }
  else if (AC_lspherical_coords__mod__cdata) {
    if (AC_lpencil__mod__cdata[i_x_mn-1]) {
      ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])*cos(AC_z__mod__cdata[AC_n__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_y_mn-1]) {
      ac_transformed_pencil_y_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])*sin(AC_z__mod__cdata[AC_n__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_z_mn-1]) {
      ac_transformed_pencil_z_mn    = AC_x__mod__cdata[vertexIdx.x]*cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_r_mn-1]) {
      ac_transformed_pencil_r_mn    = AC_x__mod__cdata[vertexIdx.x]
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn-1]) {
      ac_transformed_pencil_rcyl_mn = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
    }
    if (AC_lpencil__mod__cdata[i_phi_mn-1]) {
      ac_transformed_pencil_phi_mn  = AC_z__mod__cdata[AC_n__mod__cdata-1]
    }
    if (AC_lpencil__mod__cdata[i_rcyl_mn1-1]) {
      ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,tini)
    }
    if (AC_lpencil__mod__cdata[i_r_mn1-1]) {
      ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,tini)
    }
  }
  if (AC_lpencil__mod__cdata[i_rr-1]) {
    if (AC_lcartesian_coords__mod__cdata) {
      ac_transformed_pencil_rr.x=ac_transformed_pencil_x_mn
      ac_transformed_pencil_rr.y=ac_transformed_pencil_y_mn
      ac_transformed_pencil_rr.z=ac_transformed_pencil_z_mn
    }
    else {
    }
  }
  if (AC_lpencil__mod__cdata[i_evr-1]) {
    if (AC_lcartesian_coords__mod__cdata) {
      ac_transformed_pencil_evr.x = ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomx
      ac_transformed_pencil_evr.y = ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomy
      ac_transformed_pencil_evr.z = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
    }
    else {
    }
  }
  if (AC_lpencil__mod__cdata[i_evth-1]) {
    if (AC_lcartesian_coords__mod__cdata) {
      ac_transformed_pencil_evth.x = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomx
      ac_transformed_pencil_evth.y = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomy
      ac_transformed_pencil_evth.z = -ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1
    }
    else {
    }
  }
  if (AC_lpencil__mod__cdata[i_uu-1]) {
    ac_transformed_pencil_uu.x = 0.0
    ac_transformed_pencil_uu.y = 0.0
    ac_transformed_pencil_uu.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_u2-1]) {
    ac_transformed_pencil_u2=0.0
  }
  if (AC_lpencil__mod__cdata[i_oo-1]) {
    ac_transformed_pencil_oo.x = 0.0
    ac_transformed_pencil_oo.y = 0.0
    ac_transformed_pencil_oo.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_ou-1]) {
    ac_transformed_pencil_ou=0.0
  }
  if (AC_lpencil__mod__cdata[i_oxu-1]) {
    ac_transformed_pencil_oxu.x = 0.0
    ac_transformed_pencil_oxu.y = 0.0
    ac_transformed_pencil_oxu.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_uij-1]) {
    ac_transformed_pencil_uij=0.0
  }
  if (AC_lpencil__mod__cdata[i_sij-1]) {
    ac_transformed_pencil_sij=0.0
  }
  if (AC_lpencil__mod__cdata[i_sij2-1]) {
    ac_transformed_pencil_sij2=0.0
  }
  if (AC_lpencil__mod__cdata[i_divu-1]) {
    ac_transformed_pencil_divu=0.0
  }
  if (AC_lpencil__mod__cdata[i_uij5-1]) {
    ac_transformed_pencil_uij5=0.0
  }
  if (AC_lpencil__mod__cdata[i_graddivu-1]) {
    ac_transformed_pencil_graddivu.x = 0.0
    ac_transformed_pencil_graddivu.y = 0.0
    ac_transformed_pencil_graddivu.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_ugu-1]) {
    ac_transformed_pencil_ugu.x = 0.0
    ac_transformed_pencil_ugu.y = 0.0
    ac_transformed_pencil_ugu.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_ogu-1]) {
    ac_transformed_pencil_ogu.x = 0.0
    ac_transformed_pencil_ogu.y = 0.0
    ac_transformed_pencil_ogu.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_del2u-1]) {
    ac_transformed_pencil_del2u.x = 0.0
    ac_transformed_pencil_del2u.y = 0.0
    ac_transformed_pencil_del2u.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_curlo-1]) {
    ac_transformed_pencil_curlo.x = 0.0
    ac_transformed_pencil_curlo.y = 0.0
    ac_transformed_pencil_curlo.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_lorentz_gamma2-1]) {
    ac_transformed_pencil_lorentz_gamma2=0.0
  }
  if (AC_lpencil__mod__cdata[i_lorentz_gamma-1]) {
    ac_transformed_pencil_lorentz_gamma=0.
  }
  if (AC_lpencil__mod__cdata[i_lorentz-1]) {
    ac_transformed_pencil_lorentz=0.
  }
  if (AC_lpencil__mod__cdata[i_ss_rel-1]) {
    ac_transformed_pencil_ss_rel.x = 0.
    ac_transformed_pencil_ss_rel.y = 0.
    ac_transformed_pencil_ss_rel.z = 0.
  }
  if (AC_lpencil__mod__cdata[i_divss_rel-1]) {
    ac_transformed_pencil_divss_rel=0.
  }
  if (AC_lpencil__mod__cdata[i_hless-1]) {
    ac_transformed_pencil_hless=0.
  }
  if (AC_ldensity_nolog__mod__cdata) {
    ac_transformed_pencil_rho=value(Field(AC_irho__mod__cdata-1))
    if (AC_lreference_state__mod__cdata) {
      ac_transformed_pencil_rho=ac_transformed_pencil_rho+AC_reference_state__mod__density[vertexIdx.x][iref_rho-1]
    }
    if (AC_lpencil__mod__cdata[i_rho1-1]) {
      ac_transformed_pencil_rho1=1.0/ac_transformed_pencil_rho
    }
    if (AC_lpencil__mod__cdata[i_lnrho-1]) {
      ac_transformed_pencil_lnrho=log(ac_transformed_pencil_rho)
    }
    if (AC_lpencil__mod__cdata[i_glnrho-1] || AC_lpencil__mod__cdata[i_grho-1]) {
      ac_transformed_pencil_grho = gradient(Field(AC_irho__mod__cdata-1))
      if (AC_lreference_state__mod__cdata) {
        ac_transformed_pencil_grho.x=ac_transformed_pencil_grho.x+AC_reference_state__mod__density[vertexIdx.x][iref_grho-1]
      }
      if (AC_lpencil__mod__cdata[i_glnrho-1]) {
        ac_transformed_pencil_glnrho.x=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.x
        ac_transformed_pencil_glnrho.y=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.y
        ac_transformed_pencil_glnrho.z=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.z
      }
    }
    if (AC_lpencil__mod__cdata[i_ugrho-1]) {
      ac_transformed_pencil_ugrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_grho)
      if (AC_lupw_rho__mod__density) ac_transformed_pencil_ugrho = ac_transformed_pencil_ugrho - dot(abs(ac_transformed_pencil_uu),gradient_upwd(Field(AC_ilnrho__mod__cdata-1)))
    }
    if (AC_lpencil__mod__cdata[i_glnrho2-1]) {
      ac_transformed_pencil_glnrho2 = dot(ac_transformed_pencil_glnrho,ac_transformed_pencil_glnrho)
    }
    if (AC_lpencil__mod__cdata[i_del2rho-1]) {
      ac_transformed_pencil_del2rho = laplace(Field(AC_irho__mod__cdata-1))
      if (AC_lreference_state__mod__cdata) {
        ac_transformed_pencil_del2rho=ac_transformed_pencil_del2rho+AC_reference_state__mod__density[vertexIdx.x][iref_d2rho-1]
      }
    }
    if (AC_lpencil__mod__cdata[i_del2lnrho-1]) {
      ac_transformed_pencil_del2lnrho=ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho-ac_transformed_pencil_glnrho2
    }
    if (AC_lpencil__mod__cdata[i_del6rho-1]) {
      if (AC_ldiff_hyper3__mod__density) {
        ac_transformed_pencil_del6rho = del6(Field(AC_irho__mod__cdata-1))
        if (AC_lreference_state__mod__cdata) {
          ac_transformed_pencil_del6rho=ac_transformed_pencil_del6rho+AC_reference_state__mod__density[vertexIdx.x][iref_d6rho-1]
        }
      }
      else if (AC_ldiff_hyper3_strict__mod__density) {
        ac_transformed_pencil_del6rho=0.
        tmp_11_15_20_21_55 = der6x(Field(AC_irho__mod__cdata-1))
        ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_11_15_20_21_55
        if (1!=1) {
          tmp_11_15_20_21_55 = der6x(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (2!=1) {
          tmp_11_15_20_21_55 = der4x2y(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (3!=1) {
          tmp_11_15_20_21_55 = der4x2z(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        tmp_11_15_20_21_55 = der6y(Field(AC_irho__mod__cdata-1))
        ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_11_15_20_21_55
        if (1!=2) {
          tmp_11_15_20_21_55 = der4y2x(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (2!=2) {
          tmp_11_15_20_21_55 = der6y(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (3!=2) {
          tmp_11_15_20_21_55 = der4y2z(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        tmp_11_15_20_21_55 = der6z(Field(AC_irho__mod__cdata-1))
        ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_11_15_20_21_55
        if (1!=3) {
          tmp_11_15_20_21_55 = der4z2x(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (2!=3) {
          tmp_11_15_20_21_55 = der4z2y(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        if (3!=3) {
          tmp_11_15_20_21_55 = der6z(Field(AC_irho__mod__cdata-1))
          ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_11_15_20_21_55
        }
        tmp_11_15_20_21_55 = der2i2j2k(Field(AC_irho__mod__cdata-1))
        ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 6*tmp_11_15_20_21_55
      }
    }
    if (AC_lpencil__mod__cdata[i_del6lnrho-1]) {
      if (AC_ldiff_hyper3lnrho__mod__density) {
      }
      else if (AC_ldiff_hyper3lnrho_strict__mod__density) {
      }
    }
    if (AC_lpencil__mod__cdata[i_sglnrho-1]) {
      ac_transformed_pencil_sglnrho = ac_transformed_pencil_sij*ac_transformed_pencil_glnrho
    }
    if (AC_lpencil__mod__cdata[i_uij5glnrho-1]) {
      ac_transformed_pencil_uij5glnrho = ac_transformed_pencil_uij5*ac_transformed_pencil_glnrho
    }
    if (AC_lpencil__mod__cdata[i_transprho-1]) {
      if (AC_lreference_state__mod__cdata) {
        ac_transformed_pencil_transprho = impossible
      }
      else {
        ac_transformed_pencil_transprho = impossible
      }
    }
    if (AC_lpencil__mod__cdata[i_uuadvec_grho-1]) {
      ac_transformed_pencil_uuadvec_grho = dot(ac_transformed_pencil_uu_advec,ac_transformed_pencil_grho)
      if (AC_lupw_rho__mod__density) {
        tmp_15_20_21_55 = del_upwd(ac_transformed_pencil_uu_advec,Field(AC_irho__mod__cdata-1))
        ac_transformed_pencil_uuadvec_grho = ac_transformed_pencil_uuadvec_grho - tmp_15_20_21_55
      }
    }
    if (AC_lpencil__mod__cdata[i_divss-1]) {
      ac_transformed_pencil_divss = divergence((Field3){Field(AC_iux__mod__cdata-1), Field(AC_iux__mod__cdata), Field(AC_iux__mod__cdata+1)})
    }
    if (false) {
      if (false) {
        if (0.0==0.) {
          if(AC_t__mod__cdata < ac_transformed_pencil_hless) {
            ac_transformed_pencil_rho=ac_transformed_pencil_rho-0.0
          }
        }
        else {
          ac_transformed_pencil_rho=ac_transformed_pencil_rho-0.0*max(0.e0,min(1.e0,(ac_transformed_pencil_hless+0.5e0*0.0-AC_t__mod__cdata)/0.0))
        }
      }
      if (false) {
        ac_transformed_pencil_rho=ac_transformed_pencil_rho/(AC_cs201__mod__density*ac_transformed_pencil_lorentz-AC_cs20__mod__equationofstate)
      }
    }
  }
  else {
    ac_transformed_pencil_lnrho=value(Field(AC_ilnrho__mod__cdata-1))
    if (AC_lpencil__mod__cdata[i_rho1-1]) {
      ac_transformed_pencil_rho1=exp(-value(Field(AC_ilnrho__mod__cdata-1)))
    }
    if (AC_lpencil__mod__cdata[i_rho-1]) {
      ac_transformed_pencil_rho=exp(value(Field(AC_ilnrho__mod__cdata-1)))
    }
    if (AC_lpencil__mod__cdata[i_glnrho-1] || AC_lpencil__mod__cdata[i_grho-1]) {
      ac_transformed_pencil_glnrho = gradient(Field(AC_ilnrho__mod__cdata-1))
      if (AC_lpencil__mod__cdata[i_grho-1]) {
        ac_transformed_pencil_grho.x=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.x
        ac_transformed_pencil_grho.y=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.y
        ac_transformed_pencil_grho.z=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.z
      }
    }
    if (AC_lpencil__mod__cdata[i_uglnrho-1]) {
      if (AC_lupw_lnrho__mod__density) {
        ac_transformed_pencil_uglnrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_glnrho)
        if (AC_lupw_lnrho__mod__density) ac_transformed_pencil_uglnrho = ac_transformed_pencil_uglnrho - dot(abs(ac_transformed_pencil_uu),gradient_upwd(Field(AC_ilnrho__mod__cdata-1)))
      }
      else {
        ac_transformed_pencil_uglnrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_glnrho)
      }
    }
    if (AC_lpencil__mod__cdata[i_glnrho2-1]) {
      ac_transformed_pencil_glnrho2 = dot(ac_transformed_pencil_glnrho,ac_transformed_pencil_glnrho)
    }
    if (AC_lpencil__mod__cdata[i_del2lnrho-1]) {
      ac_transformed_pencil_del2lnrho = laplace(Field(AC_ilnrho__mod__cdata-1))
    }
    if (AC_lpencil__mod__cdata[i_del6lnrho-1]) {
      if (AC_ldiff_hyper3lnrho__mod__density) {
        ac_transformed_pencil_del6lnrho = del6(Field(AC_ilnrho__mod__cdata-1))
      }
      else if (AC_ldiff_hyper3lnrho_strict__mod__density) {
        ac_transformed_pencil_del6lnrho=0.
        tmp_16_18_20_21_55 = der6x(Field(AC_ilnrho__mod__cdata-1))
        ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_16_18_20_21_55
        if (1!=1) {
          tmp_16_18_20_21_55 = der6x(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (2!=1) {
          tmp_16_18_20_21_55 = der4x2y(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (3!=1) {
          tmp_16_18_20_21_55 = der4x2z(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        tmp_16_18_20_21_55 = der6y(Field(AC_ilnrho__mod__cdata-1))
        ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_16_18_20_21_55
        if (1!=2) {
          tmp_16_18_20_21_55 = der4y2x(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (2!=2) {
          tmp_16_18_20_21_55 = der6y(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (3!=2) {
          tmp_16_18_20_21_55 = der4y2z(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        tmp_16_18_20_21_55 = der6z(Field(AC_ilnrho__mod__cdata-1))
        ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_16_18_20_21_55
        if (1!=3) {
          tmp_16_18_20_21_55 = der4z2x(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (2!=3) {
          tmp_16_18_20_21_55 = der4z2y(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        if (3!=3) {
          tmp_16_18_20_21_55 = der6z(Field(AC_ilnrho__mod__cdata-1))
          ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_16_18_20_21_55
        }
        tmp_16_18_20_21_55 = der2i2j2k(Field(AC_ilnrho__mod__cdata-1))
        ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 6*tmp_16_18_20_21_55
      }
    }
    if (AC_lpencil__mod__cdata[i_hlnrho-1]) {
      ac_transformed_pencil_hlnrho = hessian(Field(AC_ilnrho__mod__cdata-1))
    }
    if (AC_lpencil__mod__cdata[i_sglnrho-1]) {
      ac_transformed_pencil_sglnrho = ac_transformed_pencil_sij*ac_transformed_pencil_glnrho
    }
    if (AC_lpencil__mod__cdata[i_uij5glnrho-1]) {
      ac_transformed_pencil_uij5glnrho = ac_transformed_pencil_uij5*ac_transformed_pencil_glnrho
    }
    if (AC_lpencil__mod__cdata[i_uuadvec_glnrho-1]) {
      ac_transformed_pencil_uuadvec_glnrho = dot(ac_transformed_pencil_uu_advec,ac_transformed_pencil_glnrho)
    }
  }
  if (AC_lpencil__mod__cdata[i_ekin-1]) {
    if (false) {
      ac_transformed_pencil_ekin=AC_cs201__mod__density*ac_transformed_pencil_rho*ac_transformed_pencil_lorentz*ac_transformed_pencil_u2
    }
    else {
      ac_transformed_pencil_ekin=0.5*ac_transformed_pencil_rho*ac_transformed_pencil_u2
    }
  }
  if (AC_lmultithread__mod__cdata  &&  AC_ldiff_hyper3_mesh__mod__density  &&  AC_idiag_dtv__mod__cdata!= 0) {
    if (AC_lupdate_courant_dt__mod__cdata) {
      if (AC_ldynamical_diffusion__mod__cdata) {
        diffus_diffrho3__mod__density = diffus_diffrho3__mod__density + AC_diffrho_hyper3_mesh__mod__density
        advec_hypermesh_rho_19_20_21_55=0.
      }
      else {
        advec_hypermesh_rho_19_20_21_55=AC_diffrho_hyper3_mesh__mod__density*pi5_1*sqrt(dxyz_2__mod__cdata)
      }
      advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+(advec_hypermesh_rho_19_20_21_55*advec_hypermesh_rho_19_20_21_55)
    }
  }
  if (lpscalar) {
    if (AC_lpencil__mod__cdata[i_cc-1]) {
      ac_transformed_pencil_cc=1.0
    }
    if (AC_lpencil__mod__cdata[i_cc1-1]) {
      ac_transformed_pencil_cc1=1.0
    }
    if (AC_lpencil__mod__cdata[i_gcc-1]) {
      ac_transformed_pencil_gcc=0.0
    }
  }
  if (AC_lpencil__mod__cdata[i_cv-1]) {
    ac_transformed_pencil_cv=0.0
  }
  if (AC_lpencil__mod__cdata[i_cp-1]) {
    ac_transformed_pencil_cp=0.0
  }
  if (AC_lpencil__mod__cdata[i_cv1-1]) {
    ac_transformed_pencil_cv1=0.0
  }
  if (AC_lpencil__mod__cdata[i_cp1-1]) {
    ac_transformed_pencil_cp1=0.0
  }
  if (AC_lpencil__mod__cdata[i_cs2-1]) {
    ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
  }
  if (AC_lpencil__mod__cdata[i_gtt-1]) {
    ac_transformed_pencil_gtt.x = 0.0
    ac_transformed_pencil_gtt.y = 0.0
    ac_transformed_pencil_gtt.z = 0.0
  }
  if (AC_lpencil__mod__cdata[i_mu1-1]) {
    ac_transformed_pencil_mu1=0.0
  }
  if (AC_lpencil__mod__cdata[i_glnmu-1]) {
    ac_transformed_pencil_glnmu.x = 1.
    ac_transformed_pencil_glnmu.y = 1.
    ac_transformed_pencil_glnmu.z = 1.
  }
  if (AC_lpencil__mod__cdata[i_tt-1]) {
    ac_transformed_pencil_tt=AC_tt__mod__equationofstate
  }
  if (AC_lpencil__mod__cdata[i_tt1-1]) {
    ac_transformed_pencil_tt1=1.0/AC_tt__mod__equationofstate
  }
  if (lshock) {
    if (AC_lpencil__mod__cdata[i_shock-1]) {
      ac_transformed_pencil_shock=0.
    }
    if (AC_lpencil__mod__cdata[i_gshock-1]) {
      ac_transformed_pencil_gshock.x = 0.
      ac_transformed_pencil_gshock.y = 0.
      ac_transformed_pencil_gshock.z = 0.
    }
  }
  if (AC_lpencil__mod__cdata[i_ma2-1]) {
    ac_transformed_pencil_ma2=ac_transformed_pencil_u2/ac_transformed_pencil_cs2
  }
  if (AC_lpencil__mod__cdata[i_fpres-1]) {
    if (AC_lstratz__mod__cdata) {
      ac_transformed_pencil_fpres = -ac_transformed_pencil_cs2 * ac_transformed_pencil_glnrhos
    }
    else {
      if (false) {
        lorentz_gamma_inv2_28_55 = 1. - ac_transformed_pencil_u2
      }
      if (AC_llocal_iso__mod__cdata) {
        ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.x+ac_transformed_pencil_glntt.x)
      }
      else {
        if (ldensity && false) {
          if (!false) {
            ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2/(1 + ac_transformed_pencil_cs2)*ac_transformed_pencil_glnrho.x*lorentz_gamma_inv2_28_55
          }
        }
        else if (false)   {
          ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2*ac_transformed_pencil_grho.x
        }
        else {
          ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.x
        }
      }
      if (ldensity) {
        if (AC_lffree__mod__density) {
          ac_transformed_pencil_fpres.x=ac_transformed_pencil_fpres.x*ac_real_unused_scalar*ac_unused_real_array_1d(AC_m__mod__cdata)*ac_unused_real_array_1d(AC_n__mod__cdata)
        }
      }
      if (AC_llocal_iso__mod__cdata) {
        ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.y+ac_transformed_pencil_glntt.y)
      }
      else {
        if (ldensity && false) {
          if (!false) {
            ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2/(1 + ac_transformed_pencil_cs2)*ac_transformed_pencil_glnrho.y*lorentz_gamma_inv2_28_55
          }
        }
        else if (false)   {
          ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2*ac_transformed_pencil_grho.y
        }
        else {
          ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.y
        }
      }
      if (ldensity) {
        if (AC_lffree__mod__density) {
          ac_transformed_pencil_fpres.y=ac_transformed_pencil_fpres.y*ac_real_unused_scalar*ac_unused_real_array_1d(AC_m__mod__cdata)*ac_unused_real_array_1d(AC_n__mod__cdata)
        }
      }
      if (AC_llocal_iso__mod__cdata) {
        ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.z+ac_transformed_pencil_glntt.z)
      }
      else {
        if (ldensity && false) {
          if (!false) {
            ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2/(1 + ac_transformed_pencil_cs2)*ac_transformed_pencil_glnrho.z*lorentz_gamma_inv2_28_55
          }
        }
        else if (false)   {
          ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2*ac_transformed_pencil_grho.z
        }
        else {
          ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.z
        }
      }
      if (ldensity) {
        if (AC_lffree__mod__density) {
          ac_transformed_pencil_fpres.z=ac_transformed_pencil_fpres.z*ac_real_unused_scalar*ac_unused_real_array_1d(AC_m__mod__cdata)*ac_unused_real_array_1d(AC_n__mod__cdata)
        }
      }
    }
  }
  if (AC_lpencil__mod__cdata[i_tcond-1]) {
    ac_transformed_pencil_tcond=0.
  }
  if (AC_lpencil__mod__cdata[i_sglntt-1]) {
    ac_transformed_pencil_sglntt.x = 0.
    ac_transformed_pencil_sglntt.y = 0.
    ac_transformed_pencil_sglntt.z = 0.
  }
  if (AC_lupdate_courant_dt__mod__cdata) {
    if (leos && ldensity && lhydro) {
      ac_transformed_pencil_advec_cs2=ac_transformed_pencil_cs2*dxyz_2__mod__cdata
      if (AC_lmultithread__mod__cdata) {
        advec_cs2__mod__cdata = ac_transformed_pencil_advec_cs2
      }
    }
  }
  if (lviscosity) {
    if (AC_lpencil__mod__cdata[i_fvisc-1]) {
      ac_transformed_pencil_fvisc.x = 0.0
      ac_transformed_pencil_fvisc.y = 0.0
      ac_transformed_pencil_fvisc.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_visc_heat-1]) {
      ac_transformed_pencil_visc_heat=0.0
    }
    if (AC_lpencil__mod__cdata[i_nu-1]) {
      ac_transformed_pencil_nu=0.0
    }
  }
  if (lmagnetic) {
    if (AC_lpencil__mod__cdata[i_aa-1]) {
      ac_transformed_pencil_aa.x = 0.0
      ac_transformed_pencil_aa.y = 0.0
      ac_transformed_pencil_aa.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_bb-1]) {
      ac_transformed_pencil_bb.x = 0.0
      ac_transformed_pencil_bb.y = 0.0
      ac_transformed_pencil_bb.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_bbb-1]) {
      ac_transformed_pencil_bbb.x = 0.0
      ac_transformed_pencil_bbb.y = 0.0
      ac_transformed_pencil_bbb.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_bunit-1]) {
      ac_transformed_pencil_bunit.x = 0.0
      ac_transformed_pencil_bunit.y = 0.0
      ac_transformed_pencil_bunit.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_b2-1]) {
      ac_transformed_pencil_b2=0.0
    }
    if (AC_lpencil__mod__cdata[i_jxbr-1]) {
      ac_transformed_pencil_jxbr.x = 0.0
      ac_transformed_pencil_jxbr.y = 0.0
      ac_transformed_pencil_jxbr.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_bij-1]) {
      ac_transformed_pencil_bij=0.0
    }
    if (AC_lpencil__mod__cdata[i_uxb-1]) {
      ac_transformed_pencil_uxb.x = 0.0
      ac_transformed_pencil_uxb.y = 0.0
      ac_transformed_pencil_uxb.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_jj-1]) {
      ac_transformed_pencil_jj.x = 0.0
      ac_transformed_pencil_jj.y = 0.0
      ac_transformed_pencil_jj.z = 0.0
    }
    if (AC_lpencil__mod__cdata[i_j2-1]) {
      ac_transformed_pencil_j2=0.0
    }
    if (AC_lpencil__mod__cdata[i_va2-1]) {
      ac_transformed_pencil_va2=0.0
    }
  }
  if (lgrav) {
    if (AC_lpencil__mod__cdata[i_gg-1]) {
      ac_transformed_pencil_gg.x = 0.
      ac_transformed_pencil_gg.y = 0.
      ac_transformed_pencil_gg.z = 0.
    }
  }
  if (ldustvelocity) {
    for k_38_55 in 1:ndustspec+1 {
      if (AC_lpencil__mod__cdata[i_uud-1]) {
        ac_transformed_pencil_uud[k_38_55-1]=value(F_DUST_VELOCITY[k_38_55-1])
      }
      if (AC_lpencil__mod__cdata[i_ud2-1]) {
        ac_transformed_pencil_ud2[k_38_55 -1] = dot(ac_transformed_pencil_uud[k_38_55-1],ac_transformed_pencil_uud[k_38_55-1])
      }
      if (AC_lpencil__mod__cdata[i_udij-1]) {
        ac_transformed_pencil_udij[k_38_55 -1] = gradient_tensor((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)})
      }
      if (AC_lpencil__mod__cdata[i_divud-1]) {
        ac_transformed_pencil_divud[k_38_55 -1] = ac_transformed_pencil_udij[k_38_55-1][1-1][1-1] + ac_transformed_pencil_udij[k_38_55-1][2-1][2-1] + ac_transformed_pencil_udij[k_38_55-1][3-1][3-1]
      }
      if (AC_lpencil__mod__cdata[i_udgud-1]) {
        if (AC_lspherical_coords__mod__cdata || AC_lcylindrical_coords__mod__cdata) {
          ac_transformed_pencil_udgud[k_38_55-1] = u_dot_grad((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)},ac_transformed_pencil_udij[k_38_55 -1],ac_transformed_pencil_uud[k_38_55-1])
        }
        else {
          ac_transformed_pencil_udgud[k_38_55-1] = ac_transformed_pencil_udij[k_38_55 -1]*ac_transformed_pencil_uud[k_38_55-1]
        }
      }
      if (AC_lpencil__mod__cdata[i_ood-1]) {
        ac_transformed_pencil_ood[k_38_55-1].x=ac_transformed_pencil_udij[k_38_55-1][3-1][2-1]-ac_transformed_pencil_udij[k_38_55-1][2-1][3-1]
        ac_transformed_pencil_ood[k_38_55-1].y=ac_transformed_pencil_udij[k_38_55-1][1-1][3-1]-ac_transformed_pencil_udij[k_38_55-1][3-1][1-1]
        ac_transformed_pencil_ood[k_38_55-1].z=ac_transformed_pencil_udij[k_38_55-1][2-1][1-1]-ac_transformed_pencil_udij[k_38_55-1][1-1][2-1]
      }
      if (AC_lpencil__mod__cdata[i_od2-1]) {
        ac_transformed_pencil_od2[k_38_55 -1] = dot(ac_transformed_pencil_ood[k_38_55-1],ac_transformed_pencil_ood[k_38_55-1])
      }
      if (AC_lpencil__mod__cdata[i_oud-1]) {
        ac_transformed_pencil_oud[k_38_55 -1] = dot(ac_transformed_pencil_ood[k_38_55-1],ac_transformed_pencil_uud[k_38_55-1])
      }
      if (AC_lpencil__mod__cdata[i_sdij-1]) {
        if (AC_lviscd_nud_const__mod__dustvelocity) {
          ac_transformed_pencil_sdij[k_38_55-1][1-1][1-1]=ac_transformed_pencil_udij[k_38_55-1][1-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][1-1]=0.5*(ac_transformed_pencil_udij[k_38_55-1][2-1][1-1]+ac_transformed_pencil_udij[k_38_55-1][1-1][2-1])
          ac_transformed_pencil_sdij[k_38_55-1][1-1][2-1]=ac_transformed_pencil_sdij[k_38_55-1][2-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][1-1]=0.5*(ac_transformed_pencil_udij[k_38_55-1][3-1][1-1]+ac_transformed_pencil_udij[k_38_55-1][1-1][3-1])
          ac_transformed_pencil_sdij[k_38_55-1][1-1][3-1]=ac_transformed_pencil_sdij[k_38_55-1][3-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][1-1][1-1]=ac_transformed_pencil_sdij[k_38_55-1][1-1][1-1]-(1/3.0)*ac_transformed_pencil_divud[k_38_55 -1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][2-1]=ac_transformed_pencil_udij[k_38_55-1][2-1][2-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][2-1]=0.5*(ac_transformed_pencil_udij[k_38_55-1][3-1][2-1]+ac_transformed_pencil_udij[k_38_55-1][2-1][3-1])
          ac_transformed_pencil_sdij[k_38_55-1][2-1][3-1]=ac_transformed_pencil_sdij[k_38_55-1][3-1][2-1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][2-1]=ac_transformed_pencil_sdij[k_38_55-1][2-1][2-1]-(1/3.0)*ac_transformed_pencil_divud[k_38_55 -1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][3-1]=ac_transformed_pencil_udij[k_38_55-1][3-1][3-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][3-1]=ac_transformed_pencil_sdij[k_38_55-1][3-1][3-1]-(1/3.0)*ac_transformed_pencil_divud[k_38_55 -1]
        }
        else if (AC_lviscd_hyper3_nud_const__mod__dustvelocity) {
          tmp_pencil_3x3_38_55 = gradient5((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)})
          ac_transformed_pencil_sdij[k_38_55-1][1-1][1-1]=tmp_pencil_3x3_38_55[1-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][1-1][2-1]=tmp_pencil_3x3_38_55[1-1][2-1]
          ac_transformed_pencil_sdij[k_38_55-1][1-1][3-1]=tmp_pencil_3x3_38_55[1-1][3-1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][1-1]=tmp_pencil_3x3_38_55[2-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][2-1]=tmp_pencil_3x3_38_55[2-1][2-1]
          ac_transformed_pencil_sdij[k_38_55-1][2-1][3-1]=tmp_pencil_3x3_38_55[2-1][3-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][1-1]=tmp_pencil_3x3_38_55[3-1][1-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][2-1]=tmp_pencil_3x3_38_55[3-1][2-1]
          ac_transformed_pencil_sdij[k_38_55-1][3-1][3-1]=tmp_pencil_3x3_38_55[3-1][3-1]
        }
      }
      if (AC_lpencil__mod__cdata[i_del2ud-1]) {
        ac_transformed_pencil_del2ud[k_38_55-1] = laplace((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)})
      }
      if (AC_lpencil__mod__cdata[i_del6ud-1]) {
        ac_transformed_pencil_del6ud[k_38_55-1] = del6((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)})
      }
      if (AC_lpencil__mod__cdata[i_graddivud-1]) {
        ac_transformed_pencil_graddivud[k_38_55-1] = gradient_of_divergence((Field3){Field(AC_iuud__mod__cdata[k_38_55-1]-1), Field(AC_iuud__mod__cdata[k_38_55-1]), Field(AC_iuud__mod__cdata[k_38_55-1]+1)})
      }
      if (AC_lupdate_courant_dt__mod__cdata  &&  (ldustdensity || AC_ladvection_dust__mod__dustvelocity)) {
        ac_transformed_pencil_advec_uud[k_38_55 -1]=sum(abs(ac_transformed_pencil_uud[k_38_55-1])*dline_1__mod__cdata)
      }
      if (AC_lviscd_hyper3_polar__mod__dustvelocity  ||  AC_lviscd_hyper3_mesh__mod__dustvelocity) {
        ju_38_55=1+AC_iuud__mod__cdata[k_38_55-1]-1
        grad6_uud__mod__dustvelocity[k_38_55-1][1-1][1-1] = der6x_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][2-1][1-1] = der6y_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][3-1][1-1] = der6z_ignore_spacing(Field(ju_38_55-1))
        ju_38_55=2+AC_iuud__mod__cdata[k_38_55-1]-1
        grad6_uud__mod__dustvelocity[k_38_55-1][1-1][2-1] = der6x_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][2-1][2-1] = der6y_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][3-1][2-1] = der6z_ignore_spacing(Field(ju_38_55-1))
        ju_38_55=3+AC_iuud__mod__cdata[k_38_55-1]-1
        grad6_uud__mod__dustvelocity[k_38_55-1][1-1][3-1] = der6x_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][2-1][3-1] = der6y_ignore_spacing(Field(ju_38_55-1))
        grad6_uud__mod__dustvelocity[k_38_55-1][3-1][3-1] = der6z_ignore_spacing(Field(ju_38_55-1))
      }
    }
  }
  if (ldustdensity) {
    for k_42_55 in 1:ndustspec+1 {
      if (AC_lpencil__mod__cdata[i_nd-1]) {
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_nd[k_42_55 -1]=exp(value(F_DUST_DENSITY[k_42_55-1]))
        }
        else {
          ac_transformed_pencil_nd[k_42_55 -1]=value(F_DUST_DENSITY[k_42_55-1])
        }
      }
      if (AC_lpencil__mod__cdata[i_gnd-1]) {
        if (AC_ldustdensity_log__mod__cdata) {
          tmp_pencil_3_42_55 = gradient(Field(AC_ilnnd__mod__cdata[k_42_55-1]-1))
          ac_transformed_pencil_gnd[k_42_55-1].x=ac_transformed_pencil_nd[k_42_55 -1]*tmp_pencil_3_42_55.x
          ac_transformed_pencil_gnd[k_42_55-1].y=ac_transformed_pencil_nd[k_42_55 -1]*tmp_pencil_3_42_55.y
          ac_transformed_pencil_gnd[k_42_55-1].z=ac_transformed_pencil_nd[k_42_55 -1]*tmp_pencil_3_42_55.z
        }
        else {
          ac_transformed_pencil_gnd[k_42_55-1] = gradient(Field(AC_ind__mod__cdata[k_42_55-1]-1))
        }
      }
      if (AC_lpencil__mod__cdata[i_glnnd-1]) {
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_glnnd[k_42_55-1] = gradient(Field(AC_ilnnd__mod__cdata[k_42_55-1]-1))
        }
        else {
          tmp_pencil_3_42_55 = gradient(Field(AC_ind__mod__cdata[k_42_55-1]-1))
          if (ac_transformed_pencil_nd[k_42_55 -1]!=0.0) {
            ac_transformed_pencil_glnnd[k_42_55-1].x=tmp_pencil_3_42_55.x/(ac_transformed_pencil_nd[k_42_55 -1]+1e-2)
          }
          if (ac_transformed_pencil_nd[k_42_55 -1]!=0.0) {
            ac_transformed_pencil_glnnd[k_42_55-1].y=tmp_pencil_3_42_55.y/(ac_transformed_pencil_nd[k_42_55 -1]+1e-2)
          }
          if (ac_transformed_pencil_nd[k_42_55 -1]!=0.0) {
            ac_transformed_pencil_glnnd[k_42_55-1].z=tmp_pencil_3_42_55.z/(ac_transformed_pencil_nd[k_42_55 -1]+1e-2)
          }
        }
      }
      if (AC_lpencil__mod__cdata[i_glnnd2-1]) {
        ac_transformed_pencil_glnnd2[k_42_55 -1] = dot(ac_transformed_pencil_glnnd[k_42_55-1],ac_transformed_pencil_glnnd[k_42_55-1])
      }
      if (AC_lpencil__mod__cdata[i_udgnd-1]) {
        tmp_42_55 = u_dot_grad_alt(Field(AC_ind__mod__cdata[k_42_55-1]),ac_transformed_pencil_gnd[k_42_55-1],ac_transformed_pencil_uud[k_42_55-1],AC_iadvec_ddensity__mod__dustdensity)
        ac_transformed_pencil_udgnd[k_42_55 -1]=tmp_42_55
      }
      if (AC_lpencil__mod__cdata[i_udglnnd-1]) {
        tmp_42_55 = u_dot_grad_alt(Field(AC_ind__mod__cdata[k_42_55-1]),ac_transformed_pencil_glnnd[k_42_55-1],ac_transformed_pencil_uud[k_42_55-1],AC_iadvec_ddensity__mod__dustdensity)
        ac_transformed_pencil_udglnnd[k_42_55 -1]=tmp_42_55
      }
      if (AC_lpencil__mod__cdata[i_md-1]) {
        if (AC_lmdvar__mod__cdata)  {
          ac_transformed_pencil_md[k_42_55 -1]=value(F_DUST_MASS[k_42_55-1])
          ac_transformed_pencil_ad[k_42_55 -1]=impossible
        }
        else {
          ac_transformed_pencil_md[k_42_55 -1]=AC_md__mod__dustvelocity[k_42_55-1]
          ac_transformed_pencil_ad[k_42_55 -1]=AC_ad__mod__dustvelocity[k_42_55-1]
        }
      }
      if (AC_lpencil__mod__cdata[i_rhod-1]) {
        ac_transformed_pencil_rhod[k_42_55 -1]=ac_transformed_pencil_nd[k_42_55 -1]*ac_transformed_pencil_md[k_42_55 -1]
      }
      if (AC_lpencil__mod__cdata[i_epsd-1]) {
        ac_transformed_pencil_epsd[k_42_55 -1]=ac_transformed_pencil_rhod[k_42_55 -1]*ac_transformed_pencil_rho1
      }
      if (AC_lpencil__mod__cdata[i_grhod-1]) {
        ac_transformed_pencil_grhod[k_42_55-1].x=ac_transformed_pencil_gnd[k_42_55-1].x*ac_transformed_pencil_md[k_42_55 -1]
        ac_transformed_pencil_grhod[k_42_55-1].y=ac_transformed_pencil_gnd[k_42_55-1].y*ac_transformed_pencil_md[k_42_55 -1]
        ac_transformed_pencil_grhod[k_42_55-1].z=ac_transformed_pencil_gnd[k_42_55-1].z*ac_transformed_pencil_md[k_42_55 -1]
      }
      if (AC_lpencil__mod__cdata[i_glnrhod-1]) {
        ac_transformed_pencil_glnrhod[k_42_55-1].x=ac_transformed_pencil_glnnd[k_42_55-1].x
        ac_transformed_pencil_glnrhod[k_42_55-1].y=ac_transformed_pencil_glnnd[k_42_55-1].y
        ac_transformed_pencil_glnrhod[k_42_55-1].z=ac_transformed_pencil_glnnd[k_42_55-1].z
      }
      if (AC_lpencil__mod__cdata[i_mi-1]) {
        if (AC_lmice__mod__dustdensity) {
          ac_transformed_pencil_mi[k_42_55 -1]=value(F_DUST_ICE_MASS[k_42_55-1])
        }
        else {
          ac_transformed_pencil_mi[k_42_55 -1]=0.
        }
      }
      if (AC_lpencil__mod__cdata[i_gmd-1]) {
        if (AC_lmdvar__mod__cdata) {
          ac_transformed_pencil_gmd[k_42_55-1] = gradient(Field(AC_imd__mod__cdata[k_42_55-1]-1))
        }
        else {
          ac_transformed_pencil_gmd[k_42_55-1]=0.
        }
      }
      if (AC_lpencil__mod__cdata[i_gmi-1]) {
        if (AC_lmice__mod__dustdensity) {
          ac_transformed_pencil_gmi[k_42_55-1] = gradient(Field(AC_imi__mod__cdata[k_42_55-1]-1))
        }
        else {
          ac_transformed_pencil_gmi[k_42_55-1]=0.
        }
      }
      if (AC_lpencil__mod__cdata[i_udgmd-1]) {
        tmp_42_55 = u_dot_grad_alt(Field(AC_ind__mod__cdata[k_42_55-1]),ac_transformed_pencil_gmd[k_42_55-1],ac_transformed_pencil_uud[k_42_55-1],AC_iadvec_ddensity__mod__dustdensity)
        ac_transformed_pencil_udgmd[k_42_55 -1]=tmp_42_55
      }
      if (AC_lpencil__mod__cdata[i_udgmi-1]) {
        tmp_42_55 = u_dot_grad_alt(Field(AC_ind__mod__cdata[k_42_55-1]),ac_transformed_pencil_gmi[k_42_55-1],ac_transformed_pencil_uud[k_42_55-1],AC_iadvec_ddensity__mod__dustdensity)
        ac_transformed_pencil_udgmi[k_42_55 -1]=tmp_42_55
      }
      if (AC_lpencil__mod__cdata[i_sdglnnd-1]) {
        ac_transformed_pencil_sdglnnd[k_42_55-1] = ac_transformed_pencil_sdij[k_42_55 -1]*ac_transformed_pencil_glnnd[k_42_55-1]
      }
      if (AC_lpencil__mod__cdata[i_del2nd-1]) {
        ac_transformed_pencil_del2nd[k_42_55 -1] = laplace(Field(AC_ind__mod__cdata[k_42_55-1]-1))
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_del2nd[k_42_55 -1]=ac_transformed_pencil_del2nd[k_42_55 -1]+ac_transformed_pencil_glnnd2[k_42_55 -1]
        }
      }
      if (AC_lpencil__mod__cdata[i_del6nd-1]) {
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_del6nd[k_42_55 -1] = del6_exp(Field(AC_ilnnd__mod__cdata[k_42_55-1]-1))
        }
        else {
          ac_transformed_pencil_del6nd[k_42_55 -1] = del6(Field(AC_ind__mod__cdata[k_42_55-1]-1))
        }
      }
      if (AC_lpencil__mod__cdata[i_del6lnnd-1]) {
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_del6lnnd[k_42_55 -1] = del6(Field(AC_ind__mod__cdata[k_42_55-1]-1))
        }
        else {
        }
      }
      if (AC_lpencil__mod__cdata[i_del2rhod-1]) {
        ac_transformed_pencil_del2rhod[k_42_55 -1]=ac_transformed_pencil_md[k_42_55 -1]*ac_transformed_pencil_del2nd[k_42_55 -1]
        if (AC_ldustdensity_log__mod__cdata) {
          ac_transformed_pencil_del2rhod[k_42_55 -1]=ac_transformed_pencil_del2rhod[k_42_55 -1]*ac_transformed_pencil_nd[k_42_55 -1]
        }
      }
      if (AC_lpencil__mod__cdata[i_del2md-1]) {
        if (AC_lmdvar__mod__cdata) {
          ac_transformed_pencil_del2md[k_42_55 -1] = laplace(Field(AC_imd__mod__cdata[k_42_55-1]-1))
        }
        else {
          ac_transformed_pencil_del2md[k_42_55 -1]=0.
        }
      }
      if (AC_lpencil__mod__cdata[i_del2mi-1]) {
        if (AC_lmice__mod__dustdensity) {
          ac_transformed_pencil_del2mi[k_42_55 -1] = laplace(Field(AC_imi__mod__cdata[k_42_55-1]-1))
        }
        else {
          ac_transformed_pencil_del2mi[k_42_55 -1]=0.
        }
      }
      if (AC_lpencil__mod__cdata[i_gndglnrho-1]) {
        ac_transformed_pencil_gndglnrho[k_42_55 -1] = dot(ac_transformed_pencil_gnd[k_42_55-1],ac_transformed_pencil_glnrho)
      }
      if (AC_lpencil__mod__cdata[i_glnndglnrho-1]) {
        ac_transformed_pencil_glnndglnrho[k_42_55 -1] = dot(ac_transformed_pencil_glnnd[k_42_55-1],ac_transformed_pencil_glnrho)
      }
      if (AC_lpencil__mod__cdata[i_udrop-1]) {
        if (AC_lnoaerosol__mod__dustdensity) {
          ac_transformed_pencil_udrop=0.
        }
        else {
          ac_transformed_pencil_udrop[k_42_55-1]=ac_transformed_pencil_uu
          ac_transformed_pencil_udrop[k_42_55-1].x=ac_transformed_pencil_udrop[k_42_55-1].x-1e6*(AC_dsize__mod__dustdensity[k_42_55-1]*AC_dsize__mod__dustdensity[k_42_55-1])
        }
      }
      if (AC_lpencil__mod__cdata[i_udropgnd-1]) {
        ac_transformed_pencil_udropgnd[k_42_55 -1] = dot(ac_transformed_pencil_udrop[k_42_55-1],ac_transformed_pencil_gnd[k_42_55-1])
      }
    }
    if (AC_lpencil__mod__cdata[i_fcloud-1]) {
      ttt_42_55=ac_transformed_pencil_nd*(AC_dsize__mod__dustdensity*AC_dsize__mod__dustdensity*AC_dsize__mod__dustdensity)
      if (ndustspec>1) {
        ttt_42_55=spline_integral(AC_dsize__mod__dustdensity,ttt_42_55)
      }
      ac_transformed_pencil_fcloud=4.0/3.0*pi*AC_rho_w__mod__dustdensity*ttt_42_55[ndustspec-1]
    }
    if (AC_lpencil__mod__cdata[i_ppsat-1]) {
      t_tmp_42_55=ac_transformed_pencil_tt-273.15
      ac_transformed_pencil_ppsat=(aa0_42_55 + aa1_42_55*t_tmp_42_55    + aa2_42_55*(t_tmp_42_55*t_tmp_42_55)   + aa3_42_55*(t_tmp_42_55*t_tmp_42_55*t_tmp_42_55) + aa4_42_55*(t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55)   + aa5_42_55*(t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55) + aa6_42_55*(t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55*t_tmp_42_55))*1e3
    }
    if (AC_lpencil__mod__cdata[i_ppsf-1]) {
      for k_42_55 in 1:ndustspec+1 {
        if (AC_dsize__mod__dustdensity[k_42_55-1]>0.  &&  AC_dsize__mod__dustdensity[k_42_55-1]!=1.01e-6) {
          if (!AC_ldcore__mod__cdata) {
            t_tmp_42_55 = AC_aa__mod__dustdensity*ac_transformed_pencil_tt1
            ac_transformed_pencil_ppsf[k_42_55 -1]=ac_transformed_pencil_ppsat*exp(t_tmp_42_55/(2.*AC_dsize__mod__dustdensity[k_42_55-1])-2.75e-8*0.1/(2.*(AC_dsize__mod__dustdensity[k_42_55-1]-1.01e-6)))
          }
        }
      }
    }
    if (AC_lpencil__mod__cdata[i_rhodsum-1]) {
      ac_transformed_pencil_rhodsum=sum(ac_transformed_pencil_rhod)
    }
    if (AC_lpencil__mod__cdata[i_rhodsum1-1]) {
      ac_transformed_pencil_rhodsum1=1./(ac_transformed_pencil_rhodsum+tini)
    }
    if (AC_lpencil__mod__cdata[i_grhodsum-1]) {
      ac_transformed_pencil_grhodsum.x = sum(ac_transformed_pencil_grhod)
      ac_transformed_pencil_grhodsum.y = sum(ac_transformed_pencil_grhod)
      ac_transformed_pencil_grhodsum.z = sum(ac_transformed_pencil_grhod)
    }
    if (AC_lpencil__mod__cdata[i_glnrhodsum-1]) {
      if (ndustspec==1) {
        ac_transformed_pencil_glnrhodsum=ac_transformed_pencil_glnrhod[1-1]
      }
      else {
        ac_transformed_pencil_glnrhodsum.x=ac_transformed_pencil_rhodsum1*ac_transformed_pencil_grhodsum.x
        ac_transformed_pencil_glnrhodsum.y=ac_transformed_pencil_rhodsum1*ac_transformed_pencil_grhodsum.y
        ac_transformed_pencil_glnrhodsum.z=ac_transformed_pencil_rhodsum1*ac_transformed_pencil_grhodsum.z
      }
    }
    if (AC_ldustcoagulation_simplified__mod__dustdensity) {
      if (AC_ldustcoagulation__mod__dustvelocity) {
        if (!AC_lcalcdkern__mod__dustdensity) {
          if (AC_lpiecewise_constant_kernel__mod__dustdensity) {
            dkern__mod__dustdensity = AC_dkern_cst__mod__dustdensity
          }
          else {
            dkern__mod__dustdensity = AC_dkern_cst__mod__dustdensity
            for i_41_42_55 in 1:ndustspec+1 {
              dkern__mod__dustdensity[i_41_42_55 -1][i_41_42_55 -1] = dkern__mod__dustdensity[i_41_42_55 -1][i_41_42_55 -1]*0.5
            }
          }
        }
        else {
          lgh_41_42_55=l_41_42_55+NGHOST
          for i_41_42_55 in 1:ndustspec+1 {
            for j_41_42_55 in i_41_42_55:ndustspec+1 {
              lgh_40_41_42_55=l_41_42_55+NGHOST
              if (AC_lself_collisions__mod__dustdensity) {
                if (i_41_42_55==j_41_42_55) {
                  if(AC_enum_self_collisions__mod__dustdensity == enum_average_string) {
                    fact_40_41_42_55=0.5*AC_self_collision_factor__mod__dustdensity
                    deltavd_drift2_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[j_41_42_55-1]+ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[j_41_42_55-1]+ac_transformed_pencil_uud[i_41_42_55-1]))
                  }
                  else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_string)   {
                    fact_40_41_42_55=AC_self_collision_factor__mod__dustdensity
                    if (i_41_42_55==1) {
                      deltavd_drift2_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                    }
                    else if (i_41_42_55==ndustspec) {
                      deltavd_drift2_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                    }
                    else {
                      fact_40_41_42_55=0.5*AC_self_collision_factor__mod__dustdensity
                      deltavd_drift2a_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                      deltavd_drift2a_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                      deltavd_drift2_40_41_42_55=deltavd_drift2a_40_41_42_55+deltavd_drift2b_40_41_42_55
                    }
                  }
                  else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_asymmetric_string)   {
                    fact_40_41_42_55=AC_self_collision_factor__mod__dustdensity
                    if (i_41_42_55==ndustspec) {
                      deltavd_drift2_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[i_41_42_55-1-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                    }
                    else {
                      deltavd_drift2_40_41_42_55 = dot(fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]),fact_40_41_42_55*(ac_transformed_pencil_uud[1+i_41_42_55-1]-ac_transformed_pencil_uud[i_41_42_55-1]))
                    }
                  }
                  else {
                  }
                }
                else {
                  deltavd_drift2_40_41_42_55 = dot(ac_transformed_pencil_uud[i_41_42_55-1]-ac_transformed_pencil_uud[j_41_42_55-1],ac_transformed_pencil_uud[i_41_42_55-1]-ac_transformed_pencil_uud[j_41_42_55-1])
                }
              }
              else {
                deltavd_drift2_40_41_42_55 = dot(ac_transformed_pencil_uud[i_41_42_55-1]-ac_transformed_pencil_uud[j_41_42_55-1],ac_transformed_pencil_uud[i_41_42_55-1]-ac_transformed_pencil_uud[j_41_42_55-1])
              }
              if (AC_ldeltavd_thermal__mod__dustdensity) {
                deltavd_therm_40_41_42_55 = sqrt( 8*AC_k_b__mod__cdata/(pi*ac_transformed_pencil_tt1)*(ac_transformed_pencil_md[i_41_42_55-1]+ac_transformed_pencil_md[j_41_42_55-1])/(ac_transformed_pencil_md[i_41_42_55-1]*ac_transformed_pencil_md[j_41_42_55-1]*AC_unit_md__mod__dustvelocity) )
              }
              else {
                deltavd_therm_40_41_42_55=0.
              }
              if (AC_ldeltavd_turbulent__mod__dustdensity) {
              }
              else if(AC_ldeltavd_turbulent_ormel__mod__dustdensity) {
                t_dyn_39_40_41_42_55 = sqrt(3.*pi/(32.*g_newton_cgs*ac_transformed_pencil_rho))
                nh_39_40_41_42_55 = ac_transformed_pencil_rho/(mu_gas_39_40_41_42_55*mh_39_40_41_42_55)
                cs_39_40_41_42_55 = sqrt(AC_gamma__mod__dustdensity*AC_k_b__mod__cdata*ac_transformed_pencil_tt/(mu_gas_39_40_41_42_55*mh_39_40_41_42_55))
                re_39_40_41_42_55 = 62e6*sqrt(nh_39_40_41_42_55/1e5)*sqrt(ac_transformed_pencil_tt/10.)
                t_eta_39_40_41_42_55 = t_dyn_39_40_41_42_55/sqrt(re_39_40_41_42_55)
                ts_i_39_40_41_42_55 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[i_41_42_55-1]/(ac_transformed_pencil_rho*cs_39_40_41_42_55)
                ts_j_39_40_41_42_55 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[j_41_42_55-1]/(ac_transformed_pencil_rho*cs_39_40_41_42_55)
                ts_1_39_40_41_42_55 = ts_i_39_40_41_42_55
                st_1_39_40_41_42_55    = ts_i_39_40_41_42_55/t_dyn_39_40_41_42_55
                st_2_39_40_41_42_55    = ts_j_39_40_41_42_55/t_dyn_39_40_41_42_55
                if (j_41_42_55 > i_41_42_55) {
                  ts_1_39_40_41_42_55    = ts_j_39_40_41_42_55
                  st_1_39_40_41_42_55    = ts_j_39_40_41_42_55/t_dyn_39_40_41_42_55
                  st_2_39_40_41_42_55    = ts_i_39_40_41_42_55/t_dyn_39_40_41_42_55
                }
                x_st_39_40_41_42_55    = st_2_39_40_41_42_55/st_1_39_40_41_42_55
                beta_st_39_40_41_42_55 = 3.2 - (1. + x_st_39_40_41_42_55) + 2./(1. + x_st_39_40_41_42_55) * (1./2.6 +( x_st_39_40_41_42_55* x_st_39_40_41_42_55* x_st_39_40_41_42_55)/(1.6 + x_st_39_40_41_42_55))
                if (ts_1_39_40_41_42_55 < t_eta_39_40_41_42_55) {
                  if (abs(st_1_39_40_41_42_55 - st_2_39_40_41_42_55) < epsilon(st_1_39_40_41_42_55)) {
                    res_39_40_41_42_55 = 0.
                  }
                  else {
                    res_39_40_41_42_55 = AC_alpha_turb__mod__dustdensity *( cs_39_40_41_42_55* cs_39_40_41_42_55) * (st_1_39_40_41_42_55 - st_2_39_40_41_42_55)/(st_1_39_40_41_42_55 + st_2_39_40_41_42_55) * ((st_1_39_40_41_42_55*st_1_39_40_41_42_55)/(st_1_39_40_41_42_55 + 1./sqrt(re_39_40_41_42_55)) +( st_2_39_40_41_42_55* st_2_39_40_41_42_55)/(st_2_39_40_41_42_55 + 1./sqrt(re_39_40_41_42_55)))
                  }
                }
                else if ( (t_eta_39_40_41_42_55 <= ts_1_39_40_41_42_55)  &&  (ts_1_39_40_41_42_55 < t_dyn_39_40_41_42_55) )   {
                  res_39_40_41_42_55 = AC_alpha_turb__mod__dustdensity *( cs_39_40_41_42_55* cs_39_40_41_42_55) * beta_st_39_40_41_42_55 * st_1_39_40_41_42_55
                }
                else {
                  res_39_40_41_42_55 = AC_alpha_turb__mod__dustdensity *( cs_39_40_41_42_55* cs_39_40_41_42_55) * (1./(st_1_39_40_41_42_55 + 1.) + 1./(st_2_39_40_41_42_55 + 1.))
                }
                deltavd_turbu_40_41_42_55 = sqrt(res_39_40_41_42_55)
              }
              else {
                deltavd_turbu_40_41_42_55 = 0.
              }
              deltavd_41_42_55 = sqrt(deltavd_drift2_40_41_42_55+(deltavd_therm_40_41_42_55*deltavd_therm_40_41_42_55)+(deltavd_turbu_40_41_42_55*deltavd_turbu_40_41_42_55)+(AC_deltavd_imposed__mod__dustdensity*AC_deltavd_imposed__mod__dustdensity))
              if (AC_ludstickmax__mod__dustdensity) {
                ust_40_41_42_55 = AC_ustcst__mod__dustvelocity *pow( (AC_ad__mod__dustvelocity[i_41_42_55-1]*AC_ad__mod__dustvelocity[j_41_42_55-1]/(AC_ad__mod__dustvelocity[i_41_42_55-1]+AC_ad__mod__dustvelocity[j_41_42_55-1])),(2/3.)) *pow(  ((ac_transformed_pencil_md[i_41_42_55-1]+ac_transformed_pencil_md[j_41_42_55-1])/(ac_transformed_pencil_md[i_41_42_55-1]*ac_transformed_pencil_md[j_41_42_55-1]*AC_unit_md__mod__dustvelocity)),(1/2.))
                if (deltavd_41_42_55 > ust_40_41_42_55) {
                  deltavd_41_42_55 = 0.
                }
              }
              if (AC_lkernel_mean__mod__dustdensity) {
                dkern__mod__dustdensity[i_41_42_55-1][j_41_42_55-1] = AC_kernel_mean__mod__dustdensity[i_41_42_55-1][j_41_42_55-1]
              }
              else if (AC_lzero_upper_kern__mod__dustdensity  &&  (i_41_42_55 >= ndustspec-1  ||  j_41_42_55 >= ndustspec-1)) {
                dkern__mod__dustdensity[i_41_42_55-1][j_41_42_55-1] = 0.
              }
              else if (AC_lno_deltavd__mod__dustdensity) {
                dkern__mod__dustdensity[i_41_42_55-1][j_41_42_55-1] = AC_scolld__mod__dustvelocity[i_41_42_55-1][j_41_42_55-1]*AC_deltavd_const__mod__dustdensity
              }
              else {
                dkern__mod__dustdensity[i_41_42_55-1][j_41_42_55-1] = AC_scolld__mod__dustvelocity[i_41_42_55-1][j_41_42_55-1]*deltavd_41_42_55
              }
              dkern__mod__dustdensity[j_41_42_55-1][i_41_42_55-1] = dkern__mod__dustdensity[i_41_42_55-1][j_41_42_55-1]
            }
          }
        }
      }
      else if (AC_ldustcoagulation_simplified__mod__dustdensity) {
        mu_air_41_42_55=2.e-4
        rho_air_41_42_55=1.2e-3
        tt_41_42_55=1./ac_transformed_pencil_tt1
        for i_41_42_55 in 1:ndustspec+1 {
          for k_41_42_55 in i_41_42_55:ndustspec+1 {
            rik_41_42_55=AC_dsize__mod__dustdensity[i_41_42_55-1]+AC_dsize__mod__dustdensity[k_41_42_55-1]
            kn_41_42_55=2.*mu_air_41_42_55/rho_air_41_42_55*sqrt(pi*2e-24/(2.8*AC_k_b__mod__cdata*tt_41_42_55))/(rik_41_42_55/2.)
            cor_factor_41_42_55=1.+kn_41_42_55*(1.142+0.558*exp(-0.999/kn_41_42_55))
            d_coeff_41_42_55=AC_k_b__mod__cdata*cor_factor_41_42_55*tt_41_42_55/(6.*pi*mu_air_41_42_55)
            di_41_42_55=d_coeff_41_42_55/AC_dsize__mod__dustdensity[i_41_42_55-1]
            dk_41_42_55=d_coeff_41_42_55/AC_dsize__mod__dustdensity[k_41_42_55-1]
            dik_41_42_55=(di_41_42_55+dk_41_42_55)
            kbc_41_42_55=4*pi*(AC_dsize__mod__dustdensity[i_41_42_55-1]+AC_dsize__mod__dustdensity[k_41_42_55-1])*(di_41_42_55+dk_41_42_55)
            vmean_i_41_42_55=sqrt(8.*AC_k_b__mod__cdata*tt_41_42_55/pi/(4./3*pi*(AC_dsize__mod__dustdensity[i_41_42_55-1]*AC_dsize__mod__dustdensity[i_41_42_55-1]*AC_dsize__mod__dustdensity[i_41_42_55-1])))
            vmean_k_41_42_55=sqrt(8.*AC_k_b__mod__cdata*tt_41_42_55/pi/(4./3*pi*(AC_dsize__mod__dustdensity[k_41_42_55-1]*AC_dsize__mod__dustdensity[k_41_42_55-1]*AC_dsize__mod__dustdensity[k_41_42_55-1])))
            vmean_ik_41_42_55=sqrt((vmean_i_41_42_55*vmean_i_41_42_55)+(vmean_k_41_42_55*vmean_k_41_42_55))
            gamma_i_41_42_55=8.*di_41_42_55/pi/vmean_i_41_42_55
            gamma_k_41_42_55=8.*dk_41_42_55/pi/vmean_k_41_42_55
            omega_i_41_42_55=(((rik_41_42_55+gamma_i_41_42_55)*(rik_41_42_55+gamma_i_41_42_55)*(rik_41_42_55+gamma_i_41_42_55))-pow(((rik_41_42_55*rik_41_42_55)+(gamma_i_41_42_55*gamma_i_41_42_55)),1.5))/(3.*rik_41_42_55*gamma_i_41_42_55)-rik_41_42_55
            omega_k_41_42_55=(((rik_41_42_55+gamma_k_41_42_55)*(rik_41_42_55+gamma_k_41_42_55)*(rik_41_42_55+gamma_k_41_42_55))-pow(((rik_41_42_55*rik_41_42_55)+(gamma_k_41_42_55*gamma_k_41_42_55)),1.5))/(3.*rik_41_42_55*gamma_k_41_42_55)-rik_41_42_55
            sigma_ik_41_42_55=sqrt((omega_i_41_42_55*omega_i_41_42_55)+(omega_k_41_42_55*omega_k_41_42_55))
            dkern__mod__dustdensity[i_41_42_55 -1][k_41_42_55 -1]=kbc_41_42_55/( rik_41_42_55/(rik_41_42_55+sigma_ik_41_42_55) + 4.*dik_41_42_55/(vmean_ik_41_42_55*rik_41_42_55) )
          }
        }
        for i_41_42_55 in 1:ndustspec+1 {
          for k_41_42_55 in 1:i_41_42_55-1+1 {
            dkern__mod__dustdensity[i_41_42_55 -1][k_41_42_55 -1]=dkern__mod__dustdensity[k_41_42_55 -1][i_41_42_55 -1]
          }
        }
      }
      for k_42_55 in 1:ndustspec+1 {
        nd_rho_42_55[k_42_55 -1]=ac_transformed_pencil_nd[k_42_55 -1]*AC_dsize__mod__dustdensity[k_42_55-1]*ac_transformed_pencil_rho
      }
      coags_42_55=0.
      for i_42_55 in 1:ndustspec+1 {
        for k_42_55 in 1:ndustspec+1 {
          coags_42_55[i_42_55 -1]=coags_42_55[i_42_55 -1]+nd_rho_42_55[k_42_55 -1]*dkern__mod__dustdensity[i_42_55 -1][k_42_55 -1]
        }
      }
      for k_42_55 in 1:ndustspec+1 {
        ac_transformed_pencil_nd[k_42_55 -1]=(nd_rho_42_55[k_42_55 -1]-nd_rho_42_55[k_42_55 -1]*coags_42_55[k_42_55 -1]*AC_dt__mod__cdata)/AC_dsize__mod__dustdensity[k_42_55-1]/ac_transformed_pencil_rho
      }
    }
  }
  if (! AC_lsubstepping_in_time__mod__cdata) {
    if (AC_lschur_3d3d1d__mod__density) {
      density_rhs_77_161=ac_transformed_pencil_uglnrho+ac_transformed_pencil_divu
    }
    else {
      if(AC_lcontinuity_gas__mod__density) {
        if (! AC_lweno_transport__mod__cdata  &&  ! AC_lffree__mod__density  &&  ! AC_lreduced_sound_speed__mod__density  &&   AC_enum_ieos_profile__mod__density==enum_nothing_string  &&  ! AC_lfargo_advection__mod__cdata) {
          if (false) {
            if (false) {
              density_rhs_60_77_161=impossible
              density_rhs_60_77_161=-density_rhs_60_77_161
            }
            else {
              density_rhs_60_77_161=-ac_transformed_pencil_divss
            }
            if (false) {
              density_rhs_60_77_161=density_rhs_60_77_161 + ac_transformed_pencil_ext_force[1-1]
            }
          }
          else {
            if (false) {
              prefactor_60_77_161=1./(1-AC_cs20__mod__equationofstate*ac_transformed_pencil_u2)
              lorentz_gamma_inv2_60_77_161=1.-ac_transformed_pencil_u2
              prefactor2_60_77_161=1.+ac_transformed_pencil_u2
            }
            if (AC_ldensity_nolog__mod__cdata) {
              density_rhs_60_77_161=-ac_transformed_pencil_rho*ac_transformed_pencil_divu
              if (AC_ladvection_density__mod__density) {
                density_rhs_60_77_161 = density_rhs_60_77_161 - AC_cs20_corr__mod__density*ac_transformed_pencil_ugrho
              }
            }
            else {
              density_rhs_60_77_161= - ac_transformed_pencil_divu
              if (AC_ladvection_density__mod__density) {
                density_rhs_60_77_161 = density_rhs_60_77_161 - AC_cs20_corr__mod__density*ac_transformed_pencil_uglnrho
              }
            }
            if (false) {
              u_dot_ext_force_60_77_161 = ac_transformed_pencil_uu.x*ac_transformed_pencil_ext_force[2-1]
              u_dot_ext_force_60_77_161 = ac_transformed_pencil_uu.y*ac_transformed_pencil_ext_force[3-1]
              u_dot_ext_force_60_77_161 = ac_transformed_pencil_uu.z*ac_transformed_pencil_ext_force[4-1]
            }
            if (AC_lrelativistic_eos__mod__density) {
              if (lhydro) {
                if (AC_lrelativistic_eos_term1__mod__density  &&  AC_lrelativistic_eos_term2__mod__density) {
                  density_hydro_rhs_60_77_161=density_rhs_60_77_161
                }
                else {
                  density_hydro_rhs_60_77_161=0.
                  if (AC_ldensity_nolog__mod__cdata) {
                    if (AC_lrelativistic_eos_term1__mod__density) {
                      density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161-ac_transformed_pencil_rho*ac_transformed_pencil_divu
                    }
                    if (AC_lrelativistic_eos_term2__mod__density) {
                      density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161-AC_cs20_corr__mod__density*ac_transformed_pencil_ugrho
                    }
                  }
                  else {
                    if (AC_lrelativistic_eos_term1__mod__density) {
                      density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161-ac_transformed_pencil_divu
                    }
                    if (AC_lrelativistic_eos_term2__mod__density) {
                      density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161-AC_cs20_corr__mod__density*ac_transformed_pencil_uglnrho
                    }
                  }
                }
                if (AC_ldensity_nolog__mod__cdata) {
                  density_hydro_rhs_60_77_161=AC_cs20__mod__equationofstate*density_hydro_rhs_60_77_161*ac_transformed_pencil_rho1
                }
                else {
                  density_hydro_rhs_60_77_161=AC_cs20__mod__equationofstate*density_hydro_rhs_60_77_161
                }
                if (false) {
                  density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161 - ac_transformed_pencil_rho1 * (ac_transformed_pencil_ext_force[1-1] -  2*AC_cs20__mod__equationofstate/AC_cs201__mod__density * u_dot_ext_force_60_77_161)
                  if (0.0!= 0.) {
                    density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161 + (3*AC_cs20__mod__equationofstate - 1) * 0.0
                  }
                }
                density_hydro_rhs_60_77_161=density_hydro_rhs_60_77_161*prefactor_60_77_161*lorentz_gamma_inv2_60_77_161
                tmpv_60_77_161.x=ac_transformed_pencil_uu.x*density_hydro_rhs_60_77_161
                tmpv_60_77_161.y=ac_transformed_pencil_uu.y*density_hydro_rhs_60_77_161
                tmpv_60_77_161.z=ac_transformed_pencil_uu.z*density_hydro_rhs_60_77_161
                DF_UVEC=DF_UVEC-tmpv_60_77_161
              }
            }
            density_rhs_60_77_161=AC_cs201__mod__density*density_rhs_60_77_161
            if (AC_lperturbative_reheating__mod__cdata) {
              w_eos_60_77_161=0.5 * (2.-3.*0.0)
              gamma_r_60_77_161=0.0*(1.+0.0)
              int_source_60_77_161 = gamma_r_60_77_161*0.0*pow(0.0,w_eos_60_77_161)
              if (AC_ldensity_nolog__mod__cdata) {
                density_rhs_60_77_161 = density_rhs_60_77_161 + int_source_60_77_161
              }
              else {
                density_rhs_60_77_161 = density_rhs_60_77_161 + ac_transformed_pencil_rho1 * int_source_60_77_161
              }
            }
            if (false) {
              u_dot_ext_force_60_77_161 = ac_transformed_pencil_ext_force[1-1]*prefactor2_60_77_161 - 2*u_dot_ext_force_60_77_161
              if (0.0!= 0.) {
                u_dot_ext_force_60_77_161 = u_dot_ext_force_60_77_161 + (1.0 - 3*AC_cs20__mod__equationofstate)*ac_transformed_pencil_rho*0.0*prefactor2_60_77_161
              }
              if (AC_ldensity_nolog__mod__cdata) {
                density_rhs_60_77_161=density_rhs_60_77_161 + u_dot_ext_force_60_77_161
              }
              else {
                density_rhs_60_77_161=density_rhs_60_77_161 + ac_transformed_pencil_rho1 * u_dot_ext_force_60_77_161
              }
            }
            density_rhs_60_77_161=density_rhs_60_77_161*prefactor_60_77_161
          }
        }
        else {
          density_rhs_60_77_161=0.
        }
        if (AC_lweno_transport__mod__cdata) {
          density_rhs_60_77_161= density_rhs_60_77_161 - ac_transformed_pencil_transprho
        }
        if (AC_enum_ieos_profile__mod__density==enum_surface_z_string) {
          if (AC_ldensity_nolog__mod__cdata) {
            density_rhs_60_77_161= density_rhs_60_77_161 - AC_profz_eos__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
            if (AC_ldensity_profile_masscons__mod__density) {
              density_rhs_60_77_161 = density_rhs_60_77_161-AC_dprofz_eos__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_rho*ac_transformed_pencil_uu.z
            }
          }
          else {
            density_rhs_60_77_161= density_rhs_60_77_161 - AC_profz_eos__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
            if (AC_ldensity_profile_masscons__mod__density) {
              density_rhs_60_77_161 = density_rhs_60_77_161 -AC_dprofz_eos__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z
            }
          }
        }
        if (AC_lffree__mod__density) {
          if (AC_ldensity_nolog__mod__cdata) {
            density_rhs_60_77_161= density_rhs_60_77_161 - AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
            if (AC_ldensity_profile_masscons__mod__density) {
              density_rhs_60_77_161=density_rhs_60_77_161 - ac_transformed_pencil_rho*( AC_dprofx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]   *ac_transformed_pencil_uu.x  +AC_dprofy_ffree__mod__density[AC_m__mod__cdata-1]*ac_transformed_pencil_uu.y  +AC_dprofz_ffree__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z)
            }
          }
          else {
            density_rhs_60_77_161= density_rhs_60_77_161 - AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*(AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1])*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
            if (AC_ldensity_profile_masscons__mod__density) {
              density_rhs_60_77_161=density_rhs_60_77_161-AC_dprofx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]   *ac_transformed_pencil_uu.x  -AC_dprofy_ffree__mod__density[AC_m__mod__cdata-1]*ac_transformed_pencil_uu.y  -AC_dprofz_ffree__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z
            }
          }
        }
        if (AC_lreduced_sound_speed__mod__density) {
          if (AC_ldensity_nolog__mod__cdata) {
            density_rhs_60_77_161 = density_rhs_60_77_161 - AC_reduce_cs2_profx__mod__density[vertexIdx.x-NGHOST_VAL]*AC_reduce_cs2_profz__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
          }
          else {
            density_rhs_60_77_161 = density_rhs_60_77_161 - AC_reduce_cs2_profx__mod__density[vertexIdx.x-NGHOST_VAL]*AC_reduce_cs2_profz__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
          }
        }
        if (AC_lfargo_advection__mod__cdata) {
          if (AC_ldensity_nolog__mod__cdata) {
            density_rhs_60_77_161 = density_rhs_60_77_161 - ac_transformed_pencil_uuadvec_grho   - ac_transformed_pencil_rho*ac_transformed_pencil_divu
          }
          else {
            density_rhs_60_77_161 = density_rhs_60_77_161 - ac_transformed_pencil_uuadvec_glnrho - ac_transformed_pencil_divu
          }
        }
        DF_LNRHO = DF_LNRHO + density_rhs_60_77_161
      }
      if (AC_lhubble_density__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          DF_RHO = DF_RHO - 3.*hubble__mod__cdata*pow(AC_ascale__mod__cdata,1.5)*ac_transformed_pencil_rho
        }
        else {
          DF_LNRHO = DF_LNRHO - 3.*AC_cs201__mod__density*hubble__mod__cdata*pow(AC_ascale__mod__cdata,AC_nconformal__mod__cdata)
        }
      }
      if (AC_lupdate_mass_source__mod__density) {
        if (true) {
          gamma_64_77_161=5./3.
        }
        gamma1_64_77_161=1./gamma_64_77_161
        if(AC_enum_mass_source_profile__mod__density == enum_exponential_string) {
          dlnrhodt_64_77_161=AC_mass_source_mdot__mod__density
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_bump_string)   {
          dlnrhodt_64_77_161=(AC_mass_source_mdot__mod__density/AC_fnorm__mod__density)*exp(-0.5*((ac_transformed_pencil_r_mn/AC_mass_source_sigma__mod__density)*(ac_transformed_pencil_r_mn/AC_mass_source_sigma__mod__density)))
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_bump2_string)   {
          dlnrhodt_64_77_161=AC_fprofile_z__mod__density[AC_n__mod__cdata-n1+1-1]
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_bumpr_string)   {
          radius2_64_77_161=((AC_x__mod__cdata[vertexIdx.x]-AC_xblob__mod__density[1-1])*(AC_x__mod__cdata[vertexIdx.x]-AC_xblob__mod__density[1-1]))+((AC_y__mod__cdata[AC_m__mod__cdata-1]-AC_yblob__mod__density[1-1])*(AC_y__mod__cdata[AC_m__mod__cdata-1]-AC_yblob__mod__density[1-1]))+((AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_zblob__mod__density[1-1])*(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_zblob__mod__density[1-1]))
          fprofile_64_77_161=(AC_mass_source_mdot__mod__density/AC_fnorm__mod__density)*exp(-0.5*radius2_64_77_161/(AC_mass_source_sigma__mod__density*AC_mass_source_sigma__mod__density))
          if (AC_lmass_source_random__mod__density) {
            fran_64_77_161[1-1] = rand_uniform()
            fran_64_77_161[2-1] = rand_uniform()
            tmp_64_77_161=sqrt(-2*log(fran_64_77_161[1-1]))*sin(2*pi*fran_64_77_161[2-1])
            dlnrhodt_64_77_161=fprofile_64_77_161*cos(AC_mass_source_omega__mod__density*AC_t__mod__cdata)*tmp_64_77_161
          }
          else {
            dlnrhodt_64_77_161=fprofile_64_77_161*cos(AC_mass_source_omega__mod__density*AC_t__mod__cdata)
          }
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_bumpx_string || AC_enum_mass_source_profile__mod__density == enum_sphzstepzdown_string)   {
          dlnrhodt_64_77_161=AC_fprofile_x__mod__density[vertexIdx.x-NGHOST_VAL]
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_const_string)   {
          dlnrhodt_64_77_161=-AC_mass_source_tau1__mod__density*(value(Field(AC_ilnrho__mod__cdata-1))-AC_lnrho0__mod__equationofstate)
        }
        else if(AC_enum_mass_source_profile__mod__density == enum_cylindric_string)   {
          step_vector_return_value_62_64_77_161 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_int__mod__cdata)/(AC_wdamp_rho__mod__density+tini)))
          pdamp_64_77_161=1.-step_vector_return_value_62_64_77_161
          dlnrhodt_64_77_161=-AC_damplnrho_int__mod__density*pdamp_64_77_161*(value(Field(AC_ilnrho__mod__cdata-1))-AC_lnrho_int__mod__density)
          step_vector_return_value_63_64_77_161 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_ext__mod__cdata)/(AC_wdamp_rho__mod__density+tini)))
          pdamp_64_77_161=step_vector_return_value_63_64_77_161
          dlnrhodt_64_77_161=dlnrhodt_64_77_161-AC_damplnrho_ext__mod__density*pdamp_64_77_161*(value(Field(AC_ilnrho__mod__cdata-1))-AC_lnrho_ext__mod__density)
        }
        if (AC_ldensity_nolog__mod__cdata) {
          DF_RHO=DF_RHO+ac_transformed_pencil_rho*dlnrhodt_64_77_161
        }
        else {
          DF_LNRHO=DF_LNRHO+dlnrhodt_64_77_161
        }
        if (lentropy) {
          DF_SS=DF_SS+(gamma1_64_77_161-1.0)*dlnrhodt_64_77_161
        }
      }
      diffus_diffrho__mod__density=0.
      diffus_diffrho3__mod__density=0.
      fdiff_77_161=0.0
      if (AC_ldiff_normal__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*ac_transformed_pencil_del2rho
        }
        else {
          if (AC_ldiffusion_nolog__mod__density) {
            fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho
          }
          else {
            fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*(ac_transformed_pencil_del2lnrho+ac_transformed_pencil_glnrho2)
          }
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho__mod__density
        }
      }
      if (AC_ldiff_kap_tdep__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_kap_tdep__mod__density*ac_transformed_pencil_del2rho
        }
        else {
          if (AC_ldiffusion_nolog__mod__density) {
            fdiff_77_161 = fdiff_77_161 + AC_kap_tdep__mod__density*ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho
          }
          else {
            fdiff_77_161 = fdiff_77_161 + AC_kap_tdep__mod__density*(ac_transformed_pencil_del2lnrho+ac_transformed_pencil_glnrho2)
          }
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_kap_tdep__mod__density
        }
      }
      if (AC_ldiff_cspeed__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*ac_transformed_pencil_del2rho
        }
        else {
          if (AC_ldiffusion_nolog__mod__density) {
            fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho
          }
          else {
            fdiff_77_161 = fdiff_77_161 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*(ac_transformed_pencil_del2lnrho+ac_transformed_pencil_glnrho2)
          }
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho__mod__density
        }
      }
      if (AC_ldiff_shock__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          tmp_66_77_161 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_grho)
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * ac_transformed_pencil_del2rho + tmp_66_77_161)
        }
        else {
          if (AC_ldiffusion_nolog__mod__density) {
            tmp_66_77_161 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_grho)
            fdiff_77_161 = fdiff_77_161 + ac_transformed_pencil_rho1 * AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * ac_transformed_pencil_del2rho + tmp_66_77_161)
          }
          else {
            tmp_66_77_161 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_glnrho)
            fdiff_77_161 = fdiff_77_161 + AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * (ac_transformed_pencil_del2lnrho + ac_transformed_pencil_glnrho2) + tmp_66_77_161)
            if (AC_lanti_shockdiffusion__mod__density) {
              fdiff_77_161 = fdiff_77_161 - AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock*(AC_del2lnrho_glnrho2_init_z__mod__density[AC_n__mod__cdata-1] +  2*(ac_transformed_pencil_glnrho.z-AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1])*AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1]) +  ac_transformed_pencil_gshock.z*AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1] )
            }
          }
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho_shock__mod__density*ac_transformed_pencil_shock
        }
      }
      if (AC_ldensity_slope_limited__mod__density && AC_llast__mod__cdata) {
        if(AC_lsld_every_step__mod__cdata  ||  AC_lrmv__mod__cdata) {
          if (AC_ldensity_nolog__mod__cdata) {
            tmp_65_66_77_161 = get_slope_limited_divergence(Field(AC_irho__mod__cdata-1),SLD_CHAR_SPEED,60.0,AC_h_sld_dens__mod__density,AC_nlf_sld_dens__mod__density,AC_irho__mod__cdata == AC_ilnrho__mod__cdata || AC_irho__mod__cdata == AC_ilntt__mod__cdata)
            fdiff_77_161=fdiff_77_161+tmp_65_66_77_161
          }
          else {
            tmp_65_66_77_161 = get_slope_limited_divergence(Field(AC_ilnrho__mod__cdata-1),SLD_CHAR_SPEED,60.0,AC_h_sld_dens__mod__density,AC_nlf_sld_dens__mod__density,AC_ilnrho__mod__cdata == AC_ilnrho__mod__cdata || AC_ilnrho__mod__cdata == AC_ilntt__mod__cdata)
            fdiff_77_161=fdiff_77_161+tmp_65_66_77_161*ac_transformed_pencil_rho1
          }
        }
      }
      if (AC_lmassdiff_fix__mod__density && !false) {
        if (AC_ldensity_nolog__mod__cdata) {
          tmp_69_77_161 = fdiff_77_161*ac_transformed_pencil_rho1
        }
        else {
          tmp_69_77_161 = fdiff_77_161
        }
        if (lhydro && (!lhydro_potential)) {
          DF_UX = DF_UX - ac_transformed_pencil_uu.x * tmp_69_77_161
          DF_UY = DF_UY - ac_transformed_pencil_uu.y * tmp_69_77_161
          DF_UZ = DF_UZ - ac_transformed_pencil_uu.z * tmp_69_77_161
        }
        if (lentropy && (!AC_pretend_lntt__mod__cdata)) {
          if (AC_lgamma_is_1__mod__density) {
            DF_SS = DF_SS - ac_transformed_pencil_cv*tmp_69_77_161
          }
          else {
            if (true) {
              gamma_69_77_161=5./3.
            }
            DF_SS = DF_SS - gamma_69_77_161*ac_transformed_pencil_cv*tmp_69_77_161
          }
        }
        else if (lentropy && AC_pretend_lntt__mod__cdata) {
          DF_LNTT = DF_LNTT - tmp_69_77_161
        }
        else if (ltemperature && (! AC_ltemperature_nolog__mod__cdata)) {
          DF_LNTT = DF_LNTT - tmp_69_77_161
        }
        else if (ltemperature && AC_ltemperature_nolog__mod__cdata) {
          DF_TT = DF_TT - tmp_69_77_161*ac_transformed_pencil_tt
        }
        else if (lthermal_energy) {
          DF_ETH = DF_ETH + 0.5 * fdiff_77_161 * ac_transformed_pencil_u2
        }
      }
      if (AC_ldiff_hyper3__mod__density || AC_ldiff_hyper3_strict__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_del6rho
        }
        else {
          if (AC_ldiffusion_nolog__mod__density) {
            fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_rho1*ac_transformed_pencil_del6rho
          }
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density
        }
      }
      if (AC_ldiff_hyper3_polar__mod__density) {
        if (AC_ldensity_nolog__mod__cdata) {
          tmp_71_77_161 = der6x_ignore_spacing(Field(AC_irho__mod__cdata-1))
        }
        else {
          tmp_71_77_161 = der6x_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        }
        fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*pi4_1*tmp_71_77_161*(dline_1__mod__cdata.x*dline_1__mod__cdata.x)
        if (AC_ldensity_nolog__mod__cdata) {
          tmp_71_77_161 = der6y_ignore_spacing(Field(AC_irho__mod__cdata-1))
        }
        else {
          tmp_71_77_161 = der6y_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        }
        fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*pi4_1*tmp_71_77_161*(dline_1__mod__cdata.y*dline_1__mod__cdata.y)
        if (AC_ldensity_nolog__mod__cdata) {
          tmp_71_77_161 = der6z_ignore_spacing(Field(AC_irho__mod__cdata-1))
        }
        else {
          tmp_71_77_161 = der6z_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        }
        fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*pi4_1*tmp_71_77_161*(dline_1__mod__cdata.z*dline_1__mod__cdata.z)
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density*pi4_1*(dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata)
        }
      }
      if (AC_ldiff_hyper3_mesh__mod__density) {
        tmp_71_77_161 = der6x_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        if (AC_ldynamical_diffusion__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density * tmp_71_77_161 * dline_1__mod__cdata.x
        }
        else {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density*pi5_1/60.*tmp_71_77_161*dline_1__mod__cdata.x
        }
        tmp_71_77_161 = der6y_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        if (AC_ldynamical_diffusion__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density * tmp_71_77_161 * dline_1__mod__cdata.y
        }
        else {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density*pi5_1/60.*tmp_71_77_161*dline_1__mod__cdata.y
        }
        tmp_71_77_161 = der6z_ignore_spacing(Field(AC_ilnrho__mod__cdata-1))
        if (AC_ldynamical_diffusion__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density * tmp_71_77_161 * dline_1__mod__cdata.z
        }
        else {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3_mesh__mod__density*pi5_1/60.*tmp_71_77_161*dline_1__mod__cdata.z
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          if (AC_ldynamical_diffusion__mod__cdata) {
            diffus_diffrho3__mod__density = diffus_diffrho3__mod__density + AC_diffrho_hyper3_mesh__mod__density
            advec_hypermesh_rho_70_71_77_161=0.
          }
          else {
            advec_hypermesh_rho_70_71_77_161=AC_diffrho_hyper3_mesh__mod__density*pi5_1*sqrt(dxyz_2__mod__cdata)
          }
          advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+(advec_hypermesh_rho_70_71_77_161*advec_hypermesh_rho_70_71_77_161)
        }
      }
      if (AC_ldiff_hyper3_aniso__mod__density) {
        tmp_71_77_161  = del6fj(Field(AC_ilnrho__mod__cdata-1), AC_diffrho_hyper3_aniso__mod__density)
        fdiff_77_161 = fdiff_77_161 + tmp_71_77_161
        if (AC_lsubtract_init_stratification__mod__density) {
          tmp_71_77_161  = del6fj(Field(AC_iglobal_lnrho0__mod__cdata-1), AC_diffrho_hyper3_aniso__mod__density)
          fdiff_77_161 = fdiff_77_161 - tmp_71_77_161
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho3__mod__density=diffus_diffrho3__mod__density +  (AC_diffrho_hyper3_aniso__mod__density.x*(dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x) +  AC_diffrho_hyper3_aniso__mod__density.y*(dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y) +  AC_diffrho_hyper3_aniso__mod__density.z*(dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z))/dxyz_6__mod__cdata
        }
      }
      if (AC_ldiff_hyper3lnrho__mod__density  ||  AC_ldiff_hyper3lnrho_strict__mod__density) {
        if (! AC_ldensity_nolog__mod__cdata) {
          fdiff_77_161 = fdiff_77_161 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_del6lnrho
        }
        if (AC_lupdate_courant_dt__mod__cdata) {
          diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density
        }
      }
      if (AC_ldensity_nolog__mod__cdata) {
        DF_RHO   = DF_RHO   + fdiff_77_161
      }
      else {
        DF_LNRHO = DF_LNRHO + fdiff_77_161
      }
      if (AC_lupdate_courant_dt__mod__cdata) {
        diffus_diffrho__mod__density = diffus_diffrho__mod__density*dxyz_2__mod__cdata
        if (AC_ldynamical_diffusion__mod__cdata  &&  AC_ldiff_hyper3_mesh__mod__density) {
          diffus_diffrho3__mod__density = diffus_diffrho3__mod__density * sum(abs(dline_1__mod__cdata))
        }
        else {
          diffus_diffrho3__mod__density = diffus_diffrho3__mod__density*dxyz_6__mod__cdata
        }
        maxdiffus__mod__cdata=max(maxdiffus__mod__cdata,diffus_diffrho__mod__density)
        maxdiffus3__mod__cdata=max(maxdiffus3__mod__cdata,diffus_diffrho3__mod__density)
      }
      if (lborder_profiles) {
        if(AC_enum_borderlnrho__mod__density == enum_zero_string || AC_enum_borderlnrho__mod__density == enum_0_string) {
          if (AC_ldensity_nolog__mod__cdata) {
            f_target_76_77_161=0.
          }
          else {
            f_target_76_77_161=1.
          }
        }
        else if(AC_enum_borderlnrho__mod__density == enum_constant_string)   {
          if (AC_ldensity_nolog__mod__cdata) {
            f_target_76_77_161=AC_rho_const__mod__density
          }
          else {
            f_target_76_77_161=AC_lnrho_const__mod__density
          }
        }
        else if(AC_enum_borderlnrho__mod__density == enum_initialzcondition_string)   {
        }
      }
    }
    if (lhydro && AC_lpressuregradient_gas__mod__hydro && !(false  &&  ! false)   && (!lhydro_potential)) {
      DF_UVEC=DF_UVEC+ac_transformed_pencil_fpres
      if (ac_real_unused_scalar!= 0.) {
        DF_UX = DF_UX - ac_transformed_pencil_cs2*ac_unused_real_array_1d(1)
        DF_UY = DF_UY - ac_transformed_pencil_cs2*ac_unused_real_array_1d(2)
        DF_UZ = DF_UZ - ac_transformed_pencil_cs2*ac_unused_real_array_1d(3)
      }
    }
    if (AC_lupdate_courant_dt__mod__cdata  &&  leos && ldensity && lhydro) {
      advec_cs2__mod__cdata=max(advec_cs2__mod__cdata,ac_transformed_pencil_advec_cs2)
    }
    if (ldustvelocity) {
      for k_98_161 in 1:ndustspec+1 {
        if(AC_enum_draglaw__mod__dustvelocity == enum_epstein_cst_string) {
          if (lgpu) {
            tausd1__mod__dustvelocity[k_98_161 -1] = 1.0/AC_tausd__mod__dustvelocity[k_98_161-1]
          }
        }
        else if(AC_enum_draglaw__mod__dustvelocity == enum_epstein_cst_b_string)   {
          tausd1__mod__dustvelocity[k_98_161 -1] = AC_betad__mod__dustvelocity[k_98_161-1]/ac_transformed_pencil_rhod[k_98_161 -1]
        }
        else if(AC_enum_draglaw__mod__dustvelocity == enum_stokes_cst_tausd_string)   {
          tausd1__mod__dustvelocity[k_98_161 -1] = AC_betad__mod__dustvelocity[k_98_161-1]
        }
        else if(AC_enum_draglaw__mod__dustvelocity == enum_stokes_varmass_string)   {
          tausd1__mod__dustvelocity[k_98_161 -1] = AC_betad__mod__dustvelocity[k_98_161-1]
          if (AC_lstokes_highspeed_corr__mod__dustvelocity) {
            deltaud2_90_98_161 = dot(ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu,ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu)
            rep_90_98_161=2*AC_ad__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_rho*sqrt(deltaud2_90_98_161)/AC_mu_ext__mod__dustvelocity
            tausd1__mod__dustvelocity[k_98_161 -1] = tausd1__mod__dustvelocity[k_98_161 -1]*(1+0.15*pow(rep_90_98_161,0.687))
          }
        }
        else if(AC_enum_draglaw__mod__dustvelocity == enum_epstein_var_string)   {
          deltaud2_90_98_161 = dot(ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu,ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu)
          if (AC_lpifactor1__mod__dustvelocity) {
            pifactor1_90_98_161=sqrt(8./pi)
          }
          else {
            pifactor1_90_98_161=1.
          }
          if (AC_lpifactor2__mod__dustvelocity) {
            pifactor2_90_98_161=9.*pi/128.
          }
          else {
            pifactor2_90_98_161=1.
          }
          csrho_90_98_161=pifactor1_90_98_161*sqrt(ac_transformed_pencil_cs2+pifactor2_90_98_161*deltaud2_90_98_161)*ac_transformed_pencil_rho
          tausd1__mod__dustvelocity[k_98_161 -1] = csrho_90_98_161*AC_rhodsad1__mod__dustvelocity[k_98_161-1]
        }
        else if(AC_enum_draglaw__mod__dustvelocity == enum_epstein_gaussian_z_string)   {
          tausd1__mod__dustvelocity[k_98_161 -1] = (1/AC_tausd__mod__dustvelocity[k_98_161-1])*exp(-(AC_z__mod__cdata[AC_n__mod__cdata-1]*AC_z__mod__cdata[AC_n__mod__cdata-1])/(2*(AC_scalehtaus__mod__dustvelocity*AC_scalehtaus__mod__dustvelocity)))
          if (AC_z0taus__mod__dustvelocity!=0.0) {
            tausd1__mod__dustvelocity[k_98_161 -1]=tausd1__mod__dustvelocity[k_98_161 -1]/(  0.5*(tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]+AC_z0taus__mod__dustvelocity)/AC_widthtaus__mod__dustvelocity)+tanh((-AC_z__mod__cdata[AC_n__mod__cdata-1]+AC_z0taus__mod__dustvelocity)/AC_widthtaus__mod__dustvelocity)))
          }
        }
        else {
        }
        if (AC_ldustvelocity_shorttausd__mod__dustvelocity) {
          if(tausd1__mod__dustvelocity[k_98_161-1]>=AC_shorttaus1limit__mod__dustvelocity) {
            if (lgrav) {
              aa_sfta_91_98_161=ac_transformed_pencil_gg
              if (AC_lgravx_gas__mod__cdata != AC_lgravx_dust__mod__cdata) {
                if (AC_lgravx_gas__mod__cdata) {
                  aa_sfta_91_98_161.x=aa_sfta_91_98_161.x-ac_transformed_pencil_gg[1-1]
                }
                if (AC_lgravx_dust__mod__cdata) {
                  aa_sfta_91_98_161.x=aa_sfta_91_98_161.x+ac_transformed_pencil_gg[1-1]
                }
              }
              if (AC_lgravz_gas__mod__cdata != AC_lgravz_dust__mod__cdata) {
                if (AC_lgravz_gas__mod__cdata) {
                  aa_sfta_91_98_161.z = aa_sfta_91_98_161.z-ac_transformed_pencil_gg[3-1]
                }
                if (AC_lgravz_dust__mod__cdata) {
                  aa_sfta_91_98_161.z = aa_sfta_91_98_161.z+ac_transformed_pencil_gg[3-1]
                }
              }
            }
            else {
              aa_sfta_91_98_161.x = 0.
              aa_sfta_91_98_161.y = 0.
              aa_sfta_91_98_161.z = 0.
            }
            if (ldensity) {
              aa_sfta_91_98_161=aa_sfta_91_98_161+ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho
            }
            if (lmagnetic) {
              aa_sfta_91_98_161=aa_sfta_91_98_161-ac_transformed_pencil_jxbr
            }
            DF_DUST_VELOCITY[k_98_161-1] = 1/AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]*(value(F_UVEC)-ac_transformed_pencil_uud[k_98_161-1]+aa_sfta_91_98_161/tausd1__mod__dustvelocity[k_98_161-1])
          }
          else {
            if (AC_ladvection_dust__mod__dustvelocity) {
              DF_DUST_VELOCITY[k_98_161-1] =  DF_DUST_VELOCITY[k_98_161-1] - ac_transformed_pencil_udgud[k_98_161-1]
            }
            if (AC_lcoriolisforce_dust__mod__dustvelocity) {
              if (AC_theta__mod__cdata==0) {
                c2_92_98_161=2*AC_omega__mod__cdata
                DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + c2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].y
                DF_DUST_VELOCITY[k_98_161-1].y = DF_DUST_VELOCITY[k_98_161-1].y - c2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].x
              }
              else {
                c2_92_98_161=2*AC_omega__mod__cdata*cos(AC_theta__mod__cdata*pi/180.)
                s2_92_98_161=2*AC_omega__mod__cdata*sin(AC_theta__mod__cdata*pi/180.)
                DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + c2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].y
                DF_DUST_VELOCITY[k_98_161-1].y = DF_DUST_VELOCITY[k_98_161-1].y - c2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].x + s2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].z
                DF_DUST_VELOCITY[k_98_161-1].z = DF_DUST_VELOCITY[k_98_161-1].z                    + s2_92_98_161*ac_transformed_pencil_uud[k_98_161-1].y
              }
            }
            if (AC_ldragforce_dust__mod__dustvelocity) {
              DF_DUST_VELOCITY[k_98_161-1]=DF_DUST_VELOCITY[k_98_161-1] - tausd1__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu)
              if (AC_ldragforce_gas__mod__dustvelocity) {
                tausg1_92_98_161 = ac_transformed_pencil_rhod[k_98_161-1]*tausd1__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_rho1
                if (AC_tausgmin__mod__dustvelocity!=0.0) {
                  tausg1_92_98_161=min(tausg1_92_98_161,AC_tausg1max__mod__dustvelocity)
                }
                DF_UVEC = DF_UVEC - tausg1_92_98_161*(ac_transformed_pencil_uu-ac_transformed_pencil_uud[k_98_161-1])
                if (AC_lupdate_courant_dt__mod__cdata) {
                  dt1_max__mod__cdata=max(dt1_max__mod__cdata,(tausg1_92_98_161+tausd1__mod__dustvelocity[k_98_161-1])/AC_cdtd__mod__dustvelocity)
                }
              }
              else {
                if (AC_lupdate_courant_dt__mod__cdata) {
                  dt1_max__mod__cdata=max(dt1_max__mod__cdata,tausd1__mod__dustvelocity[k_98_161-1]/AC_cdtd__mod__dustvelocity)
                }
              }
            }
            if (AC_gravx_dust__mod__dustvelocity!=0.0) {
              DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + AC_gravx_dust__mod__dustvelocity
            }
            if (AC_beta_dpdr_dust__mod__dustvelocity!=0.0) {
              DF_DUST_VELOCITY[k_98_161-1].x =  DF_DUST_VELOCITY[k_98_161-1].x + ac_transformed_pencil_cs2*AC_beta_dpdr_dust_scaled__mod__dustvelocity
            }
            if (AC_ldust_pressure__mod__dustvelocity) {
              DF_DUST_VELOCITY[k_98_161-1] = DF_DUST_VELOCITY[k_98_161-1] -  AC_dust_pressure_factor__mod__dustvelocity*ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho
            }
            fviscd_92_98_161.x = 0.0
            fviscd_92_98_161.y = 0.0
            fviscd_92_98_161.z = 0.0
            diffus_nud__mod__dustvelocity=0.0
            diffus_nud3__mod__dustvelocity=0.0
            if (AC_lviscd_simplified__mod__dustvelocity) {
              fviscd_92_98_161 = fviscd_92_98_161 + AC_nud__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_del2ud[k_98_161-1]
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud__mod__dustvelocity[k_98_161-1]*dxyz_2__mod__cdata
              }
            }
            if (AC_lviscd_nud_const__mod__dustvelocity) {
              if (ldustdensity) {
                fviscd_92_98_161 = fviscd_92_98_161 + 2*AC_nud__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_sdglnnd[k_98_161-1] +  AC_nud__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del2ud[k_98_161-1]+1/3.0*ac_transformed_pencil_graddivud[k_98_161-1])
              }
              else {
                fviscd_92_98_161 = fviscd_92_98_161 + AC_nud__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del2ud[k_98_161-1]+1/3.*ac_transformed_pencil_graddivud[k_98_161-1])
              }
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud__mod__dustvelocity[k_98_161-1]*dxyz_2__mod__cdata
              }
            }
            if (AC_lviscd_shock__mod__dustvelocity) {
              if (ldustdensity) {
                tmp2_92_98_161 = ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_glnrhod[k_98_161-1]
                tmp_92_98_161 = tmp2_92_98_161 + ac_transformed_pencil_graddivud[k_98_161-1]
              }
              else {
                tmp_92_98_161 = ac_transformed_pencil_graddivud[k_98_161-1]
              }
              tmp2_92_98_161 = AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*tmp_92_98_161
              tmp_92_98_161  = tmp2_92_98_161 + AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_gshock
              fviscd_92_98_161 = fviscd_92_98_161 + tmp_92_98_161
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*dxyz_2__mod__cdata
              }
            }
            if (AC_lviscd_shock_simplified__mod__dustvelocity) {
              tmp_92_98_161 = ac_transformed_pencil_graddivud[k_98_161-1]
              tmp2_92_98_161 = AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*tmp_92_98_161
              tmp_92_98_161  = tmp2_92_98_161 + AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_gshock
              fviscd_92_98_161 = fviscd_92_98_161 + tmp_92_98_161
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*dxyz_2__mod__cdata
              }
            }
            if (AC_lviscd_hyper3_simplified__mod__dustvelocity) {
              fviscd_92_98_161 = fviscd_92_98_161 + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_del6ud[k_98_161-1]
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
              }
            }
            if (AC_lviscd_hyper3_polar__mod__dustvelocity) {
              fviscd_92_98_161.x = fviscd_92_98_161.x + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,1,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
              fviscd_92_98_161.y = fviscd_92_98_161.y + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,2,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
              fviscd_92_98_161.z = fviscd_92_98_161.z + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,3,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*(dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata)
              }
            }
            if (AC_lviscd_hyper3_mesh__mod__dustvelocity) {
              fviscd_92_98_161.x = fviscd_92_98_161.x + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,1,k_98_161)*dline_1__mod__cdata)
              fviscd_92_98_161.y = fviscd_92_98_161.y + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,2,k_98_161)*dline_1__mod__cdata)
              fviscd_92_98_161.z = fviscd_92_98_161.z + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,3,k_98_161)*dline_1__mod__cdata)
              if (AC_lupdate_courant_dt__mod__cdata) {
                advec_hypermesh_uud__mod__dustvelocity=AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1*sqrt(dxyz_2__mod__cdata)
                advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+(advec_hypermesh_uud__mod__dustvelocity*advec_hypermesh_uud__mod__dustvelocity)
              }
            }
            if (AC_lviscd_hyper3_rhod_nud_const__mod__dustvelocity) {
              mudrhod1_92_98_161=(AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*AC_nd0__mod__dustvelocity*AC_md0__mod__dustvelocity)/ac_transformed_pencil_rhod[k_98_161-1]
              fviscd_92_98_161 = fviscd_92_98_161 + mudrhod1_92_98_161*ac_transformed_pencil_del6ud[k_98_161-1]
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
              }
            }
            if (AC_lviscd_hyper3_nud_const__mod__dustvelocity) {
              fviscd_92_98_161 = fviscd_92_98_161 + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del6ud[k_98_161-1]+ac_transformed_pencil_sdglnnd[k_98_161-1])
              if (AC_lupdate_courant_dt__mod__cdata) {
                diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
              }
            }
            DF_DUST_VELOCITY[k_98_161-1] = DF_DUST_VELOCITY[k_98_161-1] + fviscd_92_98_161
            if (AC_lupdate_courant_dt__mod__cdata) {
              maxdiffus3__mod__cdata=max(maxdiffus3__mod__cdata,diffus_nud3__mod__dustvelocity)
              maxdiffus__mod__cdata=max(maxdiffus__mod__cdata,diffus_nud__mod__dustvelocity)
            }
          }
        }
        else {
          if (AC_ladvection_dust__mod__dustvelocity) {
            DF_DUST_VELOCITY[k_98_161-1] =  DF_DUST_VELOCITY[k_98_161-1] - ac_transformed_pencil_udgud[k_98_161-1]
          }
          if (AC_lcoriolisforce_dust__mod__dustvelocity) {
            if (AC_theta__mod__cdata==0) {
              c2_93_98_161=2*AC_omega__mod__cdata
              DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + c2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].y
              DF_DUST_VELOCITY[k_98_161-1].y = DF_DUST_VELOCITY[k_98_161-1].y - c2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].x
            }
            else {
              c2_93_98_161=2*AC_omega__mod__cdata*cos(AC_theta__mod__cdata*pi/180.)
              s2_93_98_161=2*AC_omega__mod__cdata*sin(AC_theta__mod__cdata*pi/180.)
              DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + c2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].y
              DF_DUST_VELOCITY[k_98_161-1].y = DF_DUST_VELOCITY[k_98_161-1].y - c2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].x + s2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].z
              DF_DUST_VELOCITY[k_98_161-1].z = DF_DUST_VELOCITY[k_98_161-1].z                    + s2_93_98_161*ac_transformed_pencil_uud[k_98_161-1].y
            }
          }
          if (AC_ldragforce_dust__mod__dustvelocity) {
            DF_DUST_VELOCITY[k_98_161-1]=DF_DUST_VELOCITY[k_98_161-1] - tausd1__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_uud[k_98_161-1]-ac_transformed_pencil_uu)
            if (AC_ldragforce_gas__mod__dustvelocity) {
              tausg1_93_98_161 = ac_transformed_pencil_rhod[k_98_161-1]*tausd1__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_rho1
              if (AC_tausgmin__mod__dustvelocity!=0.0) {
                tausg1_93_98_161=min(tausg1_93_98_161,AC_tausg1max__mod__dustvelocity)
              }
              DF_UVEC = DF_UVEC - tausg1_93_98_161*(ac_transformed_pencil_uu-ac_transformed_pencil_uud[k_98_161-1])
              if (AC_lupdate_courant_dt__mod__cdata) {
                dt1_max__mod__cdata=max(dt1_max__mod__cdata,(tausg1_93_98_161+tausd1__mod__dustvelocity[k_98_161-1])/AC_cdtd__mod__dustvelocity)
              }
            }
            else {
              if (AC_lupdate_courant_dt__mod__cdata) {
                dt1_max__mod__cdata=max(dt1_max__mod__cdata,tausd1__mod__dustvelocity[k_98_161-1]/AC_cdtd__mod__dustvelocity)
              }
            }
          }
          if (AC_gravx_dust__mod__dustvelocity!=0.0) {
            DF_DUST_VELOCITY[k_98_161-1].x = DF_DUST_VELOCITY[k_98_161-1].x + AC_gravx_dust__mod__dustvelocity
          }
          if (AC_beta_dpdr_dust__mod__dustvelocity!=0.0) {
            DF_DUST_VELOCITY[k_98_161-1].x =  DF_DUST_VELOCITY[k_98_161-1].x + ac_transformed_pencil_cs2*AC_beta_dpdr_dust_scaled__mod__dustvelocity
          }
          if (AC_ldust_pressure__mod__dustvelocity) {
            DF_DUST_VELOCITY[k_98_161-1] = DF_DUST_VELOCITY[k_98_161-1] -  AC_dust_pressure_factor__mod__dustvelocity*ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho
          }
          fviscd_93_98_161.x = 0.0
          fviscd_93_98_161.y = 0.0
          fviscd_93_98_161.z = 0.0
          diffus_nud__mod__dustvelocity=0.0
          diffus_nud3__mod__dustvelocity=0.0
          if (AC_lviscd_simplified__mod__dustvelocity) {
            fviscd_93_98_161 = fviscd_93_98_161 + AC_nud__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_del2ud[k_98_161-1]
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud__mod__dustvelocity[k_98_161-1]*dxyz_2__mod__cdata
            }
          }
          if (AC_lviscd_nud_const__mod__dustvelocity) {
            if (ldustdensity) {
              fviscd_93_98_161 = fviscd_93_98_161 + 2*AC_nud__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_sdglnnd[k_98_161-1] +  AC_nud__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del2ud[k_98_161-1]+1/3.0*ac_transformed_pencil_graddivud[k_98_161-1])
            }
            else {
              fviscd_93_98_161 = fviscd_93_98_161 + AC_nud__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del2ud[k_98_161-1]+1/3.*ac_transformed_pencil_graddivud[k_98_161-1])
            }
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud__mod__dustvelocity[k_98_161-1]*dxyz_2__mod__cdata
            }
          }
          if (AC_lviscd_shock__mod__dustvelocity) {
            if (ldustdensity) {
              tmp2_93_98_161 = ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_glnrhod[k_98_161-1]
              tmp_93_98_161 = tmp2_93_98_161 + ac_transformed_pencil_graddivud[k_98_161-1]
            }
            else {
              tmp_93_98_161 = ac_transformed_pencil_graddivud[k_98_161-1]
            }
            tmp2_93_98_161 = AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*tmp_93_98_161
            tmp_93_98_161  = tmp2_93_98_161 + AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_gshock
            fviscd_93_98_161 = fviscd_93_98_161 + tmp_93_98_161
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*dxyz_2__mod__cdata
            }
          }
          if (AC_lviscd_shock_simplified__mod__dustvelocity) {
            tmp_93_98_161 = ac_transformed_pencil_graddivud[k_98_161-1]
            tmp2_93_98_161 = AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*tmp_93_98_161
            tmp_93_98_161  = tmp2_93_98_161 + AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_divud[k_98_161-1]*ac_transformed_pencil_gshock
            fviscd_93_98_161 = fviscd_93_98_161 + tmp_93_98_161
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud__mod__dustvelocity=diffus_nud__mod__dustvelocity+AC_nud_shock__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_shock*dxyz_2__mod__cdata
            }
          }
          if (AC_lviscd_hyper3_simplified__mod__dustvelocity) {
            fviscd_93_98_161 = fviscd_93_98_161 + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*ac_transformed_pencil_del6ud[k_98_161-1]
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
            }
          }
          if (AC_lviscd_hyper3_polar__mod__dustvelocity) {
            fviscd_93_98_161.x = fviscd_93_98_161.x + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,1,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
            fviscd_93_98_161.y = fviscd_93_98_161.y + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,2,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
            fviscd_93_98_161.z = fviscd_93_98_161.z + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,3,k_98_161)*(dline_1__mod__cdata*dline_1__mod__cdata))
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*pi4_1*(dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata)
            }
          }
          if (AC_lviscd_hyper3_mesh__mod__dustvelocity) {
            fviscd_93_98_161.x = fviscd_93_98_161.x + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,1,k_98_161)*dline_1__mod__cdata)
            fviscd_93_98_161.y = fviscd_93_98_161.y + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,2,k_98_161)*dline_1__mod__cdata)
            fviscd_93_98_161.z = fviscd_93_98_161.z + AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1/60.*sum(get_first_dim_vector(grad6_uud__mod__dustvelocity,3,k_98_161)*dline_1__mod__cdata)
            if (AC_lupdate_courant_dt__mod__cdata) {
              advec_hypermesh_uud__mod__dustvelocity=AC_nud_hyper3_mesh__mod__dustvelocity[k_98_161-1]*pi5_1*sqrt(dxyz_2__mod__cdata)
              advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+(advec_hypermesh_uud__mod__dustvelocity*advec_hypermesh_uud__mod__dustvelocity)
            }
          }
          if (AC_lviscd_hyper3_rhod_nud_const__mod__dustvelocity) {
            mudrhod1_93_98_161=(AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*AC_nd0__mod__dustvelocity*AC_md0__mod__dustvelocity)/ac_transformed_pencil_rhod[k_98_161-1]
            fviscd_93_98_161 = fviscd_93_98_161 + mudrhod1_93_98_161*ac_transformed_pencil_del6ud[k_98_161-1]
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
            }
          }
          if (AC_lviscd_hyper3_nud_const__mod__dustvelocity) {
            fviscd_93_98_161 = fviscd_93_98_161 + AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*(ac_transformed_pencil_del6ud[k_98_161-1]+ac_transformed_pencil_sdglnnd[k_98_161-1])
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_nud3__mod__dustvelocity=diffus_nud3__mod__dustvelocity+AC_nud_hyper3__mod__dustvelocity[k_98_161-1]*dxyz_6__mod__cdata
            }
          }
          DF_DUST_VELOCITY[k_98_161-1] = DF_DUST_VELOCITY[k_98_161-1] + fviscd_93_98_161
          if (AC_lupdate_courant_dt__mod__cdata) {
            maxdiffus3__mod__cdata=max(maxdiffus3__mod__cdata,diffus_nud3__mod__dustvelocity)
            maxdiffus__mod__cdata=max(maxdiffus__mod__cdata,diffus_nud__mod__dustvelocity)
          }
        }
        if (lborder_profiles) {
          if(AC_enum_borderuud__mod__dustvelocity == enum_zero_string || AC_enum_borderuud__mod__dustvelocity == enum_0_string) {
            f_target_97_98_161.x = 0.
            f_target_97_98_161.y = 0.
            f_target_97_98_161.z = 0.
            ju_97_98_161=1+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=2+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=3+AC_iuud__mod__cdata[k_98_161-1]-1
          }
          else if(AC_enum_borderuud__mod__dustvelocity == enum_initialzcondition_string)   {
            ju_97_98_161=1+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=2+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=3+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=1+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=2+AC_iuud__mod__cdata[k_98_161-1]-1
            ju_97_98_161=3+AC_iuud__mod__cdata[k_98_161-1]-1
          }
        }
      }
      if (AC_lupdate_courant_dt__mod__cdata && (ldustdensity || AC_ladvection_dust__mod__dustvelocity)) {
        maxadvec__mod__cdata=maxadvec__mod__cdata+maxval(ac_transformed_pencil_advec_uud)
      }
    }
    if (ldustdensity) {
      if (AC_ldustcontinuity__mod__dustdensity  &&  (! AC_latm_chemistry__mod__dustdensity)) {
        if (AC_ldustdensity_log__mod__cdata) {
          DF_DUST_DENSITY = DF_DUST_DENSITY - ac_transformed_pencil_udglnnd - ac_transformed_pencil_divud
        }
        else {
          DF_DUST_DENSITY = DF_DUST_DENSITY - ac_transformed_pencil_udgnd - ac_transformed_pencil_nd*ac_transformed_pencil_divud
        }
        if (AC_lmdvar__mod__cdata) {
          DF_DUST_MASS = DF_DUST_MASS - ac_transformed_pencil_udgmd
        }
        if (AC_lmice__mod__dustdensity) {
          DF_DUST_ICE_MASS = DF_DUST_ICE_MASS - ac_transformed_pencil_udgmi
        }
      }
      else if (AC_latm_chemistry__mod__dustdensity) {
        DF_DUST_DENSITY = DF_DUST_DENSITY - ac_transformed_pencil_udropgnd
      }
      if (AC_latm_chemistry__mod__dustdensity  ||  AC_lsemi_chemistry__mod__dustdensity) {
        if (AC_lnoaerosol__mod__dustdensity) {
          dndr_145_161=0.
        }
        else {
          imr_145_161=AC_dwater__mod__dustdensity*AC_m_w__mod__dustdensity*ac_transformed_pencil_ppsat/AC_rgas__mod__chemistry/ac_transformed_pencil_tt/AC_rho_w__mod__dustdensity
          if (AC_lsubstep__mod__dustdensity) {
            dndr_145_161=0.
            nd_substep_145_161=F_DUST_DENSITY
            for i_145_161 in 1:int(AC_dt__mod__cdata/AC_dt_substep__mod__dustdensity)+1 {
              nd_substep_0_145_161=nd_substep_145_161
              if (ndustspec<3) {
                dndr_tmp_145_161=0.
              }
              else {
                nd_new_103_145_161=ac_transformed_pencil_nd
                for k_103_145_161 in 1:ndustspec+1 {
                  if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string  ||  AC_enum_dust_chemistry__mod__dustvelocity==enum_pscalar_string) {
                    gs_103_145_161=AC_g_condensparam__mod__dustdensity*AC_supsatratio_given__mod__dustdensity
                  }
                  else {
                    gs_103_145_161=(ac_transformed_pencil_ppwater/ac_transformed_pencil_ppsat-ac_transformed_pencil_ppsf[k_103_145_161-1]/ac_transformed_pencil_ppsat)
                  }
                  if (AC_lsubstep__mod__dustdensity) {
                    ff_tmp_103_145_161[k_103_145_161-1]=nd_substep_145_161[k_103_145_161-1]*gs_103_145_161
                  }
                  else {
                    ff_tmp_103_145_161[k_103_145_161-1]=nd_new_103_145_161[k_103_145_161-1]*gs_103_145_161
                  }
                }
                if (ndustspec>=3) {
                  rr1_102_103_145_161=AC_dsize__mod__dustdensity[i1_102_103_145_161-1]
                  rr2_102_103_145_161=AC_dsize__mod__dustdensity[i2_102_103_145_161-1]
                  rr3_102_103_145_161=AC_dsize__mod__dustdensity[i3_102_103_145_161-1]
                  dndr_tmp_145_161[i1_102_103_145_161 -1] = (ff_tmp_103_145_161[i1_102_103_145_161 -1]*(rr1_102_103_145_161-rr2_102_103_145_161+rr1_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr2_102_103_145_161)*(rr1_102_103_145_161-rr3_102_103_145_161))   - ff_tmp_103_145_161[i2_102_103_145_161 -1]*(rr1_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr2_102_103_145_161)*(rr2_102_103_145_161-rr3_102_103_145_161))  + ff_tmp_103_145_161[i3_102_103_145_161 -1]*(rr1_102_103_145_161-rr2_102_103_145_161)/((rr1_102_103_145_161-rr3_102_103_145_161)*(rr2_102_103_145_161-rr3_102_103_145_161)) )
                  for k_102_103_145_161 in 2:ndustspec-1+1 {
                    rr1_102_103_145_161=AC_dsize__mod__dustdensity[k_102_103_145_161-1-1]
                    rr2_102_103_145_161=AC_dsize__mod__dustdensity[k_102_103_145_161-1]
                    rr3_102_103_145_161=AC_dsize__mod__dustdensity[1+k_102_103_145_161-1]
                    dndr_tmp_145_161[k_102_103_145_161 -1] =  ff_tmp_103_145_161[k_102_103_145_161-1 -1]*(rr2_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr2_102_103_145_161)*(rr1_102_103_145_161-rr3_102_103_145_161))  +ff_tmp_103_145_161[k_102_103_145_161 -1]*(2*rr2_102_103_145_161-rr1_102_103_145_161-rr3_102_103_145_161)/((rr2_102_103_145_161-rr1_102_103_145_161)*(rr2_102_103_145_161-rr3_102_103_145_161))  +ff_tmp_103_145_161[1+k_102_103_145_161 -1]*(2*rr2_102_103_145_161-rr1_102_103_145_161-rr2_102_103_145_161)/((rr3_102_103_145_161-rr1_102_103_145_161)*(rr3_102_103_145_161-rr2_102_103_145_161))
                  }
                  dndr_tmp_145_161[ndustspec -1]=-ff_tmp_103_145_161[ii3_102_103_145_161 -1]*(rr2_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr2_102_103_145_161)*(rr1_102_103_145_161-rr3_102_103_145_161))  +ff_tmp_103_145_161[ii2_102_103_145_161 -1]*(rr1_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr2_102_103_145_161)*(rr2_102_103_145_161-rr3_102_103_145_161))  -ff_tmp_103_145_161[ii1_102_103_145_161 -1]*(rr1_102_103_145_161-rr3_102_103_145_161+rr2_102_103_145_161-rr3_102_103_145_161)/((rr1_102_103_145_161-rr3_102_103_145_161)*(rr2_102_103_145_161-rr3_102_103_145_161))
                }
                else if (ndustspec==2) {
                  dndr_tmp_145_161[1 -1] = (ff_tmp_103_145_161[min(ndustspec,2) -1] - ff_tmp_103_145_161[min(ndustspec,1) -1])/(AC_dsize__mod__dustdensity[min(ndustspec,2)-1]-AC_dsize__mod__dustdensity[min(ndustspec,1)-1])
                  ndust_2nd_species_102_103_145_161 = min(ndustspec,2)
                  dndr_tmp_145_161[ndust_2nd_species_102_103_145_161 -1] = dndr_tmp_145_161[1 -1]
                }
                else {
                  dndr_tmp_145_161[1 -1] = 0.
                }
                for k_103_145_161 in 1:ndustspec+1 {
                  dndr_tmp_145_161[k_103_145_161-1]=-1./(AC_dsize__mod__dustdensity[k_103_145_161-1]*AC_dsize__mod__dustdensity[k_103_145_161-1])*ff_tmp_103_145_161[k_103_145_161-1]+dndr_tmp_145_161[k_103_145_161-1]/AC_dsize__mod__dustdensity[k_103_145_161-1]
                }
                if (ndustspec > 3) {
                  kk1_103_145_161=ndustspec-2
                  kk2_103_145_161=ndustspec
                  for k_103_145_161 in kk1_103_145_161:kk2_103_145_161+1 {
                    dndr_tmp_145_161[k_103_145_161 -1]=0.
                  }
                }
              }
              k1_145_161=0.
              for k_145_161 in 1:ndustspec+1 {
                k1_145_161[k_145_161 -1]=-imr_145_161*dndr_tmp_145_161[k_145_161 -1]
              }
              nd_substep_145_161=nd_substep_0_145_161+k1_145_161*AC_dt_substep__mod__dustdensity/2.
              if (ndustspec<3) {
                dndr_tmp_145_161=0.
              }
              else {
                nd_new_104_145_161=ac_transformed_pencil_nd
                for k_104_145_161 in 1:ndustspec+1 {
                  if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string  ||  AC_enum_dust_chemistry__mod__dustvelocity==enum_pscalar_string) {
                    gs_104_145_161=AC_g_condensparam__mod__dustdensity*AC_supsatratio_given__mod__dustdensity
                  }
                  else {
                    gs_104_145_161=(ac_transformed_pencil_ppwater/ac_transformed_pencil_ppsat-ac_transformed_pencil_ppsf[k_104_145_161-1]/ac_transformed_pencil_ppsat)
                  }
                  if (AC_lsubstep__mod__dustdensity) {
                    ff_tmp_104_145_161[k_104_145_161-1]=nd_substep_145_161[k_104_145_161-1]*gs_104_145_161
                  }
                  else {
                    ff_tmp_104_145_161[k_104_145_161-1]=nd_new_104_145_161[k_104_145_161-1]*gs_104_145_161
                  }
                }
                if (ndustspec>=3) {
                  rr1_102_104_145_161=AC_dsize__mod__dustdensity[i1_102_104_145_161-1]
                  rr2_102_104_145_161=AC_dsize__mod__dustdensity[i2_102_104_145_161-1]
                  rr3_102_104_145_161=AC_dsize__mod__dustdensity[i3_102_104_145_161-1]
                  dndr_tmp_145_161[i1_102_104_145_161 -1] = (ff_tmp_104_145_161[i1_102_104_145_161 -1]*(rr1_102_104_145_161-rr2_102_104_145_161+rr1_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr2_102_104_145_161)*(rr1_102_104_145_161-rr3_102_104_145_161))   - ff_tmp_104_145_161[i2_102_104_145_161 -1]*(rr1_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr2_102_104_145_161)*(rr2_102_104_145_161-rr3_102_104_145_161))  + ff_tmp_104_145_161[i3_102_104_145_161 -1]*(rr1_102_104_145_161-rr2_102_104_145_161)/((rr1_102_104_145_161-rr3_102_104_145_161)*(rr2_102_104_145_161-rr3_102_104_145_161)) )
                  for k_102_104_145_161 in 2:ndustspec-1+1 {
                    rr1_102_104_145_161=AC_dsize__mod__dustdensity[k_102_104_145_161-1-1]
                    rr2_102_104_145_161=AC_dsize__mod__dustdensity[k_102_104_145_161-1]
                    rr3_102_104_145_161=AC_dsize__mod__dustdensity[1+k_102_104_145_161-1]
                    dndr_tmp_145_161[k_102_104_145_161 -1] =  ff_tmp_104_145_161[k_102_104_145_161-1 -1]*(rr2_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr2_102_104_145_161)*(rr1_102_104_145_161-rr3_102_104_145_161))  +ff_tmp_104_145_161[k_102_104_145_161 -1]*(2*rr2_102_104_145_161-rr1_102_104_145_161-rr3_102_104_145_161)/((rr2_102_104_145_161-rr1_102_104_145_161)*(rr2_102_104_145_161-rr3_102_104_145_161))  +ff_tmp_104_145_161[1+k_102_104_145_161 -1]*(2*rr2_102_104_145_161-rr1_102_104_145_161-rr2_102_104_145_161)/((rr3_102_104_145_161-rr1_102_104_145_161)*(rr3_102_104_145_161-rr2_102_104_145_161))
                  }
                  dndr_tmp_145_161[ndustspec -1]=-ff_tmp_104_145_161[ii3_102_104_145_161 -1]*(rr2_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr2_102_104_145_161)*(rr1_102_104_145_161-rr3_102_104_145_161))  +ff_tmp_104_145_161[ii2_102_104_145_161 -1]*(rr1_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr2_102_104_145_161)*(rr2_102_104_145_161-rr3_102_104_145_161))  -ff_tmp_104_145_161[ii1_102_104_145_161 -1]*(rr1_102_104_145_161-rr3_102_104_145_161+rr2_102_104_145_161-rr3_102_104_145_161)/((rr1_102_104_145_161-rr3_102_104_145_161)*(rr2_102_104_145_161-rr3_102_104_145_161))
                }
                else if (ndustspec==2) {
                  dndr_tmp_145_161[1 -1] = (ff_tmp_104_145_161[min(ndustspec,2) -1] - ff_tmp_104_145_161[min(ndustspec,1) -1])/(AC_dsize__mod__dustdensity[min(ndustspec,2)-1]-AC_dsize__mod__dustdensity[min(ndustspec,1)-1])
                  ndust_2nd_species_102_104_145_161 = min(ndustspec,2)
                  dndr_tmp_145_161[ndust_2nd_species_102_104_145_161 -1] = dndr_tmp_145_161[1 -1]
                }
                else {
                  dndr_tmp_145_161[1 -1] = 0.
                }
                for k_104_145_161 in 1:ndustspec+1 {
                  dndr_tmp_145_161[k_104_145_161-1]=-1./(AC_dsize__mod__dustdensity[k_104_145_161-1]*AC_dsize__mod__dustdensity[k_104_145_161-1])*ff_tmp_104_145_161[k_104_145_161-1]+dndr_tmp_145_161[k_104_145_161-1]/AC_dsize__mod__dustdensity[k_104_145_161-1]
                }
                if (ndustspec > 3) {
                  kk1_104_145_161=ndustspec-2
                  kk2_104_145_161=ndustspec
                  for k_104_145_161 in kk1_104_145_161:kk2_104_145_161+1 {
                    dndr_tmp_145_161[k_104_145_161 -1]=0.
                  }
                }
              }
              k2_145_161=0.
              for k_145_161 in 1:ndustspec+1 {
                k2_145_161[k_145_161 -1]=-imr_145_161*dndr_tmp_145_161[k_145_161 -1]
              }
              nd_substep_145_161=nd_substep_0_145_161+k2_145_161*AC_dt_substep__mod__dustdensity/2.
              if (ndustspec<3) {
                dndr_tmp_145_161=0.
              }
              else {
                nd_new_105_145_161=ac_transformed_pencil_nd
                for k_105_145_161 in 1:ndustspec+1 {
                  if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string  ||  AC_enum_dust_chemistry__mod__dustvelocity==enum_pscalar_string) {
                    gs_105_145_161=AC_g_condensparam__mod__dustdensity*AC_supsatratio_given__mod__dustdensity
                  }
                  else {
                    gs_105_145_161=(ac_transformed_pencil_ppwater/ac_transformed_pencil_ppsat-ac_transformed_pencil_ppsf[k_105_145_161-1]/ac_transformed_pencil_ppsat)
                  }
                  if (AC_lsubstep__mod__dustdensity) {
                    ff_tmp_105_145_161[k_105_145_161-1]=nd_substep_145_161[k_105_145_161-1]*gs_105_145_161
                  }
                  else {
                    ff_tmp_105_145_161[k_105_145_161-1]=nd_new_105_145_161[k_105_145_161-1]*gs_105_145_161
                  }
                }
                if (ndustspec>=3) {
                  rr1_102_105_145_161=AC_dsize__mod__dustdensity[i1_102_105_145_161-1]
                  rr2_102_105_145_161=AC_dsize__mod__dustdensity[i2_102_105_145_161-1]
                  rr3_102_105_145_161=AC_dsize__mod__dustdensity[i3_102_105_145_161-1]
                  dndr_tmp_145_161[i1_102_105_145_161 -1] = (ff_tmp_105_145_161[i1_102_105_145_161 -1]*(rr1_102_105_145_161-rr2_102_105_145_161+rr1_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr2_102_105_145_161)*(rr1_102_105_145_161-rr3_102_105_145_161))   - ff_tmp_105_145_161[i2_102_105_145_161 -1]*(rr1_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr2_102_105_145_161)*(rr2_102_105_145_161-rr3_102_105_145_161))  + ff_tmp_105_145_161[i3_102_105_145_161 -1]*(rr1_102_105_145_161-rr2_102_105_145_161)/((rr1_102_105_145_161-rr3_102_105_145_161)*(rr2_102_105_145_161-rr3_102_105_145_161)) )
                  for k_102_105_145_161 in 2:ndustspec-1+1 {
                    rr1_102_105_145_161=AC_dsize__mod__dustdensity[k_102_105_145_161-1-1]
                    rr2_102_105_145_161=AC_dsize__mod__dustdensity[k_102_105_145_161-1]
                    rr3_102_105_145_161=AC_dsize__mod__dustdensity[1+k_102_105_145_161-1]
                    dndr_tmp_145_161[k_102_105_145_161 -1] =  ff_tmp_105_145_161[k_102_105_145_161-1 -1]*(rr2_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr2_102_105_145_161)*(rr1_102_105_145_161-rr3_102_105_145_161))  +ff_tmp_105_145_161[k_102_105_145_161 -1]*(2*rr2_102_105_145_161-rr1_102_105_145_161-rr3_102_105_145_161)/((rr2_102_105_145_161-rr1_102_105_145_161)*(rr2_102_105_145_161-rr3_102_105_145_161))  +ff_tmp_105_145_161[1+k_102_105_145_161 -1]*(2*rr2_102_105_145_161-rr1_102_105_145_161-rr2_102_105_145_161)/((rr3_102_105_145_161-rr1_102_105_145_161)*(rr3_102_105_145_161-rr2_102_105_145_161))
                  }
                  dndr_tmp_145_161[ndustspec -1]=-ff_tmp_105_145_161[ii3_102_105_145_161 -1]*(rr2_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr2_102_105_145_161)*(rr1_102_105_145_161-rr3_102_105_145_161))  +ff_tmp_105_145_161[ii2_102_105_145_161 -1]*(rr1_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr2_102_105_145_161)*(rr2_102_105_145_161-rr3_102_105_145_161))  -ff_tmp_105_145_161[ii1_102_105_145_161 -1]*(rr1_102_105_145_161-rr3_102_105_145_161+rr2_102_105_145_161-rr3_102_105_145_161)/((rr1_102_105_145_161-rr3_102_105_145_161)*(rr2_102_105_145_161-rr3_102_105_145_161))
                }
                else if (ndustspec==2) {
                  dndr_tmp_145_161[1 -1] = (ff_tmp_105_145_161[min(ndustspec,2) -1] - ff_tmp_105_145_161[min(ndustspec,1) -1])/(AC_dsize__mod__dustdensity[min(ndustspec,2)-1]-AC_dsize__mod__dustdensity[min(ndustspec,1)-1])
                  ndust_2nd_species_102_105_145_161 = min(ndustspec,2)
                  dndr_tmp_145_161[ndust_2nd_species_102_105_145_161 -1] = dndr_tmp_145_161[1 -1]
                }
                else {
                  dndr_tmp_145_161[1 -1] = 0.
                }
                for k_105_145_161 in 1:ndustspec+1 {
                  dndr_tmp_145_161[k_105_145_161-1]=-1./(AC_dsize__mod__dustdensity[k_105_145_161-1]*AC_dsize__mod__dustdensity[k_105_145_161-1])*ff_tmp_105_145_161[k_105_145_161-1]+dndr_tmp_145_161[k_105_145_161-1]/AC_dsize__mod__dustdensity[k_105_145_161-1]
                }
                if (ndustspec > 3) {
                  kk1_105_145_161=ndustspec-2
                  kk2_105_145_161=ndustspec
                  for k_105_145_161 in kk1_105_145_161:kk2_105_145_161+1 {
                    dndr_tmp_145_161[k_105_145_161 -1]=0.
                  }
                }
              }
              k3_145_161=0.
              for k_145_161 in 1:ndustspec+1 {
                k3_145_161[k_145_161 -1]=-imr_145_161*dndr_tmp_145_161[k_145_161 -1]
              }
              nd_substep_145_161=nd_substep_0_145_161+k3_145_161*AC_dt_substep__mod__dustdensity
              k4_145_161=0.
              if (ndustspec<3) {
                dndr_tmp_145_161=0.
              }
              else {
                nd_new_106_145_161=ac_transformed_pencil_nd
                for k_106_145_161 in 1:ndustspec+1 {
                  if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string  ||  AC_enum_dust_chemistry__mod__dustvelocity==enum_pscalar_string) {
                    gs_106_145_161=AC_g_condensparam__mod__dustdensity*AC_supsatratio_given__mod__dustdensity
                  }
                  else {
                    gs_106_145_161=(ac_transformed_pencil_ppwater/ac_transformed_pencil_ppsat-ac_transformed_pencil_ppsf[k_106_145_161-1]/ac_transformed_pencil_ppsat)
                  }
                  if (AC_lsubstep__mod__dustdensity) {
                    ff_tmp_106_145_161[k_106_145_161-1]=nd_substep_145_161[k_106_145_161-1]*gs_106_145_161
                  }
                  else {
                    ff_tmp_106_145_161[k_106_145_161-1]=nd_new_106_145_161[k_106_145_161-1]*gs_106_145_161
                  }
                }
                if (ndustspec>=3) {
                  rr1_102_106_145_161=AC_dsize__mod__dustdensity[i1_102_106_145_161-1]
                  rr2_102_106_145_161=AC_dsize__mod__dustdensity[i2_102_106_145_161-1]
                  rr3_102_106_145_161=AC_dsize__mod__dustdensity[i3_102_106_145_161-1]
                  dndr_tmp_145_161[i1_102_106_145_161 -1] = (ff_tmp_106_145_161[i1_102_106_145_161 -1]*(rr1_102_106_145_161-rr2_102_106_145_161+rr1_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr2_102_106_145_161)*(rr1_102_106_145_161-rr3_102_106_145_161))   - ff_tmp_106_145_161[i2_102_106_145_161 -1]*(rr1_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr2_102_106_145_161)*(rr2_102_106_145_161-rr3_102_106_145_161))  + ff_tmp_106_145_161[i3_102_106_145_161 -1]*(rr1_102_106_145_161-rr2_102_106_145_161)/((rr1_102_106_145_161-rr3_102_106_145_161)*(rr2_102_106_145_161-rr3_102_106_145_161)) )
                  for k_102_106_145_161 in 2:ndustspec-1+1 {
                    rr1_102_106_145_161=AC_dsize__mod__dustdensity[k_102_106_145_161-1-1]
                    rr2_102_106_145_161=AC_dsize__mod__dustdensity[k_102_106_145_161-1]
                    rr3_102_106_145_161=AC_dsize__mod__dustdensity[1+k_102_106_145_161-1]
                    dndr_tmp_145_161[k_102_106_145_161 -1] =  ff_tmp_106_145_161[k_102_106_145_161-1 -1]*(rr2_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr2_102_106_145_161)*(rr1_102_106_145_161-rr3_102_106_145_161))  +ff_tmp_106_145_161[k_102_106_145_161 -1]*(2*rr2_102_106_145_161-rr1_102_106_145_161-rr3_102_106_145_161)/((rr2_102_106_145_161-rr1_102_106_145_161)*(rr2_102_106_145_161-rr3_102_106_145_161))  +ff_tmp_106_145_161[1+k_102_106_145_161 -1]*(2*rr2_102_106_145_161-rr1_102_106_145_161-rr2_102_106_145_161)/((rr3_102_106_145_161-rr1_102_106_145_161)*(rr3_102_106_145_161-rr2_102_106_145_161))
                  }
                  dndr_tmp_145_161[ndustspec -1]=-ff_tmp_106_145_161[ii3_102_106_145_161 -1]*(rr2_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr2_102_106_145_161)*(rr1_102_106_145_161-rr3_102_106_145_161))  +ff_tmp_106_145_161[ii2_102_106_145_161 -1]*(rr1_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr2_102_106_145_161)*(rr2_102_106_145_161-rr3_102_106_145_161))  -ff_tmp_106_145_161[ii1_102_106_145_161 -1]*(rr1_102_106_145_161-rr3_102_106_145_161+rr2_102_106_145_161-rr3_102_106_145_161)/((rr1_102_106_145_161-rr3_102_106_145_161)*(rr2_102_106_145_161-rr3_102_106_145_161))
                }
                else if (ndustspec==2) {
                  dndr_tmp_145_161[1 -1] = (ff_tmp_106_145_161[min(ndustspec,2) -1] - ff_tmp_106_145_161[min(ndustspec,1) -1])/(AC_dsize__mod__dustdensity[min(ndustspec,2)-1]-AC_dsize__mod__dustdensity[min(ndustspec,1)-1])
                  ndust_2nd_species_102_106_145_161 = min(ndustspec,2)
                  dndr_tmp_145_161[ndust_2nd_species_102_106_145_161 -1] = dndr_tmp_145_161[1 -1]
                }
                else {
                  dndr_tmp_145_161[1 -1] = 0.
                }
                for k_106_145_161 in 1:ndustspec+1 {
                  dndr_tmp_145_161[k_106_145_161-1]=-1./(AC_dsize__mod__dustdensity[k_106_145_161-1]*AC_dsize__mod__dustdensity[k_106_145_161-1])*ff_tmp_106_145_161[k_106_145_161-1]+dndr_tmp_145_161[k_106_145_161-1]/AC_dsize__mod__dustdensity[k_106_145_161-1]
                }
                if (ndustspec > 3) {
                  kk1_106_145_161=ndustspec-2
                  kk2_106_145_161=ndustspec
                  for k_106_145_161 in kk1_106_145_161:kk2_106_145_161+1 {
                    dndr_tmp_145_161[k_106_145_161 -1]=0.
                  }
                }
              }
              for k_145_161 in 1:ndustspec+1 {
                k4_145_161[k_145_161 -1]=-imr_145_161*dndr_tmp_145_161[k_145_161 -1]
              }
              nd_substep_145_161=nd_substep_0_145_161+AC_dt_substep__mod__dustdensity/6.*(k1_145_161+2.*k2_145_161+2.*k3_145_161+k4_145_161)
            }
            for k_145_161 in 1:ndustspec+1 {
              dndr_145_161[k_145_161 -1]=(nd_substep_145_161[k_145_161 -1]-value(F_DUST_DENSITY[k_145_161-1]))/AC_dt__mod__cdata
            }
          }
          else {
            if (AC_ldustcondensation_simplified__mod__dustdensity) {
              if (ndustspec<3) {
                dndr_tmp_145_161=0.
              }
              else {
                nd_new_107_145_161=ac_transformed_pencil_nd
                for k_107_145_161 in 1:ndustspec+1 {
                  if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string  ||  AC_enum_dust_chemistry__mod__dustvelocity==enum_pscalar_string) {
                    gs_107_145_161=AC_g_condensparam__mod__dustdensity*AC_supsatratio_given__mod__dustdensity
                  }
                  else {
                    gs_107_145_161=(ac_transformed_pencil_ppwater/ac_transformed_pencil_ppsat-ac_transformed_pencil_ppsf[k_107_145_161-1]/ac_transformed_pencil_ppsat)
                  }
                  if (AC_lsubstep__mod__dustdensity) {
                    ff_tmp_107_145_161[k_107_145_161-1]=nd_substep_145_161[k_107_145_161-1]*gs_107_145_161
                  }
                  else {
                    ff_tmp_107_145_161[k_107_145_161-1]=nd_new_107_145_161[k_107_145_161-1]*gs_107_145_161
                  }
                }
                if (ndustspec>=3) {
                  rr1_102_107_145_161=AC_dsize__mod__dustdensity[i1_102_107_145_161-1]
                  rr2_102_107_145_161=AC_dsize__mod__dustdensity[i2_102_107_145_161-1]
                  rr3_102_107_145_161=AC_dsize__mod__dustdensity[i3_102_107_145_161-1]
                  dndr_tmp_145_161[i1_102_107_145_161 -1] = (ff_tmp_107_145_161[i1_102_107_145_161 -1]*(rr1_102_107_145_161-rr2_102_107_145_161+rr1_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr2_102_107_145_161)*(rr1_102_107_145_161-rr3_102_107_145_161))   - ff_tmp_107_145_161[i2_102_107_145_161 -1]*(rr1_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr2_102_107_145_161)*(rr2_102_107_145_161-rr3_102_107_145_161))  + ff_tmp_107_145_161[i3_102_107_145_161 -1]*(rr1_102_107_145_161-rr2_102_107_145_161)/((rr1_102_107_145_161-rr3_102_107_145_161)*(rr2_102_107_145_161-rr3_102_107_145_161)) )
                  for k_102_107_145_161 in 2:ndustspec-1+1 {
                    rr1_102_107_145_161=AC_dsize__mod__dustdensity[k_102_107_145_161-1-1]
                    rr2_102_107_145_161=AC_dsize__mod__dustdensity[k_102_107_145_161-1]
                    rr3_102_107_145_161=AC_dsize__mod__dustdensity[1+k_102_107_145_161-1]
                    dndr_tmp_145_161[k_102_107_145_161 -1] =  ff_tmp_107_145_161[k_102_107_145_161-1 -1]*(rr2_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr2_102_107_145_161)*(rr1_102_107_145_161-rr3_102_107_145_161))  +ff_tmp_107_145_161[k_102_107_145_161 -1]*(2*rr2_102_107_145_161-rr1_102_107_145_161-rr3_102_107_145_161)/((rr2_102_107_145_161-rr1_102_107_145_161)*(rr2_102_107_145_161-rr3_102_107_145_161))  +ff_tmp_107_145_161[1+k_102_107_145_161 -1]*(2*rr2_102_107_145_161-rr1_102_107_145_161-rr2_102_107_145_161)/((rr3_102_107_145_161-rr1_102_107_145_161)*(rr3_102_107_145_161-rr2_102_107_145_161))
                  }
                  dndr_tmp_145_161[ndustspec -1]=-ff_tmp_107_145_161[ii3_102_107_145_161 -1]*(rr2_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr2_102_107_145_161)*(rr1_102_107_145_161-rr3_102_107_145_161))  +ff_tmp_107_145_161[ii2_102_107_145_161 -1]*(rr1_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr2_102_107_145_161)*(rr2_102_107_145_161-rr3_102_107_145_161))  -ff_tmp_107_145_161[ii1_102_107_145_161 -1]*(rr1_102_107_145_161-rr3_102_107_145_161+rr2_102_107_145_161-rr3_102_107_145_161)/((rr1_102_107_145_161-rr3_102_107_145_161)*(rr2_102_107_145_161-rr3_102_107_145_161))
                }
                else if (ndustspec==2) {
                  dndr_tmp_145_161[1 -1] = (ff_tmp_107_145_161[min(ndustspec,2) -1] - ff_tmp_107_145_161[min(ndustspec,1) -1])/(AC_dsize__mod__dustdensity[min(ndustspec,2)-1]-AC_dsize__mod__dustdensity[min(ndustspec,1)-1])
                  ndust_2nd_species_102_107_145_161 = min(ndustspec,2)
                  dndr_tmp_145_161[ndust_2nd_species_102_107_145_161 -1] = dndr_tmp_145_161[1 -1]
                }
                else {
                  dndr_tmp_145_161[1 -1] = 0.
                }
                for k_107_145_161 in 1:ndustspec+1 {
                  dndr_tmp_145_161[k_107_145_161-1]=-1./(AC_dsize__mod__dustdensity[k_107_145_161-1]*AC_dsize__mod__dustdensity[k_107_145_161-1])*ff_tmp_107_145_161[k_107_145_161-1]+dndr_tmp_145_161[k_107_145_161-1]/AC_dsize__mod__dustdensity[k_107_145_161-1]
                }
                if (ndustspec > 3) {
                  kk1_107_145_161=ndustspec-2
                  kk2_107_145_161=ndustspec
                  for k_107_145_161 in kk1_107_145_161:kk2_107_145_161+1 {
                    dndr_tmp_145_161[k_107_145_161 -1]=0.
                  }
                }
              }
              if (AC_lsemi_chemistry__mod__dustdensity) {
                imr_145_161=1.
              }
              for k_145_161 in 1:ndustspec+1 {
                dndr_145_161[k_145_161 -1]=-imr_145_161*dndr_tmp_145_161[k_145_161 -1]
              }
            }
          }
        }
        DF_DUST_DENSITY = DF_DUST_DENSITY + dndr_145_161
      }
      else {
        if (AC_ldustcoagulation__mod__dustvelocity  &&  ! lcoala) {
          if (AC_ldustcoagulation__mod__dustvelocity) {
            if (!AC_lcalcdkern__mod__dustdensity) {
              if (AC_lpiecewise_constant_kernel__mod__dustdensity) {
                dkern__mod__dustdensity = AC_dkern_cst__mod__dustdensity
              }
              else {
                dkern__mod__dustdensity = AC_dkern_cst__mod__dustdensity
                for i_108_145_161 in 1:ndustspec+1 {
                  dkern__mod__dustdensity[i_108_145_161 -1][i_108_145_161 -1] = dkern__mod__dustdensity[i_108_145_161 -1][i_108_145_161 -1]*0.5
                }
              }
            }
            else {
              lgh_108_145_161=l_108_145_161+NGHOST
              for i_108_145_161 in 1:ndustspec+1 {
                for j_108_145_161 in i_108_145_161:ndustspec+1 {
                  lgh_40_108_145_161=l_108_145_161+NGHOST
                  if (AC_lself_collisions__mod__dustdensity) {
                    if (i_108_145_161==j_108_145_161) {
                      if(AC_enum_self_collisions__mod__dustdensity == enum_average_string) {
                        fact_40_108_145_161=0.5*AC_self_collision_factor__mod__dustdensity
                        deltavd_drift2_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[j_108_145_161-1]+ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[j_108_145_161-1]+ac_transformed_pencil_uud[i_108_145_161-1]))
                      }
                      else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_string)   {
                        fact_40_108_145_161=AC_self_collision_factor__mod__dustdensity
                        if (i_108_145_161==1) {
                          deltavd_drift2_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                        }
                        else if (i_108_145_161==ndustspec) {
                          deltavd_drift2_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                        }
                        else {
                          fact_40_108_145_161=0.5*AC_self_collision_factor__mod__dustdensity
                          deltavd_drift2a_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                          deltavd_drift2a_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                          deltavd_drift2_40_108_145_161=deltavd_drift2a_40_108_145_161+deltavd_drift2b_40_108_145_161
                        }
                      }
                      else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_asymmetric_string)   {
                        fact_40_108_145_161=AC_self_collision_factor__mod__dustdensity
                        if (i_108_145_161==ndustspec) {
                          deltavd_drift2_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[i_108_145_161-1-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                        }
                        else {
                          deltavd_drift2_40_108_145_161 = dot(fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]),fact_40_108_145_161*(ac_transformed_pencil_uud[1+i_108_145_161-1]-ac_transformed_pencil_uud[i_108_145_161-1]))
                        }
                      }
                      else {
                      }
                    }
                    else {
                      deltavd_drift2_40_108_145_161 = dot(ac_transformed_pencil_uud[i_108_145_161-1]-ac_transformed_pencil_uud[j_108_145_161-1],ac_transformed_pencil_uud[i_108_145_161-1]-ac_transformed_pencil_uud[j_108_145_161-1])
                    }
                  }
                  else {
                    deltavd_drift2_40_108_145_161 = dot(ac_transformed_pencil_uud[i_108_145_161-1]-ac_transformed_pencil_uud[j_108_145_161-1],ac_transformed_pencil_uud[i_108_145_161-1]-ac_transformed_pencil_uud[j_108_145_161-1])
                  }
                  if (AC_ldeltavd_thermal__mod__dustdensity) {
                    deltavd_therm_40_108_145_161 = sqrt( 8*AC_k_b__mod__cdata/(pi*ac_transformed_pencil_tt1)*(ac_transformed_pencil_md[i_108_145_161-1]+ac_transformed_pencil_md[j_108_145_161-1])/(ac_transformed_pencil_md[i_108_145_161-1]*ac_transformed_pencil_md[j_108_145_161-1]*AC_unit_md__mod__dustvelocity) )
                  }
                  else {
                    deltavd_therm_40_108_145_161=0.
                  }
                  if (AC_ldeltavd_turbulent__mod__dustdensity) {
                  }
                  else if(AC_ldeltavd_turbulent_ormel__mod__dustdensity) {
                    t_dyn_39_40_108_145_161 = sqrt(3.*pi/(32.*g_newton_cgs*ac_transformed_pencil_rho))
                    nh_39_40_108_145_161 = ac_transformed_pencil_rho/(mu_gas_39_40_108_145_161*mh_39_40_108_145_161)
                    cs_39_40_108_145_161 = sqrt(AC_gamma__mod__dustdensity*AC_k_b__mod__cdata*ac_transformed_pencil_tt/(mu_gas_39_40_108_145_161*mh_39_40_108_145_161))
                    re_39_40_108_145_161 = 62e6*sqrt(nh_39_40_108_145_161/1e5)*sqrt(ac_transformed_pencil_tt/10.)
                    t_eta_39_40_108_145_161 = t_dyn_39_40_108_145_161/sqrt(re_39_40_108_145_161)
                    ts_i_39_40_108_145_161 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[i_108_145_161-1]/(ac_transformed_pencil_rho*cs_39_40_108_145_161)
                    ts_j_39_40_108_145_161 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[j_108_145_161-1]/(ac_transformed_pencil_rho*cs_39_40_108_145_161)
                    ts_1_39_40_108_145_161 = ts_i_39_40_108_145_161
                    st_1_39_40_108_145_161    = ts_i_39_40_108_145_161/t_dyn_39_40_108_145_161
                    st_2_39_40_108_145_161    = ts_j_39_40_108_145_161/t_dyn_39_40_108_145_161
                    if (j_108_145_161 > i_108_145_161) {
                      ts_1_39_40_108_145_161    = ts_j_39_40_108_145_161
                      st_1_39_40_108_145_161    = ts_j_39_40_108_145_161/t_dyn_39_40_108_145_161
                      st_2_39_40_108_145_161    = ts_i_39_40_108_145_161/t_dyn_39_40_108_145_161
                    }
                    x_st_39_40_108_145_161    = st_2_39_40_108_145_161/st_1_39_40_108_145_161
                    beta_st_39_40_108_145_161 = 3.2 - (1. + x_st_39_40_108_145_161) + 2./(1. + x_st_39_40_108_145_161) * (1./2.6 +( x_st_39_40_108_145_161* x_st_39_40_108_145_161* x_st_39_40_108_145_161)/(1.6 + x_st_39_40_108_145_161))
                    if (ts_1_39_40_108_145_161 < t_eta_39_40_108_145_161) {
                      if (abs(st_1_39_40_108_145_161 - st_2_39_40_108_145_161) < epsilon(st_1_39_40_108_145_161)) {
                        res_39_40_108_145_161 = 0.
                      }
                      else {
                        res_39_40_108_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_40_108_145_161* cs_39_40_108_145_161) * (st_1_39_40_108_145_161 - st_2_39_40_108_145_161)/(st_1_39_40_108_145_161 + st_2_39_40_108_145_161) * ((st_1_39_40_108_145_161*st_1_39_40_108_145_161)/(st_1_39_40_108_145_161 + 1./sqrt(re_39_40_108_145_161)) +( st_2_39_40_108_145_161* st_2_39_40_108_145_161)/(st_2_39_40_108_145_161 + 1./sqrt(re_39_40_108_145_161)))
                      }
                    }
                    else if ( (t_eta_39_40_108_145_161 <= ts_1_39_40_108_145_161)  &&  (ts_1_39_40_108_145_161 < t_dyn_39_40_108_145_161) )   {
                      res_39_40_108_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_40_108_145_161* cs_39_40_108_145_161) * beta_st_39_40_108_145_161 * st_1_39_40_108_145_161
                    }
                    else {
                      res_39_40_108_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_40_108_145_161* cs_39_40_108_145_161) * (1./(st_1_39_40_108_145_161 + 1.) + 1./(st_2_39_40_108_145_161 + 1.))
                    }
                    deltavd_turbu_40_108_145_161 = sqrt(res_39_40_108_145_161)
                  }
                  else {
                    deltavd_turbu_40_108_145_161 = 0.
                  }
                  deltavd_108_145_161 = sqrt(deltavd_drift2_40_108_145_161+(deltavd_therm_40_108_145_161*deltavd_therm_40_108_145_161)+(deltavd_turbu_40_108_145_161*deltavd_turbu_40_108_145_161)+(AC_deltavd_imposed__mod__dustdensity*AC_deltavd_imposed__mod__dustdensity))
                  if (AC_ludstickmax__mod__dustdensity) {
                    ust_40_108_145_161 = AC_ustcst__mod__dustvelocity *pow( (AC_ad__mod__dustvelocity[i_108_145_161-1]*AC_ad__mod__dustvelocity[j_108_145_161-1]/(AC_ad__mod__dustvelocity[i_108_145_161-1]+AC_ad__mod__dustvelocity[j_108_145_161-1])),(2/3.)) *pow(  ((ac_transformed_pencil_md[i_108_145_161-1]+ac_transformed_pencil_md[j_108_145_161-1])/(ac_transformed_pencil_md[i_108_145_161-1]*ac_transformed_pencil_md[j_108_145_161-1]*AC_unit_md__mod__dustvelocity)),(1/2.))
                    if (deltavd_108_145_161 > ust_40_108_145_161) {
                      deltavd_108_145_161 = 0.
                    }
                  }
                  if (AC_lkernel_mean__mod__dustdensity) {
                    dkern__mod__dustdensity[i_108_145_161-1][j_108_145_161-1] = AC_kernel_mean__mod__dustdensity[i_108_145_161-1][j_108_145_161-1]
                  }
                  else if (AC_lzero_upper_kern__mod__dustdensity  &&  (i_108_145_161 >= ndustspec-1  ||  j_108_145_161 >= ndustspec-1)) {
                    dkern__mod__dustdensity[i_108_145_161-1][j_108_145_161-1] = 0.
                  }
                  else if (AC_lno_deltavd__mod__dustdensity) {
                    dkern__mod__dustdensity[i_108_145_161-1][j_108_145_161-1] = AC_scolld__mod__dustvelocity[i_108_145_161-1][j_108_145_161-1]*AC_deltavd_const__mod__dustdensity
                  }
                  else {
                    dkern__mod__dustdensity[i_108_145_161-1][j_108_145_161-1] = AC_scolld__mod__dustvelocity[i_108_145_161-1][j_108_145_161-1]*deltavd_108_145_161
                  }
                  dkern__mod__dustdensity[j_108_145_161-1][i_108_145_161-1] = dkern__mod__dustdensity[i_108_145_161-1][j_108_145_161-1]
                }
              }
            }
          }
          else if (AC_ldustcoagulation_simplified__mod__dustdensity) {
            mu_air_108_145_161=2.e-4
            rho_air_108_145_161=1.2e-3
            tt_108_145_161=1./ac_transformed_pencil_tt1
            for i_108_145_161 in 1:ndustspec+1 {
              for k_108_145_161 in i_108_145_161:ndustspec+1 {
                rik_108_145_161=AC_dsize__mod__dustdensity[i_108_145_161-1]+AC_dsize__mod__dustdensity[k_108_145_161-1]
                kn_108_145_161=2.*mu_air_108_145_161/rho_air_108_145_161*sqrt(pi*2e-24/(2.8*AC_k_b__mod__cdata*tt_108_145_161))/(rik_108_145_161/2.)
                cor_factor_108_145_161=1.+kn_108_145_161*(1.142+0.558*exp(-0.999/kn_108_145_161))
                d_coeff_108_145_161=AC_k_b__mod__cdata*cor_factor_108_145_161*tt_108_145_161/(6.*pi*mu_air_108_145_161)
                di_108_145_161=d_coeff_108_145_161/AC_dsize__mod__dustdensity[i_108_145_161-1]
                dk_108_145_161=d_coeff_108_145_161/AC_dsize__mod__dustdensity[k_108_145_161-1]
                dik_108_145_161=(di_108_145_161+dk_108_145_161)
                kbc_108_145_161=4*pi*(AC_dsize__mod__dustdensity[i_108_145_161-1]+AC_dsize__mod__dustdensity[k_108_145_161-1])*(di_108_145_161+dk_108_145_161)
                vmean_i_108_145_161=sqrt(8.*AC_k_b__mod__cdata*tt_108_145_161/pi/(4./3*pi*(AC_dsize__mod__dustdensity[i_108_145_161-1]*AC_dsize__mod__dustdensity[i_108_145_161-1]*AC_dsize__mod__dustdensity[i_108_145_161-1])))
                vmean_k_108_145_161=sqrt(8.*AC_k_b__mod__cdata*tt_108_145_161/pi/(4./3*pi*(AC_dsize__mod__dustdensity[k_108_145_161-1]*AC_dsize__mod__dustdensity[k_108_145_161-1]*AC_dsize__mod__dustdensity[k_108_145_161-1])))
                vmean_ik_108_145_161=sqrt((vmean_i_108_145_161*vmean_i_108_145_161)+(vmean_k_108_145_161*vmean_k_108_145_161))
                gamma_i_108_145_161=8.*di_108_145_161/pi/vmean_i_108_145_161
                gamma_k_108_145_161=8.*dk_108_145_161/pi/vmean_k_108_145_161
                omega_i_108_145_161=(((rik_108_145_161+gamma_i_108_145_161)*(rik_108_145_161+gamma_i_108_145_161)*(rik_108_145_161+gamma_i_108_145_161))-pow(((rik_108_145_161*rik_108_145_161)+(gamma_i_108_145_161*gamma_i_108_145_161)),1.5))/(3.*rik_108_145_161*gamma_i_108_145_161)-rik_108_145_161
                omega_k_108_145_161=(((rik_108_145_161+gamma_k_108_145_161)*(rik_108_145_161+gamma_k_108_145_161)*(rik_108_145_161+gamma_k_108_145_161))-pow(((rik_108_145_161*rik_108_145_161)+(gamma_k_108_145_161*gamma_k_108_145_161)),1.5))/(3.*rik_108_145_161*gamma_k_108_145_161)-rik_108_145_161
                sigma_ik_108_145_161=sqrt((omega_i_108_145_161*omega_i_108_145_161)+(omega_k_108_145_161*omega_k_108_145_161))
                dkern__mod__dustdensity[i_108_145_161 -1][k_108_145_161 -1]=kbc_108_145_161/( rik_108_145_161/(rik_108_145_161+sigma_ik_108_145_161) + 4.*dik_108_145_161/(vmean_ik_108_145_161*rik_108_145_161) )
              }
            }
            for i_108_145_161 in 1:ndustspec+1 {
              for k_108_145_161 in 1:i_108_145_161-1+1 {
                dkern__mod__dustdensity[i_108_145_161 -1][k_108_145_161 -1]=dkern__mod__dustdensity[k_108_145_161 -1][i_108_145_161 -1]
              }
            }
          }
          dndfac_sum__mod__dustdensity=0.
          dndfac_sum2__mod__dustdensity=0.
          momcons_sum_x__mod__dustdensity=0.
          momcons_sum_y__mod__dustdensity=0.
          momcons_sum_z__mod__dustdensity=0.
          lgh_109_145_161=l_109_145_161+NGHOST
          for i_109_145_161 in 1:ndustspec+1 {
            for j_109_145_161 in i_109_145_161:ndustspec+1 {
              dndfac_109_145_161 = -dkern__mod__dustdensity[i_109_145_161-1][j_109_145_161-1]*ac_transformed_pencil_nd[i_109_145_161-1]*ac_transformed_pencil_nd[j_109_145_161-1]
              if (AC_lmomcons2__mod__dustdensity) {
                dndfaci_109_145_161 = -dkern__mod__dustdensity[i_109_145_161-1][j_109_145_161-1]*ac_transformed_pencil_nd[j_109_145_161-1]
                dndfacj_109_145_161 = -dkern__mod__dustdensity[i_109_145_161-1][j_109_145_161-1]*ac_transformed_pencil_nd[j_109_145_161-1]
              }
              dndfac_sum__mod__dustdensity = dndfac_sum__mod__dustdensity + dndfac_109_145_161
              if (dndfac_109_145_161!=0.0) {
                if (AC_lradius_binning__mod__dustdensity) {
                  DF_DUST_DENSITY[i_109_145_161-1] = DF_DUST_DENSITY[i_109_145_161-1] + dndfac_109_145_161*ac_transformed_pencil_ad[i_109_145_161-1]*AC_dlnad__mod__dustdensity
                  DF_DUST_DENSITY[j_109_145_161-1] = DF_DUST_DENSITY[j_109_145_161-1] + dndfac_109_145_161*ac_transformed_pencil_ad[j_109_145_161-1]*AC_dlnad__mod__dustdensity
                }
                else {
                  DF_DUST_DENSITY[i_109_145_161-1] = DF_DUST_DENSITY[i_109_145_161-1] + dndfac_109_145_161
                  DF_DUST_DENSITY[j_109_145_161-1] = DF_DUST_DENSITY[j_109_145_161-1] + dndfac_109_145_161
                  if (AC_lmomcons2__mod__dustdensity) {
                    DF_DUST_VELOCITY[i_109_145_161-1].z = DF_DUST_VELOCITY[i_109_145_161-1].z - dndfaci_109_145_161*value(F_DUST_VELOCITY[i_109_145_161-1].z)
                    DF_DUST_VELOCITY[j_109_145_161-1].z = DF_DUST_VELOCITY[j_109_145_161-1].z - dndfacj_109_145_161*value(F_DUST_VELOCITY[j_109_145_161-1].z)
                  }
                }
                for k_109_145_161 in j_109_145_161:ndustspec+1 {
                  if (ac_transformed_pencil_md[i_109_145_161-1] + ac_transformed_pencil_md[j_109_145_161-1] >= AC_mdminus__mod__dustvelocity[k_109_145_161-1]  &&  ac_transformed_pencil_md[i_109_145_161-1] + ac_transformed_pencil_md[j_109_145_161-1] < AC_mdplus__mod__dustvelocity[k_109_145_161-1]) {
                    if (AC_lmdvar__mod__cdata) {
                      DF_DUST_DENSITY[k_109_145_161-1] = DF_DUST_DENSITY[k_109_145_161-1] - dndfac_109_145_161
                      dndfac_sum2__mod__dustdensity= dndfac_sum2__mod__dustdensity - dndfac_109_145_161
                      if (!lmdvar_noevolve_109_145_161) {
                        if (ac_transformed_pencil_nd[k_109_145_161-1] < AC_ndmin_for_mdvar__mod__dustdensity) {
                          DF_DUST_MASS[k_109_145_161-1] = ac_transformed_pencil_md[i_109_145_161-1] + ac_transformed_pencil_md[j_109_145_161-1]
                        }
                        else {
                          tmp_109_145_161=max(ac_transformed_pencil_nd[k_109_145_161-1],epsi)
                          DF_DUST_MASS[k_109_145_161-1] = DF_DUST_MASS[k_109_145_161-1] -  (ac_transformed_pencil_md[i_109_145_161-1] + ac_transformed_pencil_md[j_109_145_161-1] - ac_transformed_pencil_md[k_109_145_161-1])*1./tmp_109_145_161*dndfac_109_145_161
                        }
                      }
                      if (AC_lmice__mod__dustdensity) {
                        if (ac_transformed_pencil_nd[k_109_145_161-1] == 0.) {
                          DF_DUST_ICE_MASS[k_109_145_161-1] = ac_transformed_pencil_mi[i_109_145_161-1] + ac_transformed_pencil_mi[j_109_145_161-1]
                        }
                        else {
                          DF_DUST_ICE_MASS[k_109_145_161-1] = DF_DUST_ICE_MASS[k_109_145_161-1] - (ac_transformed_pencil_mi[i_109_145_161-1] + ac_transformed_pencil_mi[j_109_145_161-1] - ac_transformed_pencil_mi[k_109_145_161-1])*  1/ac_transformed_pencil_nd[k_109_145_161-1]*dndfac_109_145_161
                        }
                      }
                      break
                    }
                    else {
                      if (AC_lradius_binning__mod__dustdensity) {
                        DF_DUST_DENSITY[k_109_145_161-1] = DF_DUST_DENSITY[k_109_145_161-1] - dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])/ac_transformed_pencil_md[k_109_145_161-1]  *((ac_transformed_pencil_ad[k_109_145_161-1]/ac_transformed_pencil_ad[j_109_145_161-1])*(ac_transformed_pencil_ad[k_109_145_161-1]/ac_transformed_pencil_ad[j_109_145_161-1]))*ac_transformed_pencil_ad[i_109_145_161-1]*AC_dlnad__mod__dustdensity
                      }
                      else {
                        DF_DUST_DENSITY[k_109_145_161-1] = DF_DUST_DENSITY[k_109_145_161-1] - dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])/ac_transformed_pencil_md[k_109_145_161-1]
                        dndfac_sum2__mod__dustdensity= dndfac_sum2__mod__dustdensity - dndfac_109_145_161
                        if (AC_lmomcons2__mod__dustdensity) {
                          DF_DUST_VELOCITY[k_109_145_161-1].z = DF_DUST_VELOCITY[k_109_145_161-1].z+dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])/ac_transformed_pencil_md[k_109_145_161-1]  *value(F_DUST_VELOCITY[k_109_145_161-1].z)/((ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                        }
                        else if (AC_lmomcons3__mod__dustdensity) {
                          momcons_term_x_109_145_161= -dndfac_sum2__mod__dustdensity*value(F_DUST_VELOCITY[i_109_145_161-1].x)/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].x = DF_DUST_VELOCITY[k_109_145_161-1].x + momcons_term_x_109_145_161
                          momcons_term_y_109_145_161= -dndfac_sum2__mod__dustdensity*value(F_DUST_VELOCITY[i_109_145_161-1].y)/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].y = DF_DUST_VELOCITY[k_109_145_161-1].y + momcons_term_y_109_145_161
                          momcons_term_z_109_145_161= -dndfac_sum2__mod__dustdensity*value(F_DUST_VELOCITY[i_109_145_161-1].z)/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].z = DF_DUST_VELOCITY[k_109_145_161-1].z + momcons_term_z_109_145_161
                          momcons_sum_x__mod__dustdensity=momcons_sum_x__mod__dustdensity+momcons_term_x_109_145_161
                          momcons_sum_y__mod__dustdensity=momcons_sum_y__mod__dustdensity+momcons_term_y_109_145_161
                          momcons_sum_z__mod__dustdensity=momcons_sum_z__mod__dustdensity+momcons_term_z_109_145_161
                        }
                        else if (AC_lmomcons__mod__dustdensity) {
                          momcons_term_x_109_145_161= -dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].x)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].x)  -ac_transformed_pencil_md[k_109_145_161-1]*value(F_DUST_VELOCITY[k_109_145_161-1].x))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].x = DF_DUST_VELOCITY[k_109_145_161-1].x + momcons_term_x_109_145_161
                          momcons_term_y_109_145_161= -dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].y)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].y)  -ac_transformed_pencil_md[k_109_145_161-1]*value(F_DUST_VELOCITY[k_109_145_161-1].y))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].y = DF_DUST_VELOCITY[k_109_145_161-1].y + momcons_term_y_109_145_161
                          if (AC_lmomconsb__mod__dustdensity) {
                            momcons_term_z_109_145_161= -dndfac_109_145_161*(2*ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].z)  +2*ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].z)  -(ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])*value(F_DUST_VELOCITY[k_109_145_161-1].z))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          }
                          else {
                            momcons_term_z_109_145_161= -dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].z)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].z)  -ac_transformed_pencil_md[k_109_145_161-1]*value(F_DUST_VELOCITY[k_109_145_161-1].z))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          }
                          DF_DUST_VELOCITY[k_109_145_161-1].z = DF_DUST_VELOCITY[k_109_145_161-1].z + momcons_term_z_109_145_161 * AC_momcons_term_frac__mod__dustdensity
                          momcons_sum_x__mod__dustdensity=momcons_sum_x__mod__dustdensity+momcons_term_x_109_145_161
                          momcons_sum_y__mod__dustdensity=momcons_sum_y__mod__dustdensity+momcons_term_y_109_145_161
                          momcons_sum_z__mod__dustdensity=momcons_sum_z__mod__dustdensity+momcons_term_z_109_145_161
                        }
                        else if (AC_lmomcons3b__mod__dustdensity) {
                          DF_DUST_VELOCITY[k_109_145_161-1].x = DF_DUST_VELOCITY[k_109_145_161-1].x-dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].x)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].x)  -((ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])*value(F_DUST_VELOCITY[k_109_145_161-1].x)))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].y = DF_DUST_VELOCITY[k_109_145_161-1].y-dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].y)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].y)  -((ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])*value(F_DUST_VELOCITY[k_109_145_161-1].y)))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                          DF_DUST_VELOCITY[k_109_145_161-1].z = DF_DUST_VELOCITY[k_109_145_161-1].z-dndfac_109_145_161*(ac_transformed_pencil_md[i_109_145_161-1]*value(F_DUST_VELOCITY[i_109_145_161-1].z)  +ac_transformed_pencil_md[j_109_145_161-1]*value(F_DUST_VELOCITY[j_109_145_161-1].z)  -((ac_transformed_pencil_md[i_109_145_161-1]+ac_transformed_pencil_md[j_109_145_161-1])*value(F_DUST_VELOCITY[k_109_145_161-1].z)))/  (ac_transformed_pencil_md[k_109_145_161-1]*(ac_transformed_pencil_nd[k_109_145_161-1]+AC_dt__mod__cdata*DF_DUST_DENSITY[k_109_145_161-1]))
                        }
                      }
                      break
                    }
                  }
                }
              }
            }
          }
        }
        if (AC_ldustcondensation__mod__dustvelocity) {
          if (AC_lmdvar__mod__cdata) {
            if(AC_enum_dust_chemistry__mod__dustvelocity == enum_ice_string) {
              mu_113_114_117_145_161=0.0
              if (true) {
                pp_113_114_117_145_161=0.0
              }
              ppmon_113_114_117_145_161 = pp_113_114_117_145_161*cc_tmp_114_117_145_161*mu_113_114_117_145_161/AC_mumon__mod__dustvelocity
              ppsat_113_114_117_145_161 = 6.035e12*exp(-5938*ac_transformed_pencil_tt1)
              vth_113_114_117_145_161 =pow( (3*AC_k_b__mod__cdata/(ac_transformed_pencil_tt1*AC_mmon__mod__dustvelocity)),0.5)
              supsatratio1_113_114_117_145_161 = ppsat_113_114_117_145_161/ppmon_113_114_117_145_161
              mfluxcond_145_161 = vth_113_114_117_145_161*cc_tmp_114_117_145_161*ac_transformed_pencil_rho*(1-supsatratio1_113_114_117_145_161)
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_aerosol_string)   {
              ppmon_113_114_117_145_161=ac_transformed_pencil_pp
              ppsat_113_114_117_145_161 = 6.035e12*exp(-5938*ac_transformed_pencil_tt1)
              vth_113_114_117_145_161 =pow( (3*AC_k_b__mod__cdata/(ac_transformed_pencil_tt1*AC_mmon__mod__dustvelocity)),0.5)
              supsatratio1_113_114_117_145_161 = ppsat_113_114_117_145_161/ppmon_113_114_117_145_161
              mfluxcond_145_161 = vth_113_114_117_145_161*cc_tmp_114_117_145_161*ac_transformed_pencil_rho*(1-supsatratio1_113_114_117_145_161)
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_pscalar_string)   {
              if (AC_lpscalar_nolog__mod__cdata) {
                mfluxcond_145_161=AC_g_condensparam__mod__dustdensity*value(Field(AC_icc__mod__cdata-1))
              }
              else if (lpscalar) {
                mfluxcond_145_161=AC_g_condensparam__mod__dustdensity*exp(value(Field(AC_ilncc__mod__cdata-1)))
              }
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_condensing_species_test_string)   {
              mfluxcond_145_161=AC_g_condensparam__mod__dustdensity
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_condensing_species_string)   {
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_hatzomztz_string)   {
              mfluxcond_145_161=AC_gs_condensparam0__mod__dustdensity+AC_gs_condensparam__mod__dustdensity*tanh(20.*cos(AC_supsatratio_omega__mod__dustdensity*AC_t__mod__cdata))
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_coszomztz_string)   {
              mfluxcond_145_161=AC_gs_condensparam0__mod__dustdensity+AC_gs_condensparam__mod__dustdensity*cos(AC_supsatratio_omega__mod__dustdensity*AC_t__mod__cdata)
            }
            else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_simplified_string)   {
              mfluxcond_145_161=AC_gs_condensparam__mod__dustdensity
            }
            else {
            }
            if (AC_enum_dust_chemistry__mod__dustvelocity==enum_simplified_string) {
              lgh_114_117_145_161=l_114_117_145_161+NGHOST
              for k_114_117_145_161 in 1:ndustspec+1 {
                DF_DUST_MASS[k_114_117_145_161-1] = DF_DUST_MASS[k_114_117_145_161-1] + 4*pi*AC_ad__mod__dustvelocity[k_114_117_145_161-1]*ac_transformed_pencil_rho*mfluxcond_145_161
              }
            }
            else {
              lgh_114_117_145_161=l_114_117_145_161+NGHOST
              for k_114_117_145_161 in 1:ndustspec+1 {
                dmdfac_114_117_145_161 = AC_surfd__mod__dustvelocity[k_114_117_145_161-1]*mfluxcond_145_161/AC_unit_md__mod__dustvelocity
                if (AC_lmice__mod__dustdensity) {
                  if (ac_transformed_pencil_mi[k_114_117_145_161-1] + AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]*dmdfac_114_117_145_161 < 0.) {
                    dmdfac_114_117_145_161 = -ac_transformed_pencil_mi[k_114_117_145_161-1]/AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]
                  }
                }
                if (cc_tmp_114_117_145_161 < 1e-6  &&  dmdfac_114_117_145_161 > 0.) {
                  dmdfac_114_117_145_161=0.
                }
                if (AC_lmice__mod__dustdensity) {
                  DF_DUST_ICE_MASS[k_114_117_145_161-1] = DF_DUST_ICE_MASS[k_114_117_145_161-1] + dmdfac_114_117_145_161
                }
                DF_DUST_MASS[k_114_117_145_161-1] = DF_DUST_MASS[k_114_117_145_161-1] + dmdfac_114_117_145_161
              }
            }
          }
          else {
            if (!AC_lsemi_chemistry__mod__dustdensity) {
              if(AC_enum_dust_chemistry__mod__dustvelocity == enum_ice_string) {
                mu_115_116_117_145_161=0.0
                if (true) {
                  pp_115_116_117_145_161=0.0
                }
                ppmon_115_116_117_145_161 = pp_115_116_117_145_161*cc_tmp_116_117_145_161*mu_115_116_117_145_161/AC_mumon__mod__dustvelocity
                ppsat_115_116_117_145_161 = 6.035e12*exp(-5938*ac_transformed_pencil_tt1)
                vth_115_116_117_145_161 =pow( (3*AC_k_b__mod__cdata/(ac_transformed_pencil_tt1*AC_mmon__mod__dustvelocity)),0.5)
                supsatratio1_115_116_117_145_161 = ppsat_115_116_117_145_161/ppmon_115_116_117_145_161
                mfluxcond_145_161 = vth_115_116_117_145_161*cc_tmp_116_117_145_161*ac_transformed_pencil_rho*(1-supsatratio1_115_116_117_145_161)
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_aerosol_string)   {
                ppmon_115_116_117_145_161=ac_transformed_pencil_pp
                ppsat_115_116_117_145_161 = 6.035e12*exp(-5938*ac_transformed_pencil_tt1)
                vth_115_116_117_145_161 =pow( (3*AC_k_b__mod__cdata/(ac_transformed_pencil_tt1*AC_mmon__mod__dustvelocity)),0.5)
                supsatratio1_115_116_117_145_161 = ppsat_115_116_117_145_161/ppmon_115_116_117_145_161
                mfluxcond_145_161 = vth_115_116_117_145_161*cc_tmp_116_117_145_161*ac_transformed_pencil_rho*(1-supsatratio1_115_116_117_145_161)
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_pscalar_string)   {
                if (AC_lpscalar_nolog__mod__cdata) {
                  mfluxcond_145_161=AC_g_condensparam__mod__dustdensity*value(Field(AC_icc__mod__cdata-1))
                }
                else if (lpscalar) {
                  mfluxcond_145_161=AC_g_condensparam__mod__dustdensity*exp(value(Field(AC_ilncc__mod__cdata-1)))
                }
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_condensing_species_test_string)   {
                mfluxcond_145_161=AC_g_condensparam__mod__dustdensity
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_condensing_species_string)   {
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_hatzomztz_string)   {
                mfluxcond_145_161=AC_gs_condensparam0__mod__dustdensity+AC_gs_condensparam__mod__dustdensity*tanh(20.*cos(AC_supsatratio_omega__mod__dustdensity*AC_t__mod__cdata))
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_coszomztz_string)   {
                mfluxcond_145_161=AC_gs_condensparam0__mod__dustdensity+AC_gs_condensparam__mod__dustdensity*cos(AC_supsatratio_omega__mod__dustdensity*AC_t__mod__cdata)
              }
              else if(AC_enum_dust_chemistry__mod__dustvelocity == enum_simplified_string)   {
                mfluxcond_145_161=AC_gs_condensparam__mod__dustdensity
              }
              else {
              }
              if (AC_lradius_binning__mod__dustdensity) {
                if (AC_lfree_molecule__mod__dustdensity) {
                  if(AC_enum_dust_binning__mod__dustvelocity == enum_lin_radius_string) {
                    for k_116_117_145_161 in 2:ndustspec-1+1 {
                      coefkm_116_117_145_161=mfluxcond_145_161/0.0
                      DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  -coefkm_116_117_145_161*(value(F_DUST_DENSITY[k_116_117_145_161-1])-value(F_DUST_DENSITY[k_116_117_145_161-1-1]))
                    }
                  }
                  else if(AC_enum_dust_binning__mod__dustvelocity == enum_log_radius_string)   {
                    for k_116_117_145_161 in 2:ndustspec-1+1 {
                      coefkm_116_117_145_161=2.*mfluxcond_145_161/(0.0*(AC_ad__mod__dustvelocity[k_116_117_145_161-1]+AC_ad__mod__dustvelocity[k_116_117_145_161-1-1]))
                      DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  -coefkm_116_117_145_161*(value(F_DUST_DENSITY[k_116_117_145_161-1])-value(F_DUST_DENSITY[k_116_117_145_161-1-1]))
                    }
                  }
                }
                else {
                  k_116_117_145_161=1
                  mfluxcondp_116_117_145_161=(abs(mfluxcond_145_161)-mfluxcond_145_161)
                  mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                  coefkp_116_117_145_161=0.5*mfluxcondp_116_117_145_161/(AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]-AC_ad__mod__dustvelocity[k_116_117_145_161-1])
                  coefk0_116_117_145_161=  -mfluxcondm_116_117_145_161/ AC_ad__mod__dustvelocity[k_116_117_145_161-1]-coefkp_116_117_145_161
                  DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  +coefkp_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161+1-1])/AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/AC_ad__mod__dustvelocity[k_116_117_145_161-1]
                  k_116_117_145_161=ndustspec
                  mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                  coefkm_116_117_145_161=0.5*mfluxcondm_116_117_145_161/(AC_ad__mod__dustvelocity[k_116_117_145_161-1]-AC_ad__mod__dustvelocity[k_116_117_145_161-1-1])
                  coefk0_116_117_145_161=  -mfluxcondp_116_117_145_161/ AC_ad__mod__dustvelocity[k_116_117_145_161-1]-coefkm_116_117_145_161
                  DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  +coefkm_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1-1])/AC_ad__mod__dustvelocity[k_116_117_145_161-1-1]  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/AC_ad__mod__dustvelocity[k_116_117_145_161-1]
                  for k_116_117_145_161 in 2:ndustspec-1+1 {
                    mfluxcondp_116_117_145_161=(abs(mfluxcond_145_161)-mfluxcond_145_161)
                    mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                    coefkp_116_117_145_161=+0.5*mfluxcondp_116_117_145_161/(AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]-AC_ad__mod__dustvelocity[k_116_117_145_161-1])
                    coefkm_116_117_145_161=+0.5*mfluxcondm_116_117_145_161/(AC_ad__mod__dustvelocity[k_116_117_145_161-1]-AC_ad__mod__dustvelocity[k_116_117_145_161-1-1])
                    coefk0_116_117_145_161=-(coefkp_116_117_145_161+coefkm_116_117_145_161)
                    DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  +coefkp_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161+1-1])/AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]  +coefkm_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1-1])/AC_ad__mod__dustvelocity[k_116_117_145_161-1-1]  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/AC_ad__mod__dustvelocity[k_116_117_145_161-1]
                  }
                }
              }
              else {
                if (AC_lfree_molecule__mod__dustdensity) {
                  for k_116_117_145_161 in 2:ndustspec-1+1 {
                    coefk0_116_117_145_161=mfluxcond_145_161/0.0
                    DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  -coefk0_116_117_145_161*(value(F_DUST_DENSITY[k_116_117_145_161-1])-value(F_DUST_DENSITY[k_116_117_145_161-1-1]))
                  }
                }
                else {
                  dampfact_116_117_145_161=0.1/AC_dlnmd__mod__dustdensity*3.
                  k_116_117_145_161=1
                  mfluxcondp_116_117_145_161=(abs(mfluxcond_145_161)-mfluxcond_145_161)
                  mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                  coefkp_116_117_145_161=0.5*mfluxcondp_116_117_145_161/AC_dlnmd__mod__dustdensity*3.
                  coefk0_116_117_145_161=  -mfluxcondm_116_117_145_161*dampfact_116_117_145_161-coefkp_116_117_145_161
                  DF_DUST_DENSITY[k_116_117_145_161-1]=DF_DUST_DENSITY[k_116_117_145_161-1]+coefkp_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161+1-1])/(AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]*AC_ad__mod__dustvelocity[1+k_116_117_145_161-1])  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/(AC_ad__mod__dustvelocity[k_116_117_145_161-1]  *AC_ad__mod__dustvelocity[k_116_117_145_161-1]  )
                  k_116_117_145_161=ndustspec
                  mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                  coefkm_116_117_145_161=0.5*mfluxcondm_116_117_145_161/AC_dlnmd__mod__dustdensity*3.
                  if (AC_lzero_upper_kern__mod__dustdensity) {
                    coefk0_116_117_145_161 = 0.
                  }
                  else {
                    coefk0_116_117_145_161 = -mfluxcondp_116_117_145_161*dampfact_116_117_145_161-coefkm_116_117_145_161
                  }
                  DF_DUST_DENSITY[k_116_117_145_161-1]=DF_DUST_DENSITY[k_116_117_145_161-1]+coefkm_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1-1])/(AC_ad__mod__dustvelocity[k_116_117_145_161-1-1]*AC_ad__mod__dustvelocity[k_116_117_145_161-1-1])  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/(AC_ad__mod__dustvelocity[k_116_117_145_161-1]  *AC_ad__mod__dustvelocity[k_116_117_145_161-1]  )
                  for k_116_117_145_161 in 2:ndustspec-1+1 {
                    mfluxcondp_116_117_145_161=(abs(mfluxcond_145_161)-mfluxcond_145_161)
                    mfluxcondm_116_117_145_161=(abs(mfluxcond_145_161)+mfluxcond_145_161)
                    coefkp_116_117_145_161=+0.5*mfluxcondp_116_117_145_161/AC_dlnmd__mod__dustdensity*3.
                    coefkm_116_117_145_161=+0.5*mfluxcondm_116_117_145_161/AC_dlnmd__mod__dustdensity*3.
                    coefk0_116_117_145_161=-(coefkp_116_117_145_161+coefkm_116_117_145_161)
                    DF_DUST_DENSITY[k_116_117_145_161-1] = DF_DUST_DENSITY[k_116_117_145_161-1]  +coefkp_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161+1-1])/(AC_ad__mod__dustvelocity[1+k_116_117_145_161-1]*AC_ad__mod__dustvelocity[1+k_116_117_145_161-1])  +coefkm_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1-1])/(AC_ad__mod__dustvelocity[k_116_117_145_161-1-1]*AC_ad__mod__dustvelocity[k_116_117_145_161-1-1])  +coefk0_116_117_145_161*value(F_DUST_DENSITY[k_116_117_145_161-1])/(AC_ad__mod__dustvelocity[k_116_117_145_161-1]  *AC_ad__mod__dustvelocity[k_116_117_145_161-1]  )
                  }
                }
              }
            }
          }
        }
        if (AC_ldustnucleation__mod__dustdensity) {
          if (ac_transformed_pencil_nucl_rmin>AC_ad__mod__dustvelocity[1-1]  &&  ac_transformed_pencil_nucl_rmin<AC_ad__mod__dustvelocity[ndustspec-1]) {
            if(AC_enum_dust_binning__mod__dustvelocity == enum_lin_radius_string) {
              kk_vec_145_161=max(1,int(1+(ac_transformed_pencil_nucl_rmin-AC_ad__mod__dustvelocity[1-1])/0.0))
            }
            else if(AC_enum_dust_binning__mod__dustvelocity == enum_log_radius_string)   {
              kk_vec_145_161=max(1,int(1+log(ac_transformed_pencil_nucl_rmin/AC_ad__mod__dustvelocity[1-1])/0.0))
            }
            else if(AC_enum_dust_binning__mod__dustvelocity == enum_log_mass_string)   {
            }
            else {
            }
            DF_DUST_DENSITY[i_145_161-1]=DF_DUST_DENSITY[i_145_161-1]+ac_transformed_pencil_nucl_rate/0.0
          }
          else {
            kk_vec_145_161=0
          }
        }
      }
      if (!(AC_latm_chemistry__mod__dustdensity)  &&  dimensionality>0) {
        for k_145_161 in 1:ndustspec+1 {
          fdiffd_145_161=0.0
          diffus_diffnd_145_161=0.0
          diffus_diffnd3_145_161=0.0
          if (AC_ldiffd_simplified__mod__dustdensity) {
            fdiffd_145_161=fdiffd_145_161 + AC_diffnd_ndustspec__mod__dustdensity[k_145_161-1]*ac_transformed_pencil_del2nd[k_145_161 -1]
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd_145_161=diffus_diffnd_145_161+AC_diffnd_ndustspec__mod__dustdensity[k_145_161-1]*dxyz_2__mod__cdata
            }
          }
          if (AC_ldiffd_simpl_anisotropic__mod__dustdensity) {
            if (AC_ldustdensity_log__mod__cdata) {
              d2fdx_120_145_161 = derxx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              d2fdy_120_145_161 = deryy(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              d2fdz_120_145_161 = derzz(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              tmp1_145_161=AC_diffnd_anisotropic__mod__dustdensity.x*d2fdx_120_145_161+AC_diffnd_anisotropic__mod__dustdensity.y*d2fdy_120_145_161+AC_diffnd_anisotropic__mod__dustdensity.z*d2fdz_120_145_161
              if (AC_lcylindrical_coords__mod__cdata && AC_diffnd_anisotropic__mod__dustdensity.x!=0.) {
                tmp_120_145_161 = derx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.x*tmp_120_145_161*AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]
              }
              if (AC_lspherical_coords__mod__cdata) {
                if (AC_diffnd_anisotropic__mod__dustdensity.x!=0.) {
                  tmp_120_145_161 = derx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                  tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.x*2.*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*tmp_120_145_161
                }
                if (AC_diffnd_anisotropic__mod__dustdensity.y!=0.) {
                  tmp_120_145_161 = dery(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                  tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.y*AC_cotth__mod__cdata[AC_m__mod__cdata-1]*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*tmp_120_145_161
                }
              }
              tmp2_145_161=0.
              tmp2_145_161=tmp2_145_161+AC_diffnd_anisotropic__mod__dustdensity.x*(ac_transformed_pencil_glnnd[k_145_161-1].x*ac_transformed_pencil_glnnd[k_145_161-1].x)
              tmp2_145_161=tmp2_145_161+AC_diffnd_anisotropic__mod__dustdensity.y*(ac_transformed_pencil_glnnd[k_145_161-1].y*ac_transformed_pencil_glnnd[k_145_161-1].y)
              tmp2_145_161=tmp2_145_161+AC_diffnd_anisotropic__mod__dustdensity.z*(ac_transformed_pencil_glnnd[k_145_161-1].z*ac_transformed_pencil_glnnd[k_145_161-1].z)
              fdiffd_145_161 = fdiffd_145_161 + tmp1_145_161 + tmp2_145_161
            }
            else {
              d2fdx_122_145_161 = derxx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              d2fdy_122_145_161 = deryy(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              d2fdz_122_145_161 = derzz(Field(AC_ind__mod__cdata[k_145_161-1]-1))
              tmp1_145_161=AC_diffnd_anisotropic__mod__dustdensity.x*d2fdx_122_145_161+AC_diffnd_anisotropic__mod__dustdensity.y*d2fdy_122_145_161+AC_diffnd_anisotropic__mod__dustdensity.z*d2fdz_122_145_161
              if (AC_lcylindrical_coords__mod__cdata && AC_diffnd_anisotropic__mod__dustdensity.x!=0.) {
                tmp_122_145_161 = derx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.x*tmp_122_145_161*AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]
              }
              if (AC_lspherical_coords__mod__cdata) {
                if (AC_diffnd_anisotropic__mod__dustdensity.x!=0.) {
                  tmp_122_145_161 = derx(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                  tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.x*2.*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*tmp_122_145_161
                }
                if (AC_diffnd_anisotropic__mod__dustdensity.y!=0.) {
                  tmp_122_145_161 = dery(Field(AC_ind__mod__cdata[k_145_161-1]-1))
                  tmp1_145_161=tmp1_145_161+AC_diffnd_anisotropic__mod__dustdensity.y*AC_cotth__mod__cdata[AC_m__mod__cdata-1]*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*tmp_122_145_161
                }
              }
              fdiffd_145_161 = fdiffd_145_161 + tmp1_145_161
            }
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd_145_161=diffus_diffnd_145_161 +  (AC_diffnd_anisotropic__mod__dustdensity.x*(dline_1__mod__cdata.x*dline_1__mod__cdata.x) +  AC_diffnd_anisotropic__mod__dustdensity.y*(dline_1__mod__cdata.y*dline_1__mod__cdata.y) +  AC_diffnd_anisotropic__mod__dustdensity.z*(dline_1__mod__cdata.z*dline_1__mod__cdata.z))
            }
          }
          if (AC_ldiffd_dusttogasratio__mod__dustdensity) {
            if (AC_ldustdensity_log__mod__cdata) {
              fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_ndustspec__mod__dustdensity[k_145_161-1]*(ac_transformed_pencil_del2nd[k_145_161 -1] - ac_transformed_pencil_glnndglnrho[k_145_161 -1] - ac_transformed_pencil_del2lnrho)
            }
            else {
              fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_ndustspec__mod__dustdensity[k_145_161-1]*(ac_transformed_pencil_del2nd[k_145_161 -1] - ac_transformed_pencil_gndglnrho[k_145_161 -1] -  ac_transformed_pencil_nd[k_145_161 -1]*ac_transformed_pencil_del2lnrho)
            }
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd_145_161=diffus_diffnd_145_161+AC_diffnd_ndustspec__mod__dustdensity[k_145_161-1]*dxyz_2__mod__cdata
            }
          }
          if (AC_ldiffd_hyper3__mod__dustdensity) {
            if (AC_ldustdensity_log__mod__cdata) {
              fdiffd_145_161 = fdiffd_145_161 + 1/ac_transformed_pencil_nd[k_145_161 -1]*AC_diffnd_hyper3__mod__dustdensity*ac_transformed_pencil_del6nd[k_145_161 -1]
            }
            else {
              fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3__mod__dustdensity*ac_transformed_pencil_del6nd[k_145_161 -1]
            }
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd3_145_161=diffus_diffnd3_145_161+AC_diffnd_hyper3__mod__dustdensity*dxyz_6__mod__cdata
            }
          }
          if (AC_ldiffd_hyper3_polar__mod__dustdensity) {
            tmp1_145_161 = der6x_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3__mod__dustdensity*pi4_1*tmp1_145_161*(dline_1__mod__cdata.x*dline_1__mod__cdata.x)
            tmp1_145_161 = der6y_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3__mod__dustdensity*pi4_1*tmp1_145_161*(dline_1__mod__cdata.y*dline_1__mod__cdata.y)
            tmp1_145_161 = der6z_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3__mod__dustdensity*pi4_1*tmp1_145_161*(dline_1__mod__cdata.z*dline_1__mod__cdata.z)
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd3_145_161=diffus_diffnd3_145_161+AC_diffnd_hyper3__mod__dustdensity*pi4_1*(dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata)
            }
          }
          if (AC_ldiffd_hyper3_mesh__mod__dustdensity) {
            tmp1_145_161 = der6x_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3_mesh__mod__dustdensity*pi5_1/60.*tmp1_145_161*dline_1__mod__cdata.x
            tmp1_145_161 = der6y_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3_mesh__mod__dustdensity*pi5_1/60.*tmp1_145_161*dline_1__mod__cdata.y
            tmp1_145_161 = der6z_ignore_spacing(Field(AC_ind__mod__cdata[k_145_161-1]-1))
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3_mesh__mod__dustdensity*pi5_1/60.*tmp1_145_161*dline_1__mod__cdata.z
            if (AC_lupdate_courant_dt__mod__cdata) {
              advec_hypermesh_nd_145_161=AC_diffnd_hyper3_mesh__mod__dustdensity*pi5_1*sqrt(dxyz_2__mod__cdata)
              advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+(advec_hypermesh_nd_145_161*advec_hypermesh_nd_145_161)
            }
          }
          if (AC_ldiffd_hyper3lnnd__mod__dustdensity) {
            if (AC_ldustdensity_log__mod__cdata) {
              fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_hyper3__mod__dustdensity*ac_transformed_pencil_del6lnnd[k_145_161 -1]
            }
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd3_145_161=diffus_diffnd3_145_161+AC_diffnd_hyper3__mod__dustdensity*dxyz_6__mod__cdata
            }
          }
          if (AC_ldiffd_shock__mod__dustdensity) {
            gshockgnd_145_161 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_gnd[k_145_161-1])
            fdiffd_145_161 = fdiffd_145_161 + AC_diffnd_shock__mod__dustdensity*ac_transformed_pencil_shock*ac_transformed_pencil_del2nd[k_145_161 -1] + AC_diffnd_shock__mod__dustdensity*gshockgnd_145_161
            if (AC_lupdate_courant_dt__mod__cdata) {
              diffus_diffnd_145_161=diffus_diffnd_145_161+AC_diffnd_shock__mod__dustdensity*ac_transformed_pencil_shock*dxyz_2__mod__cdata
            }
          }
          if (AC_lupdate_courant_dt__mod__cdata) {
            maxdiffus__mod__cdata=max(maxdiffus__mod__cdata,diffus_diffnd_145_161)
            maxdiffus3__mod__cdata=max(maxdiffus3__mod__cdata,diffus_diffnd3_145_161)
          }
          if (AC_ldustdensity_log__mod__cdata) {
            DF_DUST_DENSITY[k_145_161-1] = DF_DUST_DENSITY[k_145_161-1] + fdiffd_145_161
          }
          else {
            DF_DUST_DENSITY[k_145_161-1]   = DF_DUST_DENSITY[k_145_161-1]   + fdiffd_145_161
          }
        }
        if (AC_lmdvar__mod__cdata) {
          DF_DUST_MASS = DF_DUST_MASS + AC_diffmd__mod__dustdensity*ac_transformed_pencil_del2md
        }
        if (AC_lmice__mod__dustdensity) {
          DF_DUST_ICE_MASS = DF_DUST_ICE_MASS + AC_diffmi__mod__dustdensity*ac_transformed_pencil_del2mi
        }
      }
      if (!AC_latm_chemistry__mod__dustdensity) {
        if (lborder_profiles) {
          for k_145_161 in 1:ndustspec+1 {
            if(AC_enum_bordernd__mod__dustdensity == enum_zero_string || AC_enum_bordernd__mod__dustdensity == enum_0_string) {
              if (AC_ldustdensity_log__mod__cdata) {
                f_target_126_145_161=0.
              }
              else {
                f_target_126_145_161=1.
              }
            }
            else if(AC_enum_bordernd__mod__dustdensity == enum_initialzcondition_string)   {
            }
          }
        }
        if (AC_ldust_cdtc__mod__dustdensity) {
          reac_dust__mod__cdata=0.
          for k_145_161 in 1:ndustspec+1 {
            reac_dust__mod__cdata=max(reac_dust__mod__cdata,ac_transformed_pencil_nd[k_145_161 -1])
          }
          reac_dust__mod__cdata=reac_dust__mod__cdata*AC_kern_max__mod__dustdensity
        }
      }
      if (AC_ldustcoagulation__mod__dustvelocity  &&  lcoala) {
        if(AC_llast__mod__cdata) {
          lgh_144_145_161=l_144_145_161+NGHOST
          for i_144_145_161 in 1:ndustspec+1 {
            updated_nd_144_145_161[i_144_145_161-1] = ac_transformed_pencil_nd[i_144_145_161-1] + AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]*DF_DUST_DENSITY[i_144_145_161-1]
            updated_rho_144_145_161[i_144_145_161-1] = updated_nd_144_145_161[i_144_145_161-1]*AC_md__mod__dustvelocity[i_144_145_161-1]
            ac_transformed_pencil_old_uud[i_144_145_161-1] = ac_transformed_pencil_uud[i_144_145_161-1]
            ac_transformed_pencil_uud[i_144_145_161-1] = ac_transformed_pencil_uud[i_144_145_161-1] + AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]*DF_DUST_VELOCITY[i_144_145_161-1]
          }
          for i_144_145_161 in 1:ndustspec+1 {
            for j_144_145_161 in i_144_145_161:ndustspec+1 {
              lgh_128_144_145_161=l_144_145_161+NGHOST
              if (AC_lself_collisions__mod__dustdensity) {
                if (i_144_145_161==j_144_145_161) {
                  if(AC_enum_self_collisions__mod__dustdensity == enum_average_string) {
                    fact_128_144_145_161=0.5*AC_self_collision_factor__mod__dustdensity
                    deltavd_drift2_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[j_144_145_161-1]+ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[j_144_145_161-1]+ac_transformed_pencil_uud[i_144_145_161-1]))
                  }
                  else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_string)   {
                    fact_128_144_145_161=AC_self_collision_factor__mod__dustdensity
                    if (i_144_145_161==1) {
                      deltavd_drift2_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                    }
                    else if (i_144_145_161==ndustspec) {
                      deltavd_drift2_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                    }
                    else {
                      fact_128_144_145_161=0.5*AC_self_collision_factor__mod__dustdensity
                      deltavd_drift2a_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                      deltavd_drift2a_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                      deltavd_drift2_128_144_145_161=deltavd_drift2a_128_144_145_161+deltavd_drift2b_128_144_145_161
                    }
                  }
                  else if(AC_enum_self_collisions__mod__dustdensity == enum_neighbor_asymmetric_string)   {
                    fact_128_144_145_161=AC_self_collision_factor__mod__dustdensity
                    if (i_144_145_161==ndustspec) {
                      deltavd_drift2_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[i_144_145_161-1-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                    }
                    else {
                      deltavd_drift2_128_144_145_161 = dot(fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]),fact_128_144_145_161*(ac_transformed_pencil_uud[1+i_144_145_161-1]-ac_transformed_pencil_uud[i_144_145_161-1]))
                    }
                  }
                  else {
                  }
                }
                else {
                  deltavd_drift2_128_144_145_161 = dot(ac_transformed_pencil_uud[i_144_145_161-1]-ac_transformed_pencil_uud[j_144_145_161-1],ac_transformed_pencil_uud[i_144_145_161-1]-ac_transformed_pencil_uud[j_144_145_161-1])
                }
              }
              else {
                deltavd_drift2_128_144_145_161 = dot(ac_transformed_pencil_uud[i_144_145_161-1]-ac_transformed_pencil_uud[j_144_145_161-1],ac_transformed_pencil_uud[i_144_145_161-1]-ac_transformed_pencil_uud[j_144_145_161-1])
              }
              if (AC_ldeltavd_thermal__mod__dustdensity) {
                deltavd_therm_128_144_145_161 = sqrt( 8*AC_k_b__mod__cdata/(pi*ac_transformed_pencil_tt1)*(ac_transformed_pencil_md[i_144_145_161-1]+ac_transformed_pencil_md[j_144_145_161-1])/(ac_transformed_pencil_md[i_144_145_161-1]*ac_transformed_pencil_md[j_144_145_161-1]*AC_unit_md__mod__dustvelocity) )
              }
              else {
                deltavd_therm_128_144_145_161=0.
              }
              if (AC_ldeltavd_turbulent__mod__dustdensity) {
              }
              else if(AC_ldeltavd_turbulent_ormel__mod__dustdensity) {
                t_dyn_39_128_144_145_161 = sqrt(3.*pi/(32.*g_newton_cgs*ac_transformed_pencil_rho))
                nh_39_128_144_145_161 = ac_transformed_pencil_rho/(mu_gas_39_128_144_145_161*mh_39_128_144_145_161)
                cs_39_128_144_145_161 = sqrt(AC_gamma__mod__dustdensity*AC_k_b__mod__cdata*ac_transformed_pencil_tt/(mu_gas_39_128_144_145_161*mh_39_128_144_145_161))
                re_39_128_144_145_161 = 62e6*sqrt(nh_39_128_144_145_161/1e5)*sqrt(ac_transformed_pencil_tt/10.)
                t_eta_39_128_144_145_161 = t_dyn_39_128_144_145_161/sqrt(re_39_128_144_145_161)
                ts_i_39_128_144_145_161 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[i_144_145_161-1]/(ac_transformed_pencil_rho*cs_39_128_144_145_161)
                ts_j_39_128_144_145_161 = sqrt(pi*AC_gamma__mod__dustdensity/8) * AC_rhograin__mod__dustvelocity*AC_ad__mod__dustvelocity[j_144_145_161-1]/(ac_transformed_pencil_rho*cs_39_128_144_145_161)
                ts_1_39_128_144_145_161 = ts_i_39_128_144_145_161
                st_1_39_128_144_145_161    = ts_i_39_128_144_145_161/t_dyn_39_128_144_145_161
                st_2_39_128_144_145_161    = ts_j_39_128_144_145_161/t_dyn_39_128_144_145_161
                if (j_144_145_161 > i_144_145_161) {
                  ts_1_39_128_144_145_161    = ts_j_39_128_144_145_161
                  st_1_39_128_144_145_161    = ts_j_39_128_144_145_161/t_dyn_39_128_144_145_161
                  st_2_39_128_144_145_161    = ts_i_39_128_144_145_161/t_dyn_39_128_144_145_161
                }
                x_st_39_128_144_145_161    = st_2_39_128_144_145_161/st_1_39_128_144_145_161
                beta_st_39_128_144_145_161 = 3.2 - (1. + x_st_39_128_144_145_161) + 2./(1. + x_st_39_128_144_145_161) * (1./2.6 +( x_st_39_128_144_145_161* x_st_39_128_144_145_161* x_st_39_128_144_145_161)/(1.6 + x_st_39_128_144_145_161))
                if (ts_1_39_128_144_145_161 < t_eta_39_128_144_145_161) {
                  if (abs(st_1_39_128_144_145_161 - st_2_39_128_144_145_161) < epsilon(st_1_39_128_144_145_161)) {
                    res_39_128_144_145_161 = 0.
                  }
                  else {
                    res_39_128_144_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_128_144_145_161* cs_39_128_144_145_161) * (st_1_39_128_144_145_161 - st_2_39_128_144_145_161)/(st_1_39_128_144_145_161 + st_2_39_128_144_145_161) * ((st_1_39_128_144_145_161*st_1_39_128_144_145_161)/(st_1_39_128_144_145_161 + 1./sqrt(re_39_128_144_145_161)) +( st_2_39_128_144_145_161* st_2_39_128_144_145_161)/(st_2_39_128_144_145_161 + 1./sqrt(re_39_128_144_145_161)))
                  }
                }
                else if ( (t_eta_39_128_144_145_161 <= ts_1_39_128_144_145_161)  &&  (ts_1_39_128_144_145_161 < t_dyn_39_128_144_145_161) )   {
                  res_39_128_144_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_128_144_145_161* cs_39_128_144_145_161) * beta_st_39_128_144_145_161 * st_1_39_128_144_145_161
                }
                else {
                  res_39_128_144_145_161 = AC_alpha_turb__mod__dustdensity *( cs_39_128_144_145_161* cs_39_128_144_145_161) * (1./(st_1_39_128_144_145_161 + 1.) + 1./(st_2_39_128_144_145_161 + 1.))
                }
                deltavd_turbu_128_144_145_161 = sqrt(res_39_128_144_145_161)
              }
              else {
                deltavd_turbu_128_144_145_161 = 0.
              }
              deltav_144_145_161[i_144_145_161-1][j_144_145_161-1] = sqrt(deltavd_drift2_128_144_145_161+(deltavd_therm_128_144_145_161*deltavd_therm_128_144_145_161)+(deltavd_turbu_128_144_145_161*deltavd_turbu_128_144_145_161)+(AC_deltavd_imposed__mod__dustdensity*AC_deltavd_imposed__mod__dustdensity))
              if (AC_ludstickmax__mod__dustdensity) {
                ust_128_144_145_161 = AC_ustcst__mod__dustvelocity *pow( (AC_ad__mod__dustvelocity[i_144_145_161-1]*AC_ad__mod__dustvelocity[j_144_145_161-1]/(AC_ad__mod__dustvelocity[i_144_145_161-1]+AC_ad__mod__dustvelocity[j_144_145_161-1])),(2/3.)) *pow(  ((ac_transformed_pencil_md[i_144_145_161-1]+ac_transformed_pencil_md[j_144_145_161-1])/(ac_transformed_pencil_md[i_144_145_161-1]*ac_transformed_pencil_md[j_144_145_161-1]*AC_unit_md__mod__dustvelocity)),(1/2.))
                if (deltav_144_145_161[i_144_145_161-1][j_144_145_161-1] > ust_128_144_145_161) {
                  deltav_144_145_161[i_144_145_161-1][j_144_145_161-1] = 0.
                }
              }
              deltav_144_145_161[j_144_145_161-1][i_144_145_161-1] = deltav_144_145_161[i_144_145_161-1][j_144_145_161-1]
            }
          }
          for j_142_143_144_145_161 in 1:ndustspec+1 {
            massbins_142_143_144_145_161[j_142_143_144_145_161-1] = 0.5 * (AC_massgrid__mod__coala[1+j_142_143_144_145_161-1]+AC_massgrid__mod__coala[j_142_143_144_145_161-1])
          }
          eps_gij_142_143_144_145_161 =  AC_rhodust_floor__mod__dustdensity/AC_massgrid__mod__coala[1+ndustspec-1]
          gij_142_143_144_145_161 = eps_gij_142_143_144_145_161
          for j_142_143_144_145_161 in 1:ndustspec+1 {
            if (updated_rho_144_145_161[j_142_143_144_145_161-1] > AC_rhodust_floor__mod__dustdensity) {
              gij_142_143_144_145_161[j_142_143_144_145_161-1] = updated_rho_144_145_161[j_142_143_144_145_161-1]/(AC_massgrid__mod__coala[1+j_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_142_143_144_145_161-1])
            }
          }
          iprogress_142_143_144_145_161=1
          tot_nsub_142_143_144_145_161 = 0
          tot_ndt_142_143_144_145_161 = 0
          coeff_cfl_142_143_144_145_161 = 3e-1
          nsub_142_143_144_145_161 = 0
          ndt_142_143_144_145_161 = 0
          for lp_129_131_141_142_143_144_145_161 in 1:ndustspec+1 {
            for l_129_131_141_142_143_144_145_161 in 1:ndustspec+1 {
              arr_gij_dv_131_141_142_143_144_145_161[lp_129_131_141_142_143_144_145_161-1][l_129_131_141_142_143_144_145_161-1] = gij_142_143_144_145_161[lp_129_131_141_142_143_144_145_161-1]*gij_142_143_144_145_161[l_129_131_141_142_143_144_145_161-1]*deltav_144_145_161[lp_129_131_141_142_143_144_145_161-1][l_129_131_141_142_143_144_145_161-1]
            }
          }
          flux_131_141_142_143_144_145_161 = 0.
          for i_130_131_141_142_143_144_145_161 in 1:ndustspec-1+1 {
            for j_130_131_141_142_143_144_145_161 in 1:ndustspec+1 {
              for k_130_131_141_142_143_144_145_161 in 1:ndustspec+1 {
                flux_131_141_142_143_144_145_161[i_130_131_141_142_143_144_145_161-1] = flux_131_141_142_143_144_145_161[i_130_131_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_130_131_141_142_143_144_145_161-1][j_130_131_141_142_143_144_145_161-1][k_130_131_141_142_143_144_145_161-1]*arr_gij_dv_131_141_142_143_144_145_161[j_130_131_141_142_143_144_145_161-1][k_130_131_141_142_143_144_145_161-1]
              }
            }
          }
          tabdtcfl_131_141_142_143_144_145_161 = 0.
          for j_131_141_142_143_144_145_161 in 1:ndustspec+1 {
            hj_131_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_131_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_131_141_142_143_144_145_161-1]
            if (gij_142_143_144_145_161[j_131_141_142_143_144_145_161-1] > eps_gij_142_143_144_145_161) {
              if (j_131_141_142_143_144_145_161==1) {
                tabdtcfl_131_141_142_143_144_145_161[j_131_141_142_143_144_145_161-1] = abs(gij_142_143_144_145_161[j_131_141_142_143_144_145_161-1]*hj_131_141_142_143_144_145_161/flux_131_141_142_143_144_145_161[j_131_141_142_143_144_145_161-1])
              }
              else {
                tabdtcfl_131_141_142_143_144_145_161[j_131_141_142_143_144_145_161-1] = abs(gij_142_143_144_145_161[j_131_141_142_143_144_145_161-1]*hj_131_141_142_143_144_145_161/(flux_131_141_142_143_144_145_161[j_131_141_142_143_144_145_161-1]-flux_131_141_142_143_144_145_161[j_131_141_142_143_144_145_161-1-1]))
              }
            }
          }
          cfl_return_value_131_141_142_143_144_145_161 = minval_positive(tabdtcfl_131_141_142_143_144_145_161)
          if ( cfl_return_value_131_141_142_143_144_145_161 ==0.) {
          }
          dtcflsub_141_142_143_144_145_161 = cfl_return_value_131_141_142_143_144_145_161
          dtcflsub_141_142_143_144_145_161 = coeff_cfl_142_143_144_145_161*dtcflsub_141_142_143_144_145_161
          dt_141_142_143_144_145_161 = min(dtcflsub_141_142_143_144_145_161,AC_dt__mod__cdata)
          if (dt_141_142_143_144_145_161<AC_dt__mod__cdata) {
            gijsub_in_141_142_143_144_145_161 = gij_142_143_144_145_161
            dtsub_141_142_143_144_145_161 = 0.
            while ( dtsub_141_142_143_144_145_161<AC_dt__mod__cdata  &&  AC_dt__mod__cdata-dtsub_141_142_143_144_145_161>dtcflsub_141_142_143_144_145_161){
              dtsub_141_142_143_144_145_161=dtsub_141_142_143_144_145_161+dtcflsub_141_142_143_144_145_161
              nsub_142_143_144_145_161 = nsub_142_143_144_145_161 + 1
              for lp_132_134_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                for l_132_134_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  arr_gij_dv_134_137_141_142_143_144_145_161[lp_132_134_137_141_142_143_144_145_161-1][l_132_134_137_141_142_143_144_145_161-1] = gijsub_in_141_142_143_144_145_161[lp_132_134_137_141_142_143_144_145_161-1]*gijsub_in_141_142_143_144_145_161[l_132_134_137_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_134_137_141_142_143_144_145_161-1][l_132_134_137_141_142_143_144_145_161-1]
                }
              }
              flux_134_137_141_142_143_144_145_161 = 0.
              for i_133_134_137_141_142_143_144_145_161 in 1:ndustspec-1+1 {
                for j_133_134_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  for k_133_134_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                    flux_134_137_141_142_143_144_145_161[i_133_134_137_141_142_143_144_145_161-1] = flux_134_137_141_142_143_144_145_161[i_133_134_137_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_134_137_141_142_143_144_145_161-1][j_133_134_137_141_142_143_144_145_161-1][k_133_134_137_141_142_143_144_145_161-1]*arr_gij_dv_134_137_141_142_143_144_145_161[j_133_134_137_141_142_143_144_145_161-1][k_133_134_137_141_142_143_144_145_161-1]
                  }
                }
              }
              l_k0_137_141_142_143_144_145_161 = 0.
              l_k0_137_141_142_143_144_145_161[1-1] = -flux_134_137_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
              for j_134_137_141_142_143_144_145_161 in 2:ndustspec+1 {
                hj_134_137_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_134_137_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_134_137_141_142_143_144_145_161-1]
                l_k0_137_141_142_143_144_145_161[j_134_137_141_142_143_144_145_161-1] = -(flux_134_137_141_142_143_144_145_161[j_134_137_141_142_143_144_145_161-1]/hj_134_137_141_142_143_144_145_161 - flux_134_137_141_142_143_144_145_161[j_134_137_141_142_143_144_145_161-1-1]/hj_134_137_141_142_143_144_145_161)
              }
              gij_1_137_141_142_143_144_145_161 = gijsub_in_141_142_143_144_145_161 + dtcflsub_141_142_143_144_145_161*l_k0_137_141_142_143_144_145_161
              for j_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                if ( gij_1_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] < 0.  ) {
                }
                else if ( gij_1_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                  gij_1_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
                }
              }
              for lp_132_135_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                for l_132_135_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  arr_gij_dv_135_137_141_142_143_144_145_161[lp_132_135_137_141_142_143_144_145_161-1][l_132_135_137_141_142_143_144_145_161-1] = gij_1_137_141_142_143_144_145_161[lp_132_135_137_141_142_143_144_145_161-1]*gij_1_137_141_142_143_144_145_161[l_132_135_137_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_135_137_141_142_143_144_145_161-1][l_132_135_137_141_142_143_144_145_161-1]
                }
              }
              flux_135_137_141_142_143_144_145_161 = 0.
              for i_133_135_137_141_142_143_144_145_161 in 1:ndustspec-1+1 {
                for j_133_135_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  for k_133_135_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                    flux_135_137_141_142_143_144_145_161[i_133_135_137_141_142_143_144_145_161-1] = flux_135_137_141_142_143_144_145_161[i_133_135_137_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_135_137_141_142_143_144_145_161-1][j_133_135_137_141_142_143_144_145_161-1][k_133_135_137_141_142_143_144_145_161-1]*arr_gij_dv_135_137_141_142_143_144_145_161[j_133_135_137_141_142_143_144_145_161-1][k_133_135_137_141_142_143_144_145_161-1]
                  }
                }
              }
              l_k0_1_137_141_142_143_144_145_161 = 0.
              l_k0_1_137_141_142_143_144_145_161[1-1] = -flux_135_137_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
              for j_135_137_141_142_143_144_145_161 in 2:ndustspec+1 {
                hj_135_137_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_135_137_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_135_137_141_142_143_144_145_161-1]
                l_k0_1_137_141_142_143_144_145_161[j_135_137_141_142_143_144_145_161-1] = -(flux_135_137_141_142_143_144_145_161[j_135_137_141_142_143_144_145_161-1]/hj_135_137_141_142_143_144_145_161 - flux_135_137_141_142_143_144_145_161[j_135_137_141_142_143_144_145_161-1-1]/hj_135_137_141_142_143_144_145_161)
              }
              gij_2_137_141_142_143_144_145_161 = 3.*gijsub_in_141_142_143_144_145_161/4. + (gij_1_137_141_142_143_144_145_161 + dtcflsub_141_142_143_144_145_161*l_k0_1_137_141_142_143_144_145_161)/4.
              for j_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                if ( gij_2_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] < 0.  ) {
                }
                else if ( gij_2_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                  gij_2_137_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
                }
              }
              for lp_132_136_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                for l_132_136_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  arr_gij_dv_136_137_141_142_143_144_145_161[lp_132_136_137_141_142_143_144_145_161-1][l_132_136_137_141_142_143_144_145_161-1] = gij_2_137_141_142_143_144_145_161[lp_132_136_137_141_142_143_144_145_161-1]*gij_2_137_141_142_143_144_145_161[l_132_136_137_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_136_137_141_142_143_144_145_161-1][l_132_136_137_141_142_143_144_145_161-1]
                }
              }
              flux_136_137_141_142_143_144_145_161 = 0.
              for i_133_136_137_141_142_143_144_145_161 in 1:ndustspec-1+1 {
                for j_133_136_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                  for k_133_136_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                    flux_136_137_141_142_143_144_145_161[i_133_136_137_141_142_143_144_145_161-1] = flux_136_137_141_142_143_144_145_161[i_133_136_137_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_136_137_141_142_143_144_145_161-1][j_133_136_137_141_142_143_144_145_161-1][k_133_136_137_141_142_143_144_145_161-1]*arr_gij_dv_136_137_141_142_143_144_145_161[j_133_136_137_141_142_143_144_145_161-1][k_133_136_137_141_142_143_144_145_161-1]
                  }
                }
              }
              l_k0_2_137_141_142_143_144_145_161 = 0.
              l_k0_2_137_141_142_143_144_145_161[1-1] = -flux_136_137_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
              for j_136_137_141_142_143_144_145_161 in 2:ndustspec+1 {
                hj_136_137_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_136_137_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_136_137_141_142_143_144_145_161-1]
                l_k0_2_137_141_142_143_144_145_161[j_136_137_141_142_143_144_145_161-1] = -(flux_136_137_141_142_143_144_145_161[j_136_137_141_142_143_144_145_161-1]/hj_136_137_141_142_143_144_145_161 - flux_136_137_141_142_143_144_145_161[j_136_137_141_142_143_144_145_161-1-1]/hj_136_137_141_142_143_144_145_161)
              }
              gijsub_out_141_142_143_144_145_161 = gijsub_in_141_142_143_144_145_161/3. + 2.*(gij_2_137_141_142_143_144_145_161 + dtcflsub_141_142_143_144_145_161*l_k0_2_137_141_142_143_144_145_161)/3.
              for j_137_141_142_143_144_145_161 in 1:ndustspec+1 {
                if ( gijsub_out_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] < 0. ) {
                }
                else if ( gijsub_out_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                  gijsub_out_141_142_143_144_145_161[j_137_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
                }
              }
              gijsub_in_141_142_143_144_145_161 = gijsub_out_141_142_143_144_145_161
              for lp_129_138_141_142_143_144_145_161 in 1:ndustspec+1 {
                for l_129_138_141_142_143_144_145_161 in 1:ndustspec+1 {
                  arr_gij_dv_138_141_142_143_144_145_161[lp_129_138_141_142_143_144_145_161-1][l_129_138_141_142_143_144_145_161-1] = gijsub_in_141_142_143_144_145_161[lp_129_138_141_142_143_144_145_161-1]*gijsub_in_141_142_143_144_145_161[l_129_138_141_142_143_144_145_161-1]*deltav_144_145_161[lp_129_138_141_142_143_144_145_161-1][l_129_138_141_142_143_144_145_161-1]
                }
              }
              flux_138_141_142_143_144_145_161 = 0.
              for i_130_138_141_142_143_144_145_161 in 1:ndustspec-1+1 {
                for j_130_138_141_142_143_144_145_161 in 1:ndustspec+1 {
                  for k_130_138_141_142_143_144_145_161 in 1:ndustspec+1 {
                    flux_138_141_142_143_144_145_161[i_130_138_141_142_143_144_145_161-1] = flux_138_141_142_143_144_145_161[i_130_138_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_130_138_141_142_143_144_145_161-1][j_130_138_141_142_143_144_145_161-1][k_130_138_141_142_143_144_145_161-1]*arr_gij_dv_138_141_142_143_144_145_161[j_130_138_141_142_143_144_145_161-1][k_130_138_141_142_143_144_145_161-1]
                  }
                }
              }
              tabdtcfl_138_141_142_143_144_145_161 = 0.
              for j_138_141_142_143_144_145_161 in 1:ndustspec+1 {
                hj_138_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_138_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_138_141_142_143_144_145_161-1]
                if (gijsub_in_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1] > eps_gij_142_143_144_145_161) {
                  if (j_138_141_142_143_144_145_161==1) {
                    tabdtcfl_138_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1] = abs(gijsub_in_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1]*hj_138_141_142_143_144_145_161/flux_138_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1])
                  }
                  else {
                    tabdtcfl_138_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1] = abs(gijsub_in_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1]*hj_138_141_142_143_144_145_161/(flux_138_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1]-flux_138_141_142_143_144_145_161[j_138_141_142_143_144_145_161-1-1]))
                  }
                }
              }
              cfl_return_value_138_141_142_143_144_145_161 = minval_positive(tabdtcfl_138_141_142_143_144_145_161)
              if ( cfl_return_value_138_141_142_143_144_145_161 ==0.) {
              }
              dtcflsub_141_142_143_144_145_161 = cfl_return_value_138_141_142_143_144_145_161
              dtcflsub_141_142_143_144_145_161 = coeff_cfl_142_143_144_145_161*dtcflsub_141_142_143_144_145_161
            }
            dtlast_141_142_143_144_145_161 = AC_dt__mod__cdata-dtsub_141_142_143_144_145_161
            nsub_142_143_144_145_161 = nsub_142_143_144_145_161 + 1
            for lp_132_134_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_134_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_134_139_141_142_143_144_145_161[lp_132_134_139_141_142_143_144_145_161-1][l_132_134_139_141_142_143_144_145_161-1] = gijsub_in_141_142_143_144_145_161[lp_132_134_139_141_142_143_144_145_161-1]*gijsub_in_141_142_143_144_145_161[l_132_134_139_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_134_139_141_142_143_144_145_161-1][l_132_134_139_141_142_143_144_145_161-1]
              }
            }
            flux_134_139_141_142_143_144_145_161 = 0.
            for i_133_134_139_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_134_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_134_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_134_139_141_142_143_144_145_161[i_133_134_139_141_142_143_144_145_161-1] = flux_134_139_141_142_143_144_145_161[i_133_134_139_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_134_139_141_142_143_144_145_161-1][j_133_134_139_141_142_143_144_145_161-1][k_133_134_139_141_142_143_144_145_161-1]*arr_gij_dv_134_139_141_142_143_144_145_161[j_133_134_139_141_142_143_144_145_161-1][k_133_134_139_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_139_141_142_143_144_145_161 = 0.
            l_k0_139_141_142_143_144_145_161[1-1] = -flux_134_139_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_134_139_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_134_139_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_134_139_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_134_139_141_142_143_144_145_161-1]
              l_k0_139_141_142_143_144_145_161[j_134_139_141_142_143_144_145_161-1] = -(flux_134_139_141_142_143_144_145_161[j_134_139_141_142_143_144_145_161-1]/hj_134_139_141_142_143_144_145_161 - flux_134_139_141_142_143_144_145_161[j_134_139_141_142_143_144_145_161-1-1]/hj_134_139_141_142_143_144_145_161)
            }
            gij_1_139_141_142_143_144_145_161 = gijsub_in_141_142_143_144_145_161 + dtlast_141_142_143_144_145_161*l_k0_139_141_142_143_144_145_161
            for j_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gij_1_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] < 0.  ) {
              }
              else if ( gij_1_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gij_1_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
            for lp_132_135_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_135_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_135_139_141_142_143_144_145_161[lp_132_135_139_141_142_143_144_145_161-1][l_132_135_139_141_142_143_144_145_161-1] = gij_1_139_141_142_143_144_145_161[lp_132_135_139_141_142_143_144_145_161-1]*gij_1_139_141_142_143_144_145_161[l_132_135_139_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_135_139_141_142_143_144_145_161-1][l_132_135_139_141_142_143_144_145_161-1]
              }
            }
            flux_135_139_141_142_143_144_145_161 = 0.
            for i_133_135_139_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_135_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_135_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_135_139_141_142_143_144_145_161[i_133_135_139_141_142_143_144_145_161-1] = flux_135_139_141_142_143_144_145_161[i_133_135_139_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_135_139_141_142_143_144_145_161-1][j_133_135_139_141_142_143_144_145_161-1][k_133_135_139_141_142_143_144_145_161-1]*arr_gij_dv_135_139_141_142_143_144_145_161[j_133_135_139_141_142_143_144_145_161-1][k_133_135_139_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_1_139_141_142_143_144_145_161 = 0.
            l_k0_1_139_141_142_143_144_145_161[1-1] = -flux_135_139_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_135_139_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_135_139_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_135_139_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_135_139_141_142_143_144_145_161-1]
              l_k0_1_139_141_142_143_144_145_161[j_135_139_141_142_143_144_145_161-1] = -(flux_135_139_141_142_143_144_145_161[j_135_139_141_142_143_144_145_161-1]/hj_135_139_141_142_143_144_145_161 - flux_135_139_141_142_143_144_145_161[j_135_139_141_142_143_144_145_161-1-1]/hj_135_139_141_142_143_144_145_161)
            }
            gij_2_139_141_142_143_144_145_161 = 3.*gijsub_in_141_142_143_144_145_161/4. + (gij_1_139_141_142_143_144_145_161 + dtlast_141_142_143_144_145_161*l_k0_1_139_141_142_143_144_145_161)/4.
            for j_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gij_2_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] < 0.  ) {
              }
              else if ( gij_2_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gij_2_139_141_142_143_144_145_161[j_139_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
            for lp_132_136_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_136_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_136_139_141_142_143_144_145_161[lp_132_136_139_141_142_143_144_145_161-1][l_132_136_139_141_142_143_144_145_161-1] = gij_2_139_141_142_143_144_145_161[lp_132_136_139_141_142_143_144_145_161-1]*gij_2_139_141_142_143_144_145_161[l_132_136_139_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_136_139_141_142_143_144_145_161-1][l_132_136_139_141_142_143_144_145_161-1]
              }
            }
            flux_136_139_141_142_143_144_145_161 = 0.
            for i_133_136_139_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_136_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_136_139_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_136_139_141_142_143_144_145_161[i_133_136_139_141_142_143_144_145_161-1] = flux_136_139_141_142_143_144_145_161[i_133_136_139_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_136_139_141_142_143_144_145_161-1][j_133_136_139_141_142_143_144_145_161-1][k_133_136_139_141_142_143_144_145_161-1]*arr_gij_dv_136_139_141_142_143_144_145_161[j_133_136_139_141_142_143_144_145_161-1][k_133_136_139_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_2_139_141_142_143_144_145_161 = 0.
            l_k0_2_139_141_142_143_144_145_161[1-1] = -flux_136_139_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_136_139_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_136_139_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_136_139_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_136_139_141_142_143_144_145_161-1]
              l_k0_2_139_141_142_143_144_145_161[j_136_139_141_142_143_144_145_161-1] = -(flux_136_139_141_142_143_144_145_161[j_136_139_141_142_143_144_145_161-1]/hj_136_139_141_142_143_144_145_161 - flux_136_139_141_142_143_144_145_161[j_136_139_141_142_143_144_145_161-1-1]/hj_136_139_141_142_143_144_145_161)
            }
            gijnew_142_143_144_145_161 = gijsub_in_141_142_143_144_145_161/3. + 2.*(gij_2_139_141_142_143_144_145_161 + dtlast_141_142_143_144_145_161*l_k0_2_139_141_142_143_144_145_161)/3.
            for j_139_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gijnew_142_143_144_145_161[j_139_141_142_143_144_145_161-1] < 0. ) {
              }
              else if ( gijnew_142_143_144_145_161[j_139_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gijnew_142_143_144_145_161[j_139_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
          }
          else {
            ndt_142_143_144_145_161 = ndt_142_143_144_145_161 + 1
            for lp_132_134_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_134_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_134_140_141_142_143_144_145_161[lp_132_134_140_141_142_143_144_145_161-1][l_132_134_140_141_142_143_144_145_161-1] = gij_142_143_144_145_161[lp_132_134_140_141_142_143_144_145_161-1]*gij_142_143_144_145_161[l_132_134_140_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_134_140_141_142_143_144_145_161-1][l_132_134_140_141_142_143_144_145_161-1]
              }
            }
            flux_134_140_141_142_143_144_145_161 = 0.
            for i_133_134_140_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_134_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_134_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_134_140_141_142_143_144_145_161[i_133_134_140_141_142_143_144_145_161-1] = flux_134_140_141_142_143_144_145_161[i_133_134_140_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_134_140_141_142_143_144_145_161-1][j_133_134_140_141_142_143_144_145_161-1][k_133_134_140_141_142_143_144_145_161-1]*arr_gij_dv_134_140_141_142_143_144_145_161[j_133_134_140_141_142_143_144_145_161-1][k_133_134_140_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_140_141_142_143_144_145_161 = 0.
            l_k0_140_141_142_143_144_145_161[1-1] = -flux_134_140_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_134_140_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_134_140_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_134_140_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_134_140_141_142_143_144_145_161-1]
              l_k0_140_141_142_143_144_145_161[j_134_140_141_142_143_144_145_161-1] = -(flux_134_140_141_142_143_144_145_161[j_134_140_141_142_143_144_145_161-1]/hj_134_140_141_142_143_144_145_161 - flux_134_140_141_142_143_144_145_161[j_134_140_141_142_143_144_145_161-1-1]/hj_134_140_141_142_143_144_145_161)
            }
            gij_1_140_141_142_143_144_145_161 = gij_142_143_144_145_161 + dt_141_142_143_144_145_161*l_k0_140_141_142_143_144_145_161
            for j_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gij_1_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] < 0.  ) {
              }
              else if ( gij_1_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gij_1_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
            for lp_132_135_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_135_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_135_140_141_142_143_144_145_161[lp_132_135_140_141_142_143_144_145_161-1][l_132_135_140_141_142_143_144_145_161-1] = gij_1_140_141_142_143_144_145_161[lp_132_135_140_141_142_143_144_145_161-1]*gij_1_140_141_142_143_144_145_161[l_132_135_140_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_135_140_141_142_143_144_145_161-1][l_132_135_140_141_142_143_144_145_161-1]
              }
            }
            flux_135_140_141_142_143_144_145_161 = 0.
            for i_133_135_140_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_135_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_135_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_135_140_141_142_143_144_145_161[i_133_135_140_141_142_143_144_145_161-1] = flux_135_140_141_142_143_144_145_161[i_133_135_140_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_135_140_141_142_143_144_145_161-1][j_133_135_140_141_142_143_144_145_161-1][k_133_135_140_141_142_143_144_145_161-1]*arr_gij_dv_135_140_141_142_143_144_145_161[j_133_135_140_141_142_143_144_145_161-1][k_133_135_140_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_1_140_141_142_143_144_145_161 = 0.
            l_k0_1_140_141_142_143_144_145_161[1-1] = -flux_135_140_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_135_140_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_135_140_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_135_140_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_135_140_141_142_143_144_145_161-1]
              l_k0_1_140_141_142_143_144_145_161[j_135_140_141_142_143_144_145_161-1] = -(flux_135_140_141_142_143_144_145_161[j_135_140_141_142_143_144_145_161-1]/hj_135_140_141_142_143_144_145_161 - flux_135_140_141_142_143_144_145_161[j_135_140_141_142_143_144_145_161-1-1]/hj_135_140_141_142_143_144_145_161)
            }
            gij_2_140_141_142_143_144_145_161 = 3.*gij_142_143_144_145_161/4. + (gij_1_140_141_142_143_144_145_161 + dt_141_142_143_144_145_161*l_k0_1_140_141_142_143_144_145_161)/4.
            for j_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gij_2_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] < 0.  ) {
              }
              else if ( gij_2_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gij_2_140_141_142_143_144_145_161[j_140_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
            for lp_132_136_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              for l_132_136_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                arr_gij_dv_136_140_141_142_143_144_145_161[lp_132_136_140_141_142_143_144_145_161-1][l_132_136_140_141_142_143_144_145_161-1] = gij_2_140_141_142_143_144_145_161[lp_132_136_140_141_142_143_144_145_161-1]*gij_2_140_141_142_143_144_145_161[l_132_136_140_141_142_143_144_145_161-1]*deltav_144_145_161[lp_132_136_140_141_142_143_144_145_161-1][l_132_136_140_141_142_143_144_145_161-1]
              }
            }
            flux_136_140_141_142_143_144_145_161 = 0.
            for i_133_136_140_141_142_143_144_145_161 in 1:ndustspec-1+1 {
              for j_133_136_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                for k_133_136_140_141_142_143_144_145_161 in 1:ndustspec+1 {
                  flux_136_140_141_142_143_144_145_161[i_133_136_140_141_142_143_144_145_161-1] = flux_136_140_141_142_143_144_145_161[i_133_136_140_141_142_143_144_145_161-1] + AC_tabflux_coag_k0__mod__coala[i_133_136_140_141_142_143_144_145_161-1][j_133_136_140_141_142_143_144_145_161-1][k_133_136_140_141_142_143_144_145_161-1]*arr_gij_dv_136_140_141_142_143_144_145_161[j_133_136_140_141_142_143_144_145_161-1][k_133_136_140_141_142_143_144_145_161-1]
                }
              }
            }
            l_k0_2_140_141_142_143_144_145_161 = 0.
            l_k0_2_140_141_142_143_144_145_161[1-1] = -flux_136_140_141_142_143_144_145_161[1-1]/(AC_massgrid__mod__coala[2-1]-AC_massgrid__mod__coala[1-1])
            for j_136_140_141_142_143_144_145_161 in 2:ndustspec+1 {
              hj_136_140_141_142_143_144_145_161 = AC_massgrid__mod__coala[1+j_136_140_141_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_136_140_141_142_143_144_145_161-1]
              l_k0_2_140_141_142_143_144_145_161[j_136_140_141_142_143_144_145_161-1] = -(flux_136_140_141_142_143_144_145_161[j_136_140_141_142_143_144_145_161-1]/hj_136_140_141_142_143_144_145_161 - flux_136_140_141_142_143_144_145_161[j_136_140_141_142_143_144_145_161-1-1]/hj_136_140_141_142_143_144_145_161)
            }
            gijnew_142_143_144_145_161 = gij_142_143_144_145_161/3. + 2.*(gij_2_140_141_142_143_144_145_161 + dt_141_142_143_144_145_161*l_k0_2_140_141_142_143_144_145_161)/3.
            for j_140_141_142_143_144_145_161 in 1:ndustspec+1 {
              if ( gijnew_142_143_144_145_161[j_140_141_142_143_144_145_161-1] < 0. ) {
              }
              else if ( gijnew_142_143_144_145_161[j_140_141_142_143_144_145_161-1] <= eps_gij_142_143_144_145_161  )   {
                gijnew_142_143_144_145_161[j_140_141_142_143_144_145_161-1] = eps_gij_142_143_144_145_161
              }
            }
          }
          tot_nsub_142_143_144_145_161 = tot_nsub_142_143_144_145_161 + nsub_142_143_144_145_161
          tot_ndt_142_143_144_145_161 = tot_ndt_142_143_144_145_161 + ndt_142_143_144_145_161
          new_rhod_144_145_161 = 0.
          for j_142_143_144_145_161 in 1:ndustspec+1 {
            new_rhod_144_145_161[j_142_143_144_145_161-1] = max(AC_rhodust_floor__mod__dustdensity,gijnew_142_143_144_145_161[j_142_143_144_145_161-1]*(AC_massgrid__mod__coala[1+j_142_143_144_145_161-1]-AC_massgrid__mod__coala[j_142_143_144_145_161-1]))
          }
          for i_144_145_161 in 1:ndustspec+1 {
            new_nd_144_145_161[i_144_145_161-1] = new_rhod_144_145_161[i_144_145_161-1]/AC_md__mod__dustvelocity[i_144_145_161-1]
            DF_DUST_DENSITY[i_144_145_161-1] = 1/AC_dt_beta_ts__mod__cdata[AC_itsub__mod__cdata-1]*(new_nd_144_145_161[i_144_145_161-1]-ac_transformed_pencil_nd[i_144_145_161-1])
            ac_transformed_pencil_uud[i_144_145_161-1] = ac_transformed_pencil_old_uud[i_144_145_161-1]
          }
        }
      }
    }
  }
  if (lshear) {
    #include  "../shear.h"
  }
  if (AC_lupdate_courant_dt__mod__cdata && (!AC_ldt_paronly__mod__cdata)) {
    advec2__mod__cdata=advec2__mod__cdata+advec_cs2__mod__cdata
    if (lenergy || ldensity || lmagnetic || lradiation || lneutralvelocity || lcosmicray ||   (ltestfield_z && AC_iuutest__mod__cdata>0)) {
      maxadvec__mod__cdata=maxadvec__mod__cdata+sqrt(advec2__mod__cdata)
    }
    if (ldensity || lviscosity || lmagnetic || lenergy || ldustvelocity || ldustdensity) {
      maxadvec__mod__cdata=maxadvec__mod__cdata+sqrt(advec2_hypermesh__mod__cdata)
    }
    dt1_advec_167 = maxadvec__mod__cdata/AC_cdt__mod__cdata
    dt1_diffus_167 = maxdiffus__mod__cdata/AC_cdtv__mod__cdata + maxdiffus2__mod__cdata/AC_cdtv2__mod__cdata + maxdiffus3__mod__cdata/AC_cdtv3__mod__cdata
    dt1_src_167 = maxsrc__mod__cdata/AC_cdtsrc__mod__cdata
    dt1_max_loc = sqrt((dt1_advec_167*dt1_advec_167) +( dt1_diffus_167* dt1_diffus_167) +( dt1_src_167* dt1_src_167))
    if (ldustdensity) {
      dt1_max_loc = max(dt1_max_loc,reac_dust__mod__cdata/AC_cdtc__mod__cdata)
    }
    if (lchemistry  &&  !AC_llsode__mod__cdata) {
      dt1_max_loc = max(dt1_max_loc,AC_reac_chem__mod__cdata[vertexIdx.x-NGHOST_VAL]/AC_cdtc__mod__cdata)
    }
    if (lpolymer) {
      dt1_max_loc = max(dt1_max_loc,1./(AC_trelax_poly__mod__cdata*AC_cdt_poly__mod__cdata))
    }
    if (any_AC(AC_lfreeze_varint__mod__cdata,mcom)) {
      if (AC_lcylinder_in_a_box__mod__cdata || AC_lcylindrical_coords__mod__cdata) {
        if (ac_transformed_pencil_rcyl_mn<=AC_rfreeze_int__mod__cdata) {
          dt1_max_loc=0.
          maxadvec__mod__cdata=0.
          maxdiffus__mod__cdata=0.
          maxdiffus2__mod__cdata=0.
          maxdiffus3__mod__cdata=0.
        }
      }
      else {
        if (ac_transformed_pencil_r_mn<=AC_rfreeze_int__mod__cdata) {
          dt1_max_loc=0.
          maxadvec__mod__cdata=0.
          maxdiffus__mod__cdata=0.
          maxdiffus2__mod__cdata=0.
          maxdiffus3__mod__cdata=0.
        }
      }
    }
    if (any_AC(AC_lfreeze_varext__mod__cdata,mcom)) {
      if (AC_lcylinder_in_a_box__mod__cdata || AC_lcylindrical_coords__mod__cdata) {
        if (ac_transformed_pencil_rcyl_mn>=AC_rfreeze_ext__mod__cdata) {
          dt1_max_loc=0.
          maxadvec__mod__cdata=0.
          maxdiffus__mod__cdata=0.
          maxdiffus2__mod__cdata=0.
          maxdiffus3__mod__cdata=0.
        }
      }
      else {
        if (ac_transformed_pencil_r_mn>=AC_rfreeze_ext__mod__cdata) {
          dt1_max_loc=0.
          maxadvec__mod__cdata=0.
          maxdiffus__mod__cdata=0.
          maxdiffus2__mod__cdata=0.
          maxdiffus3__mod__cdata=0.
        }
      }
    }
    if (any_AC(AC_lfreeze_varsquare__mod__cdata,mcom) && AC_y__mod__cdata[AC_m__mod__cdata-1]>AC_yfreeze_square__mod__cdata) {
      if (AC_x__mod__cdata[vertexIdx.x]>AC_xfreeze_square__mod__cdata) {
        dt1_max_loc=0.
        maxadvec__mod__cdata=0.
        maxdiffus__mod__cdata=0.
        maxdiffus2__mod__cdata=0.
        maxdiffus3__mod__cdata=0.
      }
    }
    dt1_max__mod__cdata=max(dt1_max__mod__cdata,dt1_max_loc)
  }
  if (lanelastic) {
    DF_RHSX   = ac_transformed_pencil_rho*DF_UX
    DF_RHSY = ac_transformed_pencil_rho*DF_UY
    DF_RHSY = ac_transformed_pencil_rho*DF_UZ
    DF_UVEC = df_iuu_pencil + DF_UVEC
  }

