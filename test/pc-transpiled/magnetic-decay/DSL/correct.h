const int NGHOST_VAL = 3
#include "../../../../acc-runtime/stdlib/math"
#include "../../../../acc-runtime/stdlib/derivs.h"
#include "../../../../acc-runtime/stdlib/operators.h"
#include "fieldecs.h"
#define AC_NGHOST__mod__cparam nghost__mod__cparam
//TP: nphis1 and nphis2 don't actually work. simply declared to compile the code
//
#include "var_declares.h"
run_const int AC_lpencil_int__mod__cdata[AC_npencils__mod__cparam]
Kernel twopass_solve_intermediate(){
#include "static_var_declares.h"
#include "df_declares.h"
real ac_transformed_pencil_ma2
real3 ac_transformed_pencil_fpres
real ac_transformed_pencil_tcond
real3 ac_transformed_pencil_sglntt
real ac_transformed_pencil_uglntt
real ac_transformed_pencil_advec_cs2
real ac_transformed_pencil_interdependency
real ac_transformed_pencil_among
real ac_transformed_pencil_by
real ac_transformed_pencil_the
real ac_transformed_pencil_particles_selfgrav
real ac_transformed_pencil_module
real3 ac_transformed_pencil_fcont[AC_n_forcing_cont_max__mod__cparam]
real3 ac_transformed_pencil_fvisc
real ac_transformed_pencil_diffus_total
real ac_transformed_pencil_diffus_total2
real ac_transformed_pencil_diffus_total3
real ac_transformed_pencil_visc_heat
real ac_transformed_pencil_nu
real3 ac_transformed_pencil_gradnu
real ac_transformed_pencil_nu_smag
real3 ac_transformed_pencil_gnu_smag
real ac_transformed_pencil_heat
real ac_transformed_pencil_cool
real ac_transformed_pencil_heatcool
real ac_transformed_pencil_rhod[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_udropav
real ac_transformed_pencil_rhodsum
real3 ac_transformed_pencil_glnrhodsum
real ac_transformed_pencil_divud[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_ood[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_od2[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_oud[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_ud2[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_udgud[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_uud[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_del2ud[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_del6ud[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_graddivud[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_advec_uud
real ac_transformed_pencil_dustvelocity
real ac_transformed_pencil_this
real ac_transformed_pencil_are
real ac_transformed_pencil_specified
real ac_transformed_pencil_ttp
real ac_transformed_pencil_rhop
real3 ac_transformed_pencil_grhop
real ac_transformed_pencil_peh
real ac_transformed_pencil_tauascalar
real ac_transformed_pencil_condensationrate
real ac_transformed_pencil_watermixingratio
real ac_transformed_pencil_particles
real3 ac_transformed_pencil_uun
real ac_transformed_pencil_divun
Matrix ac_transformed_pencil_snij
real ac_transformed_pencil_neutralvelocity
real ac_transformed_pencil_acc
real3 ac_transformed_pencil_gacc
real ac_transformed_pencil_ugacc
real ac_transformed_pencil_del2acc
real ac_transformed_pencil_ssat
real ac_transformed_pencil_ttc
real3 ac_transformed_pencil_gttc
real ac_transformed_pencil_ugttc
real ac_transformed_pencil_del2ttc
real ac_transformed_pencil_pscalar
real3 ac_transformed_pencil_aa
real ac_transformed_pencil_a2
Matrix ac_transformed_pencil_aij
real3 ac_transformed_pencil_bb
real3 ac_transformed_pencil_bbb
real ac_transformed_pencil_ab
real ac_transformed_pencil_ua
real3 ac_transformed_pencil_exa
real3 ac_transformed_pencil_exatotal
real ac_transformed_pencil_aps
real ac_transformed_pencil_b2
real ac_transformed_pencil_b21
real ac_transformed_pencil_bf2
Matrix ac_transformed_pencil_bij
real3 ac_transformed_pencil_del2a
real3 ac_transformed_pencil_graddiva
real3 ac_transformed_pencil_jj
real3 ac_transformed_pencil_jj_ohm
real3 ac_transformed_pencil_curlb
real3 ac_transformed_pencil_e3xa
real3 ac_transformed_pencil_el
real ac_transformed_pencil_e2
Matrix ac_transformed_pencil_bijtilde
Matrix ac_transformed_pencil_bij_cov_corr
real ac_transformed_pencil_j2
real ac_transformed_pencil_jb
real ac_transformed_pencil_va2
real3 ac_transformed_pencil_jxb
real3 ac_transformed_pencil_jxbr
real ac_transformed_pencil_jxbr2
real ac_transformed_pencil_ub
real ac_transformed_pencil_uj
real ac_transformed_pencil_ob
real3 ac_transformed_pencil_uxb
real3 ac_transformed_pencil_uxbb
real ac_transformed_pencil_uxb2
real3 ac_transformed_pencil_uxj
real ac_transformed_pencil_chibp
real ac_transformed_pencil_beta
real ac_transformed_pencil_beta1
real3 ac_transformed_pencil_uga
real3 ac_transformed_pencil_uuadvec_gaa
real ac_transformed_pencil_djuidjbi
real ac_transformed_pencil_jo
real ac_transformed_pencil_stokesi
real ac_transformed_pencil_stokesq
real ac_transformed_pencil_stokesu
real ac_transformed_pencil_stokesq1
real ac_transformed_pencil_stokesu1
real ac_transformed_pencil_ujxb
real3 ac_transformed_pencil_oxuxb
real3 ac_transformed_pencil_jxbxb
real3 ac_transformed_pencil_jxbrxb
real3 ac_transformed_pencil_gb22
real3 ac_transformed_pencil_ugb
real ac_transformed_pencil_ugb22
real3 ac_transformed_pencil_bgu
real3 ac_transformed_pencil_bgb
real3 ac_transformed_pencil_bgbp
real ac_transformed_pencil_ubgbp
real3 ac_transformed_pencil_bdivu
real3 ac_transformed_pencil_glnrhoxb
real3 ac_transformed_pencil_del4a
real3 ac_transformed_pencil_del6a
real3 ac_transformed_pencil_oxj
real ac_transformed_pencil_diva
Matrix ac_transformed_pencil_jij
real ac_transformed_pencil_sj
real ac_transformed_pencil_ss12
real ac_transformed_pencil_d6ab
real ac_transformed_pencil_etava
real ac_transformed_pencil_etaj
real ac_transformed_pencil_etaj2
real ac_transformed_pencil_etajrho
real ac_transformed_pencil_cosjb
real ac_transformed_pencil_jparallel
real ac_transformed_pencil_jperp
real ac_transformed_pencil_cosub
real3 ac_transformed_pencil_bunit
real3 ac_transformed_pencil_hjj
real ac_transformed_pencil_hj2
real ac_transformed_pencil_hjb
real ac_transformed_pencil_coshjb
real ac_transformed_pencil_hjparallel
real ac_transformed_pencil_hjperp
real ac_transformed_pencil_nu_ni1
real ac_transformed_pencil_gamma_a2
real ac_transformed_pencil_clight2
real3 ac_transformed_pencil_gva
real3 ac_transformed_pencil_vmagfric
real3 ac_transformed_pencil_bb_sph
real ac_transformed_pencil_advec_va2
real ac_transformed_pencil_lam
Matrix ac_transformed_pencil_poly
real ac_transformed_pencil_trp
real ac_transformed_pencil_fr
Matrix ac_transformed_pencil_frc
Matrix ac_transformed_pencil_u_dot_gradc
Tensor ac_transformed_pencil_cijk
Matrix ac_transformed_pencil_del2poly
real3 ac_transformed_pencil_div_frc
real3 ac_transformed_pencil_divc
real3 ac_transformed_pencil_grad_fr
real ac_transformed_pencil_divu
real3 ac_transformed_pencil_oo
real ac_transformed_pencil_o2
real ac_transformed_pencil_ou
real ac_transformed_pencil_oxu2
real3 ac_transformed_pencil_oxu
real ac_transformed_pencil_u2
Matrix ac_transformed_pencil_uij
real3 ac_transformed_pencil_uu
real3 ac_transformed_pencil_curlo
Matrix ac_transformed_pencil_sij
real ac_transformed_pencil_sij2
Matrix ac_transformed_pencil_uij5
real3 ac_transformed_pencil_ugu
real ac_transformed_pencil_ugu2
Matrix ac_transformed_pencil_oij
Matrix ac_transformed_pencil_d2uidxj
Tensor ac_transformed_pencil_uijk
real3 ac_transformed_pencil_ogu
real ac_transformed_pencil_u3u21
real ac_transformed_pencil_u1u32
real ac_transformed_pencil_u2u13
real3 ac_transformed_pencil_del2u
real3 ac_transformed_pencil_del4u
real3 ac_transformed_pencil_del6u
real ac_transformed_pencil_u2u31
real ac_transformed_pencil_u3u12
real ac_transformed_pencil_u1u23
real3 ac_transformed_pencil_graddivu
real3 ac_transformed_pencil_del6u_bulk
real3 ac_transformed_pencil_grad5divu
real3 ac_transformed_pencil_rhougu
real3 ac_transformed_pencil_der6u
real3 ac_transformed_pencil_transpurho
real ac_transformed_pencil_divu0
Matrix ac_transformed_pencil_u0ij
real3 ac_transformed_pencil_uu0
real3 ac_transformed_pencil_uu_advec
real3 ac_transformed_pencil_uuadvec_guu
real3 ac_transformed_pencil_del6u_strict
real3 ac_transformed_pencil_del4graddivu
real3 ac_transformed_pencil_uu_sph
Matrix ac_transformed_pencil_der6u_res
real ac_transformed_pencil_lorentz
real ac_transformed_pencil_hless
real ac_transformed_pencil_advec_uu
real ac_transformed_pencil_x_mn
real ac_transformed_pencil_y_mn
real ac_transformed_pencil_z_mn
real ac_transformed_pencil_r_mn
real ac_transformed_pencil_r_mn1
real ac_transformed_pencil_phix
real ac_transformed_pencil_phiy
real ac_transformed_pencil_pomx
real ac_transformed_pencil_pomy
real ac_transformed_pencil_rcyl_mn
real ac_transformed_pencil_rcyl_mn1
real ac_transformed_pencil_phi_mn
real3 ac_transformed_pencil_evr
real3 ac_transformed_pencil_rr
real3 ac_transformed_pencil_evth
real ac_transformed_pencil_pointmasses
real ac_transformed_pencil_shock
real3 ac_transformed_pencil_gshock
real ac_transformed_pencil_shock_perp
real3 ac_transformed_pencil_gshock_perp
real ac_transformed_pencil_ywater
real ac_transformed_pencil_lambda
real ac_transformed_pencil_chem_conc[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_nucl_rmin
real ac_transformed_pencil_nucl_rate
real ac_transformed_pencil_conc_satm
real ac_transformed_pencil_ff_cond
real ac_transformed_pencil_cosmicray
real3 ac_transformed_pencil_sgs_force
real ac_transformed_pencil_sgs_heat
real3 ac_transformed_pencil_glnnd[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_gmi[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_gmd[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_gnd[AC_ndustspec__mod__cparam]
real3 ac_transformed_pencil_grhod[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_ad[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_md[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_mi[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_nd[AC_ndustspec__mod__cparam]
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
real ac_transformed_pencil_fcloud
real ac_transformed_pencil_ccondens
real ac_transformed_pencil_ppwater
real ac_transformed_pencil_ppsat
real ac_transformed_pencil_ppsf[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_mu1
real3 ac_transformed_pencil_glnrhod[AC_ndustspec__mod__cparam]
real ac_transformed_pencil_rhodsum1
real3 ac_transformed_pencil_grhodsum
real ac_transformed_pencil_dustdensity
real ac_transformed_pencil_radiation
real ac_transformed_pencil_cv
real ac_transformed_pencil_cv1
real ac_transformed_pencil_cp
real ac_transformed_pencil_cp1
real3 ac_transformed_pencil_glncp
real ac_transformed_pencil_dydt_reac[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_dydt_diff[AC_nchemspec__mod__cparam]
real3 ac_transformed_pencil_glambda
real ac_transformed_pencil_diff_penc_add[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_h0_rt[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_hhk_full[AC_nchemspec__mod__cparam]
real3 ac_transformed_pencil_ghhk[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_s0_r[AC_nchemspec__mod__cparam]
real ac_transformed_pencil_neutraldensity
real ac_transformed_pencil_lnrhon
real ac_transformed_pencil_rhon
real ac_transformed_pencil_rhon1
real3 ac_transformed_pencil_glnrhon
real3 ac_transformed_pencil_grhon
real ac_transformed_pencil_unglnrhon
real ac_transformed_pencil_ungrhon
real ac_transformed_pencil_del2rhon
real ac_transformed_pencil_glnrhon2
real ac_transformed_pencil_del6lnrhon
real ac_transformed_pencil_del6rhon
real3 ac_transformed_pencil_snglnrhon
real ac_transformed_pencil_alpha
real ac_transformed_pencil_zeta
real3 ac_transformed_pencil_gg
real ac_transformed_pencil_ss
real3 ac_transformed_pencil_gss
real ac_transformed_pencil_ee
real ac_transformed_pencil_pp
real ac_transformed_pencil_lntt
real ac_transformed_pencil_cs2
real ac_transformed_pencil_cp1tilde
real3 ac_transformed_pencil_glntt
real ac_transformed_pencil_tt
real ac_transformed_pencil_tt1
real3 ac_transformed_pencil_gtt
real ac_transformed_pencil_yh
Matrix ac_transformed_pencil_hss
Matrix ac_transformed_pencil_hlntt
real ac_transformed_pencil_del2ss
real ac_transformed_pencil_del6ss
real ac_transformed_pencil_del2lntt
real ac_transformed_pencil_del6lntt
real ac_transformed_pencil_gamma
real ac_transformed_pencil_del2tt
real ac_transformed_pencil_del6tt
real3 ac_transformed_pencil_glnmumol
real ac_transformed_pencil_ppvap
real ac_transformed_pencil_csvap2
real ac_transformed_pencil_ttb
real ac_transformed_pencil_rho_anel
real ac_transformed_pencil_eth
real3 ac_transformed_pencil_geth
real ac_transformed_pencil_del2eth
Matrix ac_transformed_pencil_heth
real ac_transformed_pencil_eths
real3 ac_transformed_pencil_geths
real3 ac_transformed_pencil_rho1gpp
real3 ac_transformed_pencil_fsgs
real ac_transformed_pencil_chiral
real3 ac_transformed_pencil_bbf
real ac_transformed_pencil_visc_heatn
real ac_transformed_pencil_un2
Matrix ac_transformed_pencil_unij
real ac_transformed_pencil_snij2
real3 ac_transformed_pencil_ungun
real3 ac_transformed_pencil_del2un
real3 ac_transformed_pencil_del6un
real3 ac_transformed_pencil_graddivun
real ac_transformed_pencil_advec_uun
real ac_transformed_pencil_advec_csn2
real ac_transformed_pencil_lnrho
real ac_transformed_pencil_rho
real ac_transformed_pencil_rho1
real3 ac_transformed_pencil_glnrho
real3 ac_transformed_pencil_grho
real ac_transformed_pencil_uglnrho
real ac_transformed_pencil_ugrho
real ac_transformed_pencil_glnrho2
real ac_transformed_pencil_del2lnrho
real ac_transformed_pencil_del2rho
real ac_transformed_pencil_del6lnrho
real ac_transformed_pencil_del6rho
Matrix ac_transformed_pencil_hlnrho
real3 ac_transformed_pencil_sglnrho
real3 ac_transformed_pencil_uij5glnrho
real ac_transformed_pencil_transprho
real ac_transformed_pencil_ekin
real ac_transformed_pencil_uuadvec_glnrho
real ac_transformed_pencil_uuadvec_grho
real ac_transformed_pencil_rhos1
real3 ac_transformed_pencil_glnrhos
real ac_transformed_pencil_totenergy_rel
real ac_transformed_pencil_divss
real3 ac_transformed_pencil_mf_emf
real ac_transformed_pencil_mf_emfdotb
real3 ac_transformed_pencil_jxb_mf
real3 ac_transformed_pencil_jxbr_mf
real ac_transformed_pencil_chib_mf
real ac_transformed_pencil_mf_qp
real ac_transformed_pencil_mf_beq21
real3 df_iuu_pencil
bool lcommunicate
real tmp_4_14_15_97
real tmp2_4_14_15_97
real3 ugu0_4_14_15_97
real3 u0gu_4_14_15_97
int i_4_14_15_97
int j_4_14_15_97
int ju_4_14_15_97
real tmp_13_14_15_97
real dd_13_14_15_97
real tmp_rho_13_14_15_97
real3 tmp3_13_14_15_97
Matrix tmp33_13_14_15_97
real cs201_13_14_15_97
real outest_13_14_15_97
int i_13_14_15_97
int j_13_14_15_97
int ju_13_14_15_97
int jj_13_14_15_97
int kk_13_14_15_97
int jk_13_14_15_97
int i_4_49_50_13_14_15_97
int j_4_49_50_13_14_15_97
real bx_5_13_14_15_97
real by_5_13_14_15_97
real bz_5_13_14_15_97
real bx2_5_13_14_15_97
real by2_5_13_14_15_97
real bz2_5_13_14_15_97
real detm1_5_13_14_15_97
real tmp_20_24_25_97
int i_20_24_25_97
real tmp_16_20_24_25_97
int i_16_20_24_25_97
int j_16_20_24_25_97
int i_23_24_25_97
real tmp_21_23_24_25_97
int i_21_23_24_25_97
int j_21_23_24_25_97
real tmp_28_29_97
int i_28_29_97
int j_28_29_97
int j_32_97
real3 tmp_63_97
real3 tmp2_63_97
real3 gradnu_63_97
real3 sgradnu_63_97
real3 gradnu_shock_63_97
real murho1_63_97
real zetarho1_63_97
real mutt_63_97
real tmp3_63_97
real tmp4_63_97
real pnu_shock_63_97
real lambda_phi_63_97
real prof_63_97
real prof2_63_97
real derprof_63_97
real derprof2_63_97
real gradnu_effective_63_97
real fac_63_97
real advec_hypermesh_uu_63_97
real3 deljskl2_63_97
real3 fvisc_nnewton2_63_97
Matrix d_sld_flux_63_97
int i_63_97
int j_63_97
int ju_63_97
int ii_63_97
int jj_63_97
int kk_63_97
int ll_63_97
bool ldiffus_total_63_97
bool ldiffus_total3_63_97
real step_vector_return_value_33_35_63_97
real step_vector_return_value_33_35_63_97
real arg_34_35_63_97
real der_step_return_value_34_35_63_97
real step_vector_return_value_36_63_97
real step_vector_return_value_36_63_97
real step_vector_return_value_37_63_97
real step_vector_return_value_37_63_97
real arg_38_63_97
real der_step_return_value_38_63_97
real arg_39_63_97
real der_step_return_value_39_63_97
real step_vector_return_value_41_63_97
real step_vector_return_value_41_63_97
real step_vector_return_value_42_63_97
real step_vector_return_value_42_63_97
real arg_43_63_97
real der_step_return_value_43_63_97
real arg_44_63_97
real der_step_return_value_44_63_97
real step_vector_return_value_46_63_97
real step_vector_return_value_46_63_97
real step_vector_return_value_47_63_97
real step_vector_return_value_47_63_97
real arg_48_63_97
real der_step_return_value_48_63_97
real arg_49_63_97
real der_step_return_value_49_63_97
real step_vector_return_value_50_63_97
real step_vector_return_value_50_63_97
real arg_51_63_97
real der_step_return_value_51_63_97
int j_52_63_97
real step_vector_return_value_53_63_97
real step_vector_return_value_53_63_97
real arg_54_63_97
real der_step_return_value_54_63_97
int j_55_63_97
real step_vector_return_value_56_63_97
real step_vector_return_value_56_63_97
real arg_57_63_97
real der_step_return_value_57_63_97
int j_58_63_97
real tmp_59_63_97
int i_59_63_97
int k1_59_63_97
real tmp_60_63_97
int i_60_63_97
int k1_60_63_97
real lomega_61_63_97
real dlomega_dr_61_63_97
real dlomega_dtheta_61_63_97
real lver_61_63_97
real lhor_61_63_97
real dlver_dr_61_63_97
real dlhor_dtheta_61_63_97
real lomega_62_63_97
real dlomega_dr_62_63_97
real lver_62_63_97
real dlver_dr_62_63_97
int i_65_97
int j_65_97
real3 tmp_78_79_97
real rho1_jxb_78_79_97
real quench_78_79_97
real stokesi_ncr_78_79_97
real tmp1_78_79_97
real bbgb_78_79_97
real va2max_beta_78_79_97
real3 b_ext_78_79_97
real3 j_ext_78_79_97
real c_78_79_97
real s_78_79_97
int i_78_79_97
int j_78_79_97
int ix_78_79_97
real c_71_78_79_97
real s_71_78_79_97
real zprof_71_78_79_97
real zder_71_78_79_97
real zpostop_71_78_79_97
real zposbot_71_78_79_97
real step_scalar_return_value_67_71_78_79_97
real step_scalar_return_value_68_71_78_79_97
real arg_69_71_78_79_97
real der_step_return_value_69_71_78_79_97
real arg_70_71_78_79_97
real der_step_return_value_70_71_78_79_97
real c_72_78_79_97
real s_72_78_79_97
real zprof_72_78_79_97
real zder_72_78_79_97
real zpostop_72_78_79_97
real zposbot_72_78_79_97
real step_scalar_return_value_67_72_78_79_97
real step_scalar_return_value_68_72_78_79_97
real arg_69_72_78_79_97
real der_step_return_value_69_72_78_79_97
real arg_70_72_78_79_97
real der_step_return_value_70_72_78_79_97
real chi_diamag_75_78_79_97
real3 gchi_diamag_75_78_79_97
real3 bk_bki_75_78_79_97
real3 jj_diamag_75_78_79_97
real3 tmp_75_78_79_97
int j_74_75_78_79_97
real3 uu1_128
real3 tmpv_128
real tmp_128
real ftot_128
real ugu_schur_x_128
real ugu_schur_y_128
real ugu_schur_z_128
Matrix puij_schur_128
int i_128
int j_128
int ju_128
real c2_102_128
real s2_102_128
real c2_103_128
real s2_103_128
real om2_103_128
real cp2_103_128
real cs2_103_128
real ss2_103_128
int i_104_128
int j_104_128
int k_104_128
real3 cent_res_104_128
real3 cori_res_104_128
real c2_105_128
real s2_105_128
real c2_106_128
real s2_106_128
real c1_107_128
real c2_107_128
int i_111_128
real reshock_110_111_128
real fvisc2_110_111_128
real uus_110_111_128
real tmp_110_111_128
real qfvisc_110_111_128
real3 nud2uxb_110_111_128
real3 fluxv_110_111_128
int i_109_110_111_128
int j_109_110_111_128
real pdamp_119_128
real fint_work_119_128
real fext_work_119_128
int i_119_128
int j_119_128
real step_vector_return_value_113_119_128
real step_vector_return_value_113_119_128
real step_vector_return_value_114_119_128
real step_vector_return_value_114_119_128
real step_vector_return_value_115_119_128
real step_vector_return_value_115_119_128
real step_vector_return_value_116_119_128
real step_vector_return_value_116_119_128
real step_vector_return_value_117_119_128
real step_vector_return_value_117_119_128
real step_vector_return_value_118_119_128
real step_vector_return_value_118_119_128
real local_omega_120_128
real neg_velocity_ceiling_121_128
int j_121_128
real3 f_target_126_128
int j_126_128
int ju_126_128
real fdiff_140
real tmp_140
real3 tmpv_140
real density_rhs_140
real advec_hypermesh_rho_140
int j_140
bool ldt_up_140
int i_130_140
real fran_133_140[2]
real tmp_133_140
real dlnrhodt_133_140
real pdamp_133_140
real fprofile_133_140
real radius2_133_140
real step_vector_return_value_131_133_140
real step_vector_return_value_131_133_140
real step_vector_return_value_132_133_140
real step_vector_return_value_132_133_140
real f_target_139_140
int j_141
real3 ujiaj_190
real3 gua_190
real3 ajiuj_190
real3 aa_xyaver_190
real3 geta_190
real3 uxb_upw_190
real3 tmp2_190
real3 dadt_190
real3 gradeta_shock_190
real3 aa1_190
real3 uu1_190
Matrix d_sld_flux_190
real ftot_190
real datot_190
real peta_shock_190
real sign_jo_190
real tmp1_190
real eta_mn_190
real etass_190
real eta_heat_190
real vdrift_190
real va2max_beta_190
real del2aa_ini_190
real tanhx2_190
real advec_hall_190
real advec_hypermesh_aa_190
real eta_bb_190
real prof_190
real3 b_ext_190
real tmp_190
real eta_out1_190
real cosalp_190
real sinalp_190
real hall_term__190
const omegass_190 = 1.0
int i_190
int j_190
int k_190
int ju_190
int ix_190
int nphi_190
const nxy_190 = AC_nxgrid__mod__cparam*AC_nygrid__mod__cparam
real c_152_190
real s_152_190
real zprof_152_190
real zder_152_190
real zpostop_152_190
real zposbot_152_190
real step_scalar_return_value_67_152_190
real step_scalar_return_value_68_152_190
real arg_69_152_190
real der_step_return_value_69_152_190
real arg_70_152_190
real der_step_return_value_70_152_190
real step_vector_return_value_153_190
real step_vector_return_value_153_190
int l_164_190
real tmp1_164_190
real tmp2_164_190
real prof0_164_190
real prof1_164_190
real derprof0_164_190
real derprof1_164_190
real step_vector_return_value_154_164_190
real step_vector_return_value_154_164_190
real arg_155_164_190
real der_step_return_value_155_164_190
real step_vector_return_value_156_164_190
real step_vector_return_value_156_164_190
real step_vector_return_value_157_164_190
real step_vector_return_value_157_164_190
real arg_158_164_190
real der_step_return_value_158_164_190
real arg_159_164_190
real der_step_return_value_159_164_190
real step_vector_return_value_160_164_190
real step_vector_return_value_160_164_190
real step_vector_return_value_161_164_190
real step_vector_return_value_161_164_190
real arg_162_164_190
real der_step_return_value_162_164_190
real arg_163_164_190
real der_step_return_value_163_164_190
real tmp_165_190
int i_165_190
int k1_165_190
real prof_174_190
real eta_r_174_190
real d_int_174_190
real d_ext_174_190
real step_vector_return_value_166_174_190
real step_vector_return_value_166_174_190
real step_vector_return_value_167_174_190
real step_vector_return_value_167_174_190
real arg_168_174_190
real der_step_return_value_168_174_190
real arg_169_174_190
real der_step_return_value_169_174_190
real step_vector_return_value_170_174_190
real step_vector_return_value_170_174_190
real step_vector_return_value_171_174_190
real step_vector_return_value_171_174_190
real arg_172_174_190
real der_step_return_value_172_174_190
real arg_173_174_190
real der_step_return_value_173_174_190
real step_vector_return_value_175_190
real step_vector_return_value_175_190
real arg_176_190
real der_step_return_value_176_190
real step_vector_return_value_177_190
real step_vector_return_value_177_190
real arg_178_190
real der_step_return_value_178_190
real cubic_step_pt_return_value_180_190
real xi_180_190
real relshift_180_190
real cubic_step_pt_return_value_180_190
real del6f_upwind_181_190
int msk_181_190
real dumerfc_182_190
real t_182_190
real z_182_190
real erfunc_return_value_182_190
real scl_184_190
int j_184_190
real3 f_target_188_190
int ju_188_190
int j_188_190
if (AC_lanelastic__mod__cparam) {
df_iuu_pencil = DF_UVEC
DF_UVEC=0.0
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata && (!AC_ldt_paronly__mod__cdata)) {
advec_cs2__mod__cdata=0.0
maxadvec__mod__cdata=0.
if (AC_lenergy__mod__cparam || AC_ldensity__mod__cparam || AC_lmagnetic__mod__cparam || AC_lradiation__mod__cparam || AC_lneutralvelocity__mod__cparam || AC_lcosmicray__mod__cparam ||   (AC_ltestfield_z__mod__cparam && AC_iuutest__mod__cdata>0)) {
advec2__mod__cdata=0.
}
if (AC_ldensity__mod__cparam || AC_lviscosity__mod__cparam || AC_lmagnetic__mod__cparam || AC_lenergy__mod__cparam || AC_ldustvelocity__mod__cparam || AC_ldustdensity__mod__cparam) {
advec2_hypermesh__mod__cdata=0.0
}
maxdiffus__mod__cdata=0.
maxdiffus2__mod__cdata=0.
maxdiffus3__mod__cdata=0.
maxsrc__mod__cdata=0.
}
if (AC_lspherical_coords__mod__cdata) {
dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
dline_1__mod__cdata.y = AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
dline_1__mod__cdata.z = AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_sin1th__mod__cdata[AC_m__mod__cdata-1] * AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
}
else if (AC_lcylindrical_coords__mod__cdata) {
dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
dline_1__mod__cdata.y = AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL] * AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
}
else if (AC_lcartesian_coords__mod__cdata) {
dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
dline_1__mod__cdata.y = AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
}
else if (AC_lpipe_coords__mod__cdata) {
dline_1__mod__cdata.x = AC_dx_1__mod__cdata[vertexIdx.x]
dline_1__mod__cdata.y = AC_dy_1__mod__cdata[AC_m__mod__cdata-1]
dline_1__mod__cdata.z = AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
}
dxmax_pencil__mod__cdata = 0.
if (AC_nxgrid__mod__cparam != 1) {
dxmax_pencil__mod__cdata =     1.0 / dline_1__mod__cdata.x
}
if (AC_nygrid__mod__cparam != 1) {
dxmax_pencil__mod__cdata = max(1.0 / dline_1__mod__cdata.y, dxmax_pencil__mod__cdata)
}
if (AC_nzgrid__mod__cparam != 1) {
dxmax_pencil__mod__cdata = max(1.0 / dline_1__mod__cdata.z, dxmax_pencil__mod__cdata)
}
dxmin_pencil__mod__cdata = 0.
if (AC_nxgrid__mod__cparam != 1) {
dxmin_pencil__mod__cdata =     1.0 / dline_1__mod__cdata.x
}
if (AC_nygrid__mod__cparam != 1) {
dxmin_pencil__mod__cdata = min(1.0 / dline_1__mod__cdata.y, dxmin_pencil__mod__cdata)
}
if (AC_nzgrid__mod__cparam != 1) {
dxmin_pencil__mod__cdata = min(1.0 / dline_1__mod__cdata.z, dxmin_pencil__mod__cdata)
}
if (AC_lmaximal_cdtv__mod__cdata) {
dxyz_2__mod__cdata = max(dline_1__mod__cdata.x*dline_1__mod__cdata.x, dline_1__mod__cdata.y*dline_1__mod__cdata.y, dline_1__mod__cdata.z*dline_1__mod__cdata.z)
dxyz_4__mod__cdata = max(dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x, dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y, dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)
dxyz_6__mod__cdata = max(dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x, dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y, dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)
}
else {
dxyz_2__mod__cdata = dline_1__mod__cdata.x*dline_1__mod__cdata.x + dline_1__mod__cdata.y*dline_1__mod__cdata.y + dline_1__mod__cdata.z*dline_1__mod__cdata.z
dxyz_4__mod__cdata = dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x + dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y + dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z
dxyz_6__mod__cdata = dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x + dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y + dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z
}
dvol__mod__cdata = AC_dvol_x__mod__cdata[vertexIdx.x]*AC_dvol_y__mod__cdata[AC_m__mod__cdata-1]*AC_dvol_z__mod__cdata[AC_n__mod__cdata-1]
if (AC_lcartesian_coords__mod__cdata) {
if (AC_lpencil_int__mod__cdata[AC_i_x_mn__mod__cparam-1]) {
ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]
}
if (AC_lpencil_int__mod__cdata[AC_i_y_mn__mod__cparam-1]) {
ac_transformed_pencil_y_mn    = AC_y__mod__cdata[AC_m__mod__cdata-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_z_mn__mod__cparam-1]) {
ac_transformed_pencil_z_mn    = AC_z__mod__cdata[AC_n__mod__cdata-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn__mod__cparam-1]) {
ac_transformed_pencil_r_mn    = sqrt(AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x]+AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_y__mod__cdata[AC_m__mod__cdata-1]+AC_z__mod__cdata[AC_n__mod__cdata-1]*AC_z__mod__cdata[AC_n__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn = sqrt(AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x]+AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_y__mod__cdata[AC_m__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_phi_mn__mod__cparam-1]) {
ac_transformed_pencil_phi_mn  = atan2(AC_y__mod__cdata[AC_m__mod__cdata-1],AC_x__mod__cdata[vertexIdx.x])
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn1__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,AC_tini__mod__cparam)
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn1__mod__cparam-1]) {
ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,AC_tini__mod__cparam)
}
if (AC_lpencil_int__mod__cdata[AC_i_pomx__mod__cparam-1]) {
ac_transformed_pencil_pomx    = AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
}
if (AC_lpencil_int__mod__cdata[AC_i_pomy__mod__cparam-1]) {
ac_transformed_pencil_pomy    = AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
}
if (AC_lpencil_int__mod__cdata[AC_i_phix__mod__cparam-1]) {
ac_transformed_pencil_phix    =-AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
}
if (AC_lpencil_int__mod__cdata[AC_i_phiy__mod__cparam-1]) {
ac_transformed_pencil_phiy    = AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
}
}
else if (AC_lcylindrical_coords__mod__cdata) {
if (AC_lpencil_int__mod__cdata[AC_i_x_mn__mod__cparam-1]) {
ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]*cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_y_mn__mod__cparam-1]) {
ac_transformed_pencil_y_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_z_mn__mod__cparam-1]) {
ac_transformed_pencil_z_mn    = AC_z__mod__cdata[AC_n__mod__cdata-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn__mod__cparam-1]) {
ac_transformed_pencil_r_mn    = sqrt(AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x]+AC_z__mod__cdata[AC_n__mod__cdata-1]*AC_z__mod__cdata[AC_n__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn = AC_x__mod__cdata[vertexIdx.x]
}
if (AC_lpencil_int__mod__cdata[AC_i_phi_mn__mod__cparam-1]) {
ac_transformed_pencil_phi_mn  = AC_y__mod__cdata[AC_m__mod__cdata-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn1__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,AC_tini__mod__cparam)
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn1__mod__cparam-1]) {
ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,AC_tini__mod__cparam)
}
if (AC_lpencil_int__mod__cdata[AC_i_pomx__mod__cparam-1]) {
ac_transformed_pencil_pomx    = 1.
}
if (AC_lpencil_int__mod__cdata[AC_i_pomy__mod__cparam-1]) {
ac_transformed_pencil_pomy    = 0.
}
if (AC_lpencil_int__mod__cdata[AC_i_phix__mod__cparam-1]) {
ac_transformed_pencil_phix    = 0.
}
if (AC_lpencil_int__mod__cdata[AC_i_phiy__mod__cparam-1]) {
ac_transformed_pencil_phiy    = 1.
}
}
else if (AC_lspherical_coords__mod__cdata) {
if (AC_lpencil_int__mod__cdata[AC_i_x_mn__mod__cparam-1]) {
ac_transformed_pencil_x_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])*cos(AC_z__mod__cdata[AC_n__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_y_mn__mod__cparam-1]) {
ac_transformed_pencil_y_mn    = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])*sin(AC_z__mod__cdata[AC_n__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_z_mn__mod__cparam-1]) {
ac_transformed_pencil_z_mn    = AC_x__mod__cdata[vertexIdx.x]*cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn__mod__cparam-1]) {
ac_transformed_pencil_r_mn    = AC_x__mod__cdata[vertexIdx.x]
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn = AC_x__mod__cdata[vertexIdx.x]*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_phi_mn__mod__cparam-1]) {
ac_transformed_pencil_phi_mn  = AC_z__mod__cdata[AC_n__mod__cdata-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_rcyl_mn1__mod__cparam-1]) {
ac_transformed_pencil_rcyl_mn1=1./max(ac_transformed_pencil_rcyl_mn,AC_tini__mod__cparam)
}
if (AC_lpencil_int__mod__cdata[AC_i_r_mn1__mod__cparam-1]) {
ac_transformed_pencil_r_mn1   =1./max(ac_transformed_pencil_r_mn,AC_tini__mod__cparam)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_rr__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_rr.x=ac_transformed_pencil_x_mn
ac_transformed_pencil_rr.y=ac_transformed_pencil_y_mn
ac_transformed_pencil_rr.z=ac_transformed_pencil_z_mn
}
else {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_evr__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_evr.x = ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomx
ac_transformed_pencil_evr.y = ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomy
ac_transformed_pencil_evr.z = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
else {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_evth__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_evth.x = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomx
ac_transformed_pencil_evth.y = AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1*ac_transformed_pencil_pomy
ac_transformed_pencil_evth.z = -ac_transformed_pencil_rcyl_mn*ac_transformed_pencil_r_mn1
}
else {
}
}
if (AC_llinearized_hydro__mod__hydro) {
if (AC_lpencil_int__mod__cdata[AC_i_uu__mod__cparam-1]) {
ac_transformed_pencil_uu=value(F_UVEC)
}
if (AC_lpencil_int__mod__cdata[AC_i_uu0__mod__cparam-1]) {
ac_transformed_pencil_uu0=value(F_U0VEC)
}
if (AC_lpencil_int__mod__cdata[AC_i_uij__mod__cparam-1]) {
ac_transformed_pencil_uij = gradient_tensor((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_u0ij__mod__cparam-1]) {
ac_transformed_pencil_u0ij = gradient_tensor((Field3){Field(AC_iuu0__mod__cdata), Field(AC_iuu0__mod__cdata+1), Field(AC_iuu0__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_divu__mod__cparam-1]) {
ac_transformed_pencil_divu = divergence(ac_transformed_pencil_uij)
}
if (AC_lpencil_int__mod__cdata[AC_i_divu0__mod__cparam-1]) {
ac_transformed_pencil_divu0 = divergence(ac_transformed_pencil_u0ij)
}
if (AC_lpencil_int__mod__cdata[AC_i_sij__mod__cparam-1]) {
ac_transformed_pencil_sij = traceless_strain(ac_transformed_pencil_uij,ac_transformed_pencil_divu)
if (AC_lshear__mod__cparam  &&  AC_lshear_rateofstrain__mod__hydro) {
ac_transformed_pencil_sij[1-1][2-1] = ac_transformed_pencil_sij[1-1][2-1] + AC_sshear__mod__cdata
ac_transformed_pencil_sij[2-1][1-1] = ac_transformed_pencil_sij[2-1][1-1] + AC_sshear__mod__cdata
}
}
if (AC_lpencil_int__mod__cdata[AC_i_oo__mod__cparam-1]) {
ac_transformed_pencil_oo=curl(ac_transformed_pencil_uij)
}
if (AC_lpencil_int__mod__cdata[AC_i_ugu__mod__cparam-1]) {
ac_transformed_pencil_ugu = ugu0_4_14_15_97+u0gu_4_14_15_97
}
if (AC_lpencil_int__mod__cdata[AC_i_del4u__mod__cparam-1]) {
ac_transformed_pencil_del4u = del4((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u__mod__cparam-1]) {
ac_transformed_pencil_del6u = del6((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u_strict__mod__cparam-1]) {
ac_transformed_pencil_del6u_strict = del6_strict((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del4graddivu__mod__cparam-1]) {
not_implemented("del4graddiv")
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u_bulk__mod__cparam-1]) {
tmp_4_14_15_97 = der6x(Field(AC_iux__mod__cdata))
ac_transformed_pencil_del6u_bulk.x=tmp_4_14_15_97
tmp_4_14_15_97 = der6y(Field(AC_iuy__mod__cdata))
ac_transformed_pencil_del6u_bulk.y=tmp_4_14_15_97
tmp_4_14_15_97 = der6z(Field(AC_iuz__mod__cdata))
ac_transformed_pencil_del6u_bulk.z=tmp_4_14_15_97
}
if (!AC_lcartesian_coords__mod__cdata || AC_lalways_use_gij_etc__mod__hydro) {
if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
ac_transformed_pencil_oij = bij((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
ac_transformed_pencil_graddivu = gradient_of_divergence((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1]) {
ac_transformed_pencil_curlo=curl(ac_transformed_pencil_oij)
ac_transformed_pencil_del2u=ac_transformed_pencil_graddivu-ac_transformed_pencil_curlo
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_d2uidxj__mod__cparam-1]) {
ac_transformed_pencil_d2uidxj = d2fi_dxj((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_uijk__mod__cparam-1]) {
ac_transformed_pencil_uijk = del2fi_dxjk((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_grad5divu__mod__cparam-1]) {
tmp_4_14_15_97=0.0
ju_4_14_15_97=AC_iuu__mod__cdata+1-1
tmp2_4_14_15_97 = der6x(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+2-1
tmp2_4_14_15_97 = der5x1y(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+3-1
tmp2_4_14_15_97 = der5x1z(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ac_transformed_pencil_grad5divu.x=tmp_4_14_15_97
tmp_4_14_15_97=0.0
ju_4_14_15_97=AC_iuu__mod__cdata+1-1
tmp2_4_14_15_97 = der5y1x(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+2-1
tmp2_4_14_15_97 = der6y(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+3-1
tmp2_4_14_15_97 = der5y1z(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ac_transformed_pencil_grad5divu.y=tmp_4_14_15_97
tmp_4_14_15_97=0.0
ju_4_14_15_97=AC_iuu__mod__cdata+1-1
tmp2_4_14_15_97 = der5z1x(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+2-1
tmp2_4_14_15_97 = der5z1y(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ju_4_14_15_97=AC_iuu__mod__cdata+3-1
tmp2_4_14_15_97 = der6z(Field(ju_4_14_15_97))
tmp_4_14_15_97=tmp_4_14_15_97+tmp2_4_14_15_97
ac_transformed_pencil_grad5divu.z=tmp_4_14_15_97
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_uu__mod__cparam-1]) {
if (AC_lconservative__mod__hydro) {
tmp3_13_14_15_97=value(F_UVEC)
if (AC_lrelativistic__mod__hydro) {
cs201_13_14_15_97=AC_cs20__mod__equationofstate+1.
tmp_rho_13_14_15_97=value(F_RHO)
if (!AC_lhiggsless_old__mod__hydro && AC_lhiggsless__mod__hydro) {
if(AC_t__mod__cdata < value(F_HLESS)) {
tmp_rho_13_14_15_97=tmp_rho_13_14_15_97-AC_eps_hless__mod__hydro
}
}
if (AC_lmagnetic__mod__cparam) {
if (AC_full_3d__mod__hydro) {
dd_13_14_15_97=(value(F_RHO)-0.5*AC_b_ext2__mod__magnetic)/(1.-0.25/value(F_LORENTZ))+AC_b_ext2__mod__magnetic
bx_5_13_14_15_97=ac_transformed_pencil_bb.x
by_5_13_14_15_97=ac_transformed_pencil_bb.y
bz_5_13_14_15_97=ac_transformed_pencil_bb.z
bx2_5_13_14_15_97=bx_5_13_14_15_97*bx_5_13_14_15_97
by2_5_13_14_15_97=by_5_13_14_15_97*by_5_13_14_15_97
bz2_5_13_14_15_97=bz_5_13_14_15_97*bz_5_13_14_15_97
detm1_5_13_14_15_97=1./(dd_13_14_15_97*dd_13_14_15_97*dd_13_14_15_97-dd_13_14_15_97*dd_13_14_15_97*(bx2_5_13_14_15_97+by2_5_13_14_15_97+bz2_5_13_14_15_97)+4.*bx2_5_13_14_15_97*by2_5_13_14_15_97*bz2_5_13_14_15_97)
tmp33_13_14_15_97[1-1][1-1]=detm1_5_13_14_15_97*dd_13_14_15_97*(dd_13_14_15_97-by2_5_13_14_15_97-bz2_5_13_14_15_97)
tmp33_13_14_15_97[2-1][2-1]=detm1_5_13_14_15_97*dd_13_14_15_97*(dd_13_14_15_97-bz2_5_13_14_15_97-bx2_5_13_14_15_97)
tmp33_13_14_15_97[3-1][3-1]=detm1_5_13_14_15_97*dd_13_14_15_97*(dd_13_14_15_97-bx2_5_13_14_15_97-by2_5_13_14_15_97)
tmp33_13_14_15_97[1-1][2-1]=detm1_5_13_14_15_97*bx_5_13_14_15_97*by_5_13_14_15_97*(2.*bz2_5_13_14_15_97-dd_13_14_15_97)
tmp33_13_14_15_97[2-1][3-1]=detm1_5_13_14_15_97*by_5_13_14_15_97*bz_5_13_14_15_97*(2.*bx2_5_13_14_15_97-dd_13_14_15_97)
tmp33_13_14_15_97[3-1][1-1]=detm1_5_13_14_15_97*bz_5_13_14_15_97*bx_5_13_14_15_97*(2.*by2_5_13_14_15_97-dd_13_14_15_97)
tmp33_13_14_15_97[2-1][1-1]=tmp33_13_14_15_97[1-1][2-1]
tmp33_13_14_15_97[3-1][2-1]=tmp33_13_14_15_97[2-1][3-1]
tmp33_13_14_15_97[1-1][3-1]=tmp33_13_14_15_97[3-1][1-1]
ac_transformed_pencil_uu = tmp33_13_14_15_97*tmp3_13_14_15_97
}
else {
tmp_13_14_15_97=1./((value(F_RHO)-0.5*AC_b_ext2__mod__magnetic)/(1.-0.25/value(F_LORENTZ))+AC_b_ext2__mod__magnetic)
ac_transformed_pencil_uu = tmp_13_14_15_97*tmp3_13_14_15_97
}
}
else {
tmp_13_14_15_97=1./(tmp_rho_13_14_15_97/(1.-0.25/value(F_LORENTZ)))
ac_transformed_pencil_uu = tmp_13_14_15_97*tmp3_13_14_15_97
}
}
else {
ac_transformed_pencil_rho1=1./value(F_RHO)
ac_transformed_pencil_uu = ac_transformed_pencil_rho1*tmp3_13_14_15_97
}
}
else {
ac_transformed_pencil_uu=value(F_UVEC)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_u2__mod__cparam-1]) {
ac_transformed_pencil_u2 = dot(ac_transformed_pencil_uu,ac_transformed_pencil_uu)
}
if (AC_lpencil_int__mod__cdata[AC_i_uij__mod__cparam-1]) {
if (AC_lvv_as_aux__mod__hydro  ||  AC_lvv_as_comaux__mod__hydro) {
ac_transformed_pencil_uij = gradient_tensor((Field3){Field(AC_ivv__mod__cdata), Field(AC_ivv__mod__cdata+1), Field(AC_ivv__mod__cdata+2)})
}
else {
ac_transformed_pencil_uij = gradient_tensor((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lgradu_as_aux__mod__hydro  ||  AC_lparticles_lyapunov__mod__cparam  ||  AC_lparticles_caustics__mod__cparam  ||  AC_lparticles_tetrad__mod__cparam) {
DF_GU11 = ac_transformed_pencil_uij[1-1][1-1]
DF_GU12 = ac_transformed_pencil_uij[1-1][2-1]
DF_GU13 = ac_transformed_pencil_uij[1-1][3-1]
DF_GU21 = ac_transformed_pencil_uij[2-1][1-1]
DF_GU22 = ac_transformed_pencil_uij[2-1][2-1]
DF_GU23 = ac_transformed_pencil_uij[2-1][3-1]
DF_GU31 = ac_transformed_pencil_uij[3-1][1-1]
DF_GU32 = ac_transformed_pencil_uij[3-1][2-1]
DF_GU33 = ac_transformed_pencil_uij[3-1][3-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_divu__mod__cparam-1]) {
ac_transformed_pencil_divu = divergence(ac_transformed_pencil_uij)
}
if (AC_lpencil_int__mod__cdata[AC_i_sij__mod__cparam-1]) {
ac_transformed_pencil_sij = traceless_strain(ac_transformed_pencil_uij,ac_transformed_pencil_divu)
if (AC_lshear__mod__cparam  &&  AC_lshear_rateofstrain__mod__hydro) {
ac_transformed_pencil_sij[1-1][2-1] = ac_transformed_pencil_sij[1-1][2-1] + AC_sshear__mod__cdata
ac_transformed_pencil_sij[2-1][1-1] = ac_transformed_pencil_sij[2-1][1-1] + AC_sshear__mod__cdata
}
}
if (AC_lpencil_int__mod__cdata[AC_i_sij2__mod__cparam-1]) {
ac_transformed_pencil_sij2 = multm2_sym(ac_transformed_pencil_sij)
}
if (AC_lpencil_int__mod__cdata[AC_i_uij5__mod__cparam-1]) {
ac_transformed_pencil_uij5 = gradient5((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_oo__mod__cparam-1]) {
if (AC_ioo__mod__cdata != 0) {
ac_transformed_pencil_oo = value(F_OVEC)
}
else {
ac_transformed_pencil_oo=curl(ac_transformed_pencil_uij)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_o2__mod__cparam-1]) {
ac_transformed_pencil_o2 = dot(ac_transformed_pencil_oo,ac_transformed_pencil_oo)
}
if (AC_lpencil_int__mod__cdata[AC_i_ou__mod__cparam-1]) {
ac_transformed_pencil_ou = dot(ac_transformed_pencil_oo,ac_transformed_pencil_uu)
}
if (AC_lpencil_int__mod__cdata[AC_i_oxu__mod__cparam-1]) {
ac_transformed_pencil_oxu = cross(ac_transformed_pencil_oo,ac_transformed_pencil_uu)
}
if (AC_lpencil_int__mod__cdata[AC_i_oxu2__mod__cparam-1]) {
ac_transformed_pencil_oxu2 = dot(ac_transformed_pencil_oxu,ac_transformed_pencil_oxu)
}
if (AC_lpencil_int__mod__cdata[AC_i_ugu__mod__cparam-1]) {
ac_transformed_pencil_ugu = ac_transformed_pencil_uij*ac_transformed_pencil_uu
if (AC_lupw_uu__mod__hydro) ac_transformed_pencil_ugu = ac_transformed_pencil_ugu + del6((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
if (AC_ldensity__mod__cparam) {
if (AC_lffree__mod__density) {
tmp_13_14_15_97=AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]
ac_transformed_pencil_ugu.x=ac_transformed_pencil_ugu.x*tmp_13_14_15_97
ac_transformed_pencil_ugu.y=ac_transformed_pencil_ugu.y*tmp_13_14_15_97
ac_transformed_pencil_ugu.z=ac_transformed_pencil_ugu.z*tmp_13_14_15_97
}
}
}
if (AC_lpencil_int__mod__cdata[AC_i_ugu2__mod__cparam-1]) {
ac_transformed_pencil_ugu2 = dot(ac_transformed_pencil_ugu,ac_transformed_pencil_ugu)
}
if (AC_lpencil_int__mod__cdata[AC_i_ogu__mod__cparam-1]) {
ac_transformed_pencil_ogu = ac_transformed_pencil_uij*ac_transformed_pencil_oo
if (AC_lupw_uu__mod__hydro) ac_transformed_pencil_ogu = ac_transformed_pencil_ogu + del6((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_u3u21__mod__cparam-1]) {
ac_transformed_pencil_u3u21=ac_transformed_pencil_uu.z*ac_transformed_pencil_uij[2-1][1-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_u1u32__mod__cparam-1]) {
ac_transformed_pencil_u1u32=ac_transformed_pencil_uu.x*ac_transformed_pencil_uij[3-1][2-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_u2u13__mod__cparam-1]) {
ac_transformed_pencil_u2u13=ac_transformed_pencil_uu.y*ac_transformed_pencil_uij[1-1][3-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_u2u31__mod__cparam-1]) {
ac_transformed_pencil_u2u31=ac_transformed_pencil_uu.y*ac_transformed_pencil_uij[3-1][1-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_u3u12__mod__cparam-1]) {
ac_transformed_pencil_u3u12=ac_transformed_pencil_uu.z*ac_transformed_pencil_uij[1-1][2-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_u1u23__mod__cparam-1]) {
ac_transformed_pencil_u1u23=ac_transformed_pencil_uu.x*ac_transformed_pencil_uij[2-1][3-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_del4u__mod__cparam-1]) {
ac_transformed_pencil_del4u = del4((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u__mod__cparam-1]) {
ac_transformed_pencil_del6u = del6((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u_strict__mod__cparam-1]) {
ac_transformed_pencil_del6u_strict = del6_strict((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del4graddivu__mod__cparam-1]) {
not_implemented("del4graddiv")
}
if (AC_lpencil_int__mod__cdata[AC_i_del6u_bulk__mod__cparam-1]) {
ac_transformed_pencil_del6u_bulk.x = der6x(Field(AC_iux__mod__cdata))
ac_transformed_pencil_del6u_bulk.y = der6y(Field(AC_iuy__mod__cdata))
ac_transformed_pencil_del6u_bulk.z = der6z(Field(AC_iuz__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_der6u_res__mod__cparam-1]) {
ju_13_14_15_97=1+AC_iuu__mod__cdata-1
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[1-1][1-1] = der6x_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[2-1][1-1] = der6y_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[3-1][1-1] = der6z_ignore_spacing(Field(ju_13_14_15_97))
}
ju_13_14_15_97=2+AC_iuu__mod__cdata-1
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[1-1][2-1] = der6x_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[2-1][2-1] = der6y_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[3-1][2-1] = der6z_ignore_spacing(Field(ju_13_14_15_97))
}
ju_13_14_15_97=3+AC_iuu__mod__cdata-1
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 1==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[1-1][3-1] = der6x_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 2==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[2-1][3-1] = der6y_ignore_spacing(Field(ju_13_14_15_97))
}
if (AC_lcylindrical_coords__mod__cdata && ju_13_14_15_97==AC_iuy__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else if (AC_lspherical_coords__mod__cdata && ju_13_14_15_97==AC_iuz__mod__cdata && 3==1) {
print("not implemented der6_pencil")
}
else {
ac_transformed_pencil_der6u_res[3-1][3-1] = der6z_ignore_spacing(Field(ju_13_14_15_97))
}
}
if (!AC_lcartesian_coords__mod__cdata || AC_lalways_use_gij_etc__mod__hydro) {
if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
ac_transformed_pencil_oij = bij((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
ac_transformed_pencil_graddivu = gradient_of_divergence((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1]) {
ac_transformed_pencil_curlo=curl(ac_transformed_pencil_oij)
ac_transformed_pencil_del2u=ac_transformed_pencil_graddivu-ac_transformed_pencil_curlo
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2u__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_graddivu__mod__cparam-1]) {
}
else if (AC_lpencil_int__mod__cdata[AC_i_curlo__mod__cparam-1]) {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_d2uidxj__mod__cparam-1]) {
ac_transformed_pencil_d2uidxj = d2fi_dxj((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_uijk__mod__cparam-1]) {
ac_transformed_pencil_uijk = del2fi_dxjk((Field3){Field(AC_iuu__mod__cdata), Field(AC_iuu__mod__cdata+1), Field(AC_iuu__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_grad5divu__mod__cparam-1]) {
ac_transformed_pencil_grad5divu.x = 0.0
ju_13_14_15_97=AC_iuu__mod__cdata+1-1
tmp_13_14_15_97 = der6x(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.x = ac_transformed_pencil_grad5divu.x + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+2-1
tmp_13_14_15_97 = der5x1y(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.x = ac_transformed_pencil_grad5divu.x + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+3-1
tmp_13_14_15_97 = der5x1z(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.x = ac_transformed_pencil_grad5divu.x + tmp_13_14_15_97
ac_transformed_pencil_grad5divu.y = 0.0
ju_13_14_15_97=AC_iuu__mod__cdata+1-1
tmp_13_14_15_97 = der5y1x(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.y = ac_transformed_pencil_grad5divu.y + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+2-1
tmp_13_14_15_97 = der6y(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.y = ac_transformed_pencil_grad5divu.y + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+3-1
tmp_13_14_15_97 = der5y1z(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.y = ac_transformed_pencil_grad5divu.y + tmp_13_14_15_97
ac_transformed_pencil_grad5divu.z = 0.0
ju_13_14_15_97=AC_iuu__mod__cdata+1-1
tmp_13_14_15_97 = der5z1x(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.z = ac_transformed_pencil_grad5divu.z + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+2-1
tmp_13_14_15_97 = der5z1y(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.z = ac_transformed_pencil_grad5divu.z + tmp_13_14_15_97
ju_13_14_15_97=AC_iuu__mod__cdata+3-1
tmp_13_14_15_97 = der6z(Field(ju_13_14_15_97))
ac_transformed_pencil_grad5divu.z = ac_transformed_pencil_grad5divu.z + tmp_13_14_15_97
}
if (AC_lpencil_int__mod__cdata[AC_i_transpurho__mod__cparam-1] && AC_ldensity_nolog__mod__cdata) {
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_transpurho.x = AC_impossible__mod__cparam
ac_transformed_pencil_transpurho.y = AC_impossible__mod__cparam
ac_transformed_pencil_transpurho.z = AC_impossible__mod__cparam
}
else {
ac_transformed_pencil_transpurho.x = AC_impossible__mod__cparam
ac_transformed_pencil_transpurho.y = AC_impossible__mod__cparam
ac_transformed_pencil_transpurho.z = AC_impossible__mod__cparam
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uu_advec__mod__cparam-1]) {
ac_transformed_pencil_uu_advec.x=ac_transformed_pencil_uu.x
if (AC_lcylindrical_coords__mod__cdata) {
ac_transformed_pencil_uu_advec.y=ac_transformed_pencil_uu.y-AC_uu_average_cyl__mod__hydro[vertexIdx.x][AC_n__mod__cdata-1]
ac_transformed_pencil_uu_advec.z=ac_transformed_pencil_uu.z
}
else if (AC_lspherical_coords__mod__cdata) {
ac_transformed_pencil_uu_advec.y=ac_transformed_pencil_uu.y
ac_transformed_pencil_uu_advec.z=ac_transformed_pencil_uu.z-AC_uu_average_sph__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1]
}
tmp3_13_14_15_97 = ac_transformed_pencil_uij.row(1-1)
ac_transformed_pencil_uuadvec_guu.x = dot(ac_transformed_pencil_uu_advec,tmp3_13_14_15_97)
tmp3_13_14_15_97 = ac_transformed_pencil_uij.row(2-1)
ac_transformed_pencil_uuadvec_guu.y = dot(ac_transformed_pencil_uu_advec,tmp3_13_14_15_97)
tmp3_13_14_15_97 = ac_transformed_pencil_uij.row(3-1)
ac_transformed_pencil_uuadvec_guu.z = dot(ac_transformed_pencil_uu_advec,tmp3_13_14_15_97)
if (AC_lcylindrical_coords__mod__cdata) {
ac_transformed_pencil_uuadvec_guu.x=ac_transformed_pencil_uuadvec_guu.x-AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]*ac_transformed_pencil_uu.y*ac_transformed_pencil_uu.y
ac_transformed_pencil_uuadvec_guu.y=ac_transformed_pencil_uuadvec_guu.y+AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]*ac_transformed_pencil_uu.y*ac_transformed_pencil_uu.x
}
else if (AC_lspherical_coords__mod__cdata) {
ac_transformed_pencil_uuadvec_guu.x=ac_transformed_pencil_uuadvec_guu.x-AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*(ac_transformed_pencil_uu.y*ac_transformed_pencil_uu.y+ac_transformed_pencil_uu.z*ac_transformed_pencil_uu.z)
ac_transformed_pencil_uuadvec_guu.y=ac_transformed_pencil_uuadvec_guu.y+AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*(ac_transformed_pencil_uu.y*ac_transformed_pencil_uu.x-ac_transformed_pencil_uu.z*ac_transformed_pencil_uu.z*AC_cotth__mod__cdata[AC_m__mod__cdata-1])
ac_transformed_pencil_uuadvec_guu.z=ac_transformed_pencil_uuadvec_guu.z+AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]*(ac_transformed_pencil_uu.z*ac_transformed_pencil_uu.x+ac_transformed_pencil_uu.z*ac_transformed_pencil_uu.y*AC_cotth__mod__cdata[AC_m__mod__cdata-1])
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uu_sph__mod__cparam-1] && AC_luu_sph_as_aux__mod__hydro) {
ac_transformed_pencil_uu_sph=value(F_UU_SPHVEC)
}
if (AC_lconservative__mod__hydro) {
if (AC_ilorentz__mod__cdata != 0) {
ac_transformed_pencil_lorentz = value(F_LORENTZ)
}
if (!AC_lhiggsless_old__mod__hydro && AC_lhiggsless__mod__hydro) {
ac_transformed_pencil_hless = value(F_HLESS)
}
}
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata && AC_ladvection_velocity__mod__hydro) {
if (AC_lmaximal_cdt__mod__cdata) {
ac_transformed_pencil_advec_uu=max(abs(ac_transformed_pencil_uu.x)*dline_1__mod__cdata.x, abs(ac_transformed_pencil_uu.y)*dline_1__mod__cdata.y, abs(ac_transformed_pencil_uu.z)*dline_1__mod__cdata.z)
}
else if (AC_lfargo_advection__mod__cdata) {
ac_transformed_pencil_advec_uu=sum(abs(ac_transformed_pencil_uu_advec)*dline_1__mod__cdata)
}
else {
ac_transformed_pencil_advec_uu=sum(abs(ac_transformed_pencil_uu)*dline_1__mod__cdata)
}
if (AC_lisotropic_advection__mod__cdata) {
if (AC_dimensionality__mod__cparam<3) {
ac_transformed_pencil_advec_uu=sqrt(ac_transformed_pencil_u2*dxyz_2__mod__cdata)
}
}
}
if (AC_ldensity_nolog__mod__cdata) {
ac_transformed_pencil_rho=value(F_RHO)
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_rho=ac_transformed_pencil_rho+AC_reference_state__mod__density[vertexIdx.x][AC_iref_rho__mod__cparam-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_rho1__mod__cparam-1]) {
ac_transformed_pencil_rho1=1.0/ac_transformed_pencil_rho
}
if (AC_lpencil_int__mod__cdata[AC_i_lnrho__mod__cparam-1]) {
ac_transformed_pencil_lnrho=log(ac_transformed_pencil_rho)
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrho__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_grho__mod__cparam-1]) {
ac_transformed_pencil_grho = gradient(Field(AC_irho__mod__cdata))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_grho.x=ac_transformed_pencil_grho.x+AC_reference_state__mod__density[vertexIdx.x][AC_iref_grho__mod__cparam-1]
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrho__mod__cparam-1]) {
ac_transformed_pencil_glnrho.x=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.x
ac_transformed_pencil_glnrho.y=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.y
ac_transformed_pencil_glnrho.z=ac_transformed_pencil_rho1*ac_transformed_pencil_grho.z
}
}
if (AC_lpencil_int__mod__cdata[AC_i_ugrho__mod__cparam-1]) {
ac_transformed_pencil_ugrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_grho)
if (AC_lupw_rho__mod__density) ac_transformed_pencil_ugrho = ac_transformed_pencil_ugrho - del6_upwd(Field(AC_ilnrho__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrho2__mod__cparam-1]) {
ac_transformed_pencil_glnrho2 = dot(ac_transformed_pencil_glnrho,ac_transformed_pencil_glnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_del2rho__mod__cparam-1]) {
ac_transformed_pencil_del2rho = laplace(Field(AC_irho__mod__cdata))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_del2rho=ac_transformed_pencil_del2rho+AC_reference_state__mod__density[vertexIdx.x][AC_iref_d2rho__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lnrho__mod__cparam-1]) {
ac_transformed_pencil_del2lnrho=ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho-ac_transformed_pencil_glnrho2
}
if (AC_lpencil_int__mod__cdata[AC_i_del6rho__mod__cparam-1]) {
if (AC_ldiff_hyper3__mod__density) {
ac_transformed_pencil_del6rho = del6(Field(AC_irho__mod__cdata))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_del6rho=ac_transformed_pencil_del6rho+AC_reference_state__mod__density[vertexIdx.x][AC_iref_d6rho__mod__cparam-1]
}
}
else if (AC_ldiff_hyper3_strict__mod__density) {
ac_transformed_pencil_del6rho=0.
tmp_16_20_24_25_97 = der6x(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_16_20_24_25_97
if (1!=1) {
tmp_16_20_24_25_97 = der6x(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (2!=1) {
tmp_16_20_24_25_97 = der4x2y(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (3!=1) {
tmp_16_20_24_25_97 = der4x2z(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
tmp_16_20_24_25_97 = der6y(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_16_20_24_25_97
if (1!=2) {
tmp_16_20_24_25_97 = der4y2x(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (2!=2) {
tmp_16_20_24_25_97 = der6y(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (3!=2) {
tmp_16_20_24_25_97 = der4y2z(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
tmp_16_20_24_25_97 = der6z(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + tmp_16_20_24_25_97
if (1!=3) {
tmp_16_20_24_25_97 = der4z2x(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (2!=3) {
tmp_16_20_24_25_97 = der4z2y(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
if (3!=3) {
tmp_16_20_24_25_97 = der6z(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 3*tmp_16_20_24_25_97
}
tmp_16_20_24_25_97 = der2i2j2k(Field(AC_irho__mod__cdata))
ac_transformed_pencil_del6rho = ac_transformed_pencil_del6rho + 6*tmp_16_20_24_25_97
}
}
if (AC_lpencil_int__mod__cdata[AC_i_del6lnrho__mod__cparam-1]) {
if (AC_ldiff_hyper3lnrho__mod__density) {
}
else if (AC_ldiff_hyper3lnrho_strict__mod__density) {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_sglnrho__mod__cparam-1]) {
ac_transformed_pencil_sglnrho = ac_transformed_pencil_sij*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_uij5glnrho__mod__cparam-1]) {
ac_transformed_pencil_uij5glnrho = ac_transformed_pencil_uij5*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_transprho__mod__cparam-1]) {
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_transprho = AC_impossible__mod__cparam
}
else {
ac_transformed_pencil_transprho = AC_impossible__mod__cparam
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uuadvec_grho__mod__cparam-1]) {
ac_transformed_pencil_uuadvec_grho = dot(ac_transformed_pencil_uu_advec,ac_transformed_pencil_grho)
if (AC_lupw_rho__mod__density) {
ac_transformed_pencil_uu_advec = del6(Field(AC_irho__mod__cdata))
ac_transformed_pencil_uuadvec_grho = ac_transformed_pencil_uuadvec_grho - tmp_20_24_25_97
}
}
if (AC_lpencil_int__mod__cdata[AC_i_divss__mod__cparam-1]) {
ac_transformed_pencil_divss = divergence((Field3){Field(AC_iux__mod__cdata), Field(AC_iux__mod__cdata+1), Field(AC_iux__mod__cdata+2)})
}
if (AC_lconservative__mod__hydro) {
if (AC_lhiggsless__mod__hydro) {
if(AC_t__mod__cdata < ac_transformed_pencil_hless) {
ac_transformed_pencil_rho=ac_transformed_pencil_rho-AC_eps_hless__mod__hydro
}
ac_transformed_pencil_rho=ac_transformed_pencil_rho/(AC_fourthird__mod__cparam*ac_transformed_pencil_lorentz*(1.-0.25/ac_transformed_pencil_lorentz))
}
else {
ac_transformed_pencil_rho=ac_transformed_pencil_rho/(AC_fourthird__mod__cparam*ac_transformed_pencil_lorentz*(1.-0.25/ac_transformed_pencil_lorentz))
}
}
}
else {
ac_transformed_pencil_lnrho=value(F_LNRHO)
if (AC_lpencil_int__mod__cdata[AC_i_rho1__mod__cparam-1]) {
ac_transformed_pencil_rho1=exp(-value(F_LNRHO))
}
if (AC_lpencil_int__mod__cdata[AC_i_rho__mod__cparam-1]) {
ac_transformed_pencil_rho=1.0/ac_transformed_pencil_rho1
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrho__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_grho__mod__cparam-1]) {
ac_transformed_pencil_glnrho = gradient(Field(AC_ilnrho__mod__cdata))
if (AC_lpencil_int__mod__cdata[AC_i_grho__mod__cparam-1]) {
ac_transformed_pencil_grho.x=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.x
ac_transformed_pencil_grho.y=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.y
ac_transformed_pencil_grho.z=ac_transformed_pencil_rho*ac_transformed_pencil_glnrho.z
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uglnrho__mod__cparam-1]) {
if (AC_lupw_lnrho__mod__density) {
ac_transformed_pencil_uglnrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_glnrho)
if (AC_lupw_lnrho__mod__density) ac_transformed_pencil_uglnrho = ac_transformed_pencil_uglnrho - del6_upwd(Field(AC_ilnrho__mod__cdata))
}
else {
ac_transformed_pencil_uglnrho = dot(ac_transformed_pencil_uu,ac_transformed_pencil_glnrho)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrho2__mod__cparam-1]) {
ac_transformed_pencil_glnrho2 = dot(ac_transformed_pencil_glnrho,ac_transformed_pencil_glnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lnrho__mod__cparam-1]) {
ac_transformed_pencil_del2lnrho = laplace(Field(AC_ilnrho__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_del6lnrho__mod__cparam-1]) {
if (AC_ldiff_hyper3lnrho__mod__density) {
ac_transformed_pencil_del6lnrho = del6(Field(AC_ilnrho__mod__cdata))
}
else if (AC_ldiff_hyper3lnrho_strict__mod__density) {
ac_transformed_pencil_del6lnrho=0.
tmp_21_23_24_25_97 = der6x(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_21_23_24_25_97
if (1!=1) {
tmp_21_23_24_25_97 = der6x(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (2!=1) {
tmp_21_23_24_25_97 = der4x2y(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (3!=1) {
tmp_21_23_24_25_97 = der4x2z(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
tmp_21_23_24_25_97 = der6y(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_21_23_24_25_97
if (1!=2) {
tmp_21_23_24_25_97 = der4y2x(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (2!=2) {
tmp_21_23_24_25_97 = der6y(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (3!=2) {
tmp_21_23_24_25_97 = der4y2z(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
tmp_21_23_24_25_97 = der6z(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + tmp_21_23_24_25_97
if (1!=3) {
tmp_21_23_24_25_97 = der4z2x(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (2!=3) {
tmp_21_23_24_25_97 = der4z2y(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
if (3!=3) {
tmp_21_23_24_25_97 = der6z(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 3*tmp_21_23_24_25_97
}
tmp_21_23_24_25_97 = der2i2j2k(Field(AC_ilnrho__mod__cdata))
ac_transformed_pencil_del6lnrho = ac_transformed_pencil_del6lnrho + 6*tmp_21_23_24_25_97
}
}
if (AC_lpencil_int__mod__cdata[AC_i_hlnrho__mod__cparam-1]) {
ac_transformed_pencil_hlnrho = hessian(Field(AC_ilnrho__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_sglnrho__mod__cparam-1]) {
ac_transformed_pencil_sglnrho = ac_transformed_pencil_sij*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_uij5glnrho__mod__cparam-1]) {
ac_transformed_pencil_uij5glnrho = ac_transformed_pencil_uij5*ac_transformed_pencil_glnrho
}
ac_transformed_pencil_uuadvec_glnrho = dot(ac_transformed_pencil_uu_advec,ac_transformed_pencil_glnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_ekin__mod__cparam-1]) {
if (AC_lconservative__mod__hydro) {
ac_transformed_pencil_ekin=AC_fourthird__mod__cparam*ac_transformed_pencil_rho*ac_transformed_pencil_lorentz*ac_transformed_pencil_u2
}
else {
ac_transformed_pencil_ekin=0.5*ac_transformed_pencil_rho*ac_transformed_pencil_u2
}
}
if (AC_lpencil_int__mod__cdata[AC_i_cc__mod__cparam-1]) {
ac_transformed_pencil_cc=1.0
}
if (AC_lpencil_int__mod__cdata[AC_i_cc1__mod__cparam-1]) {
ac_transformed_pencil_cc1=1.0
}
if (AC_lpencil_int__mod__cdata[AC_i_cv1__mod__cparam-1]) {
ac_transformed_pencil_cv1=AC_cv1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_cp1__mod__cparam-1]) {
ac_transformed_pencil_cp1=AC_cp1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_cv__mod__cparam-1]) {
ac_transformed_pencil_cv=1/AC_cv1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_cp__mod__cparam-1]) {
ac_transformed_pencil_cp=1/AC_cp1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_cp1tilde__mod__cparam-1]) {
ac_transformed_pencil_cp1tilde=AC_cp1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glnmumol__mod__cparam-1]) {
ac_transformed_pencil_glnmumol.x = 0.0
ac_transformed_pencil_glnmumol.y = 0.0
ac_transformed_pencil_glnmumol.z = 0.0
}
if(AC_ieosvars__mod__equationofstate == AC_ilnrho_ss__mod__equationofstate || AC_ieosvars__mod__equationofstate == AC_irho_ss__mod__equationofstate) {
if (AC_leos_isentropic__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_gss__mod__cparam-1]) {
ac_transformed_pencil_gss.x = 0.0
ac_transformed_pencil_gss.y = 0.0
ac_transformed_pencil_gss.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_hss__mod__cparam-1]) {
ac_transformed_pencil_hss=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del2ss__mod__cparam-1]) {
ac_transformed_pencil_del2ss=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del6ss__mod__cparam-1]) {
ac_transformed_pencil_del6ss=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate*exp(AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate))
}
}
else if (AC_leos_isothermal__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_gss__mod__cparam-1]) {
ac_transformed_pencil_gss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_hss__mod__cparam-1]) {
ac_transformed_pencil_hss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_hlnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_del2ss__mod__cparam-1]) {
ac_transformed_pencil_del2ss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_del2lnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_del6ss__mod__cparam-1]) {
ac_transformed_pencil_del6ss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_del6lnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_rho1gpp__mod__cparam-1]) {
ac_transformed_pencil_rho1gpp.x = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.x
ac_transformed_pencil_rho1gpp.y = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.y
ac_transformed_pencil_rho1gpp.z = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.z
}
}
else if (AC_leos_localisothermal__mod__equationofstate) {
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=value(F_EOSVAR2)
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_ss=ac_transformed_pencil_ss+AC_reference_state__mod__density[vertexIdx.x][AC_iref_s__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_gss__mod__cparam-1]) {
ac_transformed_pencil_gss = gradient(Field(AC_ieosvar2__mod__equationofstate))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_gss.x=ac_transformed_pencil_gss.x+AC_reference_state__mod__density[vertexIdx.x][AC_iref_gs__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_hss__mod__cparam-1]) {
ac_transformed_pencil_hss = hessian(Field(AC_ieosvar2__mod__equationofstate))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_hss[1-1][1-1]=ac_transformed_pencil_hss[1-1][1-1]+AC_reference_state__mod__density[vertexIdx.x][AC_iref_d2s__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_del2ss__mod__cparam-1]) {
ac_transformed_pencil_del2ss = laplace(Field(AC_ieosvar2__mod__equationofstate))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_del2ss=ac_transformed_pencil_del2ss+AC_reference_state__mod__density[vertexIdx.x][AC_iref_d2s__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_del6ss__mod__cparam-1]) {
ac_transformed_pencil_del6ss = del6(Field(AC_ieosvar2__mod__equationofstate))
if (AC_lreference_state__mod__cdata) {
ac_transformed_pencil_del6ss=ac_transformed_pencil_del6ss+AC_reference_state__mod__density[vertexIdx.x][AC_iref_d6s__mod__cparam-1]
}
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate*exp(AC_cv1__mod__equationofstate*ac_transformed_pencil_ss+AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate))
}
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate+AC_cv1__mod__equationofstate*ac_transformed_pencil_ss+AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*exp(ac_transformed_pencil_lntt+ac_transformed_pencil_lnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_ee=AC_cv__mod__equationofstate*exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_yh__mod__cparam-1]) {
ac_transformed_pencil_yh=AC_impossible__mod__cparam
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1]) {
ac_transformed_pencil_tt1=exp(-ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_glnrho+AC_cv1__mod__equationofstate*ac_transformed_pencil_gss
}
if (AC_lpencil_int__mod__cdata[AC_i_gtt__mod__cparam-1]) {
ac_transformed_pencil_gtt.x=ac_transformed_pencil_glntt.x*ac_transformed_pencil_tt
ac_transformed_pencil_gtt.y=ac_transformed_pencil_glntt.y*ac_transformed_pencil_tt
ac_transformed_pencil_gtt.z=ac_transformed_pencil_glntt.z*ac_transformed_pencil_tt
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_del2lnrho+AC_cv1__mod__equationofstate*ac_transformed_pencil_del2ss
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_hlnrho+AC_cv1__mod__equationofstate*ac_transformed_pencil_hss
}
}
else if(AC_ieosvars__mod__equationofstate == AC_ilnrho_lntt__mod__equationofstate || AC_ieosvars__mod__equationofstate == AC_irho_lntt__mod__equationofstate) {
if (AC_leos_isentropic__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate)+AC_lntt0__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_hlnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_del2lnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate*exp(AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate))
}
}
else if (AC_leos_isothermal__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt.x = 0.0
ac_transformed_pencil_glntt.y = 0.0
ac_transformed_pencil_glntt.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
}
}
else if (AC_leos_localisothermal__mod__equationofstate) {
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=value(F_EOSVAR2)
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt = gradient(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt = hessian(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt = laplace(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_del6lntt__mod__cparam-1]) {
ac_transformed_pencil_del6lntt = del6(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cp__mod__equationofstate*exp(ac_transformed_pencil_lntt)*AC_gamma_m1__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_rho1gpp__mod__cparam-1]) {
ac_transformed_pencil_rho1gpp.x = AC_gamma1__mod__equationofstate*ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.x+ac_transformed_pencil_glntt.x)
ac_transformed_pencil_rho1gpp.y = AC_gamma1__mod__equationofstate*ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.y+ac_transformed_pencil_glntt.y)
ac_transformed_pencil_rho1gpp.z = AC_gamma1__mod__equationofstate*ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.z+ac_transformed_pencil_glntt.z)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=AC_cv__mod__equationofstate*(ac_transformed_pencil_lntt-AC_lntt0__mod__equationofstate-AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*exp(ac_transformed_pencil_lntt+ac_transformed_pencil_lnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_ee=AC_cv__mod__equationofstate*exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_yh__mod__cparam-1]) {
ac_transformed_pencil_yh=AC_impossible__mod__cparam
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1]) {
ac_transformed_pencil_tt1=exp(-ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_gss__mod__cparam-1]) {
ac_transformed_pencil_gss=AC_cv__mod__equationofstate*(ac_transformed_pencil_glntt-AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_glnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_del2ss__mod__cparam-1]) {
ac_transformed_pencil_del2ss=AC_cv__mod__equationofstate*(ac_transformed_pencil_del2lntt-AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_del2lnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_hss__mod__cparam-1]) {
ac_transformed_pencil_hss=AC_cv__mod__equationofstate*(ac_transformed_pencil_hlntt-AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_hlnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_gtt__mod__cparam-1]) {
ac_transformed_pencil_gtt.x=ac_transformed_pencil_tt*ac_transformed_pencil_glntt.x
ac_transformed_pencil_gtt.y=ac_transformed_pencil_tt*ac_transformed_pencil_glntt.y
ac_transformed_pencil_gtt.z=ac_transformed_pencil_tt*ac_transformed_pencil_glntt.z
}
}
else if(AC_ieosvars__mod__equationofstate == AC_ilnrho_tt__mod__equationofstate || AC_ieosvars__mod__equationofstate == AC_irho_tt__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=value(F_EOSVAR2)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_tt1=1/value(F_EOSVAR2)
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_lntt=log(value(F_EOSVAR2))
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cp__mod__equationofstate*AC_gamma_m1__mod__equationofstate*value(F_EOSVAR2)
}
if (AC_lpencil_int__mod__cdata[AC_i_gtt__mod__cparam-1]) {
ac_transformed_pencil_gtt = gradient(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_glntt.x=ac_transformed_pencil_gtt.x*ac_transformed_pencil_tt1
ac_transformed_pencil_glntt.y=ac_transformed_pencil_gtt.y*ac_transformed_pencil_tt1
ac_transformed_pencil_glntt.z=ac_transformed_pencil_gtt.z*ac_transformed_pencil_tt1
}
if (AC_lpencil_int__mod__cdata[AC_i_del2tt__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2tt = laplace(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
tmp_28_29_97=0.0
tmp_28_29_97=tmp_28_29_97+ac_transformed_pencil_glntt.x*ac_transformed_pencil_glntt.x
tmp_28_29_97=tmp_28_29_97+ac_transformed_pencil_glntt.y*ac_transformed_pencil_glntt.y
tmp_28_29_97=tmp_28_29_97+ac_transformed_pencil_glntt.z*ac_transformed_pencil_glntt.z
ac_transformed_pencil_del2lntt=ac_transformed_pencil_del2tt*ac_transformed_pencil_tt1-tmp_28_29_97
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt = hessian(Field(AC_itt__mod__cdata))
ac_transformed_pencil_hlntt[1-1][1-1]=ac_transformed_pencil_hlntt[1-1][1-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.x*ac_transformed_pencil_glntt.x
ac_transformed_pencil_hlntt[1-1][2-1]=ac_transformed_pencil_hlntt[1-1][2-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.x*ac_transformed_pencil_glntt.y
ac_transformed_pencil_hlntt[1-1][3-1]=ac_transformed_pencil_hlntt[1-1][3-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.x*ac_transformed_pencil_glntt.z
ac_transformed_pencil_hlntt[2-1][1-1]=ac_transformed_pencil_hlntt[2-1][1-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.y*ac_transformed_pencil_glntt.x
ac_transformed_pencil_hlntt[2-1][2-1]=ac_transformed_pencil_hlntt[2-1][2-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.y*ac_transformed_pencil_glntt.y
ac_transformed_pencil_hlntt[2-1][3-1]=ac_transformed_pencil_hlntt[2-1][3-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.y*ac_transformed_pencil_glntt.z
ac_transformed_pencil_hlntt[3-1][1-1]=ac_transformed_pencil_hlntt[3-1][1-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.z*ac_transformed_pencil_glntt.x
ac_transformed_pencil_hlntt[3-1][2-1]=ac_transformed_pencil_hlntt[3-1][2-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.z*ac_transformed_pencil_glntt.y
ac_transformed_pencil_hlntt[3-1][3-1]=ac_transformed_pencil_hlntt[3-1][3-1]*ac_transformed_pencil_tt1-ac_transformed_pencil_glntt.z*ac_transformed_pencil_glntt.z
}
if (AC_lpencil_int__mod__cdata[AC_i_del6tt__mod__cparam-1]) {
ac_transformed_pencil_del6tt = del6(Field(AC_ieosvar2__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=AC_cv__mod__equationofstate*(ac_transformed_pencil_lntt-AC_lntt0__mod__equationofstate-AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=AC_cv__mod__equationofstate*AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_rho*ac_transformed_pencil_tt
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_ee=AC_cv__mod__equationofstate*exp(ac_transformed_pencil_lntt)
}
}
else if(AC_ieosvars__mod__equationofstate == AC_ilnrho_cs2__mod__equationofstate || AC_ieosvars__mod__equationofstate == AC_irho_cs2__mod__equationofstate) {
if (AC_leos_isentropic__mod__equationofstate) {
}
else if (AC_leos_isothermal__mod__equationofstate) {
if (AC_lcs_tdep__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate*exp(-AC_cs20_tdep_rate__mod__equationofstate*AC_t__mod__cdata)
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
}
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt.x = 0.0
ac_transformed_pencil_glntt.y = 0.0
ac_transformed_pencil_glntt.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_del2ss__mod__cparam-1]) {
ac_transformed_pencil_del2ss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_del2lnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_gss__mod__cparam-1]) {
ac_transformed_pencil_gss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_glnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_hss__mod__cparam-1]) {
ac_transformed_pencil_hss=-(AC_cp__mod__equationofstate-AC_cv__mod__equationofstate)*ac_transformed_pencil_hlnrho
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=AC_gamma1__mod__equationofstate*ac_transformed_pencil_rho*AC_cs20__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_rho1gpp__mod__cparam-1]) {
ac_transformed_pencil_rho1gpp.x = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.x
ac_transformed_pencil_rho1gpp.y = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.y
ac_transformed_pencil_rho1gpp.z = ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.z
}
}
else if (AC_leos_localisothermal__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=value(F_GLOBAL_CS2)
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt=value(F_GLOBAL_GLNTVEC)
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=ac_transformed_pencil_rho*ac_transformed_pencil_cs2
}
}
else {
}
}
else if(AC_ieosvars__mod__equationofstate == AC_ipp_ss__mod__equationofstate) {
if (AC_lanelastic__mod__cparam) {
if (AC_lanelastic_lin__mod__equationofstate) {
ac_transformed_pencil_pp=value(F_PP)
ac_transformed_pencil_ss=value(F_SS)
ac_transformed_pencil_ttb=AC_cs20__mod__equationofstate*AC_cp1__mod__equationofstate*exp(AC_gamma__mod__equationofstate*value(F_SS_B)*AC_cp1__mod__equationofstate+AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_lnrho)/AC_gamma_m1__mod__equationofstate
ac_transformed_pencil_cs2=AC_cp__mod__equationofstate*ac_transformed_pencil_ttb*AC_gamma_m1__mod__equationofstate
ac_transformed_pencil_tt1=1./ac_transformed_pencil_ttb
ac_transformed_pencil_rho_anel=(value(F_PP)/(value(F_RHO_B)*ac_transformed_pencil_cs2) - value(F_SS)*AC_cp1__mod__equationofstate)
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=value(F_PP)
}
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=value(F_SS)
}
if (AC_lpencil_int__mod__cdata[AC_i_rho__mod__cparam-1]) {
ac_transformed_pencil_rho=value(F_RHO)
}
if (AC_lpencil_int__mod__cdata[AC_i_lnrho__mod__cparam-1]) {
ac_transformed_pencil_lnrho=log(ac_transformed_pencil_rho)
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_gamma__mod__equationofstate*ac_transformed_pencil_pp/ac_transformed_pencil_rho
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate+AC_cv1__mod__equationofstate*ac_transformed_pencil_ss+AC_gamma_m1__mod__equationofstate*(ac_transformed_pencil_lnrho-AC_lnrho0__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_ee=AC_cv__mod__equationofstate*exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_yh__mod__cparam-1]) {
ac_transformed_pencil_yh=AC_impossible__mod__cparam
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1]) {
ac_transformed_pencil_tt1=exp(-ac_transformed_pencil_lntt)
}
}
}
if (AC_leos_isentropic__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_ss__mod__cparam-1]) {
ac_transformed_pencil_ss=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_lnrho__mod__cparam-1]) {
ac_transformed_pencil_lnrho=log(AC_gamma__mod__equationofstate*ac_transformed_pencil_pp/(AC_rho0__mod__equationofstate*AC_cs20__mod__equationofstate))/AC_gamma__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_rho__mod__cparam-1]) {
ac_transformed_pencil_rho=exp(log(AC_gamma__mod__equationofstate*ac_transformed_pencil_pp/(AC_rho0__mod__equationofstate*AC_cs20__mod__equationofstate))/AC_gamma__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=pow((ac_transformed_pencil_pp/AC_pp0__mod__equationofstate),(1.-AC_gamma1__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=(1.-AC_gamma1__mod__equationofstate)*log(AC_gamma__mod__equationofstate*ac_transformed_pencil_pp/(AC_rho0__mod__equationofstate*AC_cs0__mod__equationofstate))
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate*pow((ac_transformed_pencil_pp/AC_pp0__mod__equationofstate),(1.-AC_gamma1__mod__equationofstate))
}
}
else if (AC_leos_isothermal__mod__equationofstate) {
if (AC_lpencil_int__mod__cdata[AC_i_lnrho__mod__cparam-1]) {
ac_transformed_pencil_lnrho=log(AC_gamma__mod__equationofstate*ac_transformed_pencil_pp/(AC_cs20__mod__equationofstate*AC_rho0__mod__equationofstate))-ac_transformed_pencil_lntt
}
if (AC_lpencil_int__mod__cdata[AC_i_rho__mod__cparam-1]) {
ac_transformed_pencil_rho=exp(ac_transformed_pencil_lnrho)
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt.x = 0.0
ac_transformed_pencil_glntt.y = 0.0
ac_transformed_pencil_glntt.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=0.0
}
}
else if (AC_leos_localisothermal__mod__equationofstate) {
}
}
else if(AC_ieosvars__mod__equationofstate == AC_ipp_cs2__mod__equationofstate) {
if (AC_leos_isentropic__mod__equationofstate) {
}
else if (AC_leos_isothermal__mod__equationofstate) {
if (AC_lanelastic__mod__cparam) {
if (AC_lanelastic_lin__mod__equationofstate) {
ac_transformed_pencil_pp=value(F_PP)
ac_transformed_pencil_rho_anel=value(F_PP)/(value(F_RHO_B)*AC_cs20__mod__equationofstate)
}
else {
ac_transformed_pencil_pp=value(F_PP)
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_cs20__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_lnrho__mod__cparam-1]) {
ac_transformed_pencil_lnrho=log(ac_transformed_pencil_pp/AC_cs20__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_rho__mod__cparam-1]) {
ac_transformed_pencil_rho=(ac_transformed_pencil_pp/AC_cs20__mod__equationofstate)
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=AC_lntt0__mod__equationofstate
}
if (AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_glntt.x = 0.0
ac_transformed_pencil_glntt.y = 0.0
ac_transformed_pencil_glntt.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_hlntt__mod__cparam-1]) {
ac_transformed_pencil_hlntt=0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_del2lntt__mod__cparam-1]) {
ac_transformed_pencil_del2lntt=0.0
}
}
}
else if (AC_leos_localisothermal__mod__equationofstate) {
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
if (AC_gamma_m1__mod__equationofstate!=0.0) {
ac_transformed_pencil_ee=(AC_gamma1__mod__equationofstate/AC_gamma_m1__mod__equationofstate)*ac_transformed_pencil_cs2
}
else {
ac_transformed_pencil_ee=ac_transformed_pencil_cs2
}
}
if (AC_lpencil_int__mod__cdata[AC_i_yh__mod__cparam-1]) {
ac_transformed_pencil_yh=AC_impossible__mod__cparam
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=exp(ac_transformed_pencil_lntt)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1]) {
ac_transformed_pencil_tt1=exp(-ac_transformed_pencil_lntt)
}
}
else if(AC_ieosvars__mod__equationofstate == AC_irho_eth__mod__equationofstate || AC_ieosvars__mod__equationofstate == AC_ilnrho_eth__mod__equationofstate) {
if (AC_lstratz__mod__cdata) {
if (AC_lpencil_int__mod__cdata[AC_i_eths__mod__cparam-1]) {
ac_transformed_pencil_eths = 1.0 + value(F_ETH)
}
if (AC_lpencil_int__mod__cdata[AC_i_geths__mod__cparam-1]) {
ac_transformed_pencil_geths = gradient(Field(AC_ieth__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_eth__mod__cparam-1]) {
ac_transformed_pencil_eth = AC_eth0z__mod__equationofstate[AC_n__mod__cdata-1] * ac_transformed_pencil_eths
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_eth__mod__cparam-1]) {
ac_transformed_pencil_eth = value(F_ETH)
}
if (AC_lpencil_int__mod__cdata[AC_i_geth__mod__cparam-1]) {
ac_transformed_pencil_geth = gradient(Field(AC_ieth__mod__cdata))
}
if (AC_lpencil_int__mod__cdata[AC_i_del2eth__mod__cparam-1]) {
ac_transformed_pencil_del2eth = laplace(Field(AC_ieth__mod__cdata))
}
}
if (AC_lpencil_int__mod__cdata[AC_i_cs2__mod__cparam-1]) {
ac_transformed_pencil_cs2=AC_gamma__mod__equationofstate*AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_eth*ac_transformed_pencil_rho1
}
if (AC_lpencil_int__mod__cdata[AC_i_pp__mod__cparam-1]) {
ac_transformed_pencil_pp=AC_gamma_m1__mod__equationofstate*ac_transformed_pencil_eth
}
if (AC_lpencil_int__mod__cdata[AC_i_ee__mod__cparam-1]) {
ac_transformed_pencil_ee=ac_transformed_pencil_rho1*ac_transformed_pencil_eth
}
if (AC_lpencil_int__mod__cdata[AC_i_tt__mod__cparam-1]) {
ac_transformed_pencil_tt=ac_transformed_pencil_cv1*ac_transformed_pencil_rho1*ac_transformed_pencil_eth
}
if (AC_lpencil_int__mod__cdata[AC_i_lntt__mod__cparam-1]) {
ac_transformed_pencil_lntt=log(ac_transformed_pencil_tt)
}
if (AC_lpencil_int__mod__cdata[AC_i_tt1__mod__cparam-1]) {
ac_transformed_pencil_tt1=1/ac_transformed_pencil_tt
}
if (AC_lpencil_int__mod__cdata[AC_i_gtt__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_glntt__mod__cparam-1]) {
ac_transformed_pencil_gtt.x=ac_transformed_pencil_rho1*(ac_transformed_pencil_cv1*ac_transformed_pencil_geth.x-ac_transformed_pencil_tt*ac_transformed_pencil_grho.x)
ac_transformed_pencil_glntt.x=ac_transformed_pencil_tt1*ac_transformed_pencil_gtt.x
ac_transformed_pencil_gtt.y=ac_transformed_pencil_rho1*(ac_transformed_pencil_cv1*ac_transformed_pencil_geth.y-ac_transformed_pencil_tt*ac_transformed_pencil_grho.y)
ac_transformed_pencil_glntt.y=ac_transformed_pencil_tt1*ac_transformed_pencil_gtt.y
ac_transformed_pencil_gtt.z=ac_transformed_pencil_rho1*(ac_transformed_pencil_cv1*ac_transformed_pencil_geth.z-ac_transformed_pencil_tt*ac_transformed_pencil_grho.z)
ac_transformed_pencil_glntt.z=ac_transformed_pencil_tt1*ac_transformed_pencil_gtt.z
}
if (AC_lpencil_int__mod__cdata[AC_i_del2tt__mod__cparam-1]) {
ac_transformed_pencil_del2tt=ac_transformed_pencil_rho1*(ac_transformed_pencil_cv1*ac_transformed_pencil_del2eth-ac_transformed_pencil_tt*ac_transformed_pencil_del2rho-2*sum(ac_transformed_pencil_grho*ac_transformed_pencil_gtt))
}
}
else {
}
if (AC_lcs_as_aux__mod__equationofstate || AC_lcs_as_comaux__mod__equationofstate) {
DF_CS=sqrt(ac_transformed_pencil_cs2)
}
if (AC_lpencil_int__mod__cdata[AC_i_shock__mod__cparam-1]) {
ac_transformed_pencil_shock=0.
}
if (AC_lpencil_int__mod__cdata[AC_i_gshock__mod__cparam-1]) {
ac_transformed_pencil_gshock.x = 0.
ac_transformed_pencil_gshock.y = 0.
ac_transformed_pencil_gshock.z = 0.
}
if (AC_lpencil_int__mod__cdata[AC_i_ma2__mod__cparam-1]) {
ac_transformed_pencil_ma2=ac_transformed_pencil_u2/ac_transformed_pencil_cs2
}
if (AC_lpencil_int__mod__cdata[AC_i_fpres__mod__cparam-1]) {
if (AC_lstratz__mod__cdata) {
ac_transformed_pencil_fpres.x = -ac_transformed_pencil_cs2 * ac_transformed_pencil_glnrhos.x
ac_transformed_pencil_fpres.y = -ac_transformed_pencil_cs2 * ac_transformed_pencil_glnrhos.y
ac_transformed_pencil_fpres.z = -ac_transformed_pencil_cs2 * ac_transformed_pencil_glnrhos.z
}
else {
if (AC_llocal_iso__mod__cdata) {
ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.x+ac_transformed_pencil_glntt.x)
}
else {
if (AC_ldensity__mod__cparam && AC_lrelativistic_eos__mod__density) {
if (!AC_lconservative__mod__hydro) {
ac_transformed_pencil_fpres.x=-0.75*ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.x
}
}
else {
ac_transformed_pencil_fpres.x=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.x
}
}
if (AC_ldensity__mod__cparam) {
if (AC_lffree__mod__density) {
ac_transformed_pencil_fpres.x=ac_transformed_pencil_fpres.x*AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]
}
}
if (AC_llocal_iso__mod__cdata) {
ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.y+ac_transformed_pencil_glntt.y)
}
else {
if (AC_ldensity__mod__cparam && AC_lrelativistic_eos__mod__density) {
if (!AC_lconservative__mod__hydro) {
ac_transformed_pencil_fpres.y=-0.75*ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.y
}
}
else {
ac_transformed_pencil_fpres.y=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.y
}
}
if (AC_ldensity__mod__cparam) {
if (AC_lffree__mod__density) {
ac_transformed_pencil_fpres.y=ac_transformed_pencil_fpres.y*AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]
}
}
if (AC_llocal_iso__mod__cdata) {
ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2*(ac_transformed_pencil_glnrho.z+ac_transformed_pencil_glntt.z)
}
else {
if (AC_ldensity__mod__cparam && AC_lrelativistic_eos__mod__density) {
if (!AC_lconservative__mod__hydro) {
ac_transformed_pencil_fpres.z=-0.75*ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.z
}
}
else {
ac_transformed_pencil_fpres.z=-ac_transformed_pencil_cs2*ac_transformed_pencil_glnrho.z
}
}
if (AC_ldensity__mod__cparam) {
if (AC_lffree__mod__density) {
ac_transformed_pencil_fpres.z=ac_transformed_pencil_fpres.z*AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]
}
}
}
}
if (AC_lpencil_int__mod__cdata[AC_i_tcond__mod__cparam-1]) {
ac_transformed_pencil_tcond=0.
}
if (AC_lpencil_int__mod__cdata[AC_i_sglntt__mod__cparam-1]) {
ac_transformed_pencil_sglntt.x = 0.
ac_transformed_pencil_sglntt.y = 0.
ac_transformed_pencil_sglntt.z = 0.
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
if (AC_leos__mod__cparam && AC_ldensity__mod__cparam && AC_lhydro__mod__cparam) {
ac_transformed_pencil_advec_cs2=ac_transformed_pencil_cs2*dxyz_2__mod__cdata
}
}
ac_transformed_pencil_fvisc.x = 0.0
ac_transformed_pencil_fvisc.y = 0.0
ac_transformed_pencil_fvisc.z = 0.0
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=0.0
}
ldiffus_total_63_97 = AC_lfirst__mod__cdata  &&  AC_ldt__mod__cdata  ||  AC_lpencil_int__mod__cdata[AC_i_diffus_total__mod__cparam-1]
ldiffus_total3_63_97 = AC_lfirst__mod__cdata  &&  AC_ldt__mod__cdata  ||  AC_lpencil_int__mod__cdata[AC_i_diffus_total3__mod__cparam-1]
if (ldiffus_total_63_97 || ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total=0.0
ac_transformed_pencil_diffus_total2=0.0
ac_transformed_pencil_diffus_total3=0.0
}
if (! AC_limplicit_viscosity__mod__viscosity  &&  AC_lvisc_simplified__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu__mod__viscosity*ac_transformed_pencil_del2u
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
if (AC_lboussinesq__mod__cparam) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2.*AC_nu__mod__viscosity*ac_transformed_pencil_sij2
}
else {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+AC_nu__mod__viscosity*ac_transformed_pencil_o2
}
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+AC_nu__mod__viscosity
}
}
if (AC_lvisc_nu_non_newtonian__mod__viscosity) {
deljskl2_63_97.x = 0.
deljskl2_63_97.y = 0.
deljskl2_63_97.z = 0.
gradnu_effective_63_97=0.
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[1-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][1-1] + ac_transformed_pencil_uijk[1-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][1-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[1-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][1-1] + ac_transformed_pencil_uijk[2-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][2-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[1-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][1-1] + ac_transformed_pencil_uijk[3-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][3-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[2-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][2-1] + ac_transformed_pencil_uijk[1-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][1-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[2-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][2-1] + ac_transformed_pencil_uijk[2-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][2-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[2-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][2-1] + ac_transformed_pencil_uijk[3-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][3-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[3-1][1-1][1-1]*ac_transformed_pencil_sij[1-1][3-1] + ac_transformed_pencil_uijk[1-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][1-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[3-1][2-1][1-1]*ac_transformed_pencil_sij[2-1][3-1] + ac_transformed_pencil_uijk[2-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][2-1]
deljskl2_63_97 .x = deljskl2_63_97 .x +  ac_transformed_pencil_uijk[3-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][3-1] + ac_transformed_pencil_uijk[3-1][3-1][1-1]*ac_transformed_pencil_sij[3-1][3-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[1-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][1-1] + ac_transformed_pencil_uijk[1-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][1-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[1-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][1-1] + ac_transformed_pencil_uijk[2-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][2-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[1-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][1-1] + ac_transformed_pencil_uijk[3-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][3-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[2-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][2-1] + ac_transformed_pencil_uijk[1-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][1-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[2-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][2-1] + ac_transformed_pencil_uijk[2-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][2-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[2-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][2-1] + ac_transformed_pencil_uijk[3-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][3-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[3-1][1-1][2-1]*ac_transformed_pencil_sij[1-1][3-1] + ac_transformed_pencil_uijk[1-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][1-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[3-1][2-1][2-1]*ac_transformed_pencil_sij[2-1][3-1] + ac_transformed_pencil_uijk[2-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][2-1]
deljskl2_63_97 .y = deljskl2_63_97 .y +  ac_transformed_pencil_uijk[3-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][3-1] + ac_transformed_pencil_uijk[3-1][3-1][2-1]*ac_transformed_pencil_sij[3-1][3-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[1-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][1-1] + ac_transformed_pencil_uijk[1-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][1-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[1-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][1-1] + ac_transformed_pencil_uijk[2-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][2-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[1-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][1-1] + ac_transformed_pencil_uijk[3-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][3-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[2-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][2-1] + ac_transformed_pencil_uijk[1-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][1-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[2-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][2-1] + ac_transformed_pencil_uijk[2-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][2-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[2-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][2-1] + ac_transformed_pencil_uijk[3-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][3-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[3-1][1-1][3-1]*ac_transformed_pencil_sij[1-1][3-1] + ac_transformed_pencil_uijk[1-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][1-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[3-1][2-1][3-1]*ac_transformed_pencil_sij[2-1][3-1] + ac_transformed_pencil_uijk[2-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][2-1]
deljskl2_63_97 .z = deljskl2_63_97 .z +  ac_transformed_pencil_uijk[3-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][3-1] + ac_transformed_pencil_uijk[3-1][3-1][3-1]*ac_transformed_pencil_sij[3-1][3-1]
fvisc_nnewton2_63_97.x = sum(ac_transformed_pencil_sij.row(1-1)*deljskl2_63_97)
fvisc_nnewton2_63_97.y = sum(ac_transformed_pencil_sij.row(2-1)*deljskl2_63_97)
fvisc_nnewton2_63_97.z = sum(ac_transformed_pencil_sij.row(3-1)*deljskl2_63_97)
if(AC_string_enum_nnewton_type__mod__viscosity == AC_string_enum_carreau_string__mod__cparam) {
ac_transformed_pencil_nu = AC_nu_infinity__mod__viscosity +  (AC_nu0__mod__viscosity-AC_nu_infinity__mod__viscosity)/pow((1+(AC_non_newton_lambda__mod__viscosity*AC_non_newton_lambda__mod__viscosity)*ac_transformed_pencil_sij2),((1.-AC_carreau_exponent__mod__viscosity)/2.))
gradnu_effective_63_97 = (AC_non_newton_lambda__mod__viscosity*AC_non_newton_lambda__mod__viscosity)*0.5*(AC_carreau_exponent__mod__viscosity-1.)*  (AC_nu0__mod__viscosity-AC_nu_infinity__mod__viscosity)/pow((1+(AC_non_newton_lambda__mod__viscosity*AC_non_newton_lambda__mod__viscosity)*ac_transformed_pencil_sij2),(1.5-AC_carreau_exponent__mod__viscosity/2.))
}
else if(AC_string_enum_nnewton_type__mod__viscosity == AC_string_enum_step_string__mod__cparam) {
step_vector_return_value_33_35_63_97 = 0.5*(1+tanh((ac_transformed_pencil_sij2-AC_nnewton_tscale__mod__viscosity*AC_nnewton_tscale__mod__viscosity)/(AC_nnewton_step_width__mod__viscosity+AC_tini__mod__cparam)))
ac_transformed_pencil_nu = AC_nu0__mod__viscosity+(AC_nu_infinity__mod__viscosity-AC_nu0__mod__viscosity)*step_vector_return_value_33_35_63_97
arg_34_35_63_97 = abs((ac_transformed_pencil_sij2-AC_nnewton_tscale__mod__viscosity*AC_nnewton_tscale__mod__viscosity)/(AC_nnewton_step_width__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_34_35_63_97)>=8.)  {
der_step_return_value_34_35_63_97 = 2./AC_nnewton_step_width__mod__viscosity*exp(-2.*abs(arg_34_35_63_97))
}
else {
der_step_return_value_34_35_63_97 = 0.5/(AC_nnewton_step_width__mod__viscosity*cosh(arg_34_35_63_97)*cosh(arg_34_35_63_97))
}
gradnu_effective_63_97 = (AC_nu_infinity__mod__viscosity-AC_nu0__mod__viscosity)*der_step_return_value_34_35_63_97
}
else {
}
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_nu*ac_transformed_pencil_del2u.x + gradnu_effective_63_97*fvisc_nnewton2_63_97.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_nu*ac_transformed_pencil_del2u.y + gradnu_effective_63_97*fvisc_nnewton2_63_97.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_nu*ac_transformed_pencil_del2u.z + gradnu_effective_63_97*fvisc_nnewton2_63_97.z
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+AC_nu__mod__viscosity
}
}
if (AC_lvisc_rho_nu_const__mod__viscosity) {
if (AC_lvisc_rho_nu_const_prefact__mod__viscosity) {
murho1_63_97=AC_nu__mod__viscosity
}
else {
murho1_63_97=AC_nu__mod__viscosity*ac_transformed_pencil_rho1
}
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + murho1_63_97*(ac_transformed_pencil_del2u.x+1.0/3.0*ac_transformed_pencil_graddivu.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + murho1_63_97*(ac_transformed_pencil_del2u.y+1.0/3.0*ac_transformed_pencil_graddivu.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + murho1_63_97*(ac_transformed_pencil_del2u.z+1.0/3.0*ac_transformed_pencil_graddivu.z)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*murho1_63_97*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+murho1_63_97
}
}
if (AC_lvisc_rho_nu_const_bulk__mod__viscosity) {
zetarho1_63_97=AC_zeta__mod__viscosity*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+zetarho1_63_97*ac_transformed_pencil_graddivu.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+zetarho1_63_97*ac_transformed_pencil_graddivu.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+zetarho1_63_97*ac_transformed_pencil_graddivu.z
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+zetarho1_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_divu
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+zetarho1_63_97
}
}
if (AC_lvisc_sqrtrho_nu_const__mod__viscosity) {
murho1_63_97=AC_nu__mod__viscosity*sqrt(ac_transformed_pencil_rho1)
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + murho1_63_97*(ac_transformed_pencil_del2u.x+1.0/3.0*ac_transformed_pencil_graddivu.x + ac_transformed_pencil_sglnrho.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + murho1_63_97*(ac_transformed_pencil_del2u.y+1.0/3.0*ac_transformed_pencil_graddivu.y + ac_transformed_pencil_sglnrho.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + murho1_63_97*(ac_transformed_pencil_del2u.z+1.0/3.0*ac_transformed_pencil_graddivu.z + ac_transformed_pencil_sglnrho.z)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*murho1_63_97*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+murho1_63_97
}
}
if (AC_lvisc_mu_cspeed__mod__viscosity) {
mutt_63_97=AC_nu__mod__viscosity*ac_transformed_pencil_rho1*exp(AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_lntt)
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + mutt_63_97*(ac_transformed_pencil_del2u.x+1.0/3.0*ac_transformed_pencil_graddivu.x+2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + mutt_63_97*(ac_transformed_pencil_del2u.y+1.0/3.0*ac_transformed_pencil_graddivu.y+2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + mutt_63_97*(ac_transformed_pencil_del2u.z+1.0/3.0*ac_transformed_pencil_graddivu.z+2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.z)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*mutt_63_97*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+mutt_63_97
}
}
if (AC_lvisc_spitzer__mod__viscosity) {
if (AC_nu_spitzer_max__mod__viscosity != 0.0) {
tmp3_63_97=AC_nu_spitzer__mod__viscosity*ac_transformed_pencil_rho1*exp(2.5*ac_transformed_pencil_lntt)
mutt_63_97=AC_nu_spitzer_max__mod__viscosity*tmp3_63_97/sqrt(pow(AC_nu_spitzer_max__mod__viscosity,2.)+pow(tmp3_63_97,2.))
}
else {
mutt_63_97=AC_nu_spitzer__mod__viscosity*ac_transformed_pencil_rho1*exp(2.5*ac_transformed_pencil_lntt)
}
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + mutt_63_97*(ac_transformed_pencil_del2u.x+1.0/3.0*ac_transformed_pencil_graddivu.x+5.*ac_transformed_pencil_sglntt.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + mutt_63_97*(ac_transformed_pencil_del2u.y+1.0/3.0*ac_transformed_pencil_graddivu.y+5.*ac_transformed_pencil_sglntt.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + mutt_63_97*(ac_transformed_pencil_del2u.z+1.0/3.0*ac_transformed_pencil_graddivu.z+5.*ac_transformed_pencil_sglntt.z)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*mutt_63_97*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+mutt_63_97
}
}
if (AC_lvisc_nu_cspeed__mod__viscosity) {
mutt_63_97=AC_nu__mod__viscosity*exp(AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_lntt)
if (AC_ldensity__mod__cparam) {
ac_transformed_pencil_fvisc.x =  ac_transformed_pencil_fvisc.x + 2*mutt_63_97*ac_transformed_pencil_sglnrho.x  + mutt_63_97*(ac_transformed_pencil_del2u.x + 1./3.*ac_transformed_pencil_graddivu.x + 2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.x)
ac_transformed_pencil_fvisc.y =  ac_transformed_pencil_fvisc.y + 2*mutt_63_97*ac_transformed_pencil_sglnrho.y  + mutt_63_97*(ac_transformed_pencil_del2u.y + 1./3.*ac_transformed_pencil_graddivu.y + 2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.y)
ac_transformed_pencil_fvisc.z =  ac_transformed_pencil_fvisc.z + 2*mutt_63_97*ac_transformed_pencil_sglnrho.z  + mutt_63_97*(ac_transformed_pencil_del2u.z + 1./3.*ac_transformed_pencil_graddivu.z + 2*AC_nu_cspeed__mod__viscosity*ac_transformed_pencil_sglntt.z)
}
else {
if (AC_lmeanfield_nu__mod__viscosity) {
if (AC_meanfield_nub__mod__viscosity!=0.) {
tmp_63_97 = mutt_63_97*ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc = ac_transformed_pencil_fvisc + 1./sqrt(1.+ac_transformed_pencil_b2/AC_meanfield_nub__mod__viscosity*AC_meanfield_nub__mod__viscosity)*ac_transformed_pencil_fvisc+tmp_63_97
}
}
else {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+mutt_63_97*(ac_transformed_pencil_del2u.x+1.0/3.0*ac_transformed_pencil_graddivu.x+ac_transformed_pencil_sglntt.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+mutt_63_97*(ac_transformed_pencil_del2u.y+1.0/3.0*ac_transformed_pencil_graddivu.y+ac_transformed_pencil_sglntt.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+mutt_63_97*(ac_transformed_pencil_del2u.z+1.0/3.0*ac_transformed_pencil_graddivu.z+ac_transformed_pencil_sglntt.z)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*mutt_63_97*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+mutt_63_97
}
}
if (AC_lvisc_nu_const__mod__viscosity) {
if (AC_ldensity__mod__cparam) {
fac_63_97=AC_nu__mod__viscosity
if (AC_damp_sound__mod__viscosity!=0.) {
fac_63_97 = fac_63_97+AC_damp_sound__mod__viscosity*abs(ac_transformed_pencil_divu)
}
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + fac_63_97*(ac_transformed_pencil_del2u.x + 2.*ac_transformed_pencil_sglnrho.x + 1./3.*ac_transformed_pencil_graddivu.x)
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + fac_63_97*(ac_transformed_pencil_del2u.y + 2.*ac_transformed_pencil_sglnrho.y + 1./3.*ac_transformed_pencil_graddivu.y)
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + fac_63_97*(ac_transformed_pencil_del2u.z + 2.*ac_transformed_pencil_sglnrho.z + 1./3.*ac_transformed_pencil_graddivu.z)
}
else {
if (AC_lmeanfield_nu__mod__viscosity) {
if (AC_meanfield_nub__mod__viscosity!=0.) {
ac_transformed_pencil_fvisc = ac_transformed_pencil_fvisc + 1./sqrt(1.+ac_transformed_pencil_b2/AC_meanfield_nub__mod__viscosity*AC_meanfield_nub__mod__viscosity)*ac_transformed_pencil_fvisc+AC_nu__mod__viscosity*(ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu)
}
}
else {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu__mod__viscosity*(ac_transformed_pencil_del2u+1.0/3.0*ac_transformed_pencil_graddivu)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*AC_nu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+AC_nu__mod__viscosity
}
}
if (AC_lvisc_nu_tdep__mod__viscosity  ||  AC_lvisc_hyper3_simplified_tdep__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+2*AC_nu_tdep__mod__viscosity*ac_transformed_pencil_sglnrho+AC_nu_tdep__mod__viscosity*(ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*AC_nu_tdep__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+AC_nu_tdep__mod__viscosity
}
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_tdep__mod__viscosity
}
}
if (AC_lvisc_mixture__mod__viscosity) {
sgradnu_63_97 = ac_transformed_pencil_sij*ac_transformed_pencil_gradnu
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_nu*(ac_transformed_pencil_del2u.x+1./3.*ac_transformed_pencil_graddivu.x) + 2*sgradnu_63_97.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_nu*(ac_transformed_pencil_del2u.y+1./3.*ac_transformed_pencil_graddivu.y) + 2*sgradnu_63_97.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_nu*(ac_transformed_pencil_del2u.z+1./3.*ac_transformed_pencil_graddivu.z) + 2*sgradnu_63_97.z
if (AC_ldensity__mod__cparam) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + 2*ac_transformed_pencil_nu*ac_transformed_pencil_sglnrho.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + 2*ac_transformed_pencil_nu*ac_transformed_pencil_sglnrho.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + 2*ac_transformed_pencil_nu*ac_transformed_pencil_sglnrho.z
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*ac_transformed_pencil_nu*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+ac_transformed_pencil_nu
}
}
if (AC_lvisc_nu_profx__mod__viscosity || AC_lvisc_nu_profr__mod__viscosity) {
if (AC_lvisc_nu_profx__mod__viscosity) {
tmp3_63_97=AC_x__mod__cdata[vertexIdx.x]
}
if (AC_lvisc_nu_profr__mod__viscosity) {
if (AC_lspherical_coords__mod__cdata || AC_lsphere_in_a_box__mod__cdata) {
tmp3_63_97=ac_transformed_pencil_r_mn
}
else {
tmp3_63_97=ac_transformed_pencil_rcyl_mn
}
}
step_vector_return_value_36_63_97 = 0.5*(1+tanh((tmp3_63_97-AC_xnu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam)))
step_vector_return_value_37_63_97 = 0.5*(1+tanh((tmp3_63_97-AC_xnu2__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam)))
pnu__mod__viscosity = AC_nu__mod__viscosity + (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*(step_vector_return_value_36_63_97  -  step_vector_return_value_37_63_97)
arg_38_63_97 = abs((tmp3_63_97-AC_xnu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_38_63_97)>=8.)  {
der_step_return_value_38_63_97 = 2./AC_widthnu__mod__viscosity*exp(-2.*abs(arg_38_63_97))
}
else {
der_step_return_value_38_63_97 = 0.5/(AC_widthnu__mod__viscosity*cosh(arg_38_63_97)*cosh(arg_38_63_97))
}
arg_39_63_97 = abs((tmp3_63_97-AC_xnu2__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_39_63_97)>=8.)  {
der_step_return_value_39_63_97 = 2./AC_widthnu__mod__viscosity*exp(-2.*abs(arg_39_63_97))
}
else {
der_step_return_value_39_63_97 = 0.5/(AC_widthnu__mod__viscosity*cosh(arg_39_63_97)*cosh(arg_39_63_97))
}
tmp4_63_97 = (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*(der_step_return_value_38_63_97-der_step_return_value_39_63_97)
gradnu_63_97.x = 0.
gradnu_63_97.y = 0.
gradnu_63_97.z = 0.
if (AC_lvisc_nu_profx__mod__viscosity || (AC_lvisc_nu_profr__mod__viscosity && (AC_lcylindrical_coords__mod__cdata || AC_lspherical_coords__mod__cdata))) {
gradnu_63_97.x=tmp4_63_97
gradnu_63_97.y=0
gradnu_63_97.z=0
}
else if (AC_lvisc_nu_profr__mod__viscosity && AC_lcylinder_in_a_box__mod__cdata) {
gradnu_63_97.x=tmp4_63_97*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
gradnu_63_97.y=tmp4_63_97*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
}
else if (AC_lvisc_nu_profr__mod__viscosity && AC_lsphere_in_a_box__mod__cdata) {
gradnu_63_97.x=tmp4_63_97*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_r_mn1
gradnu_63_97.y=tmp4_63_97*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_r_mn1
gradnu_63_97.z=tmp4_63_97*AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nu_profr_powerlaw__mod__viscosity) {
if (!AC_luse_nu_rmn_prof__mod__viscosity) {
if (AC_nu_rcyl_min__mod__viscosity==AC_impossible__mod__cparam) {
pnu__mod__viscosity = AC_nu__mod__viscosity*pow((ac_transformed_pencil_rcyl_mn/AC_xnu__mod__viscosity),(-AC_pnlaw__mod__viscosity))
}
else {
pnu__mod__viscosity = AC_nu__mod__viscosity*pow((max(AC_nu_rcyl_min__mod__viscosity,ac_transformed_pencil_rcyl_mn)/AC_xnu__mod__viscosity),(-AC_pnlaw__mod__viscosity))
}
if (AC_lcylindrical_coords__mod__cdata) {
gradnu_63_97.x = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow((ac_transformed_pencil_rcyl_mn/AC_xnu__mod__viscosity),(-AC_pnlaw__mod__viscosity-1))*1/AC_xnu__mod__viscosity
gradnu_63_97.y = 0.
gradnu_63_97.z = 0.
}
else if (AC_lspherical_coords__mod__cdata) {
if (AC_nu_rcyl_min__mod__viscosity==AC_impossible__mod__cparam) {
gradnu_63_97.x = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow(ac_transformed_pencil_rcyl_mn,(-AC_pnlaw__mod__viscosity-1))*AC_sinth__mod__cdata[AC_m__mod__cdata-1]
gradnu_63_97.y = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow(ac_transformed_pencil_rcyl_mn,(-AC_pnlaw__mod__viscosity-1))*AC_costh__mod__cdata[AC_m__mod__cdata-1]
gradnu_63_97.z = 0.
}
else {
gradnu_63_97.x = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow(max(AC_nu_rcyl_min__mod__viscosity,ac_transformed_pencil_rcyl_mn),(-AC_pnlaw__mod__viscosity-1))*AC_sinth__mod__cdata[AC_m__mod__cdata-1]
gradnu_63_97.y = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow(max(AC_nu_rcyl_min__mod__viscosity,ac_transformed_pencil_rcyl_mn),(-AC_pnlaw__mod__viscosity-1))*AC_costh__mod__cdata[AC_m__mod__cdata-1]
gradnu_63_97.z = 0.
}
}
else {
}
}
else {
pnu__mod__viscosity = AC_nu__mod__viscosity*pow((ac_transformed_pencil_r_mn/AC_xnu__mod__viscosity),(-AC_pnlaw__mod__viscosity))
if (AC_lspherical_coords__mod__cdata) {
gradnu_63_97.x = -AC_pnlaw__mod__viscosity*AC_nu__mod__viscosity*pow((ac_transformed_pencil_r_mn/AC_xnu__mod__viscosity),(-AC_pnlaw__mod__viscosity-1))*1/AC_xnu__mod__viscosity
gradnu_63_97.y = 0.
gradnu_63_97.z = 0.
}
else {
}
}
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nu_profr_twosteps__mod__viscosity) {
if (AC_lspherical_coords__mod__cdata || AC_lsphere_in_a_box__mod__cdata) {
tmp3_63_97=ac_transformed_pencil_r_mn
}
else {
tmp3_63_97=ac_transformed_pencil_rcyl_mn
}
step_vector_return_value_41_63_97 = 0.5*(1+tanh((tmp3_63_97-AC_xnu2__mod__viscosity)/(AC_widthnu2__mod__viscosity+AC_tini__mod__cparam)))
prof2_63_97    = step_vector_return_value_41_63_97
step_vector_return_value_42_63_97 = 0.5*(1+tanh((tmp3_63_97-AC_xnu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam)))
prof_63_97     = step_vector_return_value_42_63_97-prof2_63_97
arg_43_63_97 = abs((tmp3_63_97-AC_xnu2__mod__viscosity)/(AC_widthnu2__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_43_63_97)>=8.)  {
der_step_return_value_43_63_97 = 2./AC_widthnu2__mod__viscosity*exp(-2.*abs(arg_43_63_97))
}
else {
der_step_return_value_43_63_97 = 0.5/(AC_widthnu2__mod__viscosity*cosh(arg_43_63_97)*cosh(arg_43_63_97))
}
derprof2_63_97 = der_step_return_value_43_63_97
arg_44_63_97 = abs((tmp3_63_97-AC_xnu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_44_63_97)>=8.)  {
der_step_return_value_44_63_97 = 2./AC_widthnu__mod__viscosity*exp(-2.*abs(arg_44_63_97))
}
else {
der_step_return_value_44_63_97 = 0.5/(AC_widthnu__mod__viscosity*cosh(arg_44_63_97)*cosh(arg_44_63_97))
}
derprof_63_97  = der_step_return_value_44_63_97-derprof2_63_97
pnu__mod__viscosity  = AC_nu__mod__viscosity + (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*prof_63_97+(AC_nu__mod__viscosity*(AC_nu_jump2__mod__viscosity-1.))*prof2_63_97
tmp4_63_97 = AC_nu__mod__viscosity + (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*derprof_63_97+(AC_nu__mod__viscosity*(AC_nu_jump2__mod__viscosity-1.))*derprof2_63_97
gradnu_63_97.x = 0.
gradnu_63_97.y = 0.
gradnu_63_97.z = 0.
if (AC_lvisc_nu_profx__mod__viscosity || (AC_lvisc_nu_profr_twosteps__mod__viscosity && (AC_lcylindrical_coords__mod__cdata || AC_lspherical_coords__mod__cdata))) {
gradnu_63_97.x=tmp4_63_97
gradnu_63_97.y=0
gradnu_63_97.z=0
}
else if (AC_lvisc_nu_profr_twosteps__mod__viscosity && AC_lcylinder_in_a_box__mod__cdata) {
gradnu_63_97.x=tmp4_63_97*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_rcyl_mn1
gradnu_63_97.y=tmp4_63_97*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_rcyl_mn1
}
else if (AC_lvisc_nu_profr_twosteps__mod__viscosity && AC_lsphere_in_a_box__mod__cdata) {
gradnu_63_97.x=tmp4_63_97*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_r_mn1
gradnu_63_97.y=tmp4_63_97*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_r_mn1
gradnu_63_97.z=tmp4_63_97*AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nu_profy_bound__mod__viscosity) {
if (AC_lspherical_coords__mod__cdata || AC_lsphere_in_a_box__mod__cdata) {
tmp3_63_97=ac_transformed_pencil_r_mn
}
else if (AC_lcylindrical_coords__mod__cdata) {
tmp3_63_97=ac_transformed_pencil_rcyl_mn
}
else {
tmp3_63_97=1.0
}
tmp4_63_97    = AC_y__mod__cdata[AC_m__mod__cdata-1]
step_vector_return_value_46_63_97 = 0.5*(1+tanh((tmp4_63_97-AC_xyz1__mod__cdata.y-3*AC_dynu__mod__viscosity)/(AC_dynu__mod__viscosity+AC_tini__mod__cparam)))
step_vector_return_value_47_63_97 = 0.5*(1+tanh((tmp4_63_97-AC_xyz0__mod__cdata.y+3*AC_dynu__mod__viscosity)/(AC_dynu__mod__viscosity+AC_tini__mod__cparam)))
prof_63_97    =     step_vector_return_value_46_63_97 +     step_vector_return_value_47_63_97
arg_48_63_97 = abs((tmp4_63_97-AC_xyz1__mod__cdata.y-3*AC_dynu__mod__viscosity)/(AC_dynu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_48_63_97)>=8.)  {
der_step_return_value_48_63_97 = 2./AC_dynu__mod__viscosity*exp(-2.*abs(arg_48_63_97))
}
else {
der_step_return_value_48_63_97 = 0.5/(AC_dynu__mod__viscosity*cosh(arg_48_63_97)*cosh(arg_48_63_97))
}
arg_49_63_97 = abs((tmp4_63_97-AC_xyz0__mod__cdata.y+3*AC_dynu__mod__viscosity)/(AC_dynu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_49_63_97)>=8.)  {
der_step_return_value_49_63_97 = 2./AC_dynu__mod__viscosity*exp(-2.*abs(arg_49_63_97))
}
else {
der_step_return_value_49_63_97 = 0.5/(AC_dynu__mod__viscosity*cosh(arg_49_63_97)*cosh(arg_49_63_97))
}
derprof_63_97 = der_step_return_value_48_63_97 + der_step_return_value_49_63_97
pnu__mod__viscosity  = AC_nu__mod__viscosity + (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*prof_63_97
gradnu_63_97.x = 0.
gradnu_63_97.y = (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*derprof_63_97/tmp3_63_97
gradnu_63_97.z = 0.
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nut_from_magnetic__mod__viscosity) {
pnu__mod__viscosity=AC_prm_turb__mod__viscosity*AC_etat_x__mod__magnetic_meanfield[vertexIdx.x-NGHOST_VAL]*AC_etat_y__mod__magnetic_meanfield[AC_m__mod__cdata-1]*AC_etat_z__mod__magnetic_meanfield[AC_n__mod__cdata-1]
gradnu_63_97.x=AC_prm_turb__mod__viscosity*AC_detat_x__mod__magnetic_meanfield[vertexIdx.x-NGHOST_VAL]*AC_etat_y__mod__magnetic_meanfield[AC_m__mod__cdata-1]*AC_etat_z__mod__magnetic_meanfield[AC_n__mod__cdata-1]
gradnu_63_97.y=AC_prm_turb__mod__viscosity*AC_etat_x__mod__magnetic_meanfield[vertexIdx.x-NGHOST_VAL]*AC_detat_y__mod__magnetic_meanfield[AC_m__mod__cdata-1]*AC_etat_z__mod__magnetic_meanfield[AC_n__mod__cdata-1]
gradnu_63_97.z=AC_prm_turb__mod__viscosity*AC_etat_x__mod__magnetic_meanfield[vertexIdx.x-NGHOST_VAL]*AC_etat_y__mod__magnetic_meanfield[AC_m__mod__cdata-1]*AC_detat_z__mod__magnetic_meanfield[AC_n__mod__cdata-1]
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nu_prof__mod__viscosity) {
step_vector_return_value_50_63_97 = 0.5*(1+tanh((ac_transformed_pencil_z_mn-AC_znu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam)))
pnu__mod__viscosity = AC_nu__mod__viscosity + (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*step_vector_return_value_50_63_97
gradnu_63_97.x = 0.
gradnu_63_97.y = 0.
arg_51_63_97 = abs((ac_transformed_pencil_z_mn-AC_znu__mod__viscosity)/(AC_widthnu__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_51_63_97)>=8.)  {
der_step_return_value_51_63_97 = 2./AC_widthnu__mod__viscosity*exp(-2.*abs(arg_51_63_97))
}
else {
der_step_return_value_51_63_97 = 0.5/(AC_widthnu__mod__viscosity*cosh(arg_51_63_97)*cosh(arg_51_63_97))
}
gradnu_63_97.z = (AC_nu__mod__viscosity*(AC_nu_jump__mod__viscosity-1.))*der_step_return_value_51_63_97
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
tmp_63_97 = pnu__mod__viscosity*2*ac_transformed_pencil_sglnrho+ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97+2*sgradnu_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*pnu__mod__viscosity*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+pnu__mod__viscosity
}
}
if (AC_lvisc_nu_shock__mod__viscosity) {
tmp2_63_97 = ac_transformed_pencil_divu*ac_transformed_pencil_glnrho
tmp_63_97=tmp2_63_97 + ac_transformed_pencil_graddivu
tmp2_63_97 = AC_nu_shock__mod__viscosity*ac_transformed_pencil_shock*tmp_63_97
tmp_63_97.x=tmp2_63_97.x+AC_nu_shock__mod__viscosity*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.x
tmp_63_97.y=tmp2_63_97.y+AC_nu_shock__mod__viscosity*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.y
tmp_63_97.z=tmp2_63_97.z+AC_nu_shock__mod__viscosity*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.z
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+(AC_nu_shock__mod__viscosity*ac_transformed_pencil_shock)
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1] && AC_lshock_heat__mod__cdata) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+AC_nu_shock__mod__viscosity*ac_transformed_pencil_shock*ac_transformed_pencil_divu*ac_transformed_pencil_divu
}
}
if (AC_lvisc_nu_shock_profz__mod__viscosity) {
step_vector_return_value_53_63_97 = 0.5*(1+tanh((ac_transformed_pencil_z_mn-AC_znu_shock__mod__viscosity)/(AC_widthnu_shock__mod__viscosity+AC_tini__mod__cparam)))
pnu_shock_63_97 = AC_nu_shock__mod__viscosity + (AC_nu_shock__mod__viscosity*(AC_nu_jump_shock__mod__viscosity-1.))*step_vector_return_value_53_63_97
gradnu_shock_63_97.x = 0.
gradnu_shock_63_97.y = 0.
arg_54_63_97 = abs((ac_transformed_pencil_z_mn-AC_znu_shock__mod__viscosity)/(AC_widthnu_shock__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_54_63_97)>=8.)  {
der_step_return_value_54_63_97 = 2./AC_widthnu_shock__mod__viscosity*exp(-2.*abs(arg_54_63_97))
}
else {
der_step_return_value_54_63_97 = 0.5/(AC_widthnu_shock__mod__viscosity*cosh(arg_54_63_97)*cosh(arg_54_63_97))
}
gradnu_shock_63_97.z = (AC_nu_shock__mod__viscosity*(AC_nu_jump_shock__mod__viscosity-1.))*der_step_return_value_54_63_97
tmp2_63_97 = ac_transformed_pencil_divu*ac_transformed_pencil_glnrho
tmp_63_97=tmp2_63_97 + ac_transformed_pencil_graddivu
tmp2_63_97 = pnu_shock_63_97*ac_transformed_pencil_shock*tmp_63_97
tmp_63_97.x=tmp2_63_97.x+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.x
tmp_63_97.y=tmp2_63_97.y+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.y
tmp_63_97.z=tmp2_63_97.z+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.z
tmp_63_97 = tmp_63_97 + ac_transformed_pencil_shock*ac_transformed_pencil_divu*gradnu_shock_63_97
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+(pnu_shock_63_97*ac_transformed_pencil_shock)
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1] && AC_lshock_heat__mod__cdata) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+pnu_shock_63_97*ac_transformed_pencil_shock*ac_transformed_pencil_divu*ac_transformed_pencil_divu
}
}
if (AC_lvisc_nu_shock_profr__mod__viscosity) {
if (AC_lspherical_coords__mod__cdata || AC_lsphere_in_a_box__mod__cdata) {
tmp3_63_97=ac_transformed_pencil_r_mn
}
else {
tmp3_63_97=ac_transformed_pencil_rcyl_mn
}
step_vector_return_value_56_63_97 = 0.5*(1+tanh((tmp3_63_97-AC_xnu_shock__mod__viscosity)/(AC_widthnu_shock__mod__viscosity+AC_tini__mod__cparam)))
pnu_shock_63_97 = AC_nu_shock__mod__viscosity + (AC_nu_shock__mod__viscosity*(AC_nu_jump_shock__mod__viscosity-1.))*step_vector_return_value_56_63_97
arg_57_63_97 = abs((tmp3_63_97-AC_xnu_shock__mod__viscosity)/(AC_widthnu_shock__mod__viscosity+AC_tini__mod__cparam))
if (abs(arg_57_63_97)>=8.)  {
der_step_return_value_57_63_97 = 2./AC_widthnu_shock__mod__viscosity*exp(-2.*abs(arg_57_63_97))
}
else {
der_step_return_value_57_63_97 = 0.5/(AC_widthnu_shock__mod__viscosity*cosh(arg_57_63_97)*cosh(arg_57_63_97))
}
gradnu_shock_63_97.x = (AC_nu_shock__mod__viscosity*(AC_nu_jump_shock__mod__viscosity-1.))*der_step_return_value_57_63_97
gradnu_shock_63_97.y = 0.
gradnu_shock_63_97.z = 0.
tmp2_63_97 = ac_transformed_pencil_divu*ac_transformed_pencil_glnrho
tmp_63_97=tmp2_63_97 + ac_transformed_pencil_graddivu
tmp2_63_97 = pnu_shock_63_97*ac_transformed_pencil_shock*tmp_63_97
tmp_63_97.x=tmp2_63_97.x+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.x
tmp_63_97.y=tmp2_63_97.y+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.y
tmp_63_97.z=tmp2_63_97.z+pnu_shock_63_97*ac_transformed_pencil_divu*ac_transformed_pencil_gshock.z
tmp_63_97 = tmp_63_97 + ac_transformed_pencil_shock*ac_transformed_pencil_divu*gradnu_shock_63_97
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+(pnu_shock_63_97*ac_transformed_pencil_shock)
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1] && AC_lshock_heat__mod__cdata) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+pnu_shock_63_97*ac_transformed_pencil_shock*ac_transformed_pencil_divu*ac_transformed_pencil_divu
}
}
if (AC_lvisc_shock_simple__mod__viscosity) {
tmp3_63_97 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_uij.row(1-1))
tmp_63_97.x = tmp3_63_97 + ac_transformed_pencil_shock * ac_transformed_pencil_del2u.x
tmp3_63_97 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_uij.row(2-1))
tmp_63_97.y = tmp3_63_97 + ac_transformed_pencil_shock * ac_transformed_pencil_del2u.y
tmp3_63_97 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_uij.row(3-1))
tmp_63_97.z = tmp3_63_97 + ac_transformed_pencil_shock * ac_transformed_pencil_del2u.z
ac_transformed_pencil_fvisc = ac_transformed_pencil_fvisc + AC_nu_shock__mod__viscosity * tmp_63_97
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total = ac_transformed_pencil_diffus_total + AC_nu_shock__mod__viscosity * ac_transformed_pencil_shock
}
}
if (AC_lvisc_hyper2_simplified__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc-AC_nu_hyper2__mod__viscosity*ac_transformed_pencil_del4u
if (AC_lfirst__mod__cdata  &&  AC_ldt__mod__cdata) {
ac_transformed_pencil_diffus_total2=ac_transformed_pencil_diffus_total2+AC_nu_hyper2__mod__viscosity
}
}
if (AC_lvisc_hyper3_simplified__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_del6u
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity
}
}
if (AC_lvisc_hyper2_simplified_tdep__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc-AC_nu_tdep__mod__viscosity*ac_transformed_pencil_del4u
if (AC_lfirst__mod__cdata  &&  AC_ldt__mod__cdata) {
ac_transformed_pencil_diffus_total2=ac_transformed_pencil_diffus_total2+AC_nu_tdep__mod__viscosity
}
}
if (AC_lvisc_hyper3_simplified_tdep__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu_tdep__mod__viscosity*ac_transformed_pencil_del6u
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_tdep__mod__viscosity
}
}
if (AC_lvisc_hyper3_polar__mod__viscosity) {
ju_63_97=1+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.z*dline_1__mod__cdata.z
ju_63_97=2+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.z*dline_1__mod__cdata.z
ju_63_97=3+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*tmp3_63_97*dline_1__mod__cdata.z*dline_1__mod__cdata.z
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity*AC_pi4_1__mod__cparam*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata
}
}
if (AC_lvisc_hyper3_mesh__mod__viscosity) {
ju_63_97=1+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.z
}
ju_63_97=2+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.z
}
ju_63_97=3+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity * tmp3_63_97 * dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.* tmp3_63_97 *dline_1__mod__cdata.z
}
if (ldiffus_total3_63_97) {
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_diffus_total3 = ac_transformed_pencil_diffus_total3 + AC_nu_hyper3_mesh__mod__viscosity
advec_hypermesh_uu_63_97 = 0.0
}
else {
advec_hypermesh_uu_63_97=AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_uu_63_97*advec_hypermesh_uu_63_97
}
}
if (AC_lvisc_hyper3_mesh_residual__mod__viscosity) {
ju_63_97=1+AC_iuu__mod__cdata-1
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity * sum(ac_transformed_pencil_der6u_res.col(1-1) * dline_1__mod__cdata)
}
else {
ac_transformed_pencil_fvisc.x = ac_transformed_pencil_fvisc.x + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.*sum(ac_transformed_pencil_der6u_res.col(1-1)*dline_1__mod__cdata)
}
ju_63_97=2+AC_iuu__mod__cdata-1
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity * sum(ac_transformed_pencil_der6u_res.col(2-1) * dline_1__mod__cdata)
}
else {
ac_transformed_pencil_fvisc.y = ac_transformed_pencil_fvisc.y + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.*sum(ac_transformed_pencil_der6u_res.col(2-1)*dline_1__mod__cdata)
}
ju_63_97=3+AC_iuu__mod__cdata-1
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity * sum(ac_transformed_pencil_der6u_res.col(3-1) * dline_1__mod__cdata)
}
else {
ac_transformed_pencil_fvisc.z = ac_transformed_pencil_fvisc.z + AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam/60.*sum(ac_transformed_pencil_der6u_res.col(3-1)*dline_1__mod__cdata)
}
if (ldiffus_total3_63_97) {
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_diffus_total3 = ac_transformed_pencil_diffus_total3 + AC_nu_hyper3_mesh__mod__viscosity
advec_hypermesh_uu_63_97 = 0.0
}
else {
advec_hypermesh_uu_63_97=AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_uu_63_97*advec_hypermesh_uu_63_97
}
}
if (AC_lvisc_hyper3_csmesh__mod__viscosity) {
ju_63_97=1+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.z
}
ju_63_97=2+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.z
}
ju_63_97=3+AC_iuu__mod__cdata-1
tmp3_63_97 = der6x_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.x
}
else {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.x
}
tmp3_63_97 = der6y_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.y
}
else {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.y
}
tmp3_63_97 = der6z_ignore_spacing(Field(ju_63_97))
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*tmp3_63_97*dline_1__mod__cdata.z
}
else {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)*AC_pi5_1__mod__cparam/60.*tmp3_63_97*dline_1__mod__cdata.z
}
if (ldiffus_total3_63_97) {
if (AC_ldynamical_diffusion__mod__cdata) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3_mesh__mod__viscosity*sqrt(ac_transformed_pencil_cs2)
advec_hypermesh_uu_63_97=0.0
}
else {
advec_hypermesh_uu_63_97=AC_nu_hyper3_mesh__mod__viscosity*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata*ac_transformed_pencil_cs2)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_uu_63_97*advec_hypermesh_uu_63_97
}
}
if (AC_lvisc_hyper3_rho_nu_const__mod__viscosity) {
murho1_63_97=AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+murho1_63_97*ac_transformed_pencil_del6u.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+murho1_63_97*ac_transformed_pencil_del6u.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+murho1_63_97*ac_transformed_pencil_del6u.z
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+murho1_63_97
}
}
if (AC_lvisc_hyper3_rho_nu_const_symm__mod__viscosity) {
murho1_63_97=AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+murho1_63_97*(ac_transformed_pencil_del6u.x + ac_transformed_pencil_grad5divu.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+murho1_63_97*(ac_transformed_pencil_del6u.y + ac_transformed_pencil_grad5divu.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+murho1_63_97*(ac_transformed_pencil_del6u.z + ac_transformed_pencil_grad5divu.z)
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[1-1][1-1]+ac_transformed_pencil_uij5[1-1][1-1])*ac_transformed_pencil_uij[1-1][1-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[1-1][2-1]+ac_transformed_pencil_uij5[2-1][1-1])*ac_transformed_pencil_uij[1-1][2-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[1-1][3-1]+ac_transformed_pencil_uij5[3-1][1-1])*ac_transformed_pencil_uij[1-1][3-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[2-1][1-1]+ac_transformed_pencil_uij5[1-1][2-1])*ac_transformed_pencil_uij[2-1][1-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[2-1][2-1]+ac_transformed_pencil_uij5[2-1][2-1])*ac_transformed_pencil_uij[2-1][2-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[2-1][3-1]+ac_transformed_pencil_uij5[3-1][2-1])*ac_transformed_pencil_uij[2-1][3-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[3-1][1-1]+ac_transformed_pencil_uij5[1-1][3-1])*ac_transformed_pencil_uij[3-1][1-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[3-1][2-1]+ac_transformed_pencil_uij5[2-1][3-1])*ac_transformed_pencil_uij[3-1][2-1]
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat + 0.5*AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_uij5[3-1][3-1]+ac_transformed_pencil_uij5[3-1][3-1])*ac_transformed_pencil_uij[3-1][3-1]
}
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+murho1_63_97
}
}
if (AC_lvisc_hyper3_mu_const_strict__mod__viscosity) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*value(Field(F_HYPVIS-1+1))
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*value(Field(F_HYPVIS-1+2))
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*value(Field(F_HYPVIS-1+3))
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity
}
}
if (AC_lvisc_hyper3_cmu_const_strt_otf__mod__viscosity) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*(ac_transformed_pencil_del6u_strict.x + 1./3*ac_transformed_pencil_del4graddivu.x)
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*(ac_transformed_pencil_del6u_strict.y + 1./3*ac_transformed_pencil_del4graddivu.y)
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1*(ac_transformed_pencil_del6u_strict.z + 1./3*ac_transformed_pencil_del4graddivu.z)
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity
}
}
if (AC_lvisc_hyper3_nu_const_strict__mod__viscosity) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+AC_nu_hyper3__mod__viscosity*value(Field(F_HYPVIS-1+1))
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+AC_nu_hyper3__mod__viscosity*value(Field(F_HYPVIS-1+2))
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+AC_nu_hyper3__mod__viscosity*value(Field(F_HYPVIS-1+3))
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity
}
}
if (AC_lvisc_hyper3_rho_nu_const_aniso__mod__viscosity) {
k1_59_63_97=AC_iuu__mod__cdata-1
tmp_59_63_97  = del6fj(Field(k1_59_63_97+1), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.x=tmp_59_63_97
tmp_59_63_97  = del6fj(Field(k1_59_63_97+2), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.y=tmp_59_63_97
tmp_59_63_97  = del6fj(Field(k1_59_63_97+3), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.z=tmp_59_63_97
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp_63_97.x*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp_63_97.y*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp_63_97.z*ac_transformed_pencil_rho1
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3 +  (AC_nu_aniso_hyper3__mod__viscosity.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x +  AC_nu_aniso_hyper3__mod__viscosity.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y +  AC_nu_aniso_hyper3__mod__viscosity.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)/dxyz_6__mod__cdata
}
}
if (AC_lvisc_hyper3_nu_const_aniso__mod__viscosity) {
k1_60_63_97=AC_iuu__mod__cdata-1
tmp_60_63_97  = del6fj(Field(k1_60_63_97+1), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.x=tmp_60_63_97
tmp_60_63_97  = del6fj(Field(k1_60_63_97+2), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.y=tmp_60_63_97
tmp_60_63_97  = del6fj(Field(k1_60_63_97+3), AC_nu_aniso_hyper3__mod__viscosity)
tmp_63_97.z=tmp_60_63_97
tmp3_63_97=0.
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[1-1][1-1]*ac_transformed_pencil_glnrho.x*AC_nu_aniso_hyper3__mod__viscosity.x
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[1-1][2-1]*ac_transformed_pencil_glnrho.y*AC_nu_aniso_hyper3__mod__viscosity.y
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[1-1][3-1]*ac_transformed_pencil_glnrho.z*AC_nu_aniso_hyper3__mod__viscosity.z
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp_63_97.x+tmp3_63_97
tmp3_63_97=0.
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[2-1][1-1]*ac_transformed_pencil_glnrho.x*AC_nu_aniso_hyper3__mod__viscosity.x
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[2-1][2-1]*ac_transformed_pencil_glnrho.y*AC_nu_aniso_hyper3__mod__viscosity.y
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[2-1][3-1]*ac_transformed_pencil_glnrho.z*AC_nu_aniso_hyper3__mod__viscosity.z
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp_63_97.y+tmp3_63_97
tmp3_63_97=0.
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[3-1][1-1]*ac_transformed_pencil_glnrho.x*AC_nu_aniso_hyper3__mod__viscosity.x
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[3-1][2-1]*ac_transformed_pencil_glnrho.y*AC_nu_aniso_hyper3__mod__viscosity.y
tmp3_63_97=tmp3_63_97+ac_transformed_pencil_uij[3-1][3-1]*ac_transformed_pencil_glnrho.z*AC_nu_aniso_hyper3__mod__viscosity.z
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp_63_97.z+tmp3_63_97
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+  (AC_nu_aniso_hyper3__mod__viscosity.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x +  AC_nu_aniso_hyper3__mod__viscosity.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y +  AC_nu_aniso_hyper3__mod__viscosity.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)/ dxyz_6__mod__cdata
}
}
if (AC_lvisc_hyper3_rho_nu_const_bulk__mod__viscosity) {
murho1_63_97=AC_nu_hyper3__mod__viscosity*ac_transformed_pencil_rho1
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+murho1_63_97*ac_transformed_pencil_del6u_bulk.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+murho1_63_97*ac_transformed_pencil_del6u_bulk.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+murho1_63_97*ac_transformed_pencil_del6u_bulk.z
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+murho1_63_97
}
}
if (AC_lvisc_hyper3_nu_const__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu_hyper3__mod__viscosity*(ac_transformed_pencil_del6u+ac_transformed_pencil_uij5glnrho)
if (ldiffus_total3_63_97) {
ac_transformed_pencil_diffus_total3=ac_transformed_pencil_diffus_total3+AC_nu_hyper3__mod__viscosity
}
}
if (AC_lvisc_smag__mod__cdata) {
if (AC_lnusmag_as_aux__mod__viscosity) {
ac_transformed_pencil_nu_smag=value(F_NUSMAG)
gradnu_63_97 = gradient(Field(AC_inusmag__mod__cdata))
}
else {
ac_transformed_pencil_nu_smag=pow((AC_c_smag__mod__viscosity*AC_dxmax__mod__cdata),2.)*sqrt(2.*ac_transformed_pencil_sij2)
if (AC_lvisc_smag_ma__mod__viscosity) {
ac_transformed_pencil_nu_smag=ac_transformed_pencil_nu_smag*(1.+pow(ac_transformed_pencil_ma2,AC_nu_smag_ma2_power__mod__viscosity))
}
if (AC_gamma_smag__mod__viscosity!=0.) {
ac_transformed_pencil_nu_smag=ac_transformed_pencil_nu_smag/sqrt(1.+AC_gamma_smag__mod__viscosity*ac_transformed_pencil_sij2)
}
}
tmp_63_97 = ac_transformed_pencil_nu_smag*ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu+2.*ac_transformed_pencil_sglnrho
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+tmp_63_97
if (AC_lnusmag_as_aux__mod__viscosity) {
sgradnu_63_97 = ac_transformed_pencil_sij*gradnu_63_97
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+2.*sgradnu_63_97
}
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*ac_transformed_pencil_nu_smag*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+ac_transformed_pencil_nu_smag
}
}
if (AC_lvisc_smag_simplified__mod__viscosity) {
ac_transformed_pencil_nu_smag=pow((AC_c_smag__mod__viscosity*AC_dxmax__mod__cdata),2.)*sqrt(2*ac_transformed_pencil_sij2)
if (AC_gamma_smag__mod__viscosity!=0.) {
ac_transformed_pencil_nu_smag=ac_transformed_pencil_nu_smag/sqrt(1.+AC_gamma_smag__mod__viscosity*ac_transformed_pencil_sij2)
}
tmp2_63_97 = ac_transformed_pencil_nu_smag*ac_transformed_pencil_sglnrho
tmp_63_97 = ac_transformed_pencil_nu_smag*ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+2*tmp2_63_97+tmp_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+2*ac_transformed_pencil_nu_smag*ac_transformed_pencil_sij2
}
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+ac_transformed_pencil_nu_smag
}
}
if (AC_lvisc_smag_cross_simplified__mod__viscosity) {
ac_transformed_pencil_nu_smag=pow((AC_c_smag__mod__viscosity*AC_dxmax__mod__cdata),2.)*ac_transformed_pencil_ss12
tmp2_63_97 = ac_transformed_pencil_nu_smag*ac_transformed_pencil_sglnrho
tmp_63_97 = ac_transformed_pencil_nu_smag*ac_transformed_pencil_del2u+1./3.*ac_transformed_pencil_graddivu
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+2*tmp2_63_97+tmp_63_97
if (ldiffus_total_63_97) {
ac_transformed_pencil_diffus_total=ac_transformed_pencil_diffus_total+ac_transformed_pencil_nu_smag
}
}
if (AC_lvisc_slope_limited__mod__viscosity  &&  AC_llast__mod__cdata) {
if (AC_lviscosity_heat__mod__energy) {
if (AC_lcylindrical_coords__mod__cdata  ||  AC_lspherical_coords__mod__cdata) {
not_implemented("calc_slope_diff_flux")
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
not_implemented("calc_slope_diff_flux")
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
not_implemented("calc_slope_diff_flux")
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
if (AC_lcylindrical_coords__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp2_63_97.x-(d_sld_flux_63_97[2-1][2-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp2_63_97.y+(d_sld_flux_63_97[2-1][1-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp2_63_97.z
}
else if(AC_lspherical_coords__mod__cdata) {
if (AC_lsld_notensor__mod__viscosity) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp2_63_97.x
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp2_63_97.y
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp2_63_97.z
}
else {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp2_63_97.x-(d_sld_flux_63_97[2-1][2-1]+d_sld_flux_63_97[3-1][3-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp2_63_97.y+(d_sld_flux_63_97[2-1][1-1]-d_sld_flux_63_97[3-1][3-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp2_63_97.z+(d_sld_flux_63_97[3-1][1-1]+d_sld_flux_63_97[3-1][2-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
}
}
}
else {
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + tmp3_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + tmp3_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + tmp3_63_97
if (AC_lpencil_int__mod__cdata[AC_i_visc_heat__mod__cparam-1]) {
ac_transformed_pencil_visc_heat=ac_transformed_pencil_visc_heat+max(0.0,tmp4_63_97)/ac_transformed_pencil_rho
}
}
}
else {
if (AC_lcylindrical_coords__mod__cdata  ||  AC_lspherical_coords__mod__cdata) {
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
if (AC_lcylindrical_coords__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp2_63_97.x-(d_sld_flux_63_97[2-1][2-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp2_63_97.y+(d_sld_flux_63_97[2-1][1-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp2_63_97.z
}
else if(AC_lspherical_coords__mod__cdata) {
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x+tmp2_63_97.x-(d_sld_flux_63_97[2-1][2-1]+d_sld_flux_63_97[3-1][3-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y+tmp2_63_97.y+(d_sld_flux_63_97[2-1][1-1]-d_sld_flux_63_97[3-1][3-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z+tmp2_63_97.z+(d_sld_flux_63_97[3-1][1-1]+d_sld_flux_63_97[3-1][2-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
}
}
else {
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x + tmp3_63_97
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + tmp3_63_97
not_implemented("calc_slope_diff_flux")
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + tmp3_63_97
}
}
}
if (AC_lvisc_schur_223__mod__viscosity) {
ac_transformed_pencil_fvisc=ac_transformed_pencil_fvisc+AC_nu__mod__viscosity*ac_transformed_pencil_del2u
ac_transformed_pencil_fvisc.x=ac_transformed_pencil_fvisc.x-AC_nu__mod__viscosity*ac_transformed_pencil_d2uidxj[1-1][3-1]
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y-AC_nu__mod__viscosity*ac_transformed_pencil_d2uidxj[2-1][3-1]
}
if (AC_llambda_effect__mod__viscosity) {
if (AC_lspherical_coords__mod__cdata) {
lomega_61_63_97=ac_transformed_pencil_uu.z/(AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_x__mod__cdata[vertexIdx.x])+AC_omega__mod__cdata
dlomega_dr_61_63_97=(AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_uij[3-1][1-1]-ac_transformed_pencil_uu.z)/(AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x])
dlomega_dtheta_61_63_97=(ac_transformed_pencil_uij[3-1][2-1]*AC_x__mod__cdata[vertexIdx.x]-ac_transformed_pencil_uu.z*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/(AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x])
lver_61_63_97 = -(AC_lambda_v0__mod__viscosity*AC_lv0_rprof__mod__viscosity[vertexIdx.x]+AC_lambda_v1__mod__viscosity*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_lv1_rprof__mod__viscosity[vertexIdx.x] )
lhor_61_63_97 = -AC_lambda_h1__mod__viscosity*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_lh1_rprof__mod__viscosity[vertexIdx.x]
dlver_dr_61_63_97 = -(AC_lambda_v0__mod__viscosity*AC_der_lv0_rprof__mod__viscosity[vertexIdx.x]+AC_lambda_v1__mod__viscosity*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*AC_der_lv1_rprof__mod__viscosity[vertexIdx.x])
dlhor_dtheta_61_63_97 = -AC_lambda_h1__mod__viscosity*AC_lh1_rprof__mod__viscosity[vertexIdx.x]*2.*AC_costh__mod__cdata[AC_m__mod__cdata-1]*AC_sinth__mod__cdata[AC_m__mod__cdata-1]/AC_x__mod__cdata[vertexIdx.x]
lambda_phi_63_97 = AC_sinth__mod__cdata[AC_m__mod__cdata-1]*(lver_61_63_97*(lomega_61_63_97*ac_transformed_pencil_glnrho.x+3.*lomega_61_63_97/AC_x__mod__cdata[vertexIdx.x]+dlomega_dr_61_63_97)+lomega_61_63_97*dlver_dr_61_63_97)  +AC_costh__mod__cdata[AC_m__mod__cdata-1]*(lhor_61_63_97*(lomega_61_63_97*ac_transformed_pencil_glnrho.y  -1./AC_cotth__mod__cdata[AC_m__mod__cdata-1]*lomega_61_63_97/AC_x__mod__cdata[vertexIdx.x] + 2.*AC_cotth__mod__cdata[AC_m__mod__cdata-1]*lomega_61_63_97/AC_x__mod__cdata[vertexIdx.x]  +dlomega_dtheta_61_63_97) + lomega_61_63_97*dlhor_dtheta_61_63_97)
if (AC_llambda_scale_with_nu__mod__viscosity) {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + AC_nu__mod__viscosity*lambda_phi_63_97
}
else {
ac_transformed_pencil_fvisc.z=ac_transformed_pencil_fvisc.z + lambda_phi_63_97
}
}
else if (AC_lcylindrical_coords__mod__cdata) {
lomega_62_63_97=ac_transformed_pencil_uu.y/AC_x__mod__cdata[vertexIdx.x]+AC_omega__mod__cdata
dlomega_dr_62_63_97=(AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_uij[2-1][1-1]-ac_transformed_pencil_uu.y)/AC_x__mod__cdata[vertexIdx.x]*AC_x__mod__cdata[vertexIdx.x]
lver_62_63_97 = -AC_lambda_v0__mod__viscosity*AC_lv0_rprof__mod__viscosity[vertexIdx.x]
dlver_dr_62_63_97 = -AC_lambda_v0__mod__viscosity*AC_der_lv0_rprof__mod__viscosity[vertexIdx.x]
lambda_phi_63_97 = (lver_62_63_97*(lomega_62_63_97*ac_transformed_pencil_glnrho.x+2.*lomega_62_63_97/AC_x__mod__cdata[vertexIdx.x]  +dlomega_dr_62_63_97)+lomega_62_63_97*dlver_dr_62_63_97)/AC_x__mod__cdata[vertexIdx.x]
if (AC_llambda_scale_with_nu__mod__viscosity) {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + AC_nu__mod__viscosity*lambda_phi_63_97
}
else {
ac_transformed_pencil_fvisc.y=ac_transformed_pencil_fvisc.y + lambda_phi_63_97
}
}
else {
}
}
if (AC_lvisc_heat_as_aux__mod__viscosity) {
DF_VISC_HEAT = ac_transformed_pencil_visc_heat
}
if (AC_lvisc_forc_as_aux__mod__viscosity) {
DF_VISC_FORCVEC = ac_transformed_pencil_fvisc
}
if (AC_lpencil_int__mod__cdata[AC_i_fcont__mod__cparam-1]) {
for j_65_97 in 1:AC_n_forcing_cont_max__mod__cparam+1 {
ac_transformed_pencil_fcont[j_65_97-1].x=0.
}
for j_65_97 in 1:AC_n_forcing_cont_max__mod__cparam+1 {
ac_transformed_pencil_fcont[j_65_97-1].y=0.
}
for j_65_97 in 1:AC_n_forcing_cont_max__mod__cparam+1 {
ac_transformed_pencil_fcont[j_65_97-1].z=0.
}
}
if (AC_lpencil_int__mod__cdata[AC_i_aa__mod__cparam-1]) {
ac_transformed_pencil_aa=value(F_AVEC)
}
if (AC_lpencil_int__mod__cdata[AC_i_a2__mod__cparam-1]) {
ac_transformed_pencil_a2 = dot(ac_transformed_pencil_aa,ac_transformed_pencil_aa)
}
if (AC_lpencil_int__mod__cdata[AC_i_aij__mod__cparam-1]) {
ac_transformed_pencil_aij = gradient_tensor((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_diva__mod__cparam-1]) {
if (AC_lpencil_int__mod__cdata[AC_i_aij__mod__cparam-1]  &&  ! AC_lpencil_check_at_work__mod__cdata) {
ac_transformed_pencil_diva = divergence(ac_transformed_pencil_aij)
}
else {
ac_transformed_pencil_diva = divergence((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
if (AC_lcoulomb__mod__magnetic) {
DF_DIVA=ac_transformed_pencil_diva
}
}
if (AC_lpencil_int__mod__cdata[AC_i_aps__mod__cparam-1]) {
ac_transformed_pencil_aps=value(F_AZ)*ac_transformed_pencil_rcyl_mn
}
if (AC_lpencil_int__mod__cdata[AC_i_bb__mod__cparam-1]) {
if (AC_lbb_as_comaux__mod__magnetic) {
ac_transformed_pencil_bb = value(F_BVEC)
}
else if (AC_lpencil_int__mod__cdata[AC_i_aij__mod__cparam-1]  &&  ! AC_lpencil_check_at_work__mod__cdata) {
ac_transformed_pencil_bb=curl(ac_transformed_pencil_aij)
}
else {
ac_transformed_pencil_bb = curl((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
ac_transformed_pencil_bbb = ac_transformed_pencil_bb
if (! (AC_lbb_as_comaux__mod__magnetic  &&  AC_lb_ext_in_comaux__mod__magnetic)  &&  (! AC_ladd_global_field__mod__magnetic)) {
if (AC_b_ext__mod__magnetic.x != 0.0  ||  AC_b_ext__mod__magnetic.y != 0.0  ||  AC_b_ext__mod__magnetic.z != 0.0) {
if (AC_omega_bz_ext__mod__magnetic != 0.0) {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
c_71_78_79_97 = cos(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
s_71_78_79_97 = sin(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
b_ext_78_79_97.x = AC_b_ext__mod__magnetic.x * c_71_78_79_97 - AC_b_ext__mod__magnetic.y * s_71_78_79_97
b_ext_78_79_97.y = AC_b_ext__mod__magnetic.x * s_71_78_79_97 + AC_b_ext__mod__magnetic.y * c_71_78_79_97
b_ext_78_79_97.z = AC_b_ext__mod__magnetic.z
}
else {
}
}
else if (AC_lbext_moving_layer__mod__magnetic)  {
zposbot_71_78_79_97=AC_zbot_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
zpostop_71_78_79_97=AC_ztop_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
step_scalar_return_value_67_71_78_79_97 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_71_78_79_97)/AC_edge_moving_layer__mod__magnetic))
step_scalar_return_value_68_71_78_79_97 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_71_78_79_97)/AC_edge_moving_layer__mod__magnetic))
zprof_71_78_79_97 = step_scalar_return_value_67_71_78_79_97-step_scalar_return_value_68_71_78_79_97
b_ext_78_79_97.x = AC_b_ext__mod__magnetic.x*zprof_71_78_79_97
b_ext_78_79_97.y = AC_b_ext__mod__magnetic.y*zprof_71_78_79_97
b_ext_78_79_97.z=AC_b_ext__mod__magnetic.z
if (true) {
arg_69_71_78_79_97 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_71_78_79_97)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_69_71_78_79_97)>=8.)  {
der_step_return_value_69_71_78_79_97 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_69_71_78_79_97))
}
else {
der_step_return_value_69_71_78_79_97 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_69_71_78_79_97)*cosh(arg_69_71_78_79_97))
}
arg_70_71_78_79_97 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_71_78_79_97)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_70_71_78_79_97)>=8.)  {
der_step_return_value_70_71_78_79_97 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_70_71_78_79_97))
}
else {
der_step_return_value_70_71_78_79_97 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_70_71_78_79_97)*cosh(arg_70_71_78_79_97))
}
zder_71_78_79_97 = der_step_return_value_69_71_78_79_97-der_step_return_value_70_71_78_79_97
if (AC_b_ext__mod__magnetic.x!=0.) {
j_ext_78_79_97.y =  AC_b_ext__mod__magnetic.x*zder_71_78_79_97
}
if (AC_b_ext__mod__magnetic.y!=0.) {
j_ext_78_79_97.x = -AC_b_ext__mod__magnetic.y*zder_71_78_79_97
}
}
}
else {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
b_ext_78_79_97 = AC_b_ext__mod__magnetic
}
else if (AC_lcylindrical_coords__mod__cdata) {
b_ext_78_79_97.x =  AC_b_ext__mod__magnetic.x * cos(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_78_79_97.y = -AC_b_ext__mod__magnetic.x * sin(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_78_79_97.z =  AC_b_ext__mod__magnetic.z
}
else if (AC_lspherical_coords__mod__cdata) {
b_ext_78_79_97.x =  AC_b_ext__mod__magnetic.x * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.z * AC_costh__mod__cdata[AC_m__mod__cdata-1]
b_ext_78_79_97.y =  AC_b_ext__mod__magnetic.x * AC_costh__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_costh__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) - AC_b_ext__mod__magnetic.z * AC_sinth__mod__cdata[AC_m__mod__cdata-1]
b_ext_78_79_97.z = -AC_b_ext__mod__magnetic.x            * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y            * cos(AC_z__mod__cdata[AC_n__mod__cdata-1])
}
if (true) {
j_ext_78_79_97.x = 0.
j_ext_78_79_97.y = 0.
j_ext_78_79_97.z = 0.
}
}
}
else {
b_ext_78_79_97.x = 0.
b_ext_78_79_97.y = 0.
b_ext_78_79_97.z = 0.
if (true) {
j_ext_78_79_97.x = 0.
j_ext_78_79_97.y = 0.
j_ext_78_79_97.z = 0.
}
}
if (AC_t_bext__mod__magnetic > 0.0  &&  AC_t__mod__cdata < AC_t_bext__mod__magnetic) {
if (AC_t__mod__cdata <= AC_t0_bext__mod__magnetic) {
b_ext_78_79_97 = AC_b0_ext__mod__magnetic
}
else {
b_ext_78_79_97 = AC_b0_ext__mod__magnetic + 0.5*(1.-cos(AC_pi__mod__cparam*(AC_t__mod__cdata-AC_t0_bext__mod__magnetic)/(AC_t_bext__mod__magnetic-AC_t0_bext__mod__magnetic)))*(b_ext_78_79_97-AC_b0_ext__mod__magnetic)
}
if (true) {
j_ext_78_79_97.x = 0.
j_ext_78_79_97.y = 0.
j_ext_78_79_97.z = 0.
}
}
if (b_ext_78_79_97.x != 0.  ||  b_ext_78_79_97.y != 0.  ||  b_ext_78_79_97.z != 0.) {
ac_transformed_pencil_bb.x = ac_transformed_pencil_bb.x + b_ext_78_79_97.x
ac_transformed_pencil_bb.y = ac_transformed_pencil_bb.y + b_ext_78_79_97.y
ac_transformed_pencil_bb.z = ac_transformed_pencil_bb.z + b_ext_78_79_97.z
}
ac_transformed_pencil_jj.x = ac_transformed_pencil_jj.x + j_ext_78_79_97.x
ac_transformed_pencil_jj.y = ac_transformed_pencil_jj.y + j_ext_78_79_97.y
ac_transformed_pencil_jj.z = ac_transformed_pencil_jj.z + j_ext_78_79_97.z
}
if (AC_dipole_moment__mod__magnetic != 0.) {
c_78_79_97=cos(AC_inclaa__mod__magnetic*AC_pi__mod__cparam/180)
s_78_79_97=sin(AC_inclaa__mod__magnetic*AC_pi__mod__cparam/180)
ac_transformed_pencil_bb.x = ac_transformed_pencil_bb.x + AC_dipole_moment__mod__magnetic*2*(c_78_79_97*AC_costh__mod__cdata[AC_m__mod__cdata-1] + s_78_79_97*AC_sinth__mod__cdata[AC_m__mod__cdata-1]*cos(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_omega_bz_ext__mod__magnetic*AC_t__mod__cdata))*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1
ac_transformed_pencil_bb.y = ac_transformed_pencil_bb.y + AC_dipole_moment__mod__magnetic*  (c_78_79_97*AC_sinth__mod__cdata[AC_m__mod__cdata-1] - s_78_79_97*AC_costh__mod__cdata[AC_m__mod__cdata-1]*cos(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_omega_bz_ext__mod__magnetic*AC_t__mod__cdata))*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1
ac_transformed_pencil_bb.z = ac_transformed_pencil_bb.z + AC_dipole_moment__mod__magnetic*  (             s_78_79_97*         sin(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_omega_bz_ext__mod__magnetic*AC_t__mod__cdata))*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1*ac_transformed_pencil_r_mn1
}
if (AC_ladd_global_field__mod__magnetic) {
if (AC_b_ext__mod__magnetic.x != 0.0  ||  AC_b_ext__mod__magnetic.y != 0.0  ||  AC_b_ext__mod__magnetic.z != 0.0) {
if (AC_omega_bz_ext__mod__magnetic != 0.0) {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
c_72_78_79_97 = cos(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
s_72_78_79_97 = sin(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
b_ext_78_79_97.x = AC_b_ext__mod__magnetic.x * c_72_78_79_97 - AC_b_ext__mod__magnetic.y * s_72_78_79_97
b_ext_78_79_97.y = AC_b_ext__mod__magnetic.x * s_72_78_79_97 + AC_b_ext__mod__magnetic.y * c_72_78_79_97
b_ext_78_79_97.z = AC_b_ext__mod__magnetic.z
}
else {
}
}
else if (AC_lbext_moving_layer__mod__magnetic)  {
zposbot_72_78_79_97=AC_zbot_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
zpostop_72_78_79_97=AC_ztop_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
step_scalar_return_value_67_72_78_79_97 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_72_78_79_97)/AC_edge_moving_layer__mod__magnetic))
step_scalar_return_value_68_72_78_79_97 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_72_78_79_97)/AC_edge_moving_layer__mod__magnetic))
zprof_72_78_79_97 = step_scalar_return_value_67_72_78_79_97-step_scalar_return_value_68_72_78_79_97
b_ext_78_79_97.x = AC_b_ext__mod__magnetic.x*zprof_72_78_79_97
b_ext_78_79_97.y = AC_b_ext__mod__magnetic.y*zprof_72_78_79_97
b_ext_78_79_97.z=AC_b_ext__mod__magnetic.z
if (false) {
arg_69_72_78_79_97 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_72_78_79_97)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_69_72_78_79_97)>=8.)  {
der_step_return_value_69_72_78_79_97 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_69_72_78_79_97))
}
else {
der_step_return_value_69_72_78_79_97 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_69_72_78_79_97)*cosh(arg_69_72_78_79_97))
}
arg_70_72_78_79_97 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_72_78_79_97)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_70_72_78_79_97)>=8.)  {
der_step_return_value_70_72_78_79_97 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_70_72_78_79_97))
}
else {
der_step_return_value_70_72_78_79_97 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_70_72_78_79_97)*cosh(arg_70_72_78_79_97))
}
zder_72_78_79_97 = der_step_return_value_69_72_78_79_97-der_step_return_value_70_72_78_79_97
}
}
else {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
b_ext_78_79_97 = AC_b_ext__mod__magnetic
}
else if (AC_lcylindrical_coords__mod__cdata) {
b_ext_78_79_97.x =  AC_b_ext__mod__magnetic.x * cos(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_78_79_97.y = -AC_b_ext__mod__magnetic.x * sin(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_78_79_97.z =  AC_b_ext__mod__magnetic.z
}
else if (AC_lspherical_coords__mod__cdata) {
b_ext_78_79_97.x =  AC_b_ext__mod__magnetic.x * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.z * AC_costh__mod__cdata[AC_m__mod__cdata-1]
b_ext_78_79_97.y =  AC_b_ext__mod__magnetic.x * AC_costh__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_costh__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) - AC_b_ext__mod__magnetic.z * AC_sinth__mod__cdata[AC_m__mod__cdata-1]
b_ext_78_79_97.z = -AC_b_ext__mod__magnetic.x            * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y            * cos(AC_z__mod__cdata[AC_n__mod__cdata-1])
}
}
}
else {
b_ext_78_79_97.x = 0.
b_ext_78_79_97.y = 0.
b_ext_78_79_97.z = 0.
}
if (AC_t_bext__mod__magnetic > 0.0  &&  AC_t__mod__cdata < AC_t_bext__mod__magnetic) {
if (AC_t__mod__cdata <= AC_t0_bext__mod__magnetic) {
b_ext_78_79_97 = AC_b0_ext__mod__magnetic
}
else {
b_ext_78_79_97 = AC_b0_ext__mod__magnetic + 0.5*(1.-cos(AC_pi__mod__cparam*(AC_t__mod__cdata-AC_t0_bext__mod__magnetic)/(AC_t_bext__mod__magnetic-AC_t0_bext__mod__magnetic)))*(b_ext_78_79_97-AC_b0_ext__mod__magnetic)
}
}
if (AC_iglobal_ext_bx__mod__cdata!=0) {
ac_transformed_pencil_bb.x=ac_transformed_pencil_bb.x+b_ext_78_79_97.y*value(F_GLOBAL_EXT_BX)+b_ext_78_79_97.x
}
if (AC_iglobal_ext_by__mod__cdata!=0) {
ac_transformed_pencil_bb.y=ac_transformed_pencil_bb.y+b_ext_78_79_97.y*value(F_GLOBAL_EXT_BY)
}
if (AC_iglobal_ext_bz__mod__cdata!=0) {
ac_transformed_pencil_bb.z=ac_transformed_pencil_bb.z+b_ext_78_79_97.y*value(F_GLOBAL_EXT_BZ)+b_ext_78_79_97.z
}
}
}
if (AC_b0_ext_z__mod__magnetic != 0.0) {
ac_transformed_pencil_bb.z = ac_transformed_pencil_bb.z + AC_bz_stratified__mod__magnetic[AC_n__mod__cdata-1]
}
if (AC_lignore_bext_in_b2__mod__magnetic  ||  (!AC_luse_bext_in_b2__mod__magnetic) ) {
if (AC_lpencil_int__mod__cdata[AC_i_b2__mod__cparam-1]) {
ac_transformed_pencil_b2 = dot(ac_transformed_pencil_bbb,ac_transformed_pencil_bbb)
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_b2__mod__cparam-1]) {
ac_transformed_pencil_b2 = dot(ac_transformed_pencil_bb,ac_transformed_pencil_bb)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_bf2__mod__cparam-1]) {
ac_transformed_pencil_bf2 = dot(ac_transformed_pencil_bbb,ac_transformed_pencil_bbb)
}
if (AC_lpencil_int__mod__cdata[AC_i_b21__mod__cparam-1]) {
if (ac_transformed_pencil_b2>AC_tini__mod__cparam) {
ac_transformed_pencil_b21=1./ac_transformed_pencil_b2
}
else {
ac_transformed_pencil_b21=0.
}
}
if (AC_lmagneto_friction__mod__magnetic && AC_lpencil_int__mod__cdata[AC_i_rho1__mod__cparam-1]) {
ac_transformed_pencil_rho=AC_rho0__mod__equationofstate*1.0e-2+ac_transformed_pencil_b2
ac_transformed_pencil_rho1=1./ac_transformed_pencil_rho
}
if (AC_lpencil_int__mod__cdata[AC_i_bunit__mod__cparam-1]) {
quench_78_79_97 = 1.0/max(AC_tini__mod__cparam,sqrt(ac_transformed_pencil_b2))
if (AC_lignore_bext_in_b2__mod__magnetic  ||  (!AC_luse_bext_in_b2__mod__magnetic) ) {
ac_transformed_pencil_bunit.x = ac_transformed_pencil_bbb.x*quench_78_79_97
ac_transformed_pencil_bunit.y = ac_transformed_pencil_bbb.y*quench_78_79_97
ac_transformed_pencil_bunit.z = ac_transformed_pencil_bbb.z*quench_78_79_97
}
else {
ac_transformed_pencil_bunit.x = ac_transformed_pencil_bb.x*quench_78_79_97
ac_transformed_pencil_bunit.y = ac_transformed_pencil_bb.y*quench_78_79_97
ac_transformed_pencil_bunit.z = ac_transformed_pencil_bb.z*quench_78_79_97
}
}
if (AC_lpencil_int__mod__cdata[AC_i_ab__mod__cparam-1]) {
ac_transformed_pencil_ab = dot(ac_transformed_pencil_aa,ac_transformed_pencil_bbb)
}
if (AC_lpencil_int__mod__cdata[AC_i_ua__mod__cparam-1]) {
ac_transformed_pencil_ua = dot(ac_transformed_pencil_uu,ac_transformed_pencil_aa)
}
if (AC_lpencil_int__mod__cdata[AC_i_uxb__mod__cparam-1]) {
ac_transformed_pencil_uxb = cross(ac_transformed_pencil_uu,ac_transformed_pencil_bb)
if (AC_iglobal_eext__mod__cdata.x!=0) {
ac_transformed_pencil_uxb.x=ac_transformed_pencil_uxb.x+value(F_GLOBAL_EEXTVEC.x)
}
if (AC_iglobal_eext__mod__cdata.y!=0) {
ac_transformed_pencil_uxb.y=ac_transformed_pencil_uxb.y+value(F_GLOBAL_EEXTVEC.y)
}
if (AC_iglobal_eext__mod__cdata.z!=0) {
ac_transformed_pencil_uxb.z=ac_transformed_pencil_uxb.z+value(F_GLOBAL_EEXTVEC.z)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uxbb__mod__cparam-1]) {
ac_transformed_pencil_uxbb = cross(ac_transformed_pencil_uu,ac_transformed_pencil_bbb)
}
if (AC_lpencil_int__mod__cdata[AC_i_uga__mod__cparam-1]) {
ac_transformed_pencil_uga = ac_transformed_pencil_aij*ac_transformed_pencil_uu
if (AC_lupw_aa__mod__magnetic) ac_transformed_pencil_uga = ac_transformed_pencil_uga + del6((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_uuadvec_gaa__mod__cparam-1]) {
tmp_78_79_97 = ac_transformed_pencil_aij.row(1-1)
ac_transformed_pencil_uuadvec_gaa.x = dot(ac_transformed_pencil_uu_advec,tmp_78_79_97)
tmp_78_79_97 = ac_transformed_pencil_aij.row(2-1)
ac_transformed_pencil_uuadvec_gaa.y = dot(ac_transformed_pencil_uu_advec,tmp_78_79_97)
tmp_78_79_97 = ac_transformed_pencil_aij.row(3-1)
ac_transformed_pencil_uuadvec_gaa.z = dot(ac_transformed_pencil_uu_advec,tmp_78_79_97)
if (AC_lcylindrical_coords__mod__cdata) {
ac_transformed_pencil_uuadvec_gaa.x = ac_transformed_pencil_uuadvec_gaa.x - AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]*ac_transformed_pencil_uu.y*ac_transformed_pencil_aa.y
ac_transformed_pencil_uuadvec_gaa.y = ac_transformed_pencil_uuadvec_gaa.y + AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]*ac_transformed_pencil_uu.y*ac_transformed_pencil_aa.x
}
else if (AC_lspherical_coords__mod__cdata) {
}
}
if (AC_lpencil_int__mod__cdata[AC_i_bij__mod__cparam-1] && AC_lpencil_int__mod__cdata[AC_i_del2a__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_bij = bij((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
ac_transformed_pencil_del2a = laplace((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
if (AC_lpencil_int__mod__cdata[AC_i_curlb__mod__cparam-1]  &&  ! AC_ljj_as_comaux__mod__magnetic) {
ac_transformed_pencil_curlb=curl(ac_transformed_pencil_bij)
}
}
else {
not_implemented("gij_etc with more than 6 params")
if (AC_lpencil_int__mod__cdata[AC_i_curlb__mod__cparam-1]  &&  ! AC_ljj_as_comaux__mod__magnetic) {
ac_transformed_pencil_curlb=curl(ac_transformed_pencil_bij)
}
}
}
else if (AC_lpencil_int__mod__cdata[AC_i_bij__mod__cparam-1] && !AC_lpencil_int__mod__cdata[AC_i_del2a__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_bij = bij((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
if (AC_lpencil_int__mod__cdata[AC_i_curlb__mod__cparam-1] &&  ! AC_ljj_as_comaux__mod__magnetic) {
ac_transformed_pencil_curlb=curl(ac_transformed_pencil_bij)
}
}
else {
not_implemented("gij_etc with more than 6 params")
if (AC_lpencil_int__mod__cdata[AC_i_curlb__mod__cparam-1] &&  ! AC_ljj_as_comaux__mod__magnetic) {
ac_transformed_pencil_curlb=curl(ac_transformed_pencil_bij)
}
}
}
else if (AC_lpencil_int__mod__cdata[AC_i_del2a__mod__cparam-1] && !AC_lpencil_int__mod__cdata[AC_i_bij__mod__cparam-1]) {
if (AC_lcartesian_coords__mod__cdata) {
ac_transformed_pencil_del2a = laplace((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
else {
not_implemented("gij_etc with more than 6 params")
}
}
if (AC_lpencil_int__mod__cdata[AC_i_bijtilde__mod__cparam-1]) {
if (AC_lcovariant_magnetic__mod__magnetic) {
not_implemented("bij_tilde in sub.f90")
}
else {
not_implemented("bij_tilde in sub.f90")
}
}
if (AC_ljj_as_comaux__mod__magnetic  &&  AC_lpencil_int__mod__cdata[AC_i_jj__mod__cparam-1]) {
if (AC_lsmooth_jj__mod__magnetic) {
ac_transformed_pencil_jj.x = gaussian_smooth(Field(AC_ijx__mod__cdata))
ac_transformed_pencil_jj.y = gaussian_smooth(Field(AC_ijy__mod__cdata))
ac_transformed_pencil_jj.z = gaussian_smooth(Field(AC_ijz__mod__cdata))
}
else {
ac_transformed_pencil_jj=value(F_JVEC)
}
}
chi_diamag_75_78_79_97=AC_b2_diamag__mod__magnetic/ac_transformed_pencil_b2
gchi_diamag_75_78_79_97 = chi_diamag_75_78_79_97/ac_transformed_pencil_b2*ac_transformed_pencil_gb22
jj_diamag_75_78_79_97 = cross(gchi_diamag_75_78_79_97,ac_transformed_pencil_bb)
tmp_75_78_79_97.x=jj_diamag_75_78_79_97.x+chi_diamag_75_78_79_97*ac_transformed_pencil_jj.x
tmp_75_78_79_97.y=jj_diamag_75_78_79_97.y+chi_diamag_75_78_79_97*ac_transformed_pencil_jj.y
tmp_75_78_79_97.z=jj_diamag_75_78_79_97.z+chi_diamag_75_78_79_97*ac_transformed_pencil_jj.z
jj_diamag_75_78_79_97=tmp_75_78_79_97
ac_transformed_pencil_jj=ac_transformed_pencil_jj+jj_diamag_75_78_79_97
if (AC_lpencil_int__mod__cdata[AC_i_jj__mod__cparam-1]  ||  AC_lpencil_int__mod__cdata[AC_i_jj_ohm__mod__cparam-1]) {
if (AC_iex__mod__cdata>0) {
if (AC_lresi_eta_tdep__mod__magnetic) {
eta_total__mod__magnetic=AC_eta_tdep__mod__magnetic
}
else {
eta_total__mod__magnetic=AC_eta__mod__magnetic
}
if (AC_lvacuum__mod__magnetic) {
ac_transformed_pencil_jj.x = 0.
ac_transformed_pencil_jj.y = 0.
ac_transformed_pencil_jj.z = 0.
ac_transformed_pencil_jj_ohm.x = 0.
ac_transformed_pencil_jj_ohm.y = 0.
ac_transformed_pencil_jj_ohm.z = 0.
}
else {
ac_transformed_pencil_jj_ohm.x=(ac_transformed_pencil_el.x+ac_transformed_pencil_uxb.x)*AC_mu01__mod__cdata/eta_total__mod__magnetic
ac_transformed_pencil_jj_ohm.y=(ac_transformed_pencil_el.y+ac_transformed_pencil_uxb.y)*AC_mu01__mod__cdata/eta_total__mod__magnetic
ac_transformed_pencil_jj_ohm.z=(ac_transformed_pencil_el.z+ac_transformed_pencil_uxb.z)*AC_mu01__mod__cdata/eta_total__mod__magnetic
if (AC_loverride_ee2__mod__magnetic) {
if (AC_ladd_disp_current_from_aux__mod__magnetic) {
if (AC_iedotx__mod__magnetic>0  &&  AC_iedotz__mod__magnetic>0) {
ac_transformed_pencil_jj=AC_mu01__mod__cdata*ac_transformed_pencil_curlb-AC_c_light21__mod__magnetic*value(F_EDOTVEC)
}
else {
}
}
else {
ac_transformed_pencil_jj=AC_mu01__mod__cdata*ac_transformed_pencil_curlb
}
}
else {
ac_transformed_pencil_jj=ac_transformed_pencil_jj_ohm
}
}
}
else {
ac_transformed_pencil_jj=AC_mu01__mod__cdata*ac_transformed_pencil_curlb
ac_transformed_pencil_jj_ohm.x = 0.
ac_transformed_pencil_jj_ohm.y = 0.
ac_transformed_pencil_jj_ohm.z = 0.
}
if (AC_iglobal_jext__mod__cdata.x!=0) {
ac_transformed_pencil_jj.x=ac_transformed_pencil_jj.x+value(F_GLOBAL_JEXTVEC.x)
}
if (AC_iglobal_jext__mod__cdata.y!=0) {
ac_transformed_pencil_jj.y=ac_transformed_pencil_jj.y+value(F_GLOBAL_JEXTVEC.y)
}
if (AC_iglobal_jext__mod__cdata.z!=0) {
ac_transformed_pencil_jj.z=ac_transformed_pencil_jj.z+value(F_GLOBAL_JEXTVEC.z)
}
if (AC_lj_ext__mod__magnetic) {
if (AC_j_ext_quench__mod__magnetic!=0) {
quench_78_79_97=1./(1.+AC_j_ext_quench__mod__magnetic*ac_transformed_pencil_b2)
ac_transformed_pencil_jj.x=ac_transformed_pencil_jj.x-j_ext_78_79_97.x*quench_78_79_97
ac_transformed_pencil_jj.y=ac_transformed_pencil_jj.y-j_ext_78_79_97.y*quench_78_79_97
ac_transformed_pencil_jj.z=ac_transformed_pencil_jj.z-j_ext_78_79_97.z*quench_78_79_97
}
else {
ac_transformed_pencil_jj.x=ac_transformed_pencil_jj.x-j_ext_78_79_97.x
ac_transformed_pencil_jj.y=ac_transformed_pencil_jj.y-j_ext_78_79_97.y
ac_transformed_pencil_jj.z=ac_transformed_pencil_jj.z-j_ext_78_79_97.z
}
}
}
if (AC_lpencil_int__mod__cdata[AC_i_exa__mod__cparam-1]) {
ac_transformed_pencil_exa = cross(ac_transformed_pencil_uxb+AC_eta__mod__magnetic*ac_transformed_pencil_jj,ac_transformed_pencil_aa)
}
if (AC_lpencil_int__mod__cdata[AC_i_exatotal__mod__cparam-1]) {
tmp_78_79_97.x = eta_total__mod__magnetic*ac_transformed_pencil_jj.x
tmp_78_79_97.y = eta_total__mod__magnetic*ac_transformed_pencil_jj.y
tmp_78_79_97.z = eta_total__mod__magnetic*ac_transformed_pencil_jj.z
ac_transformed_pencil_exatotal = cross(ac_transformed_pencil_uxb+tmp_78_79_97,ac_transformed_pencil_aa)
}
if (AC_lpencil_int__mod__cdata[AC_i_j2__mod__cparam-1]) {
ac_transformed_pencil_j2 = dot(ac_transformed_pencil_jj,ac_transformed_pencil_jj)
}
if (AC_lpencil_int__mod__cdata[AC_i_jb__mod__cparam-1]) {
ac_transformed_pencil_jb = dot(ac_transformed_pencil_jj,ac_transformed_pencil_bbb)
}
if (AC_lpencil_int__mod__cdata[AC_i_va2__mod__cparam-1]) {
ac_transformed_pencil_va2=ac_transformed_pencil_b2*AC_mu01__mod__cdata*ac_transformed_pencil_rho1
}
if (AC_lpencil_int__mod__cdata[AC_i_etava__mod__cparam-1]) {
if (AC_lresi_vaspeed__mod__magnetic) {
ac_transformed_pencil_etava = AC_eta_va__mod__magnetic * sqrt(ac_transformed_pencil_va2)/AC_varms__mod__magnetic
if (AC_va_min__mod__magnetic > 0.) {
ac_transformed_pencil_etava = set_min_val(ac_transformed_pencil_etava,AC_va_min__mod__magnetic)
}
}
else {
ac_transformed_pencil_etava = AC_mu0__mod__cdata * AC_eta_va__mod__magnetic * AC_dxmax__mod__cdata * sqrt(ac_transformed_pencil_va2)
if (AC_eta_min__mod__magnetic > 0.) {
ac_transformed_pencil_etava = set_zero_below_threshold(ac_transformed_pencil_etava,AC_eta_min__mod__magnetic)
}
}
}
if (AC_lpencil_int__mod__cdata[AC_i_gva__mod__cparam-1] && AC_lalfven_as_aux__mod__magnetic) {
ac_transformed_pencil_gva = gradient(Field(AC_ialfven__mod__cdata))
if (AC_lresi_vaspeed__mod__magnetic) {
if (AC_va_min__mod__magnetic > 0.) {
ac_transformed_pencil_gva.x = set_zero_below_threshold(ac_transformed_pencil_etava,AC_va_min__mod__magnetic)
}
if (AC_va_min__mod__magnetic > 0.) {
ac_transformed_pencil_gva.y = set_zero_below_threshold(ac_transformed_pencil_etava,AC_va_min__mod__magnetic)
}
if (AC_va_min__mod__magnetic > 0.) {
ac_transformed_pencil_gva.z = set_zero_below_threshold(ac_transformed_pencil_etava,AC_va_min__mod__magnetic)
}
}
}
if (AC_lpencil_int__mod__cdata[AC_i_etaj__mod__cparam-1]) {
ac_transformed_pencil_etaj = AC_mu0__mod__cdata * AC_eta_j__mod__magnetic * AC_dxmax__mod__cdata*AC_dxmax__mod__cdata * sqrt(AC_mu0__mod__cdata * ac_transformed_pencil_j2 * ac_transformed_pencil_rho1)
if (AC_eta_min__mod__magnetic > 0.) {
ac_transformed_pencil_etaj = set_zero_below_threshold(ac_transformed_pencil_etaj,AC_eta_min__mod__magnetic)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_etaj2__mod__cparam-1]) {
ac_transformed_pencil_etaj2 = AC_etaj20__mod__magnetic * ac_transformed_pencil_j2 * ac_transformed_pencil_rho1
if (AC_eta_min__mod__magnetic > 0.) {
ac_transformed_pencil_etaj2 = set_zero_below_threshold(ac_transformed_pencil_etaj2,AC_eta_min__mod__magnetic)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_etajrho__mod__cparam-1]) {
ac_transformed_pencil_etajrho = AC_mu0__mod__cdata * AC_eta_jrho__mod__magnetic * AC_dxmax__mod__cdata * sqrt(ac_transformed_pencil_j2) * ac_transformed_pencil_rho1
if (AC_eta_min__mod__magnetic > 0.) {
ac_transformed_pencil_etajrho = set_zero_below_threshold(ac_transformed_pencil_etajrho,AC_eta_min__mod__magnetic)
}
}
if (AC_lpencil_int__mod__cdata[AC_i_jxb__mod__cparam-1]) {
ac_transformed_pencil_jxb = cross(ac_transformed_pencil_jj,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_cosjb__mod__cparam-1]) {
ac_transformed_pencil_cosjb = tini_sqrt_div(ac_transformed_pencil_jb,ac_transformed_pencil_j2,ac_transformed_pencil_b2)
if (AC_lpencil_check_at_work__mod__cdata) {
ac_transformed_pencil_cosjb = modulo(ac_transformed_pencil_cosjb + 1.0, 2.0) - 1
}
}
if (AC_lpencil_int__mod__cdata[AC_i_jparallel__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_jperp__mod__cparam-1]) {
ac_transformed_pencil_jparallel=sqrt(ac_transformed_pencil_j2)*ac_transformed_pencil_cosjb
ac_transformed_pencil_jperp=sqrt(ac_transformed_pencil_j2)*sqrt(abs(1-ac_transformed_pencil_cosjb*ac_transformed_pencil_cosjb))
}
if (AC_lpencil_int__mod__cdata[AC_i_jxbr__mod__cparam-1]) {
rho1_jxb_78_79_97=ac_transformed_pencil_rho1
if (AC_rhomin_jxb__mod__magnetic>0) {
rho1_jxb_78_79_97=min(rho1_jxb_78_79_97,1/AC_rhomin_jxb__mod__magnetic)
}
if (AC_va2max_jxb__mod__magnetic>0  &&  (! AC_betamin_jxb__mod__magnetic>0)) {
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1+pow((ac_transformed_pencil_va2/AC_va2max_jxb__mod__magnetic),AC_va2power_jxb__mod__magnetic)),(-1.0/AC_va2power_jxb__mod__magnetic))
}
if (AC_betamin_jxb__mod__magnetic>0) {
va2max_beta_78_79_97 = ac_transformed_pencil_cs2/AC_betamin_jxb__mod__magnetic*2.0*AC_gamma1__mod__magnetic
if (AC_va2max_jxb__mod__magnetic > 0) {
va2max_beta_78_79_97=min(va2max_beta_78_79_97,AC_va2max_jxb__mod__magnetic)
}
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1.+pow((ac_transformed_pencil_va2/va2max_beta_78_79_97),AC_va2power_jxb__mod__magnetic)),(-1.0/AC_va2power_jxb__mod__magnetic))
}
ac_transformed_pencil_jxbr = rho1_jxb_78_79_97*ac_transformed_pencil_jxb
}
if (AC_lpencil_int__mod__cdata[AC_i_jxbr2__mod__cparam-1]) {
ac_transformed_pencil_jxbr2 = dot(ac_transformed_pencil_jxbr,ac_transformed_pencil_jxbr)
}
if (AC_lpencil_int__mod__cdata[AC_i_ub__mod__cparam-1]) {
ac_transformed_pencil_ub = dot(ac_transformed_pencil_uu,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_ob__mod__cparam-1]) {
ac_transformed_pencil_ob = dot(ac_transformed_pencil_oo,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_uj__mod__cparam-1]) {
ac_transformed_pencil_uj = dot(ac_transformed_pencil_uu,ac_transformed_pencil_jj)
}
if (AC_lpencil_int__mod__cdata[AC_i_cosub__mod__cparam-1]) {
ac_transformed_pencil_cosub = tini_sqrt_div(ac_transformed_pencil_ub,ac_transformed_pencil_u2,ac_transformed_pencil_b2)
if (AC_lpencil_check__mod__cdata) {
ac_transformed_pencil_cosub = modulo(ac_transformed_pencil_cosub + 1.0, 2.0) - 1
}
}
if (AC_lpencil_int__mod__cdata[AC_i_uxb2__mod__cparam-1]) {
ac_transformed_pencil_uxb2 = dot(ac_transformed_pencil_uxb,ac_transformed_pencil_uxb)
}
if (AC_lpencil_int__mod__cdata[AC_i_uxj__mod__cparam-1]) {
ac_transformed_pencil_uxj = cross(ac_transformed_pencil_uu,ac_transformed_pencil_jj)
}
if (AC_lpencil_int__mod__cdata[AC_i_chibp__mod__cparam-1]) {
ac_transformed_pencil_chibp=atan2(ac_transformed_pencil_bb.y,ac_transformed_pencil_bb.x)+0.5*AC_pi__mod__cparam
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesi__mod__cparam-1]) {
ac_transformed_pencil_stokesi=pow((ac_transformed_pencil_bb.x*ac_transformed_pencil_bb.x+ac_transformed_pencil_bb.y*ac_transformed_pencil_bb.y),AC_exp_epspb__mod__magnetic)
}
if (AC_lncr_correlated__mod__magnetic) {
stokesi_ncr_78_79_97=ac_transformed_pencil_stokesi*ac_transformed_pencil_b2
if (AC_lpencil_int__mod__cdata[AC_i_stokesq__mod__cparam-1]) {
ac_transformed_pencil_stokesq=-stokesi_ncr_78_79_97*cos(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu__mod__cparam-1]) {
ac_transformed_pencil_stokesu=-stokesi_ncr_78_79_97*sin(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesq1__mod__cparam-1]) {
ac_transformed_pencil_stokesq1=+stokesi_ncr_78_79_97*sin(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu1__mod__cparam-1]) {
ac_transformed_pencil_stokesu1=-stokesi_ncr_78_79_97*cos(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
}
else if (AC_lncr_anticorrelated__mod__magnetic) {
stokesi_ncr_78_79_97=ac_transformed_pencil_stokesi/(1.+AC_ncr_quench__mod__magnetic*ac_transformed_pencil_b2)
if (AC_lpencil_int__mod__cdata[AC_i_stokesq__mod__cparam-1]) {
ac_transformed_pencil_stokesq=-stokesi_ncr_78_79_97*cos(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu__mod__cparam-1]) {
ac_transformed_pencil_stokesu=-stokesi_ncr_78_79_97*sin(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesq1__mod__cparam-1]) {
ac_transformed_pencil_stokesq1=+stokesi_ncr_78_79_97*sin(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu1__mod__cparam-1]) {
ac_transformed_pencil_stokesu1=-stokesi_ncr_78_79_97*cos(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
}
else {
if (AC_lpencil_int__mod__cdata[AC_i_stokesq__mod__cparam-1]) {
ac_transformed_pencil_stokesq=-ac_transformed_pencil_stokesi*cos(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu__mod__cparam-1]) {
ac_transformed_pencil_stokesu=-ac_transformed_pencil_stokesi*sin(2.*ac_transformed_pencil_chibp)
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesq1__mod__cparam-1]) {
ac_transformed_pencil_stokesq1=+ac_transformed_pencil_stokesi*sin(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
if (AC_lpencil_int__mod__cdata[AC_i_stokesu1__mod__cparam-1]) {
ac_transformed_pencil_stokesu1=-ac_transformed_pencil_stokesi*cos(2.*ac_transformed_pencil_chibp)*ac_transformed_pencil_bb.z
}
}
if (AC_lpencil_int__mod__cdata[AC_i_beta1__mod__cparam-1]) {
ac_transformed_pencil_beta1=0.5*ac_transformed_pencil_b2*AC_mu01__mod__cdata/ac_transformed_pencil_pp
}
if (AC_lpencil_int__mod__cdata[AC_i_beta__mod__cparam-1]) {
ac_transformed_pencil_beta = 2.0 * AC_mu0__mod__cdata * ac_transformed_pencil_pp / max(ac_transformed_pencil_b2, epsilon(1.0))
}
if (AC_lpencil_int__mod__cdata[AC_i_djuidjbi__mod__cparam-1]) {
ac_transformed_pencil_djuidjbi = contract(ac_transformed_pencil_uij,ac_transformed_pencil_bij)
}
if (AC_lpencil_int__mod__cdata[AC_i_jo__mod__cparam-1]) {
ac_transformed_pencil_jo = dot(ac_transformed_pencil_jj,ac_transformed_pencil_oo)
}
if (AC_lpencil_int__mod__cdata[AC_i_ujxb__mod__cparam-1]) {
ac_transformed_pencil_ujxb = dot(ac_transformed_pencil_uu,ac_transformed_pencil_jxb)
}
if (AC_lpencil_int__mod__cdata[AC_i_gb22__mod__cparam-1]) {
ac_transformed_pencil_gb22 = matmul_transpose(ac_transformed_pencil_bij,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_ugb__mod__cparam-1]) {
ac_transformed_pencil_ugb = ac_transformed_pencil_bij*ac_transformed_pencil_uu
}
if (AC_lpencil_int__mod__cdata[AC_i_ugb22__mod__cparam-1]) {
ac_transformed_pencil_ugb22 = dot(ac_transformed_pencil_uu,ac_transformed_pencil_gb22)
}
if (AC_lpencil_int__mod__cdata[AC_i_bdivu__mod__cparam-1]) {
ac_transformed_pencil_bdivu.x=ac_transformed_pencil_bb.x*ac_transformed_pencil_divu
ac_transformed_pencil_bdivu.y=ac_transformed_pencil_bb.y*ac_transformed_pencil_divu
ac_transformed_pencil_bdivu.z=ac_transformed_pencil_bb.z*ac_transformed_pencil_divu
}
if (AC_lpencil_int__mod__cdata[AC_i_bgu__mod__cparam-1]) {
ac_transformed_pencil_bgu = ac_transformed_pencil_uij*ac_transformed_pencil_bb
}
if (AC_lpencil_int__mod__cdata[AC_i_bgb__mod__cparam-1]) {
ac_transformed_pencil_bgb = ac_transformed_pencil_bij*ac_transformed_pencil_bb
}
if (AC_lpencil_int__mod__cdata[AC_i_bgbp__mod__cparam-1]) {
bbgb_78_79_97 = dot(ac_transformed_pencil_bb,ac_transformed_pencil_bgb)
ac_transformed_pencil_bgbp = bbgb_78_79_97*ac_transformed_pencil_b21*ac_transformed_pencil_bb
}
if (AC_lpencil_int__mod__cdata[AC_i_ubgbp__mod__cparam-1]) {
ac_transformed_pencil_ubgbp = dot(ac_transformed_pencil_uu,ac_transformed_pencil_bgbp)
}
if (AC_lpencil_int__mod__cdata[AC_i_oxuxb__mod__cparam-1]) {
ac_transformed_pencil_oxuxb = cross(ac_transformed_pencil_oxu,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_jxbxb__mod__cparam-1]) {
ac_transformed_pencil_jxbxb = cross(ac_transformed_pencil_jxb,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_jxbrxb__mod__cparam-1]) {
ac_transformed_pencil_jxbrxb = cross(ac_transformed_pencil_jxbr,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_glnrhoxb__mod__cparam-1]) {
ac_transformed_pencil_glnrhoxb = cross(ac_transformed_pencil_glnrho,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_del4a__mod__cparam-1]) {
ac_transformed_pencil_del4a = del4((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_hjj__mod__cparam-1]) {
ac_transformed_pencil_hjj = ac_transformed_pencil_del4a
}
if (AC_lpencil_int__mod__cdata[AC_i_hj2__mod__cparam-1]) {
ac_transformed_pencil_hj2 = dot(ac_transformed_pencil_hjj,ac_transformed_pencil_hjj)
}
if (AC_lpencil_int__mod__cdata[AC_i_hjb__mod__cparam-1]) {
ac_transformed_pencil_hjb = dot(ac_transformed_pencil_hjj,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_coshjb__mod__cparam-1]) {
ac_transformed_pencil_coshjb = tini_sqrt_div_separate(ac_transformed_pencil_hjb,ac_transformed_pencil_hj2,ac_transformed_pencil_b2)
if (AC_lpencil_check_at_work__mod__cdata) {
ac_transformed_pencil_coshjb = modulo(ac_transformed_pencil_coshjb + 1.0, 2.0) - 1
}
}
if (AC_lpencil_int__mod__cdata[AC_i_hjparallel__mod__cparam-1] || AC_lpencil_int__mod__cdata[AC_i_hjperp__mod__cparam-1]) {
ac_transformed_pencil_hjparallel=sqrt(ac_transformed_pencil_hj2)*ac_transformed_pencil_coshjb
ac_transformed_pencil_hjperp=sqrt(ac_transformed_pencil_hj2)*sqrt(abs(1-ac_transformed_pencil_coshjb*ac_transformed_pencil_coshjb))
}
if (AC_lpencil_int__mod__cdata[AC_i_del6a__mod__cparam-1]) {
ac_transformed_pencil_del6a = del6((Field3){Field(AC_iaa__mod__cdata), Field(AC_iaa__mod__cdata+1), Field(AC_iaa__mod__cdata+2)})
}
if (AC_lpencil_int__mod__cdata[AC_i_e3xa__mod__cparam-1]) {
ac_transformed_pencil_e3xa = cross(ac_transformed_pencil_uxb+AC_eta_hyper3__mod__magnetic*ac_transformed_pencil_del6a,ac_transformed_pencil_aa)
}
if (AC_lpencil_int__mod__cdata[AC_i_oxj__mod__cparam-1]) {
ac_transformed_pencil_oxj = cross(ac_transformed_pencil_oo,ac_transformed_pencil_jj)
}
if (AC_lpencil_int__mod__cdata[AC_i_jij__mod__cparam-1]) {
ac_transformed_pencil_jij[1-1][1-1]=0.5*(ac_transformed_pencil_bij[1-1][1-1]+ac_transformed_pencil_bij[1-1][1-1])
ac_transformed_pencil_jij[2-1][1-1]=0.5*(ac_transformed_pencil_bij[2-1][1-1]+ac_transformed_pencil_bij[1-1][2-1])
ac_transformed_pencil_jij[3-1][1-1]=0.5*(ac_transformed_pencil_bij[3-1][1-1]+ac_transformed_pencil_bij[1-1][3-1])
ac_transformed_pencil_jij[1-1][2-1]=0.5*(ac_transformed_pencil_bij[1-1][2-1]+ac_transformed_pencil_bij[2-1][1-1])
ac_transformed_pencil_jij[2-1][2-1]=0.5*(ac_transformed_pencil_bij[2-1][2-1]+ac_transformed_pencil_bij[2-1][2-1])
ac_transformed_pencil_jij[3-1][2-1]=0.5*(ac_transformed_pencil_bij[3-1][2-1]+ac_transformed_pencil_bij[2-1][3-1])
ac_transformed_pencil_jij[1-1][3-1]=0.5*(ac_transformed_pencil_bij[1-1][3-1]+ac_transformed_pencil_bij[3-1][1-1])
ac_transformed_pencil_jij[2-1][3-1]=0.5*(ac_transformed_pencil_bij[2-1][3-1]+ac_transformed_pencil_bij[3-1][2-1])
ac_transformed_pencil_jij[3-1][3-1]=0.5*(ac_transformed_pencil_bij[3-1][3-1]+ac_transformed_pencil_bij[3-1][3-1])
}
if (AC_lpencil_int__mod__cdata[AC_i_d6ab__mod__cparam-1]) {
ac_transformed_pencil_d6ab = dot(ac_transformed_pencil_del6a,ac_transformed_pencil_bb)
}
if (AC_lpencil_int__mod__cdata[AC_i_sj__mod__cparam-1]) {
ac_transformed_pencil_sj = contract(ac_transformed_pencil_sij,ac_transformed_pencil_jij)
}
if (AC_lpencil_int__mod__cdata[AC_i_ss12__mod__cparam-1]) {
ac_transformed_pencil_ss12=sqrt(abs(ac_transformed_pencil_sj))
}
if (AC_lpencil_int__mod__cdata[AC_i_vmagfric__mod__cparam-1] && AC_numag__mod__magnetic!=0.0) {
tmp1_78_79_97=AC_mu01__mod__cdata/(AC_numag__mod__magnetic*(AC_b0_magfric__mod__magnetic/AC_unit_magnetic__mod__cdata*AC_unit_magnetic__mod__cdata+ac_transformed_pencil_b2))
ac_transformed_pencil_vmagfric.x=abs(ac_transformed_pencil_jxb.x)*tmp1_78_79_97
ac_transformed_pencil_vmagfric.y=abs(ac_transformed_pencil_jxb.y)*tmp1_78_79_97
ac_transformed_pencil_vmagfric.z=abs(ac_transformed_pencil_jxb.z)*tmp1_78_79_97
}
if (AC_lpencil_int__mod__cdata[AC_i_lam__mod__cparam-1]) {
if (AC_lcoulomb__mod__magnetic) {
ac_transformed_pencil_lam=value(F_LAM)
}
else {
}
}
if (AC_lbb_as_aux__mod__magnetic  &&  ! AC_lbb_as_comaux__mod__magnetic) {
DF_BVEC = ac_transformed_pencil_bb
}
if (AC_ljj_as_aux__mod__magnetic  &&  ! AC_ljj_as_comaux__mod__magnetic) {
DF_JVEC = ac_transformed_pencil_jj
}
if (AC_ljxb_as_aux__mod__magnetic) {
DF_JXBVEC=ac_transformed_pencil_jxb
}
if (AC_lpencil_int__mod__cdata[AC_i_bb_sph__mod__cparam-1] && AC_lbb_sph_as_aux__mod__magnetic) {
ac_transformed_pencil_bb_sph=value(F_BB_SPHVEC)
}
if (AC_luxb_as_aux__mod__magnetic) {
DF_UXBVEC = ac_transformed_pencil_uxb
}
if (AC_lugb_as_aux__mod__magnetic) {
DF_UGBVEC = ac_transformed_pencil_ugb
}
if (AC_lbgu_as_aux__mod__magnetic) {
DF_BGUVEC = ac_transformed_pencil_bgu
}
if (AC_lbdivu_as_aux__mod__magnetic) {
DF_BDIVUVEC = ac_transformed_pencil_bdivu
}
if (AC_lpencil_int__mod__cdata[AC_i_mf_emf__mod__cparam-1]) {
ac_transformed_pencil_mf_emf.x = 0.0
ac_transformed_pencil_mf_emf.y = 0.0
ac_transformed_pencil_mf_emf.z = 0.0
}
if (AC_lpencil_int__mod__cdata[AC_i_mf_emfdotb__mod__cparam-1]) {
ac_transformed_pencil_mf_emfdotb=0.0
}
if(AC_string_enum_ambipolar_diffusion__mod__magnetic == AC_string_enum_constant_string__mod__cparam) {
ac_transformed_pencil_nu_ni1=AC_nu_ni1__mod__magnetic
}
else if(AC_string_enum_ambipolar_diffusion__mod__magnetic == AC_string_enum_ionizationzequilibrium_string__mod__cparam) {
ac_transformed_pencil_nu_ni1=AC_nu_ni1__mod__magnetic*sqrt(ac_transformed_pencil_rho1)
}
else if(AC_string_enum_ambipolar_diffusion__mod__magnetic == AC_string_enum_ionizationzyh_string__mod__cparam) {
ac_transformed_pencil_nu_ni1=AC_nu_ni1__mod__magnetic*sqrt(ac_transformed_pencil_rho1)*(1.-ac_transformed_pencil_yh)/ac_transformed_pencil_yh
}
else {
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
if (AC_lhydro__mod__cparam && AC_llorentzforce__mod__magnetic) {
rho1_jxb_78_79_97=ac_transformed_pencil_rho1
if (AC_rhomin_jxb__mod__magnetic>0) {
rho1_jxb_78_79_97=min(rho1_jxb_78_79_97,1/AC_rhomin_jxb__mod__magnetic)
}
if (AC_va2max_jxb__mod__magnetic>0  &&  (! AC_betamin_jxb__mod__magnetic>0)) {
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1+pow((ac_transformed_pencil_va2/AC_va2max_jxb__mod__magnetic),AC_va2power_jxb__mod__magnetic)),(-1.0/AC_va2power_jxb__mod__magnetic))
}
if (AC_betamin_jxb__mod__magnetic>0) {
va2max_beta_78_79_97 = ac_transformed_pencil_cs2/AC_betamin_jxb__mod__magnetic*2.0*AC_gamma1__mod__magnetic
if (AC_va2max_jxb__mod__magnetic > 0) {
va2max_beta_78_79_97=min(va2max_beta_78_79_97,AC_va2max_jxb__mod__magnetic)
}
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1+pow((ac_transformed_pencil_va2/va2max_beta_78_79_97),AC_va2power_jxb__mod__magnetic)),(-1.0/AC_va2power_jxb__mod__magnetic))
}
if (AC_lboris_correction__mod__magnetic) {
if (AC_va2max_boris__mod__magnetic>0) {
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1+pow((ac_transformed_pencil_va2/AC_va2max_boris__mod__magnetic),2.)),(-0.5))
}
if (AC_cmin__mod__magnetic>0) {
rho1_jxb_78_79_97 = rho1_jxb_78_79_97 * pow((1+pow((ac_transformed_pencil_va2/ac_transformed_pencil_clight2),2.)),(-0.5))
}
}
ac_transformed_pencil_advec_va2=sum((ac_transformed_pencil_bb*dline_1__mod__cdata)*(ac_transformed_pencil_bb*dline_1__mod__cdata))*AC_mu01__mod__cdata*rho1_jxb_78_79_97
}
else {
ac_transformed_pencil_advec_va2=0.
}
if (AC_lisotropic_advection__mod__cdata) {
if (AC_dimensionality__mod__cparam<3) {
ac_transformed_pencil_advec_va2=ac_transformed_pencil_va2*dxyz_2__mod__cdata
}
}
if (AC_lhydro__mod__cparam && AC_hall_term__mod__magnetic!=0.0) {
ac_transformed_pencil_advec_va2=( (ac_transformed_pencil_bb.x*dline_1__mod__cdata.x*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata) ) ))*(ac_transformed_pencil_bb.x*dline_1__mod__cdata.x*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.x*AC_mu01__mod__cdata) ) ))  +(ac_transformed_pencil_bb.y*dline_1__mod__cdata.y*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata) ) ))*(ac_transformed_pencil_bb.y*dline_1__mod__cdata.y*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.y*AC_mu01__mod__cdata) ) ))  +(ac_transformed_pencil_bb.z*dline_1__mod__cdata.z*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata) ) ))*(ac_transformed_pencil_bb.z*dline_1__mod__cdata.z*( AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata  +sqrt(AC_mu01__mod__cdata*ac_transformed_pencil_rho1 + (AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata)*(AC_hall_term__mod__magnetic*AC_pi__mod__cparam*dline_1__mod__cdata.z*AC_mu01__mod__cdata) ) ))  )
}
advec2__mod__cdata=advec2__mod__cdata+ac_transformed_pencil_advec_va2
if (AC_lmagneto_friction__mod__magnetic) {
tmp1_78_79_97 = dot(ac_transformed_pencil_vmagfric,ac_transformed_pencil_vmagfric)
advec2__mod__cdata=advec2__mod__cdata + tmp1_78_79_97
}
}
if (AC_lpencil_int__mod__cdata[AC_i_gg__mod__cparam-1]) {
ac_transformed_pencil_gg.x = 0.
ac_transformed_pencil_gg.y = 0.
ac_transformed_pencil_gg.z = 0.
}
fmax__mod__hydro=1./AC_impossible__mod__cparam
if (AC_ladvection_velocity__mod__hydro) {
if (! AC_lconservative__mod__hydro  &&  ! AC_lweno_transport__mod__cdata  &&   ! AC_lno_meridional_flow__mod__hydro  &&  ! AC_lfargo_advection__mod__cdata) {
if (AC_lschur_3d3d1d_uu__mod__hydro) {
ugu_schur_x_128 = dot(ac_transformed_pencil_uu,ac_transformed_pencil_uij.row(1-1))
ugu_schur_y_128 = dot(ac_transformed_pencil_uu,ac_transformed_pencil_uij.row(2-1))
ugu_schur_z_128=ac_transformed_pencil_uu.z*ac_transformed_pencil_uij[3-1][3-1]
DF_UX=DF_UX-ugu_schur_x_128
DF_UY=DF_UY-ugu_schur_y_128
DF_UZ=DF_UZ-ugu_schur_z_128
}
else if (AC_lschur_2d2d3d_uu__mod__hydro) {
puij_schur_128[1-1][1-1] = ac_transformed_pencil_uij[1-1][1-1]
puij_schur_128[2-1][1-1] = ac_transformed_pencil_uij[2-1][1-1]
puij_schur_128[3-1][1-1] = ac_transformed_pencil_uij[3-1][1-1]
puij_schur_128[1-1][2-1] = ac_transformed_pencil_uij[1-1][2-1]
puij_schur_128[2-1][2-1] = ac_transformed_pencil_uij[2-1][2-1]
puij_schur_128[3-1][2-1] = ac_transformed_pencil_uij[3-1][2-1]
puij_schur_128[1-1][3-1] = ac_transformed_pencil_uij[1-1][3-1]
puij_schur_128[2-1][3-1] = ac_transformed_pencil_uij[2-1][3-1]
puij_schur_128[3-1][3-1] = ac_transformed_pencil_uij[3-1][3-1]
puij_schur_128[1-1][3-1] = 0
puij_schur_128[2-1][3-1] = 0
ugu_schur_x_128 = dot(ac_transformed_pencil_uu,puij_schur_128.row(1-1))
ugu_schur_y_128 = dot(ac_transformed_pencil_uu,puij_schur_128.row(2-1))
ugu_schur_z_128 = dot(ac_transformed_pencil_uu,puij_schur_128.row(3-1))
DF_UX=DF_UX-ugu_schur_x_128
DF_UY=DF_UY-ugu_schur_y_128
DF_UZ=DF_UZ-ugu_schur_z_128
}
else if (AC_lschur_2d2d1d_uu__mod__hydro) {
puij_schur_128[1-1][1-1] = ac_transformed_pencil_uij[1-1][1-1]
puij_schur_128[2-1][1-1] = ac_transformed_pencil_uij[2-1][1-1]
puij_schur_128[3-1][1-1] = ac_transformed_pencil_uij[3-1][1-1]
puij_schur_128[1-1][2-1] = ac_transformed_pencil_uij[1-1][2-1]
puij_schur_128[2-1][2-1] = ac_transformed_pencil_uij[2-1][2-1]
puij_schur_128[3-1][2-1] = ac_transformed_pencil_uij[3-1][2-1]
puij_schur_128[1-1][3-1] = ac_transformed_pencil_uij[1-1][3-1]
puij_schur_128[2-1][3-1] = ac_transformed_pencil_uij[2-1][3-1]
puij_schur_128[3-1][3-1] = ac_transformed_pencil_uij[3-1][3-1]
puij_schur_128[1-1][3-1] = 0
puij_schur_128[2-1][3-1] = 0
ugu_schur_x_128 = dot(ac_transformed_pencil_uu,puij_schur_128.row(1-1))
ugu_schur_y_128 = dot(ac_transformed_pencil_uu,puij_schur_128.row(2-1))
ugu_schur_z_128=ac_transformed_pencil_uu.z*ac_transformed_pencil_uij[3-1][3-1]
DF_UX=DF_UX-ugu_schur_x_128
DF_UY=DF_UY-ugu_schur_y_128
DF_UZ=DF_UZ-ugu_schur_z_128
}
else {
DF_UVEC=DF_UVEC-ac_transformed_pencil_ugu
}
}
if (AC_ldensity__mod__cparam && AC_lconservative__mod__hydro) {
tmp_128 = derx(Field(AC_itij__mod__hydro+0))
DF_UX=DF_UX-tmp_128
tmp_128 = dery(Field(AC_itij__mod__hydro+1))
DF_UY=DF_UY-tmp_128
tmp_128 = derz(Field(AC_itij__mod__hydro+2))
DF_UZ=DF_UZ-tmp_128
tmp_128 = dery(Field(AC_itij__mod__hydro+3))
DF_UX=DF_UX-tmp_128
tmp_128 = derz(Field(AC_itij__mod__hydro+5))
DF_UX=DF_UX-tmp_128
tmp_128 = derx(Field(AC_itij__mod__hydro+3))
DF_UY=DF_UY-tmp_128
tmp_128 = derz(Field(AC_itij__mod__hydro+4))
DF_UY=DF_UY-tmp_128
tmp_128 = derx(Field(AC_itij__mod__hydro+5))
DF_UZ=DF_UZ-tmp_128
tmp_128 = dery(Field(AC_itij__mod__hydro+4))
DF_UZ=DF_UZ-tmp_128
}
if (AC_lweno_transport__mod__cdata) {
DF_UX=DF_UX-(ac_transformed_pencil_transpurho.x-ac_transformed_pencil_uu.x*ac_transformed_pencil_transprho)*ac_transformed_pencil_rho1
DF_UY=DF_UY-(ac_transformed_pencil_transpurho.y-ac_transformed_pencil_uu.y*ac_transformed_pencil_transprho)*ac_transformed_pencil_rho1
DF_UZ=DF_UZ-(ac_transformed_pencil_transpurho.z-ac_transformed_pencil_uu.z*ac_transformed_pencil_transprho)*ac_transformed_pencil_rho1
}
if (AC_lno_meridional_flow__mod__hydro) {
DF_UX=0.0
DF_UY=0.0
DF_UZ=DF_UZ-ac_transformed_pencil_ugu.z
}
if (AC_lfargo_advection__mod__cdata) {
DF_UVEC=DF_UVEC-ac_transformed_pencil_uuadvec_guu
}
}
if (AC_omega__mod__cdata!=0.) {
if (AC_lcylindrical_coords__mod__cdata) {
if (AC_lcoriolis_force__mod__hydro) {
if (AC_lomega_cyl_xy__mod__hydro) {
c2_102_128= 2*AC_omega__mod__cdata*cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
s2_102_128=-2*AC_omega__mod__cdata*sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
DF_UX=DF_UX-c2_102_128*ac_transformed_pencil_uu.z
DF_UY=DF_UY-s2_102_128*ac_transformed_pencil_uu.z
DF_UZ=DF_UZ+c2_102_128*ac_transformed_pencil_uu.x+s2_102_128*ac_transformed_pencil_uu.y
}
else {
c2_102_128=2*AC_omega__mod__cdata
DF_UX=DF_UX+c2_102_128*ac_transformed_pencil_uu.y
DF_UY=DF_UY-c2_102_128*ac_transformed_pencil_uu.x
}
}
if (AC_lcentrifugal_force__mod__hydro) {
DF_UX=DF_UX+AC_x__mod__cdata[vertexIdx.x]*AC_omega__mod__cdata*AC_omega__mod__cdata
}
}
else if (AC_lspherical_coords__mod__cdata) {
if (AC_lcoriolis_force__mod__hydro) {
c2_103_128= 2*AC_omega__mod__cdata*AC_costh__mod__cdata[AC_m__mod__cdata-1]
s2_103_128=-2*AC_omega__mod__cdata*AC_sinth__mod__cdata[AC_m__mod__cdata-1]
}
if (AC_theta__mod__cdata==0.0) {
if (AC_lcoriolis_force__mod__hydro) {
if (AC_r_omega__mod__hydro != 0.) {
DF_UX=DF_UX- s2_103_128*ac_transformed_pencil_uu.z              *AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
DF_UY=DF_UY+ c2_103_128*ac_transformed_pencil_uu.z              *AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
DF_UZ=DF_UZ-(c2_103_128*ac_transformed_pencil_uu.y-s2_103_128*ac_transformed_pencil_uu.x)*AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
}
else {
DF_UX=DF_UX-s2_103_128*ac_transformed_pencil_uu.z
DF_UY=DF_UY+c2_103_128*ac_transformed_pencil_uu.z
DF_UZ=DF_UZ-c2_103_128*ac_transformed_pencil_uu.y+s2_103_128*ac_transformed_pencil_uu.x
}
}
}
else {
cp2_103_128=2*AC_omega__mod__cdata*AC_cosph__mod__cdata[AC_n__mod__cdata-1]
cs2_103_128=c2_103_128*AC_sinph__mod__cdata[AC_n__mod__cdata-1]
ss2_103_128=s2_103_128*AC_sinph__mod__cdata[AC_n__mod__cdata-1]
if (AC_lcoriolis_force__mod__hydro) {
if (AC_r_omega__mod__hydro != 0.) {
DF_UX=DF_UX+(-cs2_103_128*ac_transformed_pencil_uu.z + cp2_103_128*ac_transformed_pencil_uu.y)*AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
DF_UY=DF_UY+(-cp2_103_128*ac_transformed_pencil_uu.x - ss2_103_128*ac_transformed_pencil_uu.z)*AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
DF_UZ=DF_UZ+(+ss2_103_128*ac_transformed_pencil_uu.y + cs2_103_128*ac_transformed_pencil_uu.x)*AC_prof_om__mod__hydro[vertexIdx.x-NGHOST_VAL]
}
else {
DF_UX=DF_UX - cs2_103_128*ac_transformed_pencil_uu.z + cp2_103_128*ac_transformed_pencil_uu.y
DF_UY=DF_UY - cp2_103_128*ac_transformed_pencil_uu.x - ss2_103_128*ac_transformed_pencil_uu.z
DF_UZ=DF_UZ + ss2_103_128*ac_transformed_pencil_uu.y + cs2_103_128*ac_transformed_pencil_uu.x
}
}
}
if (AC_lcentrifugal_force__mod__hydro) {
om2_103_128=AC_amp_centforce__mod__hydro*AC_omega__mod__cdata*AC_omega__mod__cdata
if (AC_theta__mod__cdata==0.0) {
DF_UX=DF_UX-om2_103_128*AC_x__mod__cdata[vertexIdx.x]*AC_sinth__mod__cdata[AC_m__mod__cdata-1]
}
else {
}
}
}
else if (AC_lprecession__mod__hydro) {
cent_res_104_128 = AC_mat_cent__mod__hydro*ac_transformed_pencil_rr
cori_res_104_128 = AC_mat_cent__mod__hydro*ac_transformed_pencil_uu
DF_UVEC = DF_UVEC + cent_res_104_128 + cori_res_104_128
}
else if (AC_lrotation_xaxis__mod__hydro) {
if (AC_omega__mod__cdata != 0.  &&  AC_lcoriolis_force__mod__hydro) {
c2_105_128= 2*AC_omega__mod__cdata*cos(AC_theta__mod__cdata*AC_pi__mod__cparam/180.)
s2_105_128=-2*AC_omega__mod__cdata*sin(AC_theta__mod__cdata*AC_pi__mod__cparam/180.)
DF_UX=DF_UX-s2_105_128*ac_transformed_pencil_uu.z
DF_UY=DF_UY+c2_105_128*ac_transformed_pencil_uu.z
DF_UZ=DF_UZ-c2_105_128*ac_transformed_pencil_uu.y+s2_105_128*ac_transformed_pencil_uu.x
}
}
else {
if (AC_omega__mod__cdata != 0.  &&  AC_theta__mod__cdata==0) {
if (AC_lcoriolis_force__mod__hydro) {
c2_106_128=2*AC_omega__mod__cdata
DF_UX=DF_UX+c2_106_128*ac_transformed_pencil_uu.y
DF_UY=DF_UY-c2_106_128*ac_transformed_pencil_uu.x
}
if (AC_lcentrifugal_force__mod__hydro) {
DF_UX=DF_UX+AC_x__mod__cdata[vertexIdx.x]*AC_amp_centforce__mod__hydro*AC_omega__mod__cdata*AC_omega__mod__cdata
DF_UY=DF_UY+AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_amp_centforce__mod__hydro*AC_omega__mod__cdata*AC_omega__mod__cdata
}
}
else if(AC_omega__mod__cdata != 0.) {
if (AC_lcoriolis_force__mod__hydro) {
c2_106_128= 2*AC_omega__mod__cdata*cos(AC_theta__mod__cdata*AC_dtor__mod__cparam)
s2_106_128=-2*AC_omega__mod__cdata*sin(AC_theta__mod__cdata*AC_dtor__mod__cparam)
DF_UX=DF_UX+c2_106_128*ac_transformed_pencil_uu.y
DF_UY=DF_UY-c2_106_128*ac_transformed_pencil_uu.x+s2_106_128*ac_transformed_pencil_uu.z
DF_UZ=DF_UZ           -s2_106_128*ac_transformed_pencil_uu.y
if (AC_lshear_in_coriolis__mod__hydro) {
DF_UX=DF_UX+c2_106_128*AC_sshear__mod__cdata*AC_x__mod__cdata[vertexIdx.x]
DF_UZ=DF_UZ-s2_106_128*AC_sshear__mod__cdata*AC_x__mod__cdata[vertexIdx.x]
}
}
}
}
}
c1_107_128=-2*AC_ampl_omega__mod__hydro*sin(AC_pi__mod__cparam*((AC_x__mod__cdata[vertexIdx.x])-AC_x0__mod__cdata)/AC_lx__mod__cdata)
c2_107_128= 2*AC_ampl_omega__mod__hydro*cos(AC_pi__mod__cparam*((AC_x__mod__cdata[vertexIdx.x])-AC_x0__mod__cdata)/AC_lx__mod__cdata)
DF_UX=DF_UX             +c2_107_128*ac_transformed_pencil_uu.y
DF_UY=DF_UY+c1_107_128*ac_transformed_pencil_uu.z-c2_107_128*ac_transformed_pencil_uu.x
DF_UZ=DF_UZ-c1_107_128*ac_transformed_pencil_uu.y
if (AC_lmagfield_nu__mod__viscosity) {
DF_UX = DF_UX + ac_transformed_pencil_fvisc.x/(1.+ac_transformed_pencil_b2/AC_meanfield_nub__mod__viscosity*AC_meanfield_nub__mod__viscosity)
DF_UY = DF_UY + ac_transformed_pencil_fvisc.y/(1.+ac_transformed_pencil_b2/AC_meanfield_nub__mod__viscosity*AC_meanfield_nub__mod__viscosity)
DF_UY = DF_UY + ac_transformed_pencil_fvisc.z/(1.+ac_transformed_pencil_b2/AC_meanfield_nub__mod__viscosity*AC_meanfield_nub__mod__viscosity)
}
else {
DF_UVEC = DF_UVEC + ac_transformed_pencil_fvisc
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_nu__mod__viscosity = ac_transformed_pencil_diffus_total*dxyz_2__mod__cdata
if (AC_ldynamical_diffusion__mod__cdata  &&  AC_lvisc_hyper3_mesh__mod__viscosity) {
diffus_nu3__mod__viscosity = ac_transformed_pencil_diffus_total3 * sum(abs(dline_1__mod__cdata))
}
else {
diffus_nu3__mod__viscosity = ac_transformed_pencil_diffus_total3*dxyz_6__mod__cdata
}
maxdiffus__mod__cdata =max(maxdiffus__mod__cdata ,diffus_nu__mod__viscosity)
maxdiffus2__mod__cdata=max(maxdiffus2__mod__cdata,ac_transformed_pencil_diffus_total2*dxyz_4__mod__cdata)
maxdiffus3__mod__cdata=max(maxdiffus3__mod__cdata,diffus_nu3__mod__viscosity)
}
if (AC_ldiagnos__mod__cdata) {
if (AC_idiag_reshock__mod__viscosity!=0) {
if (abs(ac_transformed_pencil_shock) > AC_tini__mod__cparam) {
reshock_110_111_128 = dxmax_pencil__mod__cdata*sqrt(ac_transformed_pencil_u2)/(AC_nu_shock__mod__viscosity*ac_transformed_pencil_shock)
}
else {
reshock_110_111_128=0.
}
}
if (AC_idiag_epsk_les__mod__viscosity!=0) {
if (AC_lvisc_smag_simplified__mod__viscosity) {
}
else if (AC_lvisc_smag_cross_simplified__mod__viscosity) {
}
}
if (AC_idiag_sijoiojm__mod__viscosity!=0) {
tmp_110_111_128=0.
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[1-1][1-1]*ac_transformed_pencil_oo.x*ac_transformed_pencil_oo.x
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[1-1][2-1]*ac_transformed_pencil_oo.x*ac_transformed_pencil_oo.y
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[1-1][3-1]*ac_transformed_pencil_oo.x*ac_transformed_pencil_oo.z
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[2-1][1-1]*ac_transformed_pencil_oo.y*ac_transformed_pencil_oo.x
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[2-1][2-1]*ac_transformed_pencil_oo.y*ac_transformed_pencil_oo.y
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[2-1][3-1]*ac_transformed_pencil_oo.y*ac_transformed_pencil_oo.z
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[3-1][1-1]*ac_transformed_pencil_oo.z*ac_transformed_pencil_oo.x
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[3-1][2-1]*ac_transformed_pencil_oo.z*ac_transformed_pencil_oo.y
tmp_110_111_128=tmp_110_111_128+ac_transformed_pencil_sij[3-1][3-1]*ac_transformed_pencil_oo.z*ac_transformed_pencil_oo.z
}
if (AC_lmagnetic__mod__cparam) {
if (AC_idiag_nud2uxbxm__mod__viscosity!=0 || AC_idiag_nud2uxbym__mod__viscosity!=0 || AC_idiag_nud2uxbzm__mod__viscosity!=0) {
nud2uxb_110_111_128 = cross(ac_transformed_pencil_fvisc,ac_transformed_pencil_bb)
}
}
if (AC_idiag_fviscm__mod__viscosity!=0  ||  AC_idiag_fviscrmsx__mod__viscosity!=0  ||  AC_idiag_fviscmin__mod__viscosity!=0  ||  AC_idiag_fviscmax__mod__viscosity!=0) {
fvisc2_110_111_128 = dot(ac_transformed_pencil_fvisc,ac_transformed_pencil_fvisc)
}
if (AC_idiag_qfviscm__mod__viscosity!=0) {
qfvisc_110_111_128 = dot(ac_transformed_pencil_curlo,ac_transformed_pencil_fvisc)
}
}
if (AC_l1davgfirst__mod__cdata) {
if (AC_idiag_viscforcezupmz__mod__viscosity!=0) {
if (ac_transformed_pencil_uu.z > 0.) {
uus_110_111_128 = ac_transformed_pencil_rho*ac_transformed_pencil_fvisc.z
}
else {
uus_110_111_128=0.
}
}
if (AC_idiag_viscforcezdownmz__mod__viscosity!=0) {
if (ac_transformed_pencil_uu.z < 0.) {
uus_110_111_128 = ac_transformed_pencil_rho*ac_transformed_pencil_fvisc.z
}
else {
uus_110_111_128=0.
}
}
}
if (AC_l2davgfirst__mod__cdata) {
if (AC_idiag_fviscmxy__mod__viscosity!=0) {
if (AC_lvisc_sqrtrho_nu_const__mod__viscosity) {
}
else if (AC_lvisc_rho_nu_const__mod__viscosity) {
}
else if (AC_lvisc_nu_profy_bound__mod__viscosity) {
}
else {
}
}
if (AC_idiag_fviscymxy__mod__viscosity!=0) {
if (AC_lyang__mod__cdata) {
fluxv_110_111_128.x=0.
fluxv_110_111_128.y=ac_transformed_pencil_uu.x*ac_transformed_pencil_sij[1-1][2-1]+ac_transformed_pencil_uu.y*ac_transformed_pencil_sij[2-1][2-1]+ac_transformed_pencil_uu.z*ac_transformed_pencil_sij[3-1][2-1]
fluxv_110_111_128.z=ac_transformed_pencil_uu.x*ac_transformed_pencil_sij[1-1][3-1]+ac_transformed_pencil_uu.y*ac_transformed_pencil_sij[2-1][3-1]+ac_transformed_pencil_uu.z*ac_transformed_pencil_sij[3-1][3-1]
}
else {
}
}
if (AC_idiag_fviscrsphmphi__mod__viscosity!=0) {
fluxv_110_111_128.x=ac_transformed_pencil_uu.x*ac_transformed_pencil_sij[1-1][1-1]+ac_transformed_pencil_uu.y*ac_transformed_pencil_sij[2-1][1-1]+ac_transformed_pencil_uu.z*ac_transformed_pencil_sij[3-1][1-1]
fluxv_110_111_128.y=ac_transformed_pencil_uu.x*ac_transformed_pencil_sij[1-1][2-1]+ac_transformed_pencil_uu.y*ac_transformed_pencil_sij[2-1][2-1]+ac_transformed_pencil_uu.z*ac_transformed_pencil_sij[3-1][2-1]
fluxv_110_111_128.z=ac_transformed_pencil_uu.x*ac_transformed_pencil_sij[1-1][3-1]+ac_transformed_pencil_uu.y*ac_transformed_pencil_sij[2-1][3-1]+ac_transformed_pencil_uu.z*ac_transformed_pencil_sij[3-1][3-1]
}
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata && AC_ladvection_velocity__mod__hydro) {
maxadvec__mod__cdata=maxadvec__mod__cdata+ac_transformed_pencil_advec_uu
}
if (AC_ekman_friction__mod__hydro!=0) {
if(AC_string_enum_friction_tdep__mod__hydro == AC_string_enum_nothing_string__mod__cparam) {
frict__mod__hydro=AC_ekman_friction__mod__hydro
}
else if(AC_string_enum_friction_tdep__mod__hydro == AC_string_enum_linear_string__mod__cparam) {
frict__mod__hydro=AC_ekman_friction__mod__hydro*max(min(AC_t__mod__cdata-AC_friction_tdep_toffset__mod__hydro/AC_friction_tdep_tau0__mod__hydro,1.),0.)
}
else if(AC_string_enum_friction_tdep__mod__hydro == AC_string_enum_inverse_string__mod__cparam) {
frict__mod__hydro=AC_ekman_friction__mod__hydro/max(AC_t__mod__cdata,AC_friction_tdep_toffset__mod__hydro)
}
else if(AC_string_enum_friction_tdep__mod__hydro == AC_string_enum_current_string__mod__cparam) {
if (AC_lmagnetic__mod__cparam) {
frict__mod__hydro=AC_ekman_friction__mod__hydro*sqrt(ac_transformed_pencil_j2)
}
else {
}
}
else {
}
tmpv_128 = frict__mod__hydro*ac_transformed_pencil_uu
DF_UVEC=DF_UVEC-tmpv_128
}
if (AC_lboussinesq__mod__cparam && AC_ltemperature__mod__cparam) {
if (AC_lsphere_in_a_box__mod__cdata) {
ju_128=1+AC_iuu__mod__cdata-1
DF_U_128=DF_U_128 + ac_transformed_pencil_r_mn*AC_ra__mod__hydro*AC_pr__mod__hydro*value(F_TT)*ac_transformed_pencil_evr.x
ju_128=2+AC_iuu__mod__cdata-1
DF_U_128=DF_U_128 + ac_transformed_pencil_r_mn*AC_ra__mod__hydro*AC_pr__mod__hydro*value(F_TT)*ac_transformed_pencil_evr.y
ju_128=3+AC_iuu__mod__cdata-1
DF_U_128=DF_U_128 + ac_transformed_pencil_r_mn*AC_ra__mod__hydro*AC_pr__mod__hydro*value(F_TT)*ac_transformed_pencil_evr.z
}
else {
DF_UZ=DF_UZ+AC_ra__mod__hydro*AC_pr__mod__hydro*value(F_TT)
}
}
if (AC_lforcing_cont_uu__mod__hydro) {
DF_UVEC=DF_UVEC + AC_ampl_fcont_uu__mod__hydro*ac_transformed_pencil_fcont[1-1]
}
if ((AC_dampu__mod__hydro != 0.)  &&  (AC_t__mod__cdata < AC_tdamp__mod__hydro)) {
if (AC_dampu__mod__hydro > 0.0) {
DF_UVEC = DF_UVEC - AC_fade_fact__mod__hydro*AC_dampu__mod__hydro*value(F_UVEC)
}
else {
DF_UVEC = DF_UVEC + AC_fade_fact__mod__hydro*AC_dampu__mod__hydro/AC_dt__mod__cdata*value(F_UVEC)
}
}
if (AC_dampuext__mod__hydro > 0.0  &&  AC_rdampext__mod__hydro != AC_impossible__mod__cparam) {
step_vector_return_value_113_119_128 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_rdampext__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = step_vector_return_value_113_119_128
DF_UX = DF_UX - AC_dampuext__mod__hydro*pdamp_119_128*value(F_UX)
DF_UY = DF_UY - AC_dampuext__mod__hydro*pdamp_119_128*value(F_UY)
DF_UZ = DF_UZ - AC_dampuext__mod__hydro*pdamp_119_128*value(F_UZ)
}
if (AC_dampuint__mod__hydro > 0.0  &&  AC_rdampint__mod__hydro != AC_impossible__mod__cparam) {
step_vector_return_value_114_119_128 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_rdampint__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = 1 - step_vector_return_value_114_119_128
DF_UX = DF_UX - AC_dampuint__mod__hydro*pdamp_119_128*value(F_UX)
DF_UY = DF_UY - AC_dampuint__mod__hydro*pdamp_119_128*value(F_UY)
DF_UZ = DF_UZ - AC_dampuint__mod__hydro*pdamp_119_128*value(F_UZ)
}
if (AC_lomega_int__mod__hydro) {
if (AC_lcylinder_in_a_box__mod__cdata) {
step_vector_return_value_115_119_128 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_rdampext__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = step_vector_return_value_115_119_128
}
else {
step_vector_return_value_116_119_128 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_rdampext__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = step_vector_return_value_116_119_128
}
fext__mod__hydro.x=-AC_dampuext__mod__hydro*pdamp_119_128*value(F_UX)
fext__mod__hydro.y=-AC_dampuext__mod__hydro*pdamp_119_128*value(F_UY)
fext__mod__hydro.z=-AC_dampuext__mod__hydro*pdamp_119_128*value(F_UZ)
DF_UVEC=DF_UVEC+fext__mod__hydro
if (AC_dampuint__mod__hydro > 0.0) {
if (AC_lcylinder_in_a_box__mod__cdata) {
step_vector_return_value_117_119_128 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_rdampint__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = 1 - step_vector_return_value_117_119_128
}
else {
step_vector_return_value_118_119_128 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_rdampint__mod__hydro)/(AC_wdamp__mod__hydro+AC_tini__mod__cparam)))
pdamp_119_128 = 1 - step_vector_return_value_118_119_128
}
fint__mod__hydro.x=-AC_dampuint__mod__hydro*pdamp_119_128*(value(F_UX)+AC_y__mod__cdata[AC_m__mod__cdata-1]*AC_omega_int__mod__hydro)
fint__mod__hydro.y=-AC_dampuint__mod__hydro*pdamp_119_128*(value(F_UY)-AC_x__mod__cdata[vertexIdx.x]*AC_omega_int__mod__hydro)
fint__mod__hydro.z=-AC_dampuint__mod__hydro*pdamp_119_128*(value(F_UZ))
DF_UVEC=DF_UVEC+fint__mod__hydro
}
}
if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_bs04_string__mod__cparam) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_bs04c_string__mod__cparam) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_bs04c1_string__mod__cparam) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_bs04m_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_hp09_string__mod__cparam) {
if (!AC_lcalc_uumeanxz__mod__hydro) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumxz__mod__hydro[vertexIdx.x][AC_n__mod__cdata-1][2-1]-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_sx_string__mod__cparam) {
if (AC_lcalc_uumeanx__mod__hydro) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumx__mod__hydro[vertexIdx.x][AC_iuy__mod__cdata-1]-AC_shearx__mod__hydro*AC_x__mod__cdata[vertexIdx.x])
}
else {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_shearx__mod__hydro*AC_x__mod__cdata[vertexIdx.x])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_solar_dc99_string__mod__cparam) {
if (AC_lcalc_uumeanxy__mod__hydro) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1])
}
else {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_string__mod__cparam) {
if (!AC_lcalc_uumean__mod__hydro) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][2-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_compression_string__mod__cparam) {
if (!AC_lcalc_uumean__mod__hydro) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][3-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_remove_vertical_shear_string__mod__cparam) {
DF_UX=value(F_UX)-AC_uumz__mod__hydro[AC_n__mod__cdata-1][1-1]
DF_UY=value(F_UY)-AC_uumz__mod__hydro[AC_n__mod__cdata-1][2-1]
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_x_string__mod__cparam) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*(value(F_UX)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_x_sinz_string__mod__cparam) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*(value(F_UX)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_z_string__mod__cparam) {
if (!AC_lcalc_uumean__mod__hydro) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(value(F_UY)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][2-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_z2_string__mod__cparam) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumxz__mod__hydro[vertexIdx.x][AC_n__mod__cdata-1][2-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_vertical_shear_linear_string__mod__cparam) {
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumxz__mod__hydro[vertexIdx.x][AC_n__mod__cdata-1][2-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_tachocline_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*value(F_UZ)
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_solar_simple_string__mod__cparam) {
if (AC_lspherical_coords__mod__cdata || AC_lcartesian_coords__mod__cdata) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1])
}
if (AC_ldiffrot_test__mod__hydro) {
DF_UX = 0.
DF_UY = 0.
if (AC_lspherical_coords__mod__cdata || AC_lcartesian_coords__mod__cdata) {
DF_UZ = AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1]
}
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_radial_uniform_shear_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_breeze_string__mod__cparam) {
if (!AC_lcalc_uumean__mod__hydro) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][3-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_slow_wind_string__mod__cparam) {
if (!AC_lcalc_uumean__mod__hydro) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(value(F_UZ)-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
else {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][3-1]-AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_radial_shear_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_radial_shear_damp_string__mod__cparam) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][1-1]-0.)
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][2-1]-0.)
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_damp_corona_string__mod__cparam) {
if (AC_lspherical_coords__mod__cdata) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][1-1]-0.)
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][2-1]-0.)
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-0.)
}
else if (AC_lcartesian_coords__mod__cdata) {
if (!AC_lcalc_uumeanz__mod__hydro) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(value(F_UX)-0.)
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(value(F_UY)-0.)
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(value(F_UZ)-0.)
}
else {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][1-1]-0.)
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][2-1]-0.)
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][3-1]-0.)
}
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_damp_horiz_vel_string__mod__cparam) {
if (AC_lcartesian_coords__mod__cdata) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*value(F_UX)
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*AC_prof_amp3__mod__hydro[AC_n__mod__cdata-1]*value(F_UY)
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_latitudinal_shear_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_damp_jets_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*AC_prof_amp4__mod__hydro[AC_m__mod__cdata-1]*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_uzjet__mod__hydro)
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_spokezlikeznssl_string__mod__cparam) {
local_omega_120_128=AC_profx_diffrot1__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_profy_diffrot1__mod__hydro[AC_m__mod__cdata-1]+AC_profx_diffrot2__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_profy_diffrot2__mod__hydro[AC_m__mod__cdata-1]+AC_profx_diffrot3__mod__hydro[vertexIdx.x-NGHOST_VAL]*AC_profy_diffrot3__mod__hydro[AC_m__mod__cdata-1]
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_prof_amp1__mod__hydro[vertexIdx.x-NGHOST_VAL]*local_omega_120_128*AC_sinth__mod__cdata[AC_m__mod__cdata-1])
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_uumz_profile_string__mod__cparam) {
DF_UX=DF_UX-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][1-1]-AC_uumz_prof__mod__hydro[AC_n__mod__cdata-AC_NGHOST__mod__cparam-1][1-1])
DF_UY=DF_UY-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][2-1]-AC_uumz_prof__mod__hydro[AC_n__mod__cdata-AC_NGHOST__mod__cparam-1][2-1])
if (!AC_limpose_only_horizontal_uumz__mod__hydro) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumz__mod__hydro[AC_n__mod__cdata-1][3-1]-AC_uumz_prof__mod__hydro[AC_n__mod__cdata-AC_NGHOST__mod__cparam-1][3-1])
}
}
else if(AC_string_enum_uuprof__mod__hydro == AC_string_enum_omega_profile_string__mod__cparam) {
DF_UZ=DF_UZ-AC_tau_diffrot1__mod__hydro*(AC_uumxy__mod__hydro[vertexIdx.x][AC_m__mod__cdata-1][3-1]-AC_omega_prof__mod__hydro[vertexIdx.x-NGHOST_VAL][AC_m__mod__cdata-AC_NGHOST__mod__cparam-1]*AC_x__mod__cdata[vertexIdx.x]*AC_sinth__mod__cdata[AC_m__mod__cdata-1])
}
else {
}
neg_velocity_ceiling_121_128 = -AC_velocity_ceiling__mod__hydro
if (AC_velocity_ceiling__mod__hydro>0.0) {
DF_UX = set_min_val(value(F_UX),neg_velocity_ceiling_121_128)
DF_UX = set_max_val(value(F_UX),AC_velocity_ceiling__mod__hydro)
DF_UY = set_min_val(value(F_UY),neg_velocity_ceiling_121_128)
DF_UY = set_max_val(value(F_UY),AC_velocity_ceiling__mod__hydro)
DF_UZ = set_min_val(value(F_UZ),neg_velocity_ceiling_121_128)
DF_UZ = set_max_val(value(F_UZ),AC_velocity_ceiling__mod__hydro)
}
if (AC_ladv_der_as_aux__mod__cdata) {
DF__ADV_DERVEC = ac_transformed_pencil_fpres + ac_transformed_pencil_fvisc
if (AC_lgrav__mod__cparam) {
DF__ADV_DERVEC = value(F__ADV_DERVEC) + ac_transformed_pencil_gg
}
}
if (AC_luu_sph_as_aux__mod__hydro && AC_lsphere_in_a_box__mod__cdata) {
DF_UU_SPHR = ac_transformed_pencil_uu.x*ac_transformed_pencil_evr.x+ac_transformed_pencil_uu.y*ac_transformed_pencil_evr.y+ac_transformed_pencil_uu.z*ac_transformed_pencil_evr.z
DF_UU_SPHT = ac_transformed_pencil_uu.x*ac_transformed_pencil_evth.x+ac_transformed_pencil_uu.y*ac_transformed_pencil_evth.y+ac_transformed_pencil_uu.z*ac_transformed_pencil_evth.z
DF_UU_SPHP = ac_transformed_pencil_uu.x*ac_transformed_pencil_phix+ac_transformed_pencil_uu.y*ac_transformed_pencil_phiy
}
if (AC_tau_damp_ruxm__mod__hydro!=0.) {
DF_UX=DF_UX-AC_ruxm__mod__hydro*ac_transformed_pencil_rho1*AC_tau_damp_ruxm1__mod__hydro
}
if (AC_tau_damp_ruym__mod__hydro!=0.) {
DF_UY=DF_UY-AC_ruym__mod__hydro*ac_transformed_pencil_rho1*AC_tau_damp_ruym1__mod__hydro
}
if (AC_tau_damp_ruzm__mod__hydro!=0.) {
DF_UZ=DF_UZ-AC_ruzm__mod__hydro*ac_transformed_pencil_rho1*AC_tau_damp_ruzm1__mod__hydro
}
ju_126_128=1+AC_iuu__mod__cdata-1
if(AC_string_enum_borderuu__mod__hydro.x == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderuu__mod__hydro.x == AC_string_enum_0_string__mod__cparam) {
f_target_126_128.x=0.
}
else if(AC_string_enum_borderuu__mod__hydro.x == AC_string_enum_constant_string__mod__cparam) {
f_target_126_128.x = AC_uu_const__mod__hydro.x
}
else if(AC_string_enum_borderuu__mod__hydro.x == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderuu__mod__hydro.x == AC_string_enum_nothing_string__mod__cparam) {
}
ju_126_128=2+AC_iuu__mod__cdata-1
if(AC_string_enum_borderuu__mod__hydro.y == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderuu__mod__hydro.y == AC_string_enum_0_string__mod__cparam) {
f_target_126_128.y=0.
}
else if(AC_string_enum_borderuu__mod__hydro.y == AC_string_enum_constant_string__mod__cparam) {
f_target_126_128.y = AC_uu_const__mod__hydro.y
}
else if(AC_string_enum_borderuu__mod__hydro.y == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderuu__mod__hydro.y == AC_string_enum_nothing_string__mod__cparam) {
}
ju_126_128=3+AC_iuu__mod__cdata-1
if(AC_string_enum_borderuu__mod__hydro.z == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderuu__mod__hydro.z == AC_string_enum_0_string__mod__cparam) {
f_target_126_128.z=0.
}
else if(AC_string_enum_borderuu__mod__hydro.z == AC_string_enum_constant_string__mod__cparam) {
f_target_126_128.z = AC_uu_const__mod__hydro.z
}
else if(AC_string_enum_borderuu__mod__hydro.z == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderuu__mod__hydro.z == AC_string_enum_nothing_string__mod__cparam) {
}
if (AC_lschur_3d3d1d__mod__density) {
}
else {
if (AC_lcontinuity_gas__mod__density) {
if (! AC_lweno_transport__mod__cdata  &&  ! AC_lffree__mod__density  &&  ! AC_lreduced_sound_speed__mod__density  &&   AC_string_enum_ieos_profile__mod__density==AC_string_enum_nothing_string__mod__cparam  &&  ! AC_lfargo_advection__mod__cdata) {
if (AC_ldensity_nolog__mod__cdata) {
if (AC_lconservative__mod__hydro) {
density_rhs_140=-ac_transformed_pencil_divss
}
else {
density_rhs_140=-ac_transformed_pencil_rho*ac_transformed_pencil_divu
if (AC_ladvection_density__mod__density) {
density_rhs_140 = density_rhs_140 - ac_transformed_pencil_ugrho
}
if (AC_lrelativistic_eos__mod__density) {
density_rhs_140=AC_fourthird__mod__cparam*density_rhs_140
}
}
}
else {
density_rhs_140= - ac_transformed_pencil_divu
if (AC_ladvection_density__mod__density) {
density_rhs_140 = density_rhs_140 - ac_transformed_pencil_uglnrho
}
if (AC_lrelativistic_eos__mod__density && !AC_lconservative__mod__hydro) {
if (AC_lhydro__mod__cparam) {
tmpv_140.x=ac_transformed_pencil_uu.x*density_rhs_140
tmpv_140.y=ac_transformed_pencil_uu.y*density_rhs_140
tmpv_140.z=ac_transformed_pencil_uu.z*density_rhs_140
DF_UVEC=DF_UVEC-AC_onethird__mod__cparam*tmpv_140
}
density_rhs_140=AC_fourthird__mod__cparam*density_rhs_140
}
}
}
else {
density_rhs_140=0.
}
if (AC_lweno_transport__mod__cdata) {
density_rhs_140= density_rhs_140 - ac_transformed_pencil_transprho
}
if (AC_string_enum_ieos_profile__mod__density==AC_string_enum_surface_z_string__mod__cparam) {
if (AC_ldensity_nolog__mod__cdata) {
density_rhs_140= density_rhs_140 - AC_profz_eos__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
if (AC_ldensity_profile_masscons__mod__density) {
density_rhs_140 = density_rhs_140-AC_dprofz_eos__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_rho*ac_transformed_pencil_uu.z
}
}
else {
density_rhs_140= density_rhs_140 - AC_profz_eos__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
if (AC_ldensity_profile_masscons__mod__density) {
density_rhs_140 = density_rhs_140 -AC_dprofz_eos__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z
}
}
}
if (AC_lffree__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
density_rhs_140= density_rhs_140 - AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
if (AC_ldensity_profile_masscons__mod__density) {
density_rhs_140=density_rhs_140 - ac_transformed_pencil_rho*( AC_dprofx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]   *ac_transformed_pencil_uu.x  +AC_dprofy_ffree__mod__density[AC_m__mod__cdata-1]*ac_transformed_pencil_uu.y  +AC_dprofz_ffree__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z)
}
}
else {
density_rhs_140= density_rhs_140 - AC_profx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]*(AC_profy_ffree__mod__density[AC_m__mod__cdata-1]*AC_profz_ffree__mod__density[AC_n__mod__cdata-1])*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
if (AC_ldensity_profile_masscons__mod__density) {
density_rhs_140=density_rhs_140-AC_dprofx_ffree__mod__density[vertexIdx.x-NGHOST_VAL]   *ac_transformed_pencil_uu.x  -AC_dprofy_ffree__mod__density[AC_m__mod__cdata-1]*ac_transformed_pencil_uu.y  -AC_dprofz_ffree__mod__density[AC_n__mod__cdata-1]*ac_transformed_pencil_uu.z
}
}
}
if (AC_lreduced_sound_speed__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
density_rhs_140 = density_rhs_140 - AC_reduce_cs2_profx__mod__density[vertexIdx.x-NGHOST_VAL]*AC_reduce_cs2_profz__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_ugrho + ac_transformed_pencil_rho*ac_transformed_pencil_divu)
}
else {
density_rhs_140 = density_rhs_140 - AC_reduce_cs2_profx__mod__density[vertexIdx.x-NGHOST_VAL]*AC_reduce_cs2_profz__mod__density[AC_n__mod__cdata-1]*(ac_transformed_pencil_uglnrho + ac_transformed_pencil_divu)
}
}
if (AC_lfargo_advection__mod__cdata) {
if (AC_ldensity_nolog__mod__cdata) {
density_rhs_140 = density_rhs_140 - ac_transformed_pencil_uuadvec_grho   - ac_transformed_pencil_rho*ac_transformed_pencil_divu
}
else {
density_rhs_140 = density_rhs_140 - ac_transformed_pencil_uuadvec_glnrho - ac_transformed_pencil_divu
}
}
DF_LNRHO = DF_LNRHO + density_rhs_140
}
if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_nothing_string__mod__cparam) {
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_exponential_string__mod__cparam) {
dlnrhodt_133_140=AC_mass_source_mdot__mod__density
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_bump_string__mod__cparam) {
dlnrhodt_133_140=(AC_mass_source_mdot__mod__density/AC_fnorm__mod__density)*exp(-0.5*(ac_transformed_pencil_r_mn/AC_mass_source_sigma__mod__density)*(ac_transformed_pencil_r_mn/AC_mass_source_sigma__mod__density))
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_bump2_string__mod__cparam) {
dlnrhodt_133_140=AC_fprofile_z__mod__density[AC_n__mod__cdata-AC_n1__mod__cparam+1-1]
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_bumpr_string__mod__cparam) {
radius2_133_140=(AC_x__mod__cdata[vertexIdx.x]-AC_xblob__mod__density[1-1])*(AC_x__mod__cdata[vertexIdx.x]-AC_xblob__mod__density[1-1])+(AC_y__mod__cdata[AC_m__mod__cdata-1]-AC_yblob__mod__density[1-1])*(AC_y__mod__cdata[AC_m__mod__cdata-1]-AC_yblob__mod__density[1-1])+(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_zblob__mod__density[1-1])*(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_zblob__mod__density[1-1])
fprofile_133_140=(AC_mass_source_mdot__mod__density/AC_fnorm__mod__density)*exp(-0.5*radius2_133_140/AC_mass_source_sigma__mod__density*AC_mass_source_sigma__mod__density)
if (AC_lmass_source_random__mod__density) {
fran_133_140[1-1] = rand_uniform()
fran_133_140[2-1] = rand_uniform()
tmp_133_140=sqrt(-2*log(fran_133_140[1-1]))*sin(2*AC_pi__mod__cparam*fran_133_140[2-1])
dlnrhodt_133_140=fprofile_133_140*cos(AC_mass_source_omega__mod__density*AC_t__mod__cdata)*tmp_133_140
}
else {
dlnrhodt_133_140=fprofile_133_140*cos(AC_mass_source_omega__mod__density*AC_t__mod__cdata)
}
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_bumpx_string__mod__cparam || AC_string_enum_mass_source_profile__mod__density == AC_string_enum_sphzstepzdown_string__mod__cparam) {
dlnrhodt_133_140=AC_fprofile_x__mod__density[vertexIdx.x-NGHOST_VAL]
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_const_string__mod__cparam) {
dlnrhodt_133_140=-AC_mass_source_tau1__mod__density*(value(F_LNRHO)-AC_lnrho0__mod__equationofstate)
}
else if(AC_string_enum_mass_source_profile__mod__density == AC_string_enum_cylindric_string__mod__cparam) {
step_vector_return_value_131_133_140 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_int__mod__cdata)/(AC_wdamp__mod__density+AC_tini__mod__cparam)))
pdamp_133_140=1.-step_vector_return_value_131_133_140
dlnrhodt_133_140=-AC_damplnrho_int__mod__density*pdamp_133_140*(value(F_LNRHO)-AC_lnrho_int__mod__density)
step_vector_return_value_132_133_140 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_ext__mod__cdata)/(AC_wdamp__mod__density+AC_tini__mod__cparam)))
pdamp_133_140=step_vector_return_value_132_133_140
dlnrhodt_133_140=dlnrhodt_133_140-AC_damplnrho_ext__mod__density*pdamp_133_140*(value(F_LNRHO)-AC_lnrho_ext__mod__density)
}
else {
}
if (AC_ldensity_nolog__mod__cdata) {
DF_RHO=DF_RHO+ac_transformed_pencil_rho*dlnrhodt_133_140
}
else {
DF_LNRHO=DF_LNRHO+dlnrhodt_133_140
}
if (AC_lentropy__mod__cparam) {
DF_SS=DF_SS+(AC_gamma1__mod__density-1.0)*dlnrhodt_133_140
}
diffus_diffrho__mod__density=0.
diffus_diffrho3__mod__density=0.
fdiff_140=0.0
ldt_up_140 = AC_lfirst__mod__cdata && AC_ldt__mod__cdata
if (AC_ldiff_normal__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*ac_transformed_pencil_del2rho
}
else {
if (AC_ldiffusion_nolog__mod__density) {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho
}
else {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*(ac_transformed_pencil_del2lnrho+ac_transformed_pencil_glnrho2)
}
}
if (ldt_up_140) {
diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho__mod__density
}
}
if (AC_ldiff_cspeed__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*ac_transformed_pencil_del2rho
}
else {
if (AC_ldiffusion_nolog__mod__density) {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*ac_transformed_pencil_rho1*ac_transformed_pencil_del2rho
}
else {
fdiff_140 = fdiff_140 + AC_diffrho__mod__density*pow(ac_transformed_pencil_tt,AC_diff_cspeed__mod__density)*(ac_transformed_pencil_del2lnrho+ac_transformed_pencil_glnrho2)
}
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho__mod__density
}
}
if (AC_ldiff_shock__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
tmp_140 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_grho)
fdiff_140 = fdiff_140 + AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * ac_transformed_pencil_del2rho + tmp_140)
}
else {
if (AC_ldiffusion_nolog__mod__density) {
tmp_140 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_grho)
fdiff_140 = fdiff_140 + ac_transformed_pencil_rho1 * AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * ac_transformed_pencil_del2rho + tmp_140)
}
else {
tmp_140 = dot(ac_transformed_pencil_gshock,ac_transformed_pencil_glnrho)
fdiff_140 = fdiff_140 + AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock * (ac_transformed_pencil_del2lnrho + ac_transformed_pencil_glnrho2) + tmp_140)
if (AC_lanti_shockdiffusion__mod__density) {
fdiff_140 = fdiff_140 - AC_diffrho_shock__mod__density * (ac_transformed_pencil_shock*(AC_del2lnrho_glnrho2_init_z__mod__density[AC_n__mod__cdata-1] +  2*(ac_transformed_pencil_glnrho.z-AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1])*AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1]) +  ac_transformed_pencil_gshock.z*AC_dlnrhodz_init_z__mod__density[AC_n__mod__cdata-1] )
}
}
}
if (ldt_up_140) {
diffus_diffrho__mod__density=diffus_diffrho__mod__density+AC_diffrho_shock__mod__density*ac_transformed_pencil_shock
}
}
if (AC_ldensity_slope_limited__mod__density && AC_llast__mod__cdata) {
if (AC_ldensity_nolog__mod__cdata) {
not_implemented("calc_slope_diff_flux")
fdiff_140=fdiff_140+tmp_140
}
else {
not_implemented("calc_slope_diff_flux")
fdiff_140=fdiff_140+tmp_140*ac_transformed_pencil_rho1
}
}
if (AC_lmassdiff_fix__mod__density && !AC_lconservative__mod__hydro) {
if (AC_ldensity_nolog__mod__cdata) {
tmp_140 = fdiff_140*ac_transformed_pencil_rho1
}
else {
tmp_140 = fdiff_140
}
if (AC_lhydro__mod__cparam && (!AC_lhydro_potential__mod__cparam)) {
DF_UX = DF_UX - ac_transformed_pencil_uu.x * tmp_140
DF_UY = DF_UY - ac_transformed_pencil_uu.y * tmp_140
DF_UZ = DF_UZ - ac_transformed_pencil_uu.z * tmp_140
}
if (AC_lentropy__mod__cparam && (!AC_pretend_lntt__mod__cdata)) {
DF_SS = DF_SS - ac_transformed_pencil_cv*tmp_140
}
else if (AC_lentropy__mod__cparam && AC_pretend_lntt__mod__cdata) {
DF_LNTT = DF_LNTT - tmp_140
}
else if (AC_ltemperature__mod__cparam && (! AC_ltemperature_nolog__mod__cdata)) {
DF_LNTT = DF_LNTT - tmp_140
}
else if (AC_ltemperature__mod__cparam && AC_ltemperature_nolog__mod__cdata) {
DF_TT = DF_TT - tmp_140*ac_transformed_pencil_tt
}
else if (AC_lthermal_energy__mod__cparam) {
DF_ETH = DF_ETH + 0.5 * fdiff_140 * ac_transformed_pencil_u2
}
}
if (AC_ldiff_hyper3__mod__density || AC_ldiff_hyper3_strict__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_del6rho
}
else {
if (AC_ldiffusion_nolog__mod__density) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_rho1*ac_transformed_pencil_del6rho
}
}
if (ldt_up_140) {
diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density
}
}
if (AC_ldiff_hyper3_polar__mod__density) {
if (AC_ldensity_nolog__mod__cdata) {
tmp_140 = der6x_ignore_spacing(Field(AC_irho__mod__cdata))
}
else {
tmp_140 = der6x_ignore_spacing(Field(AC_ilnrho__mod__cdata))
}
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*AC_pi4_1__mod__cparam*tmp_140*dline_1__mod__cdata.x*dline_1__mod__cdata.x
if (AC_ldensity_nolog__mod__cdata) {
tmp_140 = der6y_ignore_spacing(Field(AC_irho__mod__cdata))
}
else {
tmp_140 = der6y_ignore_spacing(Field(AC_ilnrho__mod__cdata))
}
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*AC_pi4_1__mod__cparam*tmp_140*dline_1__mod__cdata.y*dline_1__mod__cdata.y
if (AC_ldensity_nolog__mod__cdata) {
tmp_140 = der6z_ignore_spacing(Field(AC_irho__mod__cdata))
}
else {
tmp_140 = der6z_ignore_spacing(Field(AC_ilnrho__mod__cdata))
}
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*AC_pi4_1__mod__cparam*tmp_140*dline_1__mod__cdata.z*dline_1__mod__cdata.z
if (ldt_up_140) {
diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density*AC_pi4_1__mod__cparam*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata
}
}
if (AC_ldiff_hyper3_mesh__mod__density) {
tmp_140 = der6x_ignore_spacing(Field(AC_ilnrho__mod__cdata))
if (AC_ldynamical_diffusion__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density * tmp_140 * dline_1__mod__cdata.x
}
else {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density*AC_pi5_1__mod__cparam/60.*tmp_140*dline_1__mod__cdata.x
}
tmp_140 = der6y_ignore_spacing(Field(AC_ilnrho__mod__cdata))
if (AC_ldynamical_diffusion__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density * tmp_140 * dline_1__mod__cdata.y
}
else {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density*AC_pi5_1__mod__cparam/60.*tmp_140*dline_1__mod__cdata.y
}
tmp_140 = der6z_ignore_spacing(Field(AC_ilnrho__mod__cdata))
if (AC_ldynamical_diffusion__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density * tmp_140 * dline_1__mod__cdata.z
}
else {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3_mesh__mod__density*AC_pi5_1__mod__cparam/60.*tmp_140*dline_1__mod__cdata.z
}
if (ldt_up_140) {
if (AC_ldynamical_diffusion__mod__cdata) {
diffus_diffrho3__mod__density = diffus_diffrho3__mod__density + AC_diffrho_hyper3_mesh__mod__density
advec_hypermesh_rho_140=0.
}
else {
advec_hypermesh_rho_140=AC_diffrho_hyper3_mesh__mod__density*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_rho_140*advec_hypermesh_rho_140
}
}
if (AC_ldiff_hyper3_aniso__mod__density) {
tmp_140  = del6fj(Field(AC_ilnrho__mod__cdata), AC_diffrho_hyper3_aniso__mod__density)
fdiff_140 = fdiff_140 + tmp_140
if (AC_lsubtract_init_stratification__mod__density) {
tmp_140  = del6fj(Field(AC_iglobal_lnrho0__mod__cdata), AC_diffrho_hyper3_aniso__mod__density)
fdiff_140 = fdiff_140 - tmp_140
}
if (ldt_up_140) {
diffus_diffrho3__mod__density=diffus_diffrho3__mod__density +  (AC_diffrho_hyper3_aniso__mod__density.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x +  AC_diffrho_hyper3_aniso__mod__density.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y +  AC_diffrho_hyper3_aniso__mod__density.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)/dxyz_6__mod__cdata
}
}
if (AC_ldiff_hyper3lnrho__mod__density  ||  AC_ldiff_hyper3lnrho_strict__mod__density) {
if (! AC_ldensity_nolog__mod__cdata) {
fdiff_140 = fdiff_140 + AC_diffrho_hyper3__mod__density*ac_transformed_pencil_del6lnrho
}
if (ldt_up_140) {
diffus_diffrho3__mod__density=diffus_diffrho3__mod__density+AC_diffrho_hyper3__mod__density
}
}
if (AC_ldensity_nolog__mod__cdata) {
DF_RHO   = DF_RHO   + fdiff_140
}
else {
DF_LNRHO = DF_LNRHO + fdiff_140
}
if (ldt_up_140) {
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
if(AC_string_enum_borderlnrho__mod__density == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderlnrho__mod__density == AC_string_enum_0_string__mod__cparam) {
if (AC_ldensity_nolog__mod__cdata) {
f_target_139_140=0.
}
else {
f_target_139_140=1.
}
}
else if(AC_string_enum_borderlnrho__mod__density == AC_string_enum_constant_string__mod__cparam) {
if (AC_ldensity_nolog__mod__cdata) {
f_target_139_140=AC_rho_const__mod__density
}
else {
f_target_139_140=AC_lnrho_const__mod__density
}
}
else if(AC_string_enum_borderlnrho__mod__density == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderlnrho__mod__density == AC_string_enum_nothing_string__mod__cparam) {
}
}
if (AC_lhydro__mod__cparam && AC_lpressuregradient_gas__mod__hydro && !AC_lconservative__mod__hydro   && (!AC_lhydro_potential__mod__cparam)) {
DF_UVEC=DF_UVEC+ac_transformed_pencil_fpres
if (AC_beta_glnrho_scaled__mod__density.x != 0.  ||  AC_beta_glnrho_scaled__mod__density.y != 0.  ||  AC_beta_glnrho_scaled__mod__density.z != 0.) {
DF_UX = DF_UX - ac_transformed_pencil_cs2*AC_beta_glnrho_scaled__mod__density.x
DF_UY = DF_UY - ac_transformed_pencil_cs2*AC_beta_glnrho_scaled__mod__density.y
DF_UZ = DF_UZ - ac_transformed_pencil_cs2*AC_beta_glnrho_scaled__mod__density.z
}
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata && AC_leos__mod__cparam && AC_ldensity__mod__cparam && AC_lhydro__mod__cparam) {
advec_cs2__mod__cdata = ac_transformed_pencil_advec_cs2
}
dadt_190.x = 0.
dadt_190.y = 0.
dadt_190.z = 0.
fmax__mod__magnetic=1./AC_impossible__mod__cparam
damax__mod__magnetic=1./AC_impossible__mod__cparam
ssmax__mod__magnetic=1./AC_impossible__mod__cparam
if (AC_b_ext__mod__magnetic.x != 0.0  ||  AC_b_ext__mod__magnetic.y != 0.0  ||  AC_b_ext__mod__magnetic.z != 0.0) {
if (AC_omega_bz_ext__mod__magnetic != 0.0) {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
c_152_190 = cos(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
s_152_190 = sin(AC_omega_bz_ext__mod__magnetic * AC_t__mod__cdata)
b_ext_190.x = AC_b_ext__mod__magnetic.x * c_152_190 - AC_b_ext__mod__magnetic.y * s_152_190
b_ext_190.y = AC_b_ext__mod__magnetic.x * s_152_190 + AC_b_ext__mod__magnetic.y * c_152_190
b_ext_190.z = AC_b_ext__mod__magnetic.z
}
else {
}
}
else if (AC_lbext_moving_layer__mod__magnetic)  {
zposbot_152_190=AC_zbot_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
zpostop_152_190=AC_ztop_moving_layer__mod__magnetic + AC_t__mod__cdata*AC_speed_moving_layer__mod__magnetic
step_scalar_return_value_67_152_190 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_152_190)/AC_edge_moving_layer__mod__magnetic))
step_scalar_return_value_68_152_190 = 0.5*(1+tanh((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_152_190)/AC_edge_moving_layer__mod__magnetic))
zprof_152_190 = step_scalar_return_value_67_152_190-step_scalar_return_value_68_152_190
b_ext_190.x = AC_b_ext__mod__magnetic.x*zprof_152_190
b_ext_190.y = AC_b_ext__mod__magnetic.y*zprof_152_190
b_ext_190.z=AC_b_ext__mod__magnetic.z
if (false) {
arg_69_152_190 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zposbot_152_190)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_69_152_190)>=8.)  {
der_step_return_value_69_152_190 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_69_152_190))
}
else {
der_step_return_value_69_152_190 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_69_152_190)*cosh(arg_69_152_190))
}
arg_70_152_190 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-zpostop_152_190)/(AC_edge_moving_layer__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_70_152_190)>=8.)  {
der_step_return_value_70_152_190 = 2./AC_edge_moving_layer__mod__magnetic*exp(-2.*abs(arg_70_152_190))
}
else {
der_step_return_value_70_152_190 = 0.5/(AC_edge_moving_layer__mod__magnetic*cosh(arg_70_152_190)*cosh(arg_70_152_190))
}
zder_152_190 = der_step_return_value_69_152_190-der_step_return_value_70_152_190
}
}
else {
if (AC_lcartesian_coords__mod__cdata  ||  AC_lbext_curvilinear__mod__magnetic) {
b_ext_190 = AC_b_ext__mod__magnetic
}
else if (AC_lcylindrical_coords__mod__cdata) {
b_ext_190.x =  AC_b_ext__mod__magnetic.x * cos(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * sin(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_190.y = -AC_b_ext__mod__magnetic.x * sin(AC_y__mod__cdata[AC_m__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * cos(AC_y__mod__cdata[AC_m__mod__cdata-1])
b_ext_190.z =  AC_b_ext__mod__magnetic.z
}
else if (AC_lspherical_coords__mod__cdata) {
b_ext_190.x =  AC_b_ext__mod__magnetic.x * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_sinth__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.z * AC_costh__mod__cdata[AC_m__mod__cdata-1]
b_ext_190.y =  AC_b_ext__mod__magnetic.x * AC_costh__mod__cdata[AC_m__mod__cdata-1] * cos(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y * AC_costh__mod__cdata[AC_m__mod__cdata-1] * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) - AC_b_ext__mod__magnetic.z * AC_sinth__mod__cdata[AC_m__mod__cdata-1]
b_ext_190.z = -AC_b_ext__mod__magnetic.x            * sin(AC_z__mod__cdata[AC_n__mod__cdata-1]) + AC_b_ext__mod__magnetic.y            * cos(AC_z__mod__cdata[AC_n__mod__cdata-1])
}
}
}
else {
b_ext_190.x = 0.
b_ext_190.y = 0.
b_ext_190.z = 0.
}
if (AC_t_bext__mod__magnetic > 0.0  &&  AC_t__mod__cdata < AC_t_bext__mod__magnetic) {
if (AC_t__mod__cdata <= AC_t0_bext__mod__magnetic) {
b_ext_190 = AC_b0_ext__mod__magnetic
}
else {
b_ext_190 = AC_b0_ext__mod__magnetic + 0.5*(1.-cos(AC_pi__mod__cparam*(AC_t__mod__cdata-AC_t0_bext__mod__magnetic)/(AC_t_bext__mod__magnetic-AC_t0_bext__mod__magnetic)))*(b_ext_190-AC_b0_ext__mod__magnetic)
}
}
if (AC_lhydro__mod__cparam) {
if (!AC_lkinematic__mod__magnetic) {
if (AC_llorentzforce__mod__magnetic) {
if (AC_lboris_correction__mod__magnetic) {
DF_UX=DF_UX+ac_transformed_pencil_gamma_a2*ac_transformed_pencil_jxbr.x+ (ac_transformed_pencil_ugu.x+ac_transformed_pencil_rho1gpp.x-ac_transformed_pencil_gg.x)*(1-ac_transformed_pencil_gamma_a2)- AC_mu01__mod__cdata*(ac_transformed_pencil_gamma_a2*ac_transformed_pencil_gamma_a2*ac_transformed_pencil_rho1/ac_transformed_pencil_clight2)*  (ac_transformed_pencil_bb.x*ac_transformed_pencil_bb.x*(ac_transformed_pencil_ugu.x+ac_transformed_pencil_rho1gpp.x-ac_transformed_pencil_gg.x)+ ac_transformed_pencil_bb.x*ac_transformed_pencil_bb.y*(ac_transformed_pencil_ugu.y+ac_transformed_pencil_rho1gpp.y-ac_transformed_pencil_gg.y)+ ac_transformed_pencil_bb.x*ac_transformed_pencil_bb.z*(ac_transformed_pencil_ugu.z+ac_transformed_pencil_rho1gpp.z-ac_transformed_pencil_gg.z))
DF_UY=DF_UY+ac_transformed_pencil_gamma_a2*ac_transformed_pencil_jxbr.y+ (ac_transformed_pencil_ugu.y+ac_transformed_pencil_rho1gpp.y-ac_transformed_pencil_gg.y)*(1-ac_transformed_pencil_gamma_a2)- AC_mu01__mod__cdata*(ac_transformed_pencil_gamma_a2*ac_transformed_pencil_gamma_a2*ac_transformed_pencil_rho1/ac_transformed_pencil_clight2)*  (ac_transformed_pencil_bb.y*ac_transformed_pencil_bb.y*(ac_transformed_pencil_ugu.y+ac_transformed_pencil_rho1gpp.y-ac_transformed_pencil_gg.y)+ ac_transformed_pencil_bb.y*ac_transformed_pencil_bb.x*(ac_transformed_pencil_ugu.x+ac_transformed_pencil_rho1gpp.x-ac_transformed_pencil_gg.x)+ ac_transformed_pencil_bb.y*ac_transformed_pencil_bb.z*(ac_transformed_pencil_ugu.z+ac_transformed_pencil_rho1gpp.z-ac_transformed_pencil_gg.z))
DF_UZ=DF_UZ+ac_transformed_pencil_gamma_a2*ac_transformed_pencil_jxbr.z+ (ac_transformed_pencil_ugu.z+ac_transformed_pencil_rho1gpp.z-ac_transformed_pencil_gg.z)*(1-ac_transformed_pencil_gamma_a2)- AC_mu01__mod__cdata*(ac_transformed_pencil_gamma_a2*ac_transformed_pencil_gamma_a2*ac_transformed_pencil_rho1/ac_transformed_pencil_clight2)*  (ac_transformed_pencil_bb.z*ac_transformed_pencil_bb.z*(ac_transformed_pencil_ugu.z+ac_transformed_pencil_rho1gpp.z-ac_transformed_pencil_gg.z)+ ac_transformed_pencil_bb.z*ac_transformed_pencil_bb.x*(ac_transformed_pencil_ugu.x+ac_transformed_pencil_rho1gpp.x-ac_transformed_pencil_gg.x)+ ac_transformed_pencil_bb.z*ac_transformed_pencil_bb.y*(ac_transformed_pencil_ugu.y+ac_transformed_pencil_rho1gpp.y-ac_transformed_pencil_gg.y))
}
else if (AC_llorentz_rhoref__mod__magnetic) {
DF_UVEC=DF_UVEC+ac_transformed_pencil_jxb*AC_rhoref1__mod__magnetic
}
else {
if (AC_lrelativistic_eos__mod__density) {
tmp1_190 = dot(ac_transformed_pencil_uu,ac_transformed_pencil_jxbr)
DF_LNRHO=DF_LNRHO+tmp1_190
DF_UVEC=DF_UVEC+0.75*ac_transformed_pencil_jxbr
}
else {
if (AC_iphiuu__mod__cdata==0) {
if (AC_lconservative__mod__hydro) {
DF_UVEC=DF_UVEC+ac_transformed_pencil_jxb
}
else {
if (AC_lignore_1rho_in_lorentz__mod__magnetic) {
DF_UVEC=DF_UVEC+ac_transformed_pencil_jxb
}
else {
DF_UVEC=DF_UVEC+ac_transformed_pencil_jxbr
}
}
}
else {
DF_PHIUU=DF_PHIUU-0.5*(ac_transformed_pencil_b2-AC_b_ext2__mod__magnetic)
}
}
}
}
}
}
if (AC_iex__mod__cdata==0 || AC_loverride_ee__mod__magnetic) {
fres__mod__magnetic.x = 0.
fres__mod__magnetic.y = 0.
fres__mod__magnetic.z = 0.
eta_total__mod__magnetic=0.
diffus_eta2__mod__magnetic=0.
diffus_eta3__mod__magnetic=0.
if (AC_lresi_eta_const__mod__magnetic) {
if (! AC_limplicit_resistivity__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic = fres__mod__magnetic - AC_eta__mod__magnetic * AC_mu0__mod__cdata * ac_transformed_pencil_jj
}
else {
fres__mod__magnetic = fres__mod__magnetic + AC_eta__mod__magnetic * ac_transformed_pencil_del2a
}
if (AC_ladd_efield__mod__magnetic) {
tanhx2_190 = tanh( AC_x__mod__cdata[vertexIdx.x] )*tanh( AC_x__mod__cdata[vertexIdx.x] )
del2aa_ini_190 = AC_ampl_efield__mod__magnetic*(-2 + 8*tanhx2_190 - 6*tanhx2_190*tanhx2_190 )
fres__mod__magnetic.z = fres__mod__magnetic.z - AC_eta__mod__magnetic*AC_mu0__mod__cdata*del2aa_ini_190
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta__mod__magnetic
}
}
if (AC_lresi_eta_tdep__mod__magnetic) {
if (AC_lresi_eta_ztdep__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic = fres__mod__magnetic                 -AC_eta_tdep__mod__magnetic* AC_feta_ztdep__mod__magnetic[AC_n__mod__cdata-1]    *AC_mu0__mod__cdata*ac_transformed_pencil_jj
}
else {
fres__mod__magnetic = fres__mod__magnetic+AC_eta_tdep__mod__magnetic*ac_transformed_pencil_del2a-AC_eta_tdep__mod__magnetic*(AC_feta_ztdep__mod__magnetic[AC_n__mod__cdata-1]-1.)*AC_mu0__mod__cdata*ac_transformed_pencil_jj
}
}
else {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic = fres__mod__magnetic - AC_eta_tdep__mod__magnetic * AC_mu0__mod__cdata * ac_transformed_pencil_jj
}
else {
fres__mod__magnetic = fres__mod__magnetic + AC_eta_tdep__mod__magnetic * ac_transformed_pencil_del2a
}
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta_tdep__mod__magnetic
}
if (AC_lresi_zdep__mod__magnetic) {
if (! AC_limplicit_resistivity__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic = fres__mod__magnetic - AC_eta_z__mod__magnetic[AC_n__mod__cdata-1] * AC_mu0__mod__cdata * ac_transformed_pencil_jj
}
else {
fres__mod__magnetic.x = fres__mod__magnetic.x + AC_eta_z__mod__magnetic[AC_n__mod__cdata-1] * ac_transformed_pencil_del2a.x
fres__mod__magnetic.y = fres__mod__magnetic.y + AC_eta_z__mod__magnetic[AC_n__mod__cdata-1] * ac_transformed_pencil_del2a.y
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_eta_z__mod__magnetic[AC_n__mod__cdata-1] * ac_transformed_pencil_del2a.z
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_geta_z__mod__magnetic[AC_n__mod__cdata-1] * ac_transformed_pencil_diva
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta_z__mod__magnetic[AC_n__mod__cdata-1]
}
else {
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_geta_z__mod__magnetic[AC_n__mod__cdata-1] * ac_transformed_pencil_diva
if (AC_lfirst__mod__cdata  &&  AC_ldt__mod__cdata) {
maxadvec__mod__cdata = maxadvec__mod__cdata + abs(AC_geta_z__mod__magnetic[AC_n__mod__cdata-1]) * AC_dz_1__mod__cdata[AC_n__mod__cdata-1]
}
}
}
if (AC_lresi_sqrtrhoeta_const__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1) * (ac_transformed_pencil_del2a.x-0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glnrho.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1) * (ac_transformed_pencil_del2a.y-0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glnrho.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1) * (ac_transformed_pencil_del2a.z-0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glnrho.z)
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta__mod__magnetic*sqrt(ac_transformed_pencil_rho1)
}
if (AC_lresi_eta_aniso__mod__magnetic) {
cosalp_190=cos(AC_alp_aniso__mod__magnetic*AC_dtor__mod__cparam)
sinalp_190=sin(AC_alp_aniso__mod__magnetic*AC_dtor__mod__cparam)
if (AC_eta1_aniso_r__mod__magnetic==0.) {
prof_190=AC_eta1_aniso__mod__magnetic
}
else {
step_vector_return_value_153_190 = 0.5*(1+tanh((AC_x__mod__cdata[vertexIdx.x]-AC_eta1_aniso_r__mod__magnetic)/(AC_eta1_aniso_d__mod__magnetic+AC_tini__mod__cparam)))
prof_190=AC_eta1_aniso__mod__magnetic*(1.-step_vector_return_value_153_190)
}
if (AC_lquench_eta_aniso__mod__magnetic) {
prof_190=prof_190/(1.+AC_quench_aniso__mod__magnetic*AC_arms__mod__magnetic)
}
fres__mod__magnetic.x=fres__mod__magnetic.x-prof_190*cosalp_190*(cosalp_190*ac_transformed_pencil_jj.x+sinalp_190*ac_transformed_pencil_jj.y)
fres__mod__magnetic.y=fres__mod__magnetic.y-prof_190*sinalp_190*(cosalp_190*ac_transformed_pencil_jj.x+sinalp_190*ac_transformed_pencil_jj.y)
eta_total__mod__magnetic=eta_total__mod__magnetic+abs(AC_eta1_aniso__mod__magnetic)
}
if (AC_lresi_etass__mod__magnetic) {
etass_190=AC_alphassm__mod__magnetic*ac_transformed_pencil_cs2/omegass_190
fres__mod__magnetic.x=fres__mod__magnetic.x-etass_190*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-etass_190*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-etass_190*ac_transformed_pencil_jj.z
eta_total__mod__magnetic=eta_total__mod__magnetic+etass_190
}
if (AC_lresi_xydep__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.x+AC_geta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1][1-1]*ac_transformed_pencil_diva
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.y+AC_geta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1][2-1]*ac_transformed_pencil_diva
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.z+AC_geta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1][3-1]*ac_transformed_pencil_diva
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_xy__mod__magnetic[vertexIdx.x][AC_m__mod__cdata-1]
}
if (AC_lresi_xdep__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - AC_eta_x__mod__magnetic[vertexIdx.x] * AC_mu0__mod__cdata * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - AC_eta_x__mod__magnetic[vertexIdx.x] * AC_mu0__mod__cdata * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - AC_eta_x__mod__magnetic[vertexIdx.x] * AC_mu0__mod__cdata * ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_x__mod__magnetic[vertexIdx.x]*ac_transformed_pencil_del2a.x
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_x__mod__magnetic[vertexIdx.x]*ac_transformed_pencil_del2a.y
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_x__mod__magnetic[vertexIdx.x]*ac_transformed_pencil_del2a.z
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_geta_x__mod__magnetic[vertexIdx.x]*ac_transformed_pencil_diva
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_x__mod__magnetic[vertexIdx.x]
}
if (AC_lresi_rdep__mod__magnetic) {
if(AC_string_enum_rdep_profile__mod__magnetic == AC_string_enum_step_string__mod__cparam) {
tmp1_164_190=ac_transformed_pencil_r_mn
step_vector_return_value_154_164_190 = 0.5*(1+tanh((tmp1_164_190-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth__mod__magnetic+AC_tini__mod__cparam)))
eta_r__mod__magnetic = AC_eta__mod__magnetic + AC_eta__mod__magnetic*(AC_eta_jump__mod__magnetic-1.)*step_vector_return_value_154_164_190
arg_155_164_190 = abs((tmp1_164_190-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_155_164_190)>=8.)  {
der_step_return_value_155_164_190 = 2./AC_eta_rwidth__mod__magnetic*exp(-2.*abs(arg_155_164_190))
}
else {
der_step_return_value_155_164_190 = 0.5/(AC_eta_rwidth__mod__magnetic*cosh(arg_155_164_190)*cosh(arg_155_164_190))
}
tmp2_164_190 = AC_eta__mod__magnetic*(AC_eta_jump__mod__magnetic-1.)*der_step_return_value_155_164_190
geta_r__mod__magnetic.x=tmp2_164_190*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.y=tmp2_164_190*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.z=tmp2_164_190*AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
else if(AC_string_enum_rdep_profile__mod__magnetic == AC_string_enum_two_step_string__mod__cparam || AC_string_enum_rdep_profile__mod__magnetic == AC_string_enum_twozstep_string__mod__cparam) {
step_vector_return_value_156_164_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth0__mod__magnetic+AC_tini__mod__cparam)))
step_vector_return_value_157_164_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_eta_r1__mod__magnetic)/(AC_eta_rwidth1__mod__magnetic+AC_tini__mod__cparam)))
eta_r__mod__magnetic = AC_eta__mod__magnetic + AC_eta__mod__magnetic*(AC_eta_jump__mod__magnetic - 1.)*  (step_vector_return_value_156_164_190 - step_vector_return_value_157_164_190)
arg_158_164_190 = abs((ac_transformed_pencil_r_mn-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth0__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_158_164_190)>=8.)  {
der_step_return_value_158_164_190 = 2./AC_eta_rwidth0__mod__magnetic*exp(-2.*abs(arg_158_164_190))
}
else {
der_step_return_value_158_164_190 = 0.5/(AC_eta_rwidth0__mod__magnetic*cosh(arg_158_164_190)*cosh(arg_158_164_190))
}
arg_159_164_190 = abs((ac_transformed_pencil_r_mn-AC_eta_r1__mod__magnetic)/(AC_eta_rwidth1__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_159_164_190)>=8.)  {
der_step_return_value_159_164_190 = 2./AC_eta_rwidth1__mod__magnetic*exp(-2.*abs(arg_159_164_190))
}
else {
der_step_return_value_159_164_190 = 0.5/(AC_eta_rwidth1__mod__magnetic*cosh(arg_159_164_190)*cosh(arg_159_164_190))
}
tmp1_164_190 = AC_eta__mod__magnetic*(AC_eta_jump__mod__magnetic-1.)*(  der_step_return_value_158_164_190 - der_step_return_value_159_164_190)
geta_r__mod__magnetic.x=tmp1_164_190*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.y=tmp1_164_190*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.z=tmp1_164_190*AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
else if(AC_string_enum_rdep_profile__mod__magnetic == AC_string_enum_two_step2_string__mod__cparam || AC_string_enum_rdep_profile__mod__magnetic == AC_string_enum_twozstep2_string__mod__cparam) {
step_vector_return_value_160_164_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_eta_r1__mod__magnetic)/(AC_eta_rwidth1__mod__magnetic+AC_tini__mod__cparam)))
prof1_164_190    = step_vector_return_value_160_164_190
step_vector_return_value_161_164_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth0__mod__magnetic+AC_tini__mod__cparam)))
prof0_164_190    = step_vector_return_value_161_164_190 - prof1_164_190
arg_162_164_190 = abs((ac_transformed_pencil_r_mn-AC_eta_r1__mod__magnetic)/(AC_eta_rwidth1__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_162_164_190)>=8.)  {
der_step_return_value_162_164_190 = 2./AC_eta_rwidth1__mod__magnetic*exp(-2.*abs(arg_162_164_190))
}
else {
der_step_return_value_162_164_190 = 0.5/(AC_eta_rwidth1__mod__magnetic*cosh(arg_162_164_190)*cosh(arg_162_164_190))
}
derprof1_164_190 = der_step_return_value_162_164_190
arg_163_164_190 = abs((ac_transformed_pencil_r_mn-AC_eta_r0__mod__magnetic)/(AC_eta_rwidth0__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_163_164_190)>=8.)  {
der_step_return_value_163_164_190 = 2./AC_eta_rwidth0__mod__magnetic*exp(-2.*abs(arg_163_164_190))
}
else {
der_step_return_value_163_164_190 = 0.5/(AC_eta_rwidth0__mod__magnetic*cosh(arg_163_164_190)*cosh(arg_163_164_190))
}
derprof0_164_190 = der_step_return_value_163_164_190 - derprof1_164_190
eta_r__mod__magnetic = AC_eta__mod__magnetic + (AC_eta__mod__magnetic*(AC_eta_jump0__mod__magnetic-1.))*prof0_164_190 + (AC_eta__mod__magnetic*(AC_eta_jump1__mod__magnetic-1.))*prof1_164_190
tmp1_164_190  = AC_eta__mod__magnetic + (AC_eta__mod__magnetic*(AC_eta_jump0__mod__magnetic-1.))*derprof0_164_190 + (AC_eta__mod__magnetic*(AC_eta_jump1__mod__magnetic-1.))*derprof1_164_190
geta_r__mod__magnetic.x=tmp1_164_190*AC_x__mod__cdata[vertexIdx.x]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.y=tmp1_164_190*AC_y__mod__cdata[AC_m__mod__cdata-1]*ac_transformed_pencil_r_mn1
geta_r__mod__magnetic.z=tmp1_164_190*AC_z__mod__cdata[AC_n__mod__cdata-1]*ac_transformed_pencil_r_mn1
}
fres__mod__magnetic.x=fres__mod__magnetic.x+eta_r__mod__magnetic*ac_transformed_pencil_del2a.x+geta_r__mod__magnetic.x*ac_transformed_pencil_diva
fres__mod__magnetic.y=fres__mod__magnetic.y+eta_r__mod__magnetic*ac_transformed_pencil_del2a.y+geta_r__mod__magnetic.y*ac_transformed_pencil_diva
fres__mod__magnetic.z=fres__mod__magnetic.z+eta_r__mod__magnetic*ac_transformed_pencil_del2a.z+geta_r__mod__magnetic.z*ac_transformed_pencil_diva
eta_total__mod__magnetic=eta_total__mod__magnetic+eta_r__mod__magnetic
}
if (AC_lresi_ydep__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_y__mod__magnetic[AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.x
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_y__mod__magnetic[AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.y
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_y__mod__magnetic[AC_m__mod__cdata-1]*ac_transformed_pencil_del2a.z
if (AC_lspherical_coords__mod__cdata) {
fres__mod__magnetic.y=fres__mod__magnetic.y+ac_transformed_pencil_r_mn1*AC_geta_y__mod__magnetic[AC_m__mod__cdata-1]*ac_transformed_pencil_diva
}
else {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_geta_y__mod__magnetic[AC_m__mod__cdata-1]*ac_transformed_pencil_diva
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_y__mod__magnetic[AC_m__mod__cdata-1]
}
if (AC_lresi_hyper2__mod__magnetic) {
fres__mod__magnetic=fres__mod__magnetic+AC_eta_hyper2__mod__magnetic*ac_transformed_pencil_del4a
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta2__mod__magnetic=diffus_eta2__mod__magnetic+AC_eta_hyper2__mod__magnetic
}
}
if (AC_lresi_hyper3__mod__magnetic) {
fres__mod__magnetic=fres__mod__magnetic+AC_eta_hyper3__mod__magnetic*ac_transformed_pencil_del6a
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic+AC_eta_hyper3__mod__magnetic
}
}
if (AC_lresi_hyper2_tdep__mod__magnetic) {
fres__mod__magnetic=fres__mod__magnetic-AC_eta_tdep__mod__magnetic*ac_transformed_pencil_del4a
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta2__mod__magnetic=diffus_eta2__mod__magnetic+AC_eta_tdep__mod__magnetic
}
}
if (AC_lresi_hyper3_tdep__mod__magnetic) {
fres__mod__magnetic=fres__mod__magnetic+AC_eta_tdep__mod__magnetic*ac_transformed_pencil_del6a
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic+AC_eta_tdep__mod__magnetic
}
}
if (AC_lresi_hyper3_polar__mod__magnetic) {
ju_190=1+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.z*dline_1__mod__cdata.z
ju_190=2+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.z*dline_1__mod__cdata.z
ju_190=3+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.x*dline_1__mod__cdata.x
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.y*dline_1__mod__cdata.y
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*tmp1_190*dline_1__mod__cdata.z*dline_1__mod__cdata.z
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic+AC_eta_hyper3__mod__magnetic*AC_pi4_1__mod__cparam*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata*dxmin_pencil__mod__cdata
}
}
if (AC_lresi_hyper3_mesh__mod__magnetic) {
ju_190=1+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x = fres__mod__magnetic.x + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.x = fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x = fres__mod__magnetic.x + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.x = fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x = fres__mod__magnetic.x + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.x = fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
ju_190=2+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y = fres__mod__magnetic.y + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.y = fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y = fres__mod__magnetic.y + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.y = fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y = fres__mod__magnetic.y + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.y = fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
ju_190=3+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.z = fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.z = fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_eta_hyper3_mesh__mod__magnetic * tmp1_190 * dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.z = fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
if (AC_ldynamical_diffusion__mod__cdata) {
diffus_eta3__mod__magnetic = diffus_eta3__mod__magnetic + AC_eta_hyper3_mesh__mod__magnetic
advec_hypermesh_aa_190 = 0.0
}
else {
advec_hypermesh_aa_190=AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_aa_190*advec_hypermesh_aa_190
}
}
if (AC_lresi_hyper3_csmesh__mod__magnetic) {
ju_190=1+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
ju_190=2+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
ju_190=3+AC_iaa__mod__cdata-1
tmp1_190 = der6x_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.x
}
else {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.x
}
tmp1_190 = der6y_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.y
}
else {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.y
}
tmp1_190 = der6z_ignore_spacing(Field(ju_190))
if (AC_ldynamical_diffusion__mod__cdata) {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * tmp1_190*dline_1__mod__cdata.z
}
else {
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2) * AC_pi5_1__mod__cparam/60.*tmp1_190*dline_1__mod__cdata.z
}
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
if (AC_ldynamical_diffusion__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic+AC_eta_hyper3_mesh__mod__magnetic*sqrt(ac_transformed_pencil_cs2)
advec_hypermesh_aa_190=0.0
}
else {
advec_hypermesh_aa_190=AC_eta_hyper3_mesh__mod__magnetic*AC_pi5_1__mod__cparam*sqrt(dxyz_2__mod__cdata*ac_transformed_pencil_cs2)
}
advec2_hypermesh__mod__cdata=advec2_hypermesh__mod__cdata+advec_hypermesh_aa_190*advec_hypermesh_aa_190
}
}
if (AC_lresi_hyper3_strict__mod__magnetic) {
fres__mod__magnetic=fres__mod__magnetic+AC_eta_hyper3__mod__magnetic*value(F_HYPREVEC)
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic+AC_eta_hyper3__mod__magnetic
}
}
if (AC_lresi_hyper3_aniso__mod__magnetic) {
k1_165_190=AC_iaa__mod__cdata-1
tmp_165_190  = del6fj(Field(k1_165_190+1), AC_eta_aniso_hyper3__mod__magnetic)
tmp2_190.x=tmp_165_190
tmp_165_190  = del6fj(Field(k1_165_190+2), AC_eta_aniso_hyper3__mod__magnetic)
tmp2_190.y=tmp_165_190
tmp_165_190  = del6fj(Field(k1_165_190+3), AC_eta_aniso_hyper3__mod__magnetic)
tmp2_190.z=tmp_165_190
fres__mod__magnetic=fres__mod__magnetic+tmp2_190
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
diffus_eta3__mod__magnetic=diffus_eta3__mod__magnetic +  (AC_eta_aniso_hyper3__mod__magnetic.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x*dline_1__mod__cdata.x +  AC_eta_aniso_hyper3__mod__magnetic.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y*dline_1__mod__cdata.y +  AC_eta_aniso_hyper3__mod__magnetic.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z*dline_1__mod__cdata.z)/dxyz_6__mod__cdata
}
}
if (AC_lresi_shell__mod__magnetic) {
eta_r_174_190=0.
if (AC_eta_int__mod__magnetic > 0.) {
d_int_174_190 = AC_eta_int__mod__magnetic - AC_eta__mod__magnetic
}
else {
d_int_174_190 = 0.
}
if (AC_eta_ext__mod__magnetic > 0.) {
d_ext_174_190 = AC_eta_ext__mod__magnetic - AC_eta__mod__magnetic
}
else {
d_ext_174_190 = 0.
}
if (AC_lcylinder_in_a_box__mod__cdata || AC_lcylindrical_coords__mod__cdata) {
step_vector_return_value_166_174_190 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_int__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam)))
prof_174_190=step_vector_return_value_166_174_190
eta_mn_190=d_int_174_190*(1-prof_174_190)
step_vector_return_value_167_174_190 = 0.5*(1+tanh((ac_transformed_pencil_rcyl_mn-AC_r_ext__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam)))
prof_174_190=step_vector_return_value_167_174_190
eta_mn_190=AC_eta__mod__magnetic+eta_mn_190+d_ext_174_190*prof_174_190
arg_168_174_190 = abs((ac_transformed_pencil_rcyl_mn-AC_r_int__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_168_174_190)>=8.)  {
der_step_return_value_168_174_190 = 2./AC_wresistivity__mod__magnetic*exp(-2.*abs(arg_168_174_190))
}
else {
der_step_return_value_168_174_190 = 0.5/(AC_wresistivity__mod__magnetic*cosh(arg_168_174_190)*cosh(arg_168_174_190))
}
prof_174_190=der_step_return_value_168_174_190
eta_r_174_190=-d_int_174_190*prof_174_190
arg_169_174_190 = abs((ac_transformed_pencil_rcyl_mn-AC_r_ext__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_169_174_190)>=8.)  {
der_step_return_value_169_174_190 = 2./AC_wresistivity__mod__magnetic*exp(-2.*abs(arg_169_174_190))
}
else {
der_step_return_value_169_174_190 = 0.5/(AC_wresistivity__mod__magnetic*cosh(arg_169_174_190)*cosh(arg_169_174_190))
}
prof_174_190=der_step_return_value_169_174_190
eta_r_174_190=eta_r_174_190+d_ext_174_190*prof_174_190
geta_190.x=ac_transformed_pencil_evr.x*eta_r_174_190
geta_190.y=ac_transformed_pencil_evr.y*eta_r_174_190
geta_190.z=ac_transformed_pencil_evr.z*eta_r_174_190
}
else if (AC_lsphere_in_a_box__mod__cdata || AC_lspherical_coords__mod__cdata) {
step_vector_return_value_170_174_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_r_int__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam)))
prof_174_190=step_vector_return_value_170_174_190
eta_mn_190=d_int_174_190*(1-prof_174_190)
step_vector_return_value_171_174_190 = 0.5*(1+tanh((ac_transformed_pencil_r_mn-AC_r_ext__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam)))
prof_174_190=step_vector_return_value_171_174_190
eta_mn_190=AC_eta__mod__magnetic+eta_mn_190+d_ext_174_190*prof_174_190
arg_172_174_190 = abs((ac_transformed_pencil_r_mn-AC_r_int__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_172_174_190)>=8.)  {
der_step_return_value_172_174_190 = 2./AC_wresistivity__mod__magnetic*exp(-2.*abs(arg_172_174_190))
}
else {
der_step_return_value_172_174_190 = 0.5/(AC_wresistivity__mod__magnetic*cosh(arg_172_174_190)*cosh(arg_172_174_190))
}
prof_174_190=der_step_return_value_172_174_190
eta_r_174_190=-d_int_174_190*prof_174_190
arg_173_174_190 = abs((ac_transformed_pencil_r_mn-AC_r_ext__mod__cdata)/(AC_wresistivity__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_173_174_190)>=8.)  {
der_step_return_value_173_174_190 = 2./AC_wresistivity__mod__magnetic*exp(-2.*abs(arg_173_174_190))
}
else {
der_step_return_value_173_174_190 = 0.5/(AC_wresistivity__mod__magnetic*cosh(arg_173_174_190)*cosh(arg_173_174_190))
}
prof_174_190=der_step_return_value_173_174_190
eta_r_174_190=eta_r_174_190+d_ext_174_190*prof_174_190
geta_190.x=ac_transformed_pencil_evr.x*eta_r_174_190
geta_190.y=ac_transformed_pencil_evr.y*eta_r_174_190
geta_190.z=ac_transformed_pencil_evr.z*eta_r_174_190
}
else {
}
fres__mod__magnetic.x=fres__mod__magnetic.x+eta_mn_190*ac_transformed_pencil_del2a.x+geta_190.x*ac_transformed_pencil_diva
fres__mod__magnetic.y=fres__mod__magnetic.y+eta_mn_190*ac_transformed_pencil_del2a.y+geta_190.y*ac_transformed_pencil_diva
fres__mod__magnetic.z=fres__mod__magnetic.z+eta_mn_190*ac_transformed_pencil_del2a.z+geta_190.z*ac_transformed_pencil_diva
eta_total__mod__magnetic=eta_total__mod__magnetic+eta_mn_190
}
if (AC_lresi_eta_shock__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.x+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.y+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.z+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.z)
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock
}
if (AC_lresi_eta_shock2__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_shock2__mod__magnetic*ac_transformed_pencil_shock*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_shock2__mod__magnetic*ac_transformed_pencil_shock*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_shock2__mod__magnetic*ac_transformed_pencil_shock*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_shock2__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_shock*ac_transformed_pencil_del2a.x+2*ac_transformed_pencil_shock*ac_transformed_pencil_diva*ac_transformed_pencil_gshock.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_shock2__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_shock*ac_transformed_pencil_del2a.y+2*ac_transformed_pencil_shock*ac_transformed_pencil_diva*ac_transformed_pencil_gshock.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_shock2__mod__magnetic*(ac_transformed_pencil_shock*ac_transformed_pencil_shock*ac_transformed_pencil_del2a.z+2*ac_transformed_pencil_shock*ac_transformed_pencil_diva*ac_transformed_pencil_gshock.z)
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_shock2__mod__magnetic*ac_transformed_pencil_shock*ac_transformed_pencil_shock
}
if (AC_lresi_eta_shock_profz__mod__magnetic) {
step_vector_return_value_175_190 = 0.5*(1+tanh((ac_transformed_pencil_z_mn-AC_eta_zshock__mod__magnetic)/(AC_eta_width_shock__mod__magnetic+AC_tini__mod__cparam)))
peta_shock_190 = AC_eta_shock__mod__magnetic + AC_eta_shock_jump1__mod__magnetic*step_vector_return_value_175_190
gradeta_shock_190.x = 0.
gradeta_shock_190.y = 0.
arg_176_190 = abs((ac_transformed_pencil_z_mn-AC_eta_zshock__mod__magnetic)/(AC_eta_width_shock__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_176_190)>=8.)  {
der_step_return_value_176_190 = 2./AC_eta_width_shock__mod__magnetic*exp(-2.*abs(arg_176_190))
}
else {
der_step_return_value_176_190 = 0.5/(AC_eta_width_shock__mod__magnetic*cosh(arg_176_190)*cosh(arg_176_190))
}
gradeta_shock_190.z = AC_eta_shock_jump1__mod__magnetic*der_step_return_value_176_190
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+  peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.x+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.x)+ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.x
fres__mod__magnetic.y=fres__mod__magnetic.y+  peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.y+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.y)+ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.y
fres__mod__magnetic.z=fres__mod__magnetic.z+  peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.z+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.z)+ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.z
}
eta_total__mod__magnetic=eta_total__mod__magnetic+peta_shock_190*ac_transformed_pencil_shock
}
if (AC_lresi_eta_shock_profr__mod__magnetic) {
if (AC_lspherical_coords__mod__cdata || AC_lsphere_in_a_box__mod__cdata) {
tmp1_190=ac_transformed_pencil_r_mn
}
else {
tmp1_190=ac_transformed_pencil_rcyl_mn
}
step_vector_return_value_177_190 = 0.5*(1+tanh((tmp1_190-AC_eta_xshock__mod__magnetic)/(AC_eta_width_shock__mod__magnetic+AC_tini__mod__cparam)))
peta_shock_190 = AC_eta_shock__mod__magnetic + AC_eta_shock_jump1__mod__magnetic*step_vector_return_value_177_190
arg_178_190 = abs((tmp1_190-AC_eta_xshock__mod__magnetic)/(AC_eta_width_shock__mod__magnetic+AC_tini__mod__cparam))
if (abs(arg_178_190)>=8.)  {
der_step_return_value_178_190 = 2./AC_eta_width_shock__mod__magnetic*exp(-2.*abs(arg_178_190))
}
else {
der_step_return_value_178_190 = 0.5/(AC_eta_width_shock__mod__magnetic*cosh(arg_178_190)*cosh(arg_178_190))
}
gradeta_shock_190.x = AC_eta_shock_jump1__mod__magnetic*der_step_return_value_178_190
gradeta_shock_190.y = 0.
gradeta_shock_190.z = 0.
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-peta_shock_190*ac_transformed_pencil_shock*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x + peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.x+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.x)+  ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.x
fres__mod__magnetic.y=fres__mod__magnetic.y + peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.y+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.y)+  ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.y
fres__mod__magnetic.z=fres__mod__magnetic.z + peta_shock_190*(ac_transformed_pencil_shock*ac_transformed_pencil_del2a.z+ac_transformed_pencil_diva*ac_transformed_pencil_gshock.z)+  ac_transformed_pencil_diva*ac_transformed_pencil_shock*gradeta_shock_190.z
}
eta_total__mod__magnetic=eta_total__mod__magnetic+peta_shock_190*ac_transformed_pencil_shock
}
if (AC_lresi_eta_shock_perp__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock_perp*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock_perp*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock_perp*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+ AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock_perp*ac_transformed_pencil_del2a.x+ac_transformed_pencil_diva*ac_transformed_pencil_gshock_perp.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+ AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock_perp*ac_transformed_pencil_del2a.y+ac_transformed_pencil_diva*ac_transformed_pencil_gshock_perp.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+ AC_eta_shock__mod__magnetic*(ac_transformed_pencil_shock_perp*ac_transformed_pencil_del2a.z+ac_transformed_pencil_diva*ac_transformed_pencil_gshock_perp.z)
}
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_shock__mod__magnetic*ac_transformed_pencil_shock_perp
}
if (AC_lresi_etava__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - ac_transformed_pencil_etava * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - ac_transformed_pencil_etava * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - ac_transformed_pencil_etava * ac_transformed_pencil_jj.z
}
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_etava
}
if (AC_lresi_vaspeed__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - ac_transformed_pencil_etava * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - ac_transformed_pencil_etava * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - ac_transformed_pencil_etava * ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x = fres__mod__magnetic.x + AC_mu0__mod__cdata * ac_transformed_pencil_etava * ac_transformed_pencil_del2a.x + AC_eta_va__mod__magnetic/AC_varms__mod__magnetic * ac_transformed_pencil_diva * ac_transformed_pencil_gva.x
fres__mod__magnetic.y = fres__mod__magnetic.y + AC_mu0__mod__cdata * ac_transformed_pencil_etava * ac_transformed_pencil_del2a.y + AC_eta_va__mod__magnetic/AC_varms__mod__magnetic * ac_transformed_pencil_diva * ac_transformed_pencil_gva.y
fres__mod__magnetic.z = fres__mod__magnetic.z + AC_mu0__mod__cdata * ac_transformed_pencil_etava * ac_transformed_pencil_del2a.z + AC_eta_va__mod__magnetic/AC_varms__mod__magnetic * ac_transformed_pencil_diva * ac_transformed_pencil_gva.z
}
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_etava
}
if (AC_lresi_etaj__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - ac_transformed_pencil_etaj * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - ac_transformed_pencil_etaj * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - ac_transformed_pencil_etaj * ac_transformed_pencil_jj.z
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_etaj
}
if (AC_lresi_etaj2__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - ac_transformed_pencil_etaj2 * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - ac_transformed_pencil_etaj2 * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - ac_transformed_pencil_etaj2 * ac_transformed_pencil_jj.z
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_etaj2
}
if (AC_lresi_etajrho__mod__magnetic) {
fres__mod__magnetic.x = fres__mod__magnetic.x - ac_transformed_pencil_etajrho * ac_transformed_pencil_jj.x
fres__mod__magnetic.y = fres__mod__magnetic.y - ac_transformed_pencil_etajrho * ac_transformed_pencil_jj.y
fres__mod__magnetic.z = fres__mod__magnetic.z - ac_transformed_pencil_etajrho * ac_transformed_pencil_jj.z
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_etajrho
}
if (AC_lresi_smagorinsky__mod__magnetic) {
if (!AC_lweyl_gauge__mod__magnetic) {
if (AC_letasmag_as_aux__mod__magnetic) {
eta_smag__mod__magnetic=pow((AC_d_smag__mod__magnetic*AC_dxmax__mod__cdata),2.)*sqrt(ac_transformed_pencil_j2)
fres__mod__magnetic = eta_smag__mod__magnetic+AC_eta__mod__magnetic*ac_transformed_pencil_del2a
geta_190 = gradient(Field(AC_ietasmag__mod__cdata))
fres__mod__magnetic.x=fres__mod__magnetic.x+geta_190.x*ac_transformed_pencil_diva
fres__mod__magnetic.y=fres__mod__magnetic.y+geta_190.y*ac_transformed_pencil_diva
fres__mod__magnetic.z=fres__mod__magnetic.z+geta_190.z*ac_transformed_pencil_diva
}
else {
eta_smag__mod__magnetic=pow((AC_d_smag__mod__magnetic*AC_dxmax__mod__cdata),2.)*sqrt(ac_transformed_pencil_j2)
fres__mod__magnetic = eta_smag__mod__magnetic+AC_eta__mod__magnetic*ac_transformed_pencil_del2a
}
}
else {
eta_smag__mod__magnetic=pow((AC_d_smag__mod__magnetic*AC_dxmax__mod__cdata),2.)*sqrt(ac_transformed_pencil_j2)
fres__mod__magnetic.x=fres__mod__magnetic.x-eta_smag__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-eta_smag__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-eta_smag__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
}
if (AC_lresi_smagorinsky_nusmag__mod__magnetic) {
eta_smag__mod__magnetic=AC_pm_smag1__mod__magnetic*ac_transformed_pencil_nu_smag
fres__mod__magnetic = eta_smag__mod__magnetic+AC_eta__mod__magnetic*ac_transformed_pencil_del2a
geta_190 = gradient(Field(AC_inusmag__mod__cdata))
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_pm_smag1__mod__magnetic*geta_190.x*ac_transformed_pencil_diva
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_pm_smag1__mod__magnetic*geta_190.y*ac_transformed_pencil_diva
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_pm_smag1__mod__magnetic*geta_190.z*ac_transformed_pencil_diva
}
if (AC_lresi_smagorinsky_cross__mod__magnetic) {
sign_jo_190=1.
if (ac_transformed_pencil_jo < 0) {
sign_jo_190=-1.
}
eta_smag__mod__magnetic=pow((AC_d_smag__mod__magnetic*AC_dxmax__mod__cdata),2.)*sign_jo_190*sqrt(ac_transformed_pencil_jo*sign_jo_190)
fres__mod__magnetic = eta_smag__mod__magnetic+AC_eta__mod__magnetic*ac_transformed_pencil_del2a
}
if (AC_lresi_smagorinsky__mod__magnetic  ||  AC_lresi_smagorinsky_nusmag__mod__magnetic  ||  AC_lresi_smagorinsky_cross__mod__magnetic) {
eta_total__mod__magnetic = eta_total__mod__magnetic + eta_smag__mod__magnetic
}
if (AC_lresi_anomalous__mod__magnetic) {
vdrift_190=sqrt(sum(ac_transformed_pencil_jj*ac_transformed_pencil_jj))*ac_transformed_pencil_rho1
if (AC_lweyl_gauge__mod__magnetic) {
if (AC_eta_anom_thresh__mod__magnetic!=0) {
if (AC_eta_anom__mod__magnetic*vdrift_190 > AC_eta_anom_thresh__mod__magnetic*AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_anom_thresh__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
}
}
else {
if (vdrift_190>AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
}
}
if (AC_eta_anom_thresh__mod__magnetic!=0) {
if (AC_eta_anom__mod__magnetic*vdrift_190 > AC_eta_anom_thresh__mod__magnetic*AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_anom_thresh__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
}
else {
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
}
}
else {
if (vdrift_190>AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
}
}
if (AC_eta_anom_thresh__mod__magnetic!=0) {
if (AC_eta_anom__mod__magnetic*vdrift_190 > AC_eta_anom_thresh__mod__magnetic*AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_anom_thresh__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
}
else {
if (vdrift_190>AC_vcrit_anom__mod__magnetic) {
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
}
}
else {
}
if (AC_eta_anom_thresh__mod__magnetic!=0) {
if (AC_eta_anom__mod__magnetic*vdrift_190 > AC_eta_anom_thresh__mod__magnetic*AC_vcrit_anom__mod__magnetic) {
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_anom_thresh__mod__magnetic
}
else {
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic
}
}
else {
if (vdrift_190>AC_vcrit_anom__mod__magnetic) {
eta_total__mod__magnetic=eta_total__mod__magnetic+AC_eta_anom__mod__magnetic*vdrift_190/AC_vcrit_anom__mod__magnetic
}
}
}
if (AC_lresi_spitzer__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.x-1.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.y-1.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.z-1.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.z)
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta_spitzer__mod__magnetic*exp(-1.5*ac_transformed_pencil_lntt)
}
if (AC_lresi_cspeed__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.x+0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.x)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.y+0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.y)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)*(ac_transformed_pencil_del2a.z+0.5*ac_transformed_pencil_diva*ac_transformed_pencil_glntt.z)
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta__mod__magnetic*exp(AC_eta_cspeed__mod__magnetic*ac_transformed_pencil_lntt)
}
if (AC_lresi_eta_proptouz__mod__magnetic) {
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*ac_transformed_pencil_uu.z*AC_mu0__mod__cdata*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*ac_transformed_pencil_uu.z*AC_mu0__mod__cdata*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*ac_transformed_pencil_uu.z*AC_mu0__mod__cdata*ac_transformed_pencil_jj.z
}
else {
fres__mod__magnetic.x=fres__mod__magnetic.x+AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*(ac_transformed_pencil_uu.z*ac_transformed_pencil_del2a.x+ac_transformed_pencil_uij[3-1][1-1]*ac_transformed_pencil_diva)
fres__mod__magnetic.y=fres__mod__magnetic.y+AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*(ac_transformed_pencil_uu.z*ac_transformed_pencil_del2a.y+ac_transformed_pencil_uij[3-1][2-1]*ac_transformed_pencil_diva)
fres__mod__magnetic.z=fres__mod__magnetic.z+AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*(ac_transformed_pencil_uu.z*ac_transformed_pencil_del2a.z+ac_transformed_pencil_uij[3-1][3-1]*ac_transformed_pencil_diva)
}
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta__mod__magnetic*AC_ampl_eta_uz__mod__magnetic*ac_transformed_pencil_uu.z
}
if (AC_lresi_magfield__mod__magnetic) {
eta_bb_190=AC_eta__mod__magnetic/(1.+AC_etab__mod__magnetic*ac_transformed_pencil_bb.y*ac_transformed_pencil_bb.y)
if (AC_lweyl_gauge__mod__magnetic) {
fres__mod__magnetic.x=fres__mod__magnetic.x-AC_mu0__mod__cdata*eta_bb_190*ac_transformed_pencil_jj.x
fres__mod__magnetic.y=fres__mod__magnetic.y-AC_mu0__mod__cdata*eta_bb_190*ac_transformed_pencil_jj.y
fres__mod__magnetic.z=fres__mod__magnetic.z-AC_mu0__mod__cdata*eta_bb_190*ac_transformed_pencil_jj.z
}
eta_total__mod__magnetic = eta_total__mod__magnetic + eta_bb_190
}
if (AC_eta_aniso_bb__mod__magnetic!=0.0) {
if (ac_transformed_pencil_b2==0.) {
tmp1_190=0.
}
else {
tmp1_190=AC_eta_aniso_bb__mod__magnetic/ac_transformed_pencil_b2
}
if (AC_lquench_eta_aniso__mod__magnetic) {
tmp1_190=tmp1_190/(1.+AC_quench_aniso__mod__magnetic*AC_arms__mod__magnetic)
}
ju_190=1-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190-tmp1_190*ac_transformed_pencil_jb*ac_transformed_pencil_bb.x
ju_190=2-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190-tmp1_190*ac_transformed_pencil_jb*ac_transformed_pencil_bb.y
ju_190=3-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190-tmp1_190*ac_transformed_pencil_jb*ac_transformed_pencil_bb.z
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_eta_aniso_bb__mod__magnetic
}
if (AC_lambipolar_diffusion__mod__magnetic) {
ju_190=1-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190+ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbrxb.x
ju_190=2-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190+ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbrxb.y
ju_190=3-1+AC_iaa__mod__cdata
DF_U_190=DF_U_190+ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbrxb.z
if (AC_lentropy__mod__cparam  &&  AC_lneutralion_heat__mod__magnetic) {
if (AC_pretend_lntt__mod__cdata) {
DF_SS = DF_SS + ac_transformed_pencil_cv1*ac_transformed_pencil_tt1*ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbr2
}
else {
DF_SS = DF_SS + ac_transformed_pencil_tt1*ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbr2
}
}
else if (AC_ltemperature__mod__cparam  &&  AC_lneutralion_heat__mod__magnetic) {
DF_LNTT = DF_LNTT + ac_transformed_pencil_cv1*ac_transformed_pencil_tt1*ac_transformed_pencil_nu_ni1*ac_transformed_pencil_jxbr2
}
eta_total__mod__magnetic = eta_total__mod__magnetic + ac_transformed_pencil_nu_ni1*ac_transformed_pencil_va2
}
if (AC_lmean_friction__mod__magnetic) {
}
else if (AC_llocal_friction__mod__magnetic) {
dadt_190 = dadt_190-AC_llambda_aa__mod__magnetic*ac_transformed_pencil_aa
}
if (AC_lmagnetic_slope_limited__mod__magnetic && AC_llast__mod__cdata) {
if (AC_lsld_bb__mod__magnetic) {
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
tmp2_190.x= (-d_sld_flux_190[2-1][3-1] + d_sld_flux_190[3-1][2-1])*AC_fac_sld_magn__mod__magnetic
tmp2_190.y= (-d_sld_flux_190[3-1][1-1] + d_sld_flux_190[1-1][3-1])*AC_fac_sld_magn__mod__magnetic
tmp2_190.z= (-d_sld_flux_190[1-1][2-1] + d_sld_flux_190[2-1][1-1])*AC_fac_sld_magn__mod__magnetic
fres__mod__magnetic=fres__mod__magnetic + tmp2_190
}
else {
if (AC_lcylindrical_coords__mod__cdata  ||  AC_lspherical_coords__mod__cdata) {
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
if (AC_lcylindrical_coords__mod__cdata) {
fres__mod__magnetic.x=fres__mod__magnetic.x+tmp2_190.x-(d_sld_flux_190[2-1][2-1])/AC_x__mod__cdata[vertexIdx.x]
fres__mod__magnetic.y=fres__mod__magnetic.y+tmp2_190.y+(d_sld_flux_190[2-1][1-1])/AC_x__mod__cdata[vertexIdx.x]
fres__mod__magnetic.z=fres__mod__magnetic.z+tmp2_190.z
}
else if(AC_lspherical_coords__mod__cdata) {
fres__mod__magnetic.x=fres__mod__magnetic.x+tmp2_190.x-(d_sld_flux_190[2-1][2-1]+d_sld_flux_190[3-1][3-1])/AC_x__mod__cdata[vertexIdx.x]
fres__mod__magnetic.y=fres__mod__magnetic.y+tmp2_190.y+(d_sld_flux_190[2-1][1-1]-d_sld_flux_190[3-1][3-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
fres__mod__magnetic.z=fres__mod__magnetic.z+tmp2_190.z+(d_sld_flux_190[3-1][1-1]+d_sld_flux_190[3-1][2-1]*AC_cotth__mod__cdata[AC_m__mod__cdata-1])/AC_x__mod__cdata[vertexIdx.x]
}
}
else {
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
not_implemented("calc_slope_diff_flux")
fres__mod__magnetic=fres__mod__magnetic+tmp2_190
}
}
if (AC_lohmic_heat__mod__magnetic) {
tmp1_190 = dot(tmp2_190,ac_transformed_pencil_jj)
if (AC_lentropy__mod__cparam) {
if (AC_pretend_lntt__mod__cdata) {
DF_SS = DF_SS + ac_transformed_pencil_cv1*max(0.0,tmp1_190)*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
else {
DF_SS = DF_SS + max(0.0,tmp1_190)*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
}
else if (AC_ltemperature__mod__cparam) {
if (AC_ltemperature_nolog__mod__cdata) {
DF_TT   = DF_TT + ac_transformed_pencil_cv1*max(0.0,tmp1_190)*ac_transformed_pencil_rho1
}
else {
DF_LNTT = DF_LNTT + ac_transformed_pencil_cv1*max(0.0,tmp1_190)*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
}
else if (AC_lthermal_energy__mod__cparam) {
DF_ETH = DF_ETH + max(0.0,tmp1_190)
}
}
}
if (AC_lno_ohmic_heat_bound_z__mod__magnetic && AC_lohmic_heat__mod__magnetic) {
if (false) {
}
else {
relshift_180_190=0.0
}
xi_180_190 = (AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_no_ohmic_heat_z0__mod__magnetic)/(AC_no_ohmic_heat_zwidth__mod__magnetic+AC_tini__mod__cparam) - relshift_180_190
xi_180_190 = max(xi_180_190,-1.0)
xi_180_190 = min(xi_180_190, 1.0)
cubic_step_pt_return_value_180_190 = 0.5 + xi_180_190*(0.75-xi_180_190*xi_180_190*0.25)
eta_heat_190=eta_total__mod__magnetic*cubic_step_pt_return_value_180_190
}
else {
eta_heat_190=eta_total__mod__magnetic
}
if (!AC_lkinematic__mod__magnetic && AC_lohmic_heat__mod__magnetic) {
if (AC_lentropy__mod__cparam) {
if (AC_pretend_lntt__mod__cdata) {
DF_SS = DF_SS + ac_transformed_pencil_cv1*eta_heat_190*AC_mu0__mod__cdata*ac_transformed_pencil_j2*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
else {
DF_SS = DF_SS + eta_heat_190*AC_mu0__mod__cdata*ac_transformed_pencil_j2*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
}
else if (AC_ltemperature__mod__cparam) {
if (AC_ltemperature_nolog__mod__cdata) {
DF_TT   = DF_TT + ac_transformed_pencil_cv1*eta_heat_190*AC_mu0__mod__cdata*ac_transformed_pencil_j2*ac_transformed_pencil_rho1
}
else {
DF_LNTT = DF_LNTT + ac_transformed_pencil_cv1*eta_heat_190*AC_mu0__mod__cdata*ac_transformed_pencil_j2*ac_transformed_pencil_rho1*ac_transformed_pencil_tt1
}
}
else if (AC_lthermal_energy__mod__cparam) {
DF_ETH = DF_ETH + eta_heat_190*AC_mu0__mod__cdata*ac_transformed_pencil_j2
}
}
if (AC_lfrozen_bb_bot__mod__magnetic.x && AC_lfirst_proc_z__mod__cdata && AC_n__mod__cdata==AC_n1__mod__cparam) {
fres__mod__magnetic.x=0.
}
if (AC_lfrozen_bb_top__mod__magnetic.x && AC_llast_proc_z__mod__cdata && AC_n__mod__cdata==AC_n2__mod__cdata) {
fres__mod__magnetic.x=0.
}
if (AC_lfrozen_bb_bot__mod__magnetic.y && AC_lfirst_proc_z__mod__cdata && AC_n__mod__cdata==AC_n1__mod__cparam) {
fres__mod__magnetic.y=0.
}
if (AC_lfrozen_bb_top__mod__magnetic.y && AC_llast_proc_z__mod__cdata && AC_n__mod__cdata==AC_n2__mod__cdata) {
fres__mod__magnetic.y=0.
}
if (AC_lfrozen_bb_bot__mod__magnetic.z && AC_lfirst_proc_z__mod__cdata && AC_n__mod__cdata==AC_n1__mod__cparam) {
fres__mod__magnetic.z=0.
}
if (AC_lfrozen_bb_top__mod__magnetic.z && AC_llast_proc_z__mod__cdata && AC_n__mod__cdata==AC_n2__mod__cdata) {
fres__mod__magnetic.z=0.
}
if (!AC_lupw_aa__mod__magnetic) {
if (AC_linduction__mod__magnetic) {
if (AC_ladvective_gauge__mod__magnetic) {
if (b_ext_190.x != 0.  ||  b_ext_190.y != 0.  ||  b_ext_190.z != 0.) {
ujiaj_190 = cross(ac_transformed_pencil_uu,b_ext_190)
}
else {
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.x = 0.
ajiuj_190.y = 0.
ajiuj_190.z = 0.
}
else {
ujiaj_190.x = 0.
ujiaj_190.y = 0.
ujiaj_190.z = 0.
}
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.x=ajiuj_190.x+ac_transformed_pencil_uu.x*ac_transformed_pencil_aij[1-1][1-1]
}
else {
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][1-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.x=ajiuj_190.x+ac_transformed_pencil_uu.y*ac_transformed_pencil_aij[2-1][1-1]
}
else {
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][1-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.x=ajiuj_190.x+ac_transformed_pencil_uu.z*ac_transformed_pencil_aij[3-1][1-1]
}
else {
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][1-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.y=ajiuj_190.y+ac_transformed_pencil_uu.x*ac_transformed_pencil_aij[1-1][2-1]
}
else {
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][2-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.y=ajiuj_190.y+ac_transformed_pencil_uu.y*ac_transformed_pencil_aij[2-1][2-1]
}
else {
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][2-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.y=ajiuj_190.y+ac_transformed_pencil_uu.z*ac_transformed_pencil_aij[3-1][2-1]
}
else {
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][2-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.z=ajiuj_190.z+ac_transformed_pencil_uu.x*ac_transformed_pencil_aij[1-1][3-1]
}
else {
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][3-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.z=ajiuj_190.z+ac_transformed_pencil_uu.y*ac_transformed_pencil_aij[2-1][3-1]
}
else {
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][3-1]
}
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.z=ajiuj_190.z+ac_transformed_pencil_uu.z*ac_transformed_pencil_aij[3-1][3-1]
}
else {
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][3-1]
}
if (AC_lcylindrical_coords__mod__cdata) {
if (AC_lfargo_advection__mod__cdata) {
ajiuj_190.y = ajiuj_190.y + (ac_transformed_pencil_aa.x*ac_transformed_pencil_uu.y - ac_transformed_pencil_aa.y*ac_transformed_pencil_uu.x)*AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]
}
else {
ujiaj_190.y = ujiaj_190.y + (ac_transformed_pencil_uu.x*ac_transformed_pencil_aa.y - ac_transformed_pencil_uu.y*ac_transformed_pencil_aa.x)*AC_rcyl_mn1__mod__cdata[vertexIdx.x-NGHOST_VAL]
}
}
else if (AC_lspherical_coords__mod__cdata) {
ujiaj_190.y = ujiaj_190.y + (ac_transformed_pencil_uu.x*ac_transformed_pencil_aa.y - ac_transformed_pencil_uu.y*ac_transformed_pencil_aa.x)*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]
ujiaj_190.z = ujiaj_190.z + (ac_transformed_pencil_uu.x*ac_transformed_pencil_aa.z          -  ac_transformed_pencil_uu.z*ac_transformed_pencil_aa.x          +  ac_transformed_pencil_uu.y*ac_transformed_pencil_aa.z*AC_cotth__mod__cdata[AC_m__mod__cdata-1] -  ac_transformed_pencil_uu.z*ac_transformed_pencil_aa.z*AC_cotth__mod__cdata[AC_m__mod__cdata-1])*AC_r1_mn__mod__cdata[vertexIdx.x-NGHOST_VAL]
}
if (!AC_lfargo_advection__mod__cdata) {
dadt_190 = dadt_190-ac_transformed_pencil_uga-ujiaj_190+fres__mod__magnetic
}
else {
dadt_190 = dadt_190-ac_transformed_pencil_uuadvec_gaa+ajiuj_190+fres__mod__magnetic
}
}
else if (AC_ladvective_gauge2__mod__magnetic) {
if (AC_lua_as_aux__mod__magnetic) {
gua_190 = gradient(Field(AC_iua__mod__magnetic))
dadt_190 = dadt_190 + ac_transformed_pencil_uxb+fres__mod__magnetic-gua_190
}
else {
}
}
else {
dadt_190 = dadt_190+ ac_transformed_pencil_uxb+fres__mod__magnetic
}
if (AC_lnoinduction__mod__magnetic) {
dadt_190 = dadt_190 - ac_transformed_pencil_uxb
}
}
}
else {
if (b_ext_190.x != 0.  ||  b_ext_190.y != 0.  ||  b_ext_190.z != 0.) {
uxb_upw_190 = cross(ac_transformed_pencil_uu,b_ext_190)
}
else {
uxb_upw_190.x = 0.
uxb_upw_190.y = 0.
uxb_upw_190.z = 0.
}
if (AC_ladvective_gauge__mod__magnetic) {
ujiaj_190.x = 0.
ujiaj_190.y = 0.
ujiaj_190.z = 0.
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][1-1]
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][1-1]
ujiaj_190.x=ujiaj_190.x+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][1-1]
uxb_upw_190.x=uxb_upw_190.x-ac_transformed_pencil_uga.x-ujiaj_190.x
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][2-1]
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][2-1]
ujiaj_190.y=ujiaj_190.y+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][2-1]
uxb_upw_190.y=uxb_upw_190.y-ac_transformed_pencil_uga.y-ujiaj_190.y
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.x*ac_transformed_pencil_uij[1-1][3-1]
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.y*ac_transformed_pencil_uij[2-1][3-1]
ujiaj_190.z=ujiaj_190.z+ac_transformed_pencil_aa.z*ac_transformed_pencil_uij[3-1][3-1]
uxb_upw_190.z=uxb_upw_190.z-ac_transformed_pencil_uga.z-ujiaj_190.z
}
else {
if (1!=1) {
uxb_upw_190.x=uxb_upw_190.x+ac_transformed_pencil_uu.x*(ac_transformed_pencil_aij[1-1][1-1]-ac_transformed_pencil_aij[1-1][1-1])
}
if (2!=1) {
uxb_upw_190.x=uxb_upw_190.x+ac_transformed_pencil_uu.y*(ac_transformed_pencil_aij[2-1][1-1]-ac_transformed_pencil_aij[1-1][2-1])
}
if (3!=1) {
uxb_upw_190.x=uxb_upw_190.x+ac_transformed_pencil_uu.z*(ac_transformed_pencil_aij[3-1][1-1]-ac_transformed_pencil_aij[1-1][3-1])
}
msk_181_190=0
if (true) {
if ( 1>=1  &&  1 <=3 ) {
msk_181_190=1
}
}
del6f_upwind_181_190 = del6_masked(Field(AC_iaa__mod__cdata+1-1), msk_181_190)
if (msk_181_190>0) {
uxb_upw_190.x = uxb_upw_190.x+del6f_upwind_181_190
}
else {
uxb_upw_190.x = uxb_upw_190.x-del6f_upwind_181_190
}
if (1!=2) {
uxb_upw_190.y=uxb_upw_190.y+ac_transformed_pencil_uu.x*(ac_transformed_pencil_aij[1-1][2-1]-ac_transformed_pencil_aij[2-1][1-1])
}
if (2!=2) {
uxb_upw_190.y=uxb_upw_190.y+ac_transformed_pencil_uu.y*(ac_transformed_pencil_aij[2-1][2-1]-ac_transformed_pencil_aij[2-1][2-1])
}
if (3!=2) {
uxb_upw_190.y=uxb_upw_190.y+ac_transformed_pencil_uu.z*(ac_transformed_pencil_aij[3-1][2-1]-ac_transformed_pencil_aij[2-1][3-1])
}
msk_181_190=0
if (true) {
if ( 2>=1  &&  2 <=3 ) {
msk_181_190=2
}
}
del6f_upwind_181_190 = del6_masked(Field(AC_iaa__mod__cdata+2-1), msk_181_190)
if (msk_181_190>0) {
uxb_upw_190.y = uxb_upw_190.y+del6f_upwind_181_190
}
else {
uxb_upw_190.y = uxb_upw_190.y-del6f_upwind_181_190
}
if (1!=3) {
uxb_upw_190.z=uxb_upw_190.z+ac_transformed_pencil_uu.x*(ac_transformed_pencil_aij[1-1][3-1]-ac_transformed_pencil_aij[3-1][1-1])
}
if (2!=3) {
uxb_upw_190.z=uxb_upw_190.z+ac_transformed_pencil_uu.y*(ac_transformed_pencil_aij[2-1][3-1]-ac_transformed_pencil_aij[3-1][2-1])
}
if (3!=3) {
uxb_upw_190.z=uxb_upw_190.z+ac_transformed_pencil_uu.z*(ac_transformed_pencil_aij[3-1][3-1]-ac_transformed_pencil_aij[3-1][3-1])
}
msk_181_190=0
if (true) {
if ( 3>=1  &&  3 <=3 ) {
msk_181_190=3
}
}
del6f_upwind_181_190 = del6_masked(Field(AC_iaa__mod__cdata+3-1), msk_181_190)
if (msk_181_190>0) {
uxb_upw_190.z = uxb_upw_190.z+del6f_upwind_181_190
}
else {
uxb_upw_190.z = uxb_upw_190.z-del6f_upwind_181_190
}
}
if (AC_linduction__mod__magnetic) {
dadt_190= dadt_190 + uxb_upw_190 + fres__mod__magnetic
}
}
if(AC_limp_alpha__mod__magnetic) {
if (abs(AC_z__mod__cdata[AC_n__mod__cdata-1])<=AC_imp_halpha__mod__magnetic/2) {
dadt_190 = dadt_190+AC_imp_alpha0__mod__magnetic*sin(AC_pi__mod__cparam*AC_z__mod__cdata[AC_n__mod__cdata-1]/AC_imp_halpha__mod__magnetic)*ac_transformed_pencil_bb
}
else {
dadt_190 = dadt_190+sign(AC_imp_alpha0__mod__magnetic,AC_z__mod__cdata[AC_n__mod__cdata-1])*exp(-((2*AC_z__mod__cdata[AC_n__mod__cdata-1]-sign(AC_imp_halpha__mod__magnetic,AC_z__mod__cdata[AC_n__mod__cdata-1]))/AC_imp_halpha__mod__magnetic)*((2*AC_z__mod__cdata[AC_n__mod__cdata-1]-sign(AC_imp_halpha__mod__magnetic,AC_z__mod__cdata[AC_n__mod__cdata-1]))/AC_imp_halpha__mod__magnetic))*ac_transformed_pencil_bb
}
}
if (AC_hall_term__mod__magnetic!=0.0) {
if(AC_string_enum_ihall_term__mod__magnetic == AC_string_enum_const_string__mod__cparam) {
hall_term__190=AC_hall_term__mod__magnetic
}
else if(AC_string_enum_ihall_term__mod__magnetic == AC_string_enum_tzdep_string__mod__cparam) {
hall_term__190=AC_hall_term__mod__magnetic*pow(max(AC_t__mod__cdata,AC_hall_tdep_t0__mod__magnetic),AC_hall_tdep_exponent__mod__magnetic)
}
else if(AC_string_enum_ihall_term__mod__magnetic == AC_string_enum_zzdep_string__mod__cparam) {
hall_term__190=AC_hall_term__mod__magnetic/pow((1.-(AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_xyz1__mod__cdata.z)/AC_hhall__mod__magnetic),AC_hall_zdep_exponent__mod__magnetic)
}
dadt_190=dadt_190-hall_term__190*ac_transformed_pencil_jxb
if (AC_lfirst__mod__cdata && AC_ldt__mod__cdata) {
advec_hall_190=sum(abs(ac_transformed_pencil_uu-hall_term__190*ac_transformed_pencil_jj)*dline_1__mod__cdata)
advec2__mod__cdata=advec2__mod__cdata+advec_hall_190*advec_hall_190
}
}
if (AC_battery_term__mod__magnetic!=0.0) {
dadt_190 = dadt_190-AC_battery_term__mod__magnetic*ac_transformed_pencil_fpres
}
if (AC_lambipolar_strong_coupling__mod__magnetic && AC_tauad__mod__magnetic!=0.0) {
dadt_190=dadt_190+AC_tauad__mod__magnetic*ac_transformed_pencil_jxbxb
eta_total__mod__magnetic = eta_total__mod__magnetic + AC_tauad__mod__magnetic*AC_mu01__mod__cdata*ac_transformed_pencil_b2
}
if (AC_lmagneto_friction__mod__magnetic && (!AC_lhydro__mod__cparam) && AC_numag__mod__magnetic!=0.0) {
tmp1_190=AC_mu01__mod__cdata/(AC_numag__mod__magnetic*(AC_b0_magfric__mod__magnetic/AC_unit_magnetic__mod__cdata*AC_unit_magnetic__mod__cdata+ac_transformed_pencil_b2))
dadt_190.x = dadt_190.x + ac_transformed_pencil_jxbxb.x*tmp1_190
dadt_190.y = dadt_190.y + ac_transformed_pencil_jxbxb.y*tmp1_190
dadt_190.z = dadt_190.z + ac_transformed_pencil_jxbxb.z*tmp1_190
if (! AC_linduction__mod__magnetic) {
dadt_190 = dadt_190 + fres__mod__magnetic
}
}
if (AC_height_eta__mod__magnetic!=0.0) {
if (AC_lhalox__mod__magnetic) {
tmp_190 = (AC_x__mod__cdata[AC_nx__mod__cparam-1]/AC_height_eta__mod__magnetic)*(AC_x__mod__cdata[AC_nx__mod__cparam-1]/AC_height_eta__mod__magnetic)
eta_out1_190 = AC_eta_out__mod__magnetic*(1.0-exp(-tmp_190*tmp_190*tmp_190*tmp_190*tmp_190*tmp_190/max(1.0-tmp_190,1.0e-5)))-AC_eta__mod__magnetic
}
else {
z_182_190 = abs((AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_height_eta__mod__magnetic)/AC_eta_zwidth__mod__magnetic)
t_182_190 = 1.0 / ( 1.0 + 0.5 * z_182_190 )
dumerfc_182_190 =  t_182_190 * exp(-z_182_190 * z_182_190 - 1.26551223 + t_182_190 *         ( 1.00002368 + t_182_190 * ( 0.37409196 + t_182_190 *            ( 0.09678418 + t_182_190 * (-0.18628806 + t_182_190 *            ( 0.27886807 + t_182_190 * (-1.13520398 + t_182_190 *            ( 1.48851587 + t_182_190 * (-0.82215223 + t_182_190 * 0.17087277 )))))))))
if ((AC_z__mod__cdata[AC_n__mod__cdata-1]-AC_height_eta__mod__magnetic)/AC_eta_zwidth__mod__magnetic<0.0) {
dumerfc_182_190 = 2.0 - dumerfc_182_190
}
erfunc_return_value_182_190 = 1.0 - dumerfc_182_190
eta_out1_190=(AC_eta_out__mod__magnetic-AC_eta__mod__magnetic)*0.5*(1.+erfunc_return_value_182_190)
}
dadt_190 = dadt_190-(eta_out1_190*AC_mu0__mod__cdata)*ac_transformed_pencil_jj
eta_total__mod__magnetic = eta_total__mod__magnetic + eta_out1_190*AC_mu0__mod__cdata
}
if (AC_ekman_friction_aa__mod__magnetic!=0) {
DF_AVEC=DF_AVEC-AC_ekman_friction_aa__mod__magnetic*ac_transformed_pencil_aa
}
if (AC_lforcing_cont_aa__mod__magnetic) {
dadt_190=dadt_190+ AC_ampl_fcont_aa__mod__magnetic*ac_transformed_pencil_fcont[AC_iforcing_cont_aa__mod__magnetic-1]
}
if (AC_z__mod__cdata[AC_n__mod__cdata-1]>AC_zgrav__mod__gravity) {
scl_184_190=1./AC_tau_aa_exterior__mod__magnetic
DF_AX=DF_AX-scl_184_190*value(F_AX)
DF_AY=DF_AY-scl_184_190*value(F_AY)
DF_AZ=DF_AZ-scl_184_190*value(F_AZ)
}
if (AC_tau_relprof__mod__magnetic!=0.0) {
if (AC_la_relprof_global__mod__magnetic) {
dadt_190= dadt_190-(ac_transformed_pencil_aa-value(F_GLOBAL_EXT_AVEC))*AC_tau_relprof1__mod__magnetic
}
else {
dadt_190.x= dadt_190.x-(ac_transformed_pencil_aa.x-AC_a_relprof__mod__magnetic[vertexIdx.x-NGHOST_VAL][AC_m__mod__cdata-AC_m1__mod__cparam+1-1][AC_n__mod__cdata-AC_n1__mod__cparam+1-1][1-1])*AC_tau_relprof1__mod__magnetic
dadt_190.y= dadt_190.y-(ac_transformed_pencil_aa.x-AC_a_relprof__mod__magnetic[vertexIdx.x-NGHOST_VAL][AC_m__mod__cdata-AC_m1__mod__cparam+1-1][AC_n__mod__cdata-AC_n1__mod__cparam+1-1][2-1])*AC_tau_relprof1__mod__magnetic
dadt_190.z= dadt_190.z-(ac_transformed_pencil_aa.x-AC_a_relprof__mod__magnetic[vertexIdx.x-NGHOST_VAL][AC_m__mod__cdata-AC_m1__mod__cparam+1-1][AC_n__mod__cdata-AC_n1__mod__cparam+1-1][3-1])*AC_tau_relprof1__mod__magnetic
}
}
ju_188_190=1+AC_iaa__mod__cdata-1
if(AC_string_enum_borderaa__mod__magnetic.x == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderaa__mod__magnetic.x == AC_string_enum_0_string__mod__cparam) {
f_target_188_190.x=0.
}
else if(AC_string_enum_borderaa__mod__magnetic.x == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderaa__mod__magnetic.x == AC_string_enum_nothing_string__mod__cparam) {
}
ju_188_190=2+AC_iaa__mod__cdata-1
if(AC_string_enum_borderaa__mod__magnetic.y == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderaa__mod__magnetic.y == AC_string_enum_0_string__mod__cparam) {
f_target_188_190.y=0.
}
else if(AC_string_enum_borderaa__mod__magnetic.y == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderaa__mod__magnetic.y == AC_string_enum_nothing_string__mod__cparam) {
}
ju_188_190=3+AC_iaa__mod__cdata-1
if(AC_string_enum_borderaa__mod__magnetic.z == AC_string_enum_zero_string__mod__cparam || AC_string_enum_borderaa__mod__magnetic.z == AC_string_enum_0_string__mod__cparam) {
f_target_188_190.z=0.
}
else if(AC_string_enum_borderaa__mod__magnetic.z == AC_string_enum_initialzcondition_string__mod__cparam) {
}
else if(AC_string_enum_borderaa__mod__magnetic.z == AC_string_enum_nothing_string__mod__cparam) {
}
if (AC_lee_as_aux__mod__magnetic) {
DF_EVEC=-dadt_190
}
if (AC_lbb_sph_as_aux__mod__magnetic && AC_lsphere_in_a_box__mod__cdata) {
DF_BB_SPHR = ac_transformed_pencil_bb.x*ac_transformed_pencil_evr.x+ac_transformed_pencil_bb.y*ac_transformed_pencil_evr.y+ac_transformed_pencil_bb.z*ac_transformed_pencil_evr.z
DF_BB_SPHT = ac_transformed_pencil_bb.x*ac_transformed_pencil_evth.x+ac_transformed_pencil_bb.y*ac_transformed_pencil_evth.y+ac_transformed_pencil_bb.z*ac_transformed_pencil_evth.z
DF_BB_SPHP = ac_transformed_pencil_bb.x*ac_transformed_pencil_phix+ac_transformed_pencil_bb.y*ac_transformed_pencil_phiy
}
DF_AVEC=DF_AVEC+dadt_190
}
if (AC_lanelastic__mod__cparam) {
DF_RHX   = ac_transformed_pencil_rho*DF_UX
DF_RHY = ac_transformed_pencil_rho*DF_UY
DF_RHZ = ac_transformed_pencil_rho*DF_UZ
DF_UVEC = df_iuu_pencil + DF_UVEC
}
headtt__mod__cdata=false
lfirstpoint__mod__cdata=false
}

#include "taskgraph.h"
