// automatically generated; do not edit!

run_const int AC_lpressuregradient_gas // from hydro
#define lpressuregradient_gas AC_lpressuregradient_gas
run_const int AC_lupw_uu // from hydro
#define lupw_uu AC_lupw_uu
run_const int AC_ladvection_velocity // from hydro
#define ladvection_velocity AC_ladvection_velocity

run_const int AC_ldiff_shock // from density
#define ldiff_shock AC_ldiff_shock
run_const real AC_diffrho_shock // from density
#define diffrho_shock AC_diffrho_shock
run_const int AC_ldiff_hyper3lnrho // from density
#define ldiff_hyper3lnrho AC_ldiff_hyper3lnrho
run_const real AC_diffrho_hyper3 // from density
#define diffrho_hyper3 AC_diffrho_hyper3
run_const int AC_lupw_lnrho // from density
#define lupw_lnrho AC_lupw_lnrho

run_const real AC_eta // from magnetic
#define eta AC_eta
run_const real AC_eta_hyper2 // from magnetic
#define eta_hyper2 AC_eta_hyper2
run_const real AC_eta_hyper3 // from magnetic
#define eta_hyper3 AC_eta_hyper3
run_const int AC_lresi_eta_const // from magnetic
#define lresi_eta_const AC_lresi_eta_const
run_const int AC_lresi_hyper2 // from magnetic
#define lresi_hyper2 AC_lresi_hyper2
run_const int AC_lresi_hyper3 // from magnetic
#define lresi_hyper3 AC_lresi_hyper3
run_const int AC_lupw_aa // from magnetic
#define lupw_aa AC_lupw_aa
run_const int AC_llorentzforce // from magnetic
#define llorentzforce AC_llorentzforce
run_const int AC_linduction // from magnetic
#define linduction AC_linduction

run_const real AC_cs20 // from equationofstate
#define cs20 AC_cs20
run_const real AC_gamma // from equationofstate
#define gamma AC_gamma
run_const real AC_cv // from equationofstate
#define cv AC_cv
run_const real AC_cp // from equationofstate
#define cp AC_cp
run_const real AC_lnrho0 // from equationofstate
#define lnrho0 AC_lnrho0
run_const real AC_lnTT0 // from equationofstate
#define lnTT0 AC_lnTT0
run_const real AC_gamma_m1 // from equationofstate
#define gamma_m1 AC_gamma_m1
run_const real AC_gamma1 // from equationofstate
#define gamma1 AC_gamma1
run_const real AC_cv1 // from equationofstate
#define cv1 AC_cv1
run_const real AC_cs2bot // from equationofstate
#define cs2bot AC_cs2bot
run_const real AC_cs2top // from equationofstate
#define cs2top AC_cs2top
run_const int AC_leos_isothermal // from equationofstate
#define leos_isothermal AC_leos_isothermal

run_const real AC_nu // from viscosity
#define nu AC_nu
run_const real AC_zeta // from viscosity
#define zeta AC_zeta
run_const real AC_nu_hyper3 // from viscosity
#define nu_hyper3 AC_nu_hyper3
run_const real AC_nu_shock // from viscosity
#define nu_shock AC_nu_shock
run_const int AC_lvisc_nu_const // from viscosity
#define lvisc_nu_const AC_lvisc_nu_const
run_const int AC_lvisc_hyper3_nu_const // from viscosity
#define lvisc_hyper3_nu_const AC_lvisc_hyper3_nu_const
run_const int AC_lvisc_nu_shock // from viscosity
#define lvisc_nu_shock AC_lvisc_nu_shock
run_const real AC_nu_hyper2 // from viscosity
#define nu_hyper2 AC_nu_hyper2

run_const int AC_ncoarse // from run_module
#define ncoarse AC_ncoarse
run_const int AC_lcoarse // from run_module
#define lcoarse AC_lcoarse
run_const int AC_lcoarse_mn // from run_module
#define lcoarse_mn AC_lcoarse_mn
run_const real AC_tslice // from run_module
#define tslice AC_tslice
run_const real AC_eps_rkf // from run_module
#define eps_rkf AC_eps_rkf
run_const real AC_eps_stiff // from run_module
#define eps_stiff AC_eps_stiff
run_const real AC_eps_rkf0 // from run_module
#define eps_rkf0 AC_eps_rkf0
run_const real AC_dsound // from run_module
#define dsound AC_dsound
run_const real AC_tsound // from run_module
#define tsound AC_tsound
run_const real AC_soundeps // from run_module
#define soundeps AC_soundeps
run_const int AC_lfix_unit_std // from run_module
#define lfix_unit_std AC_lfix_unit_std
run_const int AC_m // from run_module
#define m AC_m
run_const int AC_n // from run_module
#define n AC_n
run_const int AC_lfirstpoint // from run_module
#define lfirstpoint AC_lfirstpoint
gmem real AC_dxyz_2[AC_nx] // from run_module
#define dxyz_2 AC_dxyz_2
gmem real AC_dxyz_4[AC_nx] // from run_module
#define dxyz_4 AC_dxyz_4
gmem real AC_dxyz_6[AC_nx] // from run_module
#define dxyz_6 AC_dxyz_6
gmem real AC_dvol[AC_nx] // from run_module
#define dvol AC_dvol
run_const real AC_dvol_glob // from run_module
#define dvol_glob AC_dvol_glob

gmem real AC_dx_1[AC_mx] // from run_module
#define dx_1 AC_dx_1
gmem real AC_dx2[AC_mx] // from run_module
#define dx2 AC_dx2
gmem real AC_dx_tilde[AC_mx] // from run_module
#define dx_tilde AC_dx_tilde
gmem real AC_xprim[AC_mx] // from run_module
#define xprim AC_xprim
gmem real AC_dvol_x[AC_mx] // from run_module
#define dvol_x AC_dvol_x
gmem real AC_dvol1_x[AC_mx] // from run_module
#define dvol1_x AC_dvol1_x

gmem real AC_dy_1[AC_my] // from run_module
#define dy_1 AC_dy_1
gmem real AC_dy2[AC_my] // from run_module
#define dy2 AC_dy2
gmem real AC_dy_tilde[AC_my] // from run_module
#define dy_tilde AC_dy_tilde
gmem real AC_yprim[AC_my] // from run_module
#define yprim AC_yprim
gmem real AC_dvol_y[AC_my] // from run_module
#define dvol_y AC_dvol_y
gmem real AC_dvol1_y[AC_my] // from run_module
#define dvol1_y AC_dvol1_y

gmem real AC_dz_1[AC_mz] // from run_module
#define dz_1 AC_dz_1
gmem real AC_dz2[AC_mz] // from run_module
#define dz2 AC_dz2
gmem real AC_dz_tilde[AC_mz] // from run_module
#define dz_tilde AC_dz_tilde
gmem real AC_zprim[AC_mz] // from run_module
#define zprim AC_zprim
gmem real AC_dvol_z[AC_mz] // from run_module
#define dvol_z AC_dvol_z
gmem real AC_dvol1_z[AC_mz] // from run_module
#define dvol1_z AC_dvol1_z
run_const real AC_dx // from run_module
#define dx AC_dx
run_const real AC_dy // from run_module
#define dy AC_dy
run_const real AC_dz // from run_module
#define dz AC_dz
run_const real AC_dxmin // from run_module
#define dxmin AC_dxmin
run_const real AC_dxmax // from run_module
#define dxmax AC_dxmax
gmem real AC_xgrid[AC_nxgrid] // from run_module
#define xgrid AC_xgrid
gmem real AC_dx1grid[AC_nxgrid] // from run_module
#define dx1grid AC_dx1grid
gmem real AC_dxtgrid[AC_nxgrid] // from run_module
#define dxtgrid AC_dxtgrid
gmem real AC_ygrid[AC_nygrid] // from run_module
#define ygrid AC_ygrid
gmem real AC_dy1grid[AC_nygrid] // from run_module
#define dy1grid AC_dy1grid
gmem real AC_dytgrid[AC_nygrid] // from run_module
#define dytgrid AC_dytgrid
gmem real AC_zgrid[AC_nzgrid] // from run_module
#define zgrid AC_zgrid
gmem real AC_dz1grid[AC_nzgrid] // from run_module
#define dz1grid AC_dz1grid
gmem real AC_dztgrid[AC_nzgrid] // from run_module
#define dztgrid AC_dztgrid
gmem real AC_xglobal[AC_mxgrid] // from run_module
#define xglobal AC_xglobal
gmem real AC_yglobal[AC_mygrid] // from run_module
#define yglobal AC_yglobal
gmem real AC_zglobal[AC_mzgrid] // from run_module
#define zglobal AC_zglobal
run_const real AC_kx_nyq // from run_module
#define kx_nyq AC_kx_nyq
run_const real AC_ky_nyq // from run_module
#define ky_nyq AC_ky_nyq
run_const real AC_kz_nyq // from run_module
#define kz_nyq AC_kz_nyq
run_const int AC_lcartesian_coords // from run_module
#define lcartesian_coords AC_lcartesian_coords
run_const int AC_lspherical_coords // from run_module
#define lspherical_coords AC_lspherical_coords
run_const int AC_lcylindrical_coords // from run_module
#define lcylindrical_coords AC_lcylindrical_coords
run_const int AC_lpipe_coords // from run_module
#define lpipe_coords AC_lpipe_coords
run_const int AC_lsphere_in_a_box // from run_module
#define lsphere_in_a_box AC_lsphere_in_a_box
run_const int AC_lcylinder_in_a_box // from run_module
#define lcylinder_in_a_box AC_lcylinder_in_a_box
run_const int AC_luse_latitude // from run_module
#define luse_latitude AC_luse_latitude
run_const int AC_luse_oldgrid // from run_module
#define luse_oldgrid AC_luse_oldgrid
run_const int AC_luse_xyz1 // from run_module
#define luse_xyz1 AC_luse_xyz1
run_const int AC_lcylindrical_gravity // from run_module
#define lcylindrical_gravity AC_lcylindrical_gravity
run_const int AC_luniform_z_mesh_aspect_ratio // from run_module
#define luniform_z_mesh_aspect_ratio AC_luniform_z_mesh_aspect_ratio
run_const int AC_lconcurrent // from run_module
#define lconcurrent AC_lconcurrent
run_const int AC_tag_foreign // from run_module
#define tag_foreign AC_tag_foreign
run_const int AC_lforeign // from run_module
#define lforeign AC_lforeign
run_const int AC_lforeign_comm_nblckg // from run_module
#define lforeign_comm_nblckg AC_lforeign_comm_nblckg
run_const int AC_lyinyang // from run_module
#define lyinyang AC_lyinyang
run_const int AC_lyang // from run_module
#define lyang AC_lyang
run_const int AC_lcutoff_corners // from run_module
#define lcutoff_corners AC_lcutoff_corners
run_const int AC_iyinyang_intpol_type // from run_module
#define iyinyang_intpol_type AC_iyinyang_intpol_type
run_const int AC_nzgrid_eff // from run_module
#define nzgrid_eff AC_nzgrid_eff
run_const real AC_yy_biquad_weights[4] // from run_module
#define yy_biquad_weights AC_yy_biquad_weights
run_const int AC_nycut // from run_module
#define nycut AC_nycut
run_const int AC_nzcut // from run_module
#define nzcut AC_nzcut
run_const real AC_rel_dang // from run_module
#define rel_dang AC_rel_dang
run_const int AC_lcubed_sphere // from run_module
#define lcubed_sphere AC_lcubed_sphere
run_const real AC_drcyl // from run_module
#define drcyl AC_drcyl
run_const real AC_dsurfxy // from run_module
#define dsurfxy AC_dsurfxy
run_const real AC_dsurfyz // from run_module
#define dsurfyz AC_dsurfyz
run_const real AC_dsurfzx // from run_module
#define dsurfzx AC_dsurfzx
gmem real AC_r_mn[AC_nx] // from run_module
#define r_mn AC_r_mn
gmem real AC_r1_mn[AC_nx] // from run_module
#define r1_mn AC_r1_mn
gmem real AC_r2_mn[AC_nx] // from run_module
#define r2_mn AC_r2_mn
gmem real AC_r2_weight[AC_nx] // from run_module
#define r2_weight AC_r2_weight
gmem real AC_sinth[AC_my] // from run_module
#define sinth AC_sinth
gmem real AC_sin1th[AC_my] // from run_module
#define sin1th AC_sin1th
gmem real AC_sin2th[AC_my] // from run_module
#define sin2th AC_sin2th
gmem real AC_costh[AC_my] // from run_module
#define costh AC_costh
gmem real AC_cotth[AC_my] // from run_module
#define cotth AC_cotth
gmem real AC_sinth_weight[AC_my] // from run_module
#define sinth_weight AC_sinth_weight
gmem real AC_sinph[AC_mz] // from run_module
#define sinph AC_sinph
gmem real AC_cosph[AC_mz] // from run_module
#define cosph AC_cosph
gmem real AC_cos1th[AC_my] // from run_module
#define cos1th AC_cos1th
gmem real AC_tanth[AC_my] // from run_module
#define tanth AC_tanth
gmem real AC_sinth_weight_across_proc[AC_nygrid] // from run_module
#define sinth_weight_across_proc AC_sinth_weight_across_proc
gmem real AC_rcyl_mn[AC_nx] // from run_module
#define rcyl_mn AC_rcyl_mn
gmem real AC_rcyl_mn1[AC_nx] // from run_module
#define rcyl_mn1 AC_rcyl_mn1
gmem real AC_rcyl_mn2[AC_nx] // from run_module
#define rcyl_mn2 AC_rcyl_mn2
gmem real AC_rcyl_weight[AC_nx] // from run_module
#define rcyl_weight AC_rcyl_weight
gmem real AC_glncrosssec[AC_nx] // from run_module
#define glncrosssec AC_glncrosssec
run_const real AC_rcyl[nrcyl] // from run_module
#define rcyl AC_rcyl
gmem real AC_x12[AC_mx] // from run_module
#define x12 AC_x12
gmem real AC_y12[AC_my] // from run_module
#define y12 AC_y12
gmem real AC_z12[AC_mz] // from run_module
#define z12 AC_z12
run_const real AC_coeff_grid[3] // from run_module
#define coeff_grid AC_coeff_grid
run_const real AC_dxi_fact[3] // from run_module
#define dxi_fact AC_dxi_fact
run_const real AC_trans_width[3] // from run_module
#define trans_width AC_trans_width
run_const real AC_zeta_grid0 // from run_module
#define zeta_grid0 AC_zeta_grid0
run_const real AC_xbot_slice // from run_module
#define xbot_slice AC_xbot_slice
run_const real AC_xtop_slice // from run_module
#define xtop_slice AC_xtop_slice
run_const real AC_ybot_slice // from run_module
#define ybot_slice AC_ybot_slice
run_const real AC_ytop_slice // from run_module
#define ytop_slice AC_ytop_slice
run_const real AC_zbot_slice // from run_module
#define zbot_slice AC_zbot_slice
run_const real AC_ztop_slice // from run_module
#define ztop_slice AC_ztop_slice
run_const real AC_r_rslice // from run_module
#define r_rslice AC_r_rslice
run_const int AC_nth_rslice // from run_module
#define nth_rslice AC_nth_rslice
run_const int AC_nph_rslice // from run_module
#define nph_rslice AC_nph_rslice
run_const real AC_glncrosssec0 // from run_module
#define glncrosssec0 AC_glncrosssec0
run_const real AC_crosssec_x1 // from run_module
#define crosssec_x1 AC_crosssec_x1
run_const real AC_crosssec_x2 // from run_module
#define crosssec_x2 AC_crosssec_x2
run_const real AC_crosssec_w // from run_module
#define crosssec_w AC_crosssec_w
run_const int AC_lignore_nonequi // from run_module
#define lignore_nonequi AC_lignore_nonequi
run_const int AC_lcart_equi // from run_module
#define lcart_equi AC_lcart_equi
run_const int AC_nghost_read_fewer // from run_module
#define nghost_read_fewer AC_nghost_read_fewer
run_const int AC_lall_onesided // from run_module
#define lall_onesided AC_lall_onesided
run_const real AC_lxyz[3] // from run_module
#define lxyz AC_lxyz
run_const real AC_xyz0[3] // from run_module
#define xyz0 AC_xyz0
run_const real AC_xyz1[3] // from run_module
#define xyz1 AC_xyz1
run_const real AC_xyz_star[3] // from run_module
#define xyz_star AC_xyz_star
run_const real AC_lxyz_loc[3] // from run_module
#define lxyz_loc AC_lxyz_loc
run_const real AC_xyz0_loc[3] // from run_module
#define xyz0_loc AC_xyz0_loc
run_const real AC_xyz1_loc[3] // from run_module
#define xyz1_loc AC_xyz1_loc
run_const real AC_r_int // from run_module
#define r_int AC_r_int
run_const real AC_r_ext // from run_module
#define r_ext AC_r_ext
run_const real AC_r_int_border // from run_module
#define r_int_border AC_r_int_border
run_const real AC_r_ext_border // from run_module
#define r_ext_border AC_r_ext_border
run_const real AC_r_ref // from run_module
#define r_ref AC_r_ref
run_const real AC_rsmooth // from run_module
#define rsmooth AC_rsmooth
run_const real AC_box_volume // from run_module
#define box_volume AC_box_volume
run_const real AC_area_xy // from run_module
#define area_xy AC_area_xy
run_const real AC_area_yz // from run_module
#define area_yz AC_area_yz
run_const real AC_area_xz // from run_module
#define area_xz AC_area_xz
run_const real AC_mu0 // from run_module
#define mu0 AC_mu0
run_const real AC_mu01 // from run_module
#define mu01 AC_mu01
run_const int AC_lfirst // from run_module
#define lfirst AC_lfirst
run_const int AC_llast // from run_module
#define llast AC_llast
run_const int AC_ldt_paronly // from run_module
#define ldt_paronly AC_ldt_paronly
run_const int AC_ldt // from run_module
#define ldt AC_ldt
run_const real AC_tmax // from run_module
#define tmax AC_tmax
run_const real AC_tstart // from run_module
#define tstart AC_tstart
run_const real AC_max_walltime // from run_module
#define max_walltime AC_max_walltime
run_const real AC_dt_incr // from run_module
#define dt_incr AC_dt_incr
run_const real AC_dt0 // from run_module
#define dt0 AC_dt0
run_const real AC_cdt // from run_module
#define cdt AC_cdt
run_const real AC_cdts // from run_module
#define cdts AC_cdts
run_const real AC_cdtr // from run_module
#define cdtr AC_cdtr
run_const real AC_cdtc // from run_module
#define cdtc AC_cdtc
run_const real AC_cdt_poly // from run_module
#define cdt_poly AC_cdt_poly
run_const real AC_cdtv // from run_module
#define cdtv AC_cdtv
run_const real AC_cdtv2 // from run_module
#define cdtv2 AC_cdtv2
run_const real AC_cdtv3 // from run_module
#define cdtv3 AC_cdtv3
run_const real AC_cdtsrc // from run_module
#define cdtsrc AC_cdtsrc
run_const real AC_cdtf // from run_module
#define cdtf AC_cdtf
run_const real AC_ddt // from run_module
#define ddt AC_ddt
run_const real AC_dtinc // from run_module
#define dtinc AC_dtinc
run_const real AC_dtdec // from run_module
#define dtdec AC_dtdec
run_const real AC_dtmin // from run_module
#define dtmin AC_dtmin
run_const real AC_dtmax // from run_module
#define dtmax AC_dtmax
run_const real AC_dt_epsi // from run_module
#define dt_epsi AC_dt_epsi
run_const real AC_dt_ratio // from run_module
#define dt_ratio AC_dt_ratio
run_const real AC_nu_sts // from run_module
#define nu_sts AC_nu_sts
run_const int AC_permute_sts // from run_module
#define permute_sts AC_permute_sts
run_const int AC_ireset_tstart // from run_module
#define ireset_tstart AC_ireset_tstart
run_const int AC_nt // from run_module
#define nt AC_nt
run_const int AC_it // from run_module
#define it AC_it
run_const int AC_itorder // from run_module
#define itorder AC_itorder
run_const int AC_itsub // from run_module
#define itsub AC_itsub
run_const int AC_it_timing // from run_module
#define it_timing AC_it_timing
run_const int AC_it_rmv // from run_module
#define it_rmv AC_it_rmv
run_const int AC_nproc_comm // from run_module
#define nproc_comm AC_nproc_comm
run_const int AC_ix // from run_module
#define ix AC_ix
run_const int AC_iy // from run_module
#define iy AC_iy
run_const int AC_iy2 // from run_module
#define iy2 AC_iy2
run_const int AC_iz // from run_module
#define iz AC_iz
run_const int AC_iz2 // from run_module
#define iz2 AC_iz2
run_const int AC_iz3 // from run_module
#define iz3 AC_iz3
run_const int AC_iz4 // from run_module
#define iz4 AC_iz4
run_const int AC_ix_loc // from run_module
#define ix_loc AC_ix_loc
run_const int AC_iy_loc // from run_module
#define iy_loc AC_iy_loc
run_const int AC_iy2_loc // from run_module
#define iy2_loc AC_iy2_loc
run_const int AC_iz_loc // from run_module
#define iz_loc AC_iz_loc
run_const int AC_iz2_loc // from run_module
#define iz2_loc AC_iz2_loc
run_const int AC_iz3_loc // from run_module
#define iz3_loc AC_iz3_loc
run_const int AC_iz4_loc // from run_module
#define iz4_loc AC_iz4_loc
run_const int AC_iproc // from run_module
#define iproc AC_iproc
run_const int AC_ipx // from run_module
#define ipx AC_ipx
run_const int AC_ipy // from run_module
#define ipy AC_ipy
run_const int AC_ipz // from run_module
#define ipz AC_ipz
run_const int AC_iproc_world // from run_module
#define iproc_world AC_iproc_world
run_const int AC_ipatch // from run_module
#define ipatch AC_ipatch
run_const int AC_lprocz_slowest // from run_module
#define lprocz_slowest AC_lprocz_slowest
run_const int AC_lzorder // from run_module
#define lzorder AC_lzorder
run_const int AC_lmorton_curve // from run_module
#define lmorton_curve AC_lmorton_curve
run_const int AC_xlneigh // from run_module
#define xlneigh AC_xlneigh
run_const int AC_ylneigh // from run_module
#define ylneigh AC_ylneigh
run_const int AC_zlneigh // from run_module
#define zlneigh AC_zlneigh
run_const int AC_xuneigh // from run_module
#define xuneigh AC_xuneigh
run_const int AC_yuneigh // from run_module
#define yuneigh AC_yuneigh
run_const int AC_zuneigh // from run_module
#define zuneigh AC_zuneigh
run_const int AC_poleneigh // from run_module
#define poleneigh AC_poleneigh
run_const int AC_nprocx_node // from run_module
#define nprocx_node AC_nprocx_node
run_const int AC_nprocy_node // from run_module
#define nprocy_node AC_nprocy_node
run_const int AC_nprocz_node // from run_module
#define nprocz_node AC_nprocz_node
run_const int AC_num_after_timestep // from run_module
#define num_after_timestep AC_num_after_timestep
run_const int AC_ighosts_updated // from run_module
#define ighosts_updated AC_ighosts_updated
run_const real AC_x0 // from run_module
#define x0 AC_x0
run_const real AC_y0 // from run_module
#define y0 AC_y0
run_const real AC_z0 // from run_module
#define z0 AC_z0
run_const real AC_lx // from run_module
#define lx AC_lx
run_const real AC_ly // from run_module
#define ly AC_ly
run_const real AC_lz // from run_module
#define lz AC_lz
run_const real AC_wav1 // from run_module
#define wav1 AC_wav1
run_const real AC_wav1z // from run_module
#define wav1z AC_wav1z
run_const int AC_lini_t_eq_zero // from run_module
#define lini_t_eq_zero AC_lini_t_eq_zero
run_const int AC_lini_t_eq_zero_once // from run_module
#define lini_t_eq_zero_once AC_lini_t_eq_zero_once
gmem real AC_reac_chem[AC_nx] // from run_module
#define reac_chem AC_reac_chem
gmem real AC_reac_dust[AC_nx] // from run_module
#define reac_dust AC_reac_dust
run_const real AC_trelax_poly // from run_module
#define trelax_poly AC_trelax_poly
run_const real AC_reac_pchem // from run_module
#define reac_pchem AC_reac_pchem
run_const real AC_alpha_ts[5] // from run_module
#define alpha_ts AC_alpha_ts
run_const real AC_beta_ts[5] // from run_module
#define beta_ts AC_beta_ts
run_const real AC_dt_beta_ts[5] // from run_module
#define dt_beta_ts AC_dt_beta_ts
run_const int AC_lfractional_tstep_advance // from run_module
#define lfractional_tstep_advance AC_lfractional_tstep_advance
run_const int AC_lfractional_tstep_negative // from run_module
#define lfractional_tstep_negative AC_lfractional_tstep_negative
run_const int AC_lmaxadvec_sum // from run_module
#define lmaxadvec_sum AC_lmaxadvec_sum
run_const int AC_old_cdtv // from run_module
#define old_cdtv AC_old_cdtv
run_const int AC_leps_fixed // from run_module
#define leps_fixed AC_leps_fixed
run_const int AC_lmaximal_cdtv // from run_module
#define lmaximal_cdtv AC_lmaximal_cdtv
run_const int AC_lmaximal_cdt // from run_module
#define lmaximal_cdt AC_lmaximal_cdt
run_const int AC_llsode // from run_module
#define llsode AC_llsode
run_const int AC_lstep1 // from run_module
#define lstep1 AC_lstep1
run_const int AC_lchemonly // from run_module
#define lchemonly AC_lchemonly
run_const int AC_lsplit_second // from run_module
#define lsplit_second AC_lsplit_second
run_const int AC_lsnap // from run_module
#define lsnap AC_lsnap
run_const int AC_lsnap_down // from run_module
#define lsnap_down AC_lsnap_down
run_const int AC_lspec // from run_module
#define lspec AC_lspec
run_const int AC_lspec_start // from run_module
#define lspec_start AC_lspec_start
run_const int AC_lspec_at_tplusdt // from run_module
#define lspec_at_tplusdt AC_lspec_at_tplusdt
run_const real AC_dsnap // from run_module
#define dsnap AC_dsnap
run_const real AC_dsnap_down // from run_module
#define dsnap_down AC_dsnap_down
run_const real AC_d1davg // from run_module
#define d1davg AC_d1davg
run_const real AC_d2davg // from run_module
#define d2davg AC_d2davg
run_const real AC_dvid // from run_module
#define dvid AC_dvid
run_const real AC_dspec // from run_module
#define dspec AC_dspec
run_const real AC_dtracers // from run_module
#define dtracers AC_dtracers
run_const real AC_dfixed_points // from run_module
#define dfixed_points AC_dfixed_points
run_const real AC_crash_file_dtmin_factor // from run_module
#define crash_file_dtmin_factor AC_crash_file_dtmin_factor
run_const int AC_farray_smooth_width // from run_module
#define farray_smooth_width AC_farray_smooth_width
run_const int AC_isave // from run_module
#define isave AC_isave
run_const int AC_ialive // from run_module
#define ialive AC_ialive
run_const int AC_isaveglobal // from run_module
#define isaveglobal AC_isaveglobal
run_const int AC_nv1_capitalvar // from run_module
#define nv1_capitalvar AC_nv1_capitalvar
run_const int AC_lwrite_ts_hdf5 // from run_module
#define lwrite_ts_hdf5 AC_lwrite_ts_hdf5
run_const int AC_lsave // from run_module
#define lsave AC_lsave
run_const int AC_lread_aux // from run_module
#define lread_aux AC_lread_aux
run_const int AC_lwrite_aux // from run_module
#define lwrite_aux AC_lwrite_aux
run_const int AC_lwrite_dvar // from run_module
#define lwrite_dvar AC_lwrite_dvar
run_const int AC_lenforce_maux_check // from run_module
#define lenforce_maux_check AC_lenforce_maux_check
run_const int AC_lwrite_avg1d_binary // from run_module
#define lwrite_avg1d_binary AC_lwrite_avg1d_binary
run_const int AC_lread_oldsnap // from run_module
#define lread_oldsnap AC_lread_oldsnap
run_const int AC_lwrite_var_anyway // from run_module
#define lwrite_var_anyway AC_lwrite_var_anyway
run_const int AC_lwrite_last_powersnap // from run_module
#define lwrite_last_powersnap AC_lwrite_last_powersnap
run_const int AC_lwrite_fsum // from run_module
#define lwrite_fsum AC_lwrite_fsum
run_const int AC_lread_oldsnap_rho2lnrho // from run_module
#define lread_oldsnap_rho2lnrho AC_lread_oldsnap_rho2lnrho
run_const int AC_lread_oldsnap_nomag // from run_module
#define lread_oldsnap_nomag AC_lread_oldsnap_nomag
run_const int AC_lread_oldsnap_lnrho2rho // from run_module
#define lread_oldsnap_lnrho2rho AC_lread_oldsnap_lnrho2rho
run_const int AC_lread_oldsnap_noshear // from run_module
#define lread_oldsnap_noshear AC_lread_oldsnap_noshear
run_const int AC_lread_oldsnap_nohydro // from run_module
#define lread_oldsnap_nohydro AC_lread_oldsnap_nohydro
run_const int AC_lread_oldsnap_nohydro_nomu5 // from run_module
#define lread_oldsnap_nohydro_nomu5 AC_lread_oldsnap_nohydro_nomu5
run_const int AC_lread_oldsnap_onlya // from run_module
#define lread_oldsnap_onlya AC_lread_oldsnap_onlya
run_const int AC_lread_oldsnap_mskipvar // from run_module
#define lread_oldsnap_mskipvar AC_lread_oldsnap_mskipvar
run_const int AC_lread_oldsnap_nohydro_efield // from run_module
#define lread_oldsnap_nohydro_efield AC_lread_oldsnap_nohydro_efield
run_const int AC_lread_oldsnap_nohydro_ekfield // from run_module
#define lread_oldsnap_nohydro_ekfield AC_lread_oldsnap_nohydro_ekfield
run_const int AC_ldivu_perp // from run_module
#define ldivu_perp AC_ldivu_perp
run_const int AC_lread_oldsnap_nopscalar // from run_module
#define lread_oldsnap_nopscalar AC_lread_oldsnap_nopscalar
run_const int AC_lread_oldsnap_notestfield // from run_module
#define lread_oldsnap_notestfield AC_lread_oldsnap_notestfield
run_const int AC_lread_oldsnap_notestflow // from run_module
#define lread_oldsnap_notestflow AC_lread_oldsnap_notestflow
run_const int AC_lread_oldsnap_notestscalar // from run_module
#define lread_oldsnap_notestscalar AC_lread_oldsnap_notestscalar
run_const int AC_lread_oldsnap_noisothmhd // from run_module
#define lread_oldsnap_noisothmhd AC_lread_oldsnap_noisothmhd
run_const int AC_lread_oldsnap_nosink // from run_module
#define lread_oldsnap_nosink AC_lread_oldsnap_nosink
run_const int AC_lnamelist_error // from run_module
#define lnamelist_error AC_lnamelist_error
run_const int AC_ltolerate_namelist_errors // from run_module
#define ltolerate_namelist_errors AC_ltolerate_namelist_errors
run_const int AC_lparam_nml // from run_module
#define lparam_nml AC_lparam_nml
run_const int AC_lwrite_dim_again // from run_module
#define lwrite_dim_again AC_lwrite_dim_again
run_const int AC_allproc_print // from run_module
#define allproc_print AC_allproc_print
run_const int AC_lproc_print // from run_module
#define lproc_print AC_lproc_print
run_const int AC_lseparate_persist // from run_module
#define lseparate_persist AC_lseparate_persist
run_const int AC_ldistribute_persist // from run_module
#define ldistribute_persist AC_ldistribute_persist
run_const int AC_lpersist // from run_module
#define lpersist AC_lpersist
run_const int AC_lomit_add_data // from run_module
#define lomit_add_data AC_lomit_add_data
run_const int AC_save_lastsnap // from run_module
#define save_lastsnap AC_save_lastsnap
run_const int AC_noghost_for_isave // from run_module
#define noghost_for_isave AC_noghost_for_isave
run_const int AC_ltec // from run_module
#define ltec AC_ltec
run_const int AC_lformat // from run_module
#define lformat AC_lformat
run_const int AC_lread_less // from run_module
#define lread_less AC_lread_less
run_const int AC_lread_nogrid // from run_module
#define lread_nogrid AC_lread_nogrid
run_const int AC_lread_global // from run_module
#define lread_global AC_lread_global
run_const int AC_loutput_varn_at_exact_tsnap // from run_module
#define loutput_varn_at_exact_tsnap AC_loutput_varn_at_exact_tsnap
run_const int AC_ldirect_access // from run_module
#define ldirect_access AC_ldirect_access
run_const int AC_lread_from_other_prec // from run_module
#define lread_from_other_prec AC_lread_from_other_prec
run_const int AC_ldownsampl // from run_module
#define ldownsampl AC_ldownsampl
run_const int AC_ldownsampling // from run_module
#define ldownsampling AC_ldownsampling
run_const int AC_lrepair_snap // from run_module
#define lrepair_snap AC_lrepair_snap
run_const int AC_linterpol_on_repair // from run_module
#define linterpol_on_repair AC_linterpol_on_repair
run_const int AC_lastaroth_output // from run_module
#define lastaroth_output AC_lastaroth_output
run_const int AC_lzaver_on_input // from run_module
#define lzaver_on_input AC_lzaver_on_input
run_const int AC_lfatal_num_vector_369 // from run_module
#define lfatal_num_vector_369 AC_lfatal_num_vector_369
run_const int AC_lsmooth_farray // from run_module
#define lsmooth_farray AC_lsmooth_farray
run_const int AC_lread_scl_factor_file // from run_module
#define lread_scl_factor_file AC_lread_scl_factor_file
run_const int AC_lread_scl_factor_file_new // from run_module
#define lread_scl_factor_file_new AC_lread_scl_factor_file_new
run_const real AC_scl_factor_target // from run_module
#define scl_factor_target AC_scl_factor_target
run_const real AC_hp_target // from run_module
#define hp_target AC_hp_target
run_const real AC_appa_target // from run_module
#define appa_target AC_appa_target
run_const real AC_wweos_target // from run_module
#define wweos_target AC_wweos_target
run_const int AC_ip // from run_module
#define ip AC_ip
run_const real AC_omega // from run_module
#define omega AC_omega
run_const real AC_theta // from run_module
#define theta AC_theta
run_const real AC_phi // from run_module
#define phi AC_phi
run_const real AC_qshear // from run_module
#define qshear AC_qshear
run_const real AC_sshear // from run_module
#define sshear AC_sshear
run_const real AC_deltay // from run_module
#define deltay AC_deltay
run_const int AC_ldensity_nolog // from run_module
#define ldensity_nolog AC_ldensity_nolog
run_const int AC_lreference_state // from run_module
#define lreference_state AC_lreference_state
run_const int AC_lfullvar_in_slices // from run_module
#define lfullvar_in_slices AC_lfullvar_in_slices
run_const int AC_lsubstract_reference_state // from run_module
#define lsubstract_reference_state AC_lsubstract_reference_state
run_const int AC_ldensity_linearstart // from run_module
#define ldensity_linearstart AC_ldensity_linearstart
run_const int AC_lforcing_cont // from run_module
#define lforcing_cont AC_lforcing_cont
run_const int AC_lwrite_slices // from run_module
#define lwrite_slices AC_lwrite_slices
run_const int AC_lwrite_1daverages // from run_module
#define lwrite_1daverages AC_lwrite_1daverages
run_const int AC_lwrite_2daverages // from run_module
#define lwrite_2daverages AC_lwrite_2daverages
run_const int AC_lwrite_tracers // from run_module
#define lwrite_tracers AC_lwrite_tracers
run_const int AC_lwrite_fixed_points // from run_module
#define lwrite_fixed_points AC_lwrite_fixed_points
run_const int AC_lwrite_sound // from run_module
#define lwrite_sound AC_lwrite_sound
run_const int AC_lwrite_slice_xy2 // from run_module
#define lwrite_slice_xy2 AC_lwrite_slice_xy2
run_const int AC_lwrite_slice_xy // from run_module
#define lwrite_slice_xy AC_lwrite_slice_xy
run_const int AC_lwrite_slice_xz // from run_module
#define lwrite_slice_xz AC_lwrite_slice_xz
run_const int AC_lwrite_slice_yz // from run_module
#define lwrite_slice_yz AC_lwrite_slice_yz
run_const int AC_lwrite_slice_xy3 // from run_module
#define lwrite_slice_xy3 AC_lwrite_slice_xy3
run_const int AC_lwrite_slice_xy4 // from run_module
#define lwrite_slice_xy4 AC_lwrite_slice_xy4
run_const int AC_lwrite_slice_xz2 // from run_module
#define lwrite_slice_xz2 AC_lwrite_slice_xz2
run_const int AC_lwrite_slice_r // from run_module
#define lwrite_slice_r AC_lwrite_slice_r
run_const int AC_lgravx // from run_module
#define lgravx AC_lgravx
run_const int AC_lgravy // from run_module
#define lgravy AC_lgravy
run_const int AC_lgravz // from run_module
#define lgravz AC_lgravz
run_const int AC_lgravx_gas // from run_module
#define lgravx_gas AC_lgravx_gas
run_const int AC_lgravy_gas // from run_module
#define lgravy_gas AC_lgravy_gas
run_const int AC_lgravz_gas // from run_module
#define lgravz_gas AC_lgravz_gas
run_const int AC_lgravx_dust // from run_module
#define lgravx_dust AC_lgravx_dust
run_const int AC_lgravy_dust // from run_module
#define lgravy_dust AC_lgravy_dust
run_const int AC_lgravz_dust // from run_module
#define lgravz_dust AC_lgravz_dust
run_const int AC_lgravr // from run_module
#define lgravr AC_lgravr
run_const int AC_lwrite_ic // from run_module
#define lwrite_ic AC_lwrite_ic
run_const int AC_lnowrite // from run_module
#define lnowrite AC_lnowrite
run_const int AC_lserial_io // from run_module
#define lserial_io AC_lserial_io
run_const int AC_lmodify // from run_module
#define lmodify AC_lmodify
run_const int AC_lroot // from run_module
#define lroot AC_lroot
run_const int AC_lcaproot // from run_module
#define lcaproot AC_lcaproot
run_const int AC_ldebug // from run_module
#define ldebug AC_ldebug
run_const int AC_lfft // from run_module
#define lfft AC_lfft
run_const int AC_lproc_pt // from run_module
#define lproc_pt AC_lproc_pt
run_const int AC_lproc_p2 // from run_module
#define lproc_p2 AC_lproc_p2
run_const int AC_lfirst_proc_x // from run_module
#define lfirst_proc_x AC_lfirst_proc_x
run_const int AC_lfirst_proc_y // from run_module
#define lfirst_proc_y AC_lfirst_proc_y
run_const int AC_lfirst_proc_z // from run_module
#define lfirst_proc_z AC_lfirst_proc_z
run_const int AC_lfirst_proc_xy // from run_module
#define lfirst_proc_xy AC_lfirst_proc_xy
run_const int AC_lfirst_proc_yz // from run_module
#define lfirst_proc_yz AC_lfirst_proc_yz
run_const int AC_lfirst_proc_xz // from run_module
#define lfirst_proc_xz AC_lfirst_proc_xz
run_const int AC_lfirst_proc_xyz // from run_module
#define lfirst_proc_xyz AC_lfirst_proc_xyz
run_const int AC_llast_proc_x // from run_module
#define llast_proc_x AC_llast_proc_x
run_const int AC_llast_proc_y // from run_module
#define llast_proc_y AC_llast_proc_y
run_const int AC_llast_proc_z // from run_module
#define llast_proc_z AC_llast_proc_z
run_const int AC_llast_proc_xy // from run_module
#define llast_proc_xy AC_llast_proc_xy
run_const int AC_llast_proc_yz // from run_module
#define llast_proc_yz AC_llast_proc_yz
run_const int AC_llast_proc_xz // from run_module
#define llast_proc_xz AC_llast_proc_xz
run_const int AC_llast_proc_xyz // from run_module
#define llast_proc_xyz AC_llast_proc_xyz
run_const int AC_lnorth_pole // from run_module
#define lnorth_pole AC_lnorth_pole
run_const int AC_lsouth_pole // from run_module
#define lsouth_pole AC_lsouth_pole
run_const int AC_lpscalar_nolog // from run_module
#define lpscalar_nolog AC_lpscalar_nolog
run_const int AC_lalpm // from run_module
#define lalpm AC_lalpm
run_const int AC_lalpm_alternate // from run_module
#define lalpm_alternate AC_lalpm_alternate
run_const int AC_ldustdensity_log // from run_module
#define ldustdensity_log AC_ldustdensity_log
run_const int AC_lmdvar // from run_module
#define lmdvar AC_lmdvar
run_const int AC_ldcore // from run_module
#define ldcore AC_ldcore
run_const int AC_lneutraldensity_nolog // from run_module
#define lneutraldensity_nolog AC_lneutraldensity_nolog
run_const int AC_lvisc_smag // from run_module
#define lvisc_smag AC_lvisc_smag
run_const int AC_lslope_limit_diff // from run_module
#define lslope_limit_diff AC_lslope_limit_diff
run_const int AC_ltemperature_nolog // from run_module
#define ltemperature_nolog AC_ltemperature_nolog
run_const int AC_ltestperturb // from run_module
#define ltestperturb AC_ltestperturb
run_const int AC_lweno_transport // from run_module
#define lweno_transport AC_lweno_transport
run_const int AC_lstart // from run_module
#define lstart AC_lstart
run_const int AC_lrun // from run_module
#define lrun AC_lrun
run_const int AC_lreloading // from run_module
#define lreloading AC_lreloading
run_const int AC_ladv_der_as_aux // from run_module
#define ladv_der_as_aux AC_ladv_der_as_aux
run_const int AC_lghostfold_usebspline // from run_module
#define lghostfold_usebspline AC_lghostfold_usebspline
run_const int AC_lcooling_ss_mz // from run_module
#define lcooling_ss_mz AC_lcooling_ss_mz
run_const int AC_lshock_heat // from run_module
#define lshock_heat AC_lshock_heat
run_const real AC_density_scale_factor // from run_module
#define density_scale_factor AC_density_scale_factor
run_const int AC_pretend_lntt // from run_module
#define pretend_lntt AC_pretend_lntt
run_const int AC_nvar // from run_module
#define nvar AC_nvar
run_const int AC_naux // from run_module
#define naux AC_naux
run_const int AC_naux_com // from run_module
#define naux_com AC_naux_com
run_const int AC_nscratch // from run_module
#define nscratch AC_nscratch
run_const int AC_nglobal // from run_module
#define nglobal AC_nglobal
run_const int AC_n_odevars // from run_module
#define n_odevars AC_n_odevars
run_const int AC_lode // from run_module
#define lode AC_lode
run_const int AC_ilnrho // from run_module
#define ilnrho AC_ilnrho
run_const int AC_irho // from run_module
#define irho AC_irho
run_const int AC_irho_b // from run_module
#define irho_b AC_irho_b
run_const int AC_iss_b // from run_module
#define iss_b AC_iss_b
run_const int AC_ipp // from run_module
#define ipp AC_ipp
run_const int AC_irhs // from run_module
#define irhs AC_irhs
run_const int AC_ittold // from run_module
#define ittold AC_ittold
run_const int AC_ipoly // from run_module
#define ipoly AC_ipoly
run_const int AC_ip11 // from run_module
#define ip11 AC_ip11
run_const int AC_ip12 // from run_module
#define ip12 AC_ip12
run_const int AC_ip13 // from run_module
#define ip13 AC_ip13
run_const int AC_ip21 // from run_module
#define ip21 AC_ip21
run_const int AC_ip22 // from run_module
#define ip22 AC_ip22
run_const int AC_ip23 // from run_module
#define ip23 AC_ip23
run_const int AC_ip31 // from run_module
#define ip31 AC_ip31
run_const int AC_ip32 // from run_module
#define ip32 AC_ip32
run_const int AC_ip33 // from run_module
#define ip33 AC_ip33
run_const int AC_ipoly_fr // from run_module
#define ipoly_fr AC_ipoly_fr
run_const int AC_iuu // from run_module
#define iuu AC_iuu
run_const int AC_iux // from run_module
#define iux AC_iux
run_const int AC_iuy // from run_module
#define iuy AC_iuy
run_const int AC_iuz // from run_module
#define iuz AC_iuz
run_const int AC_iss // from run_module
#define iss AC_iss
run_const int AC_iphiuu // from run_module
#define iphiuu AC_iphiuu
run_const int AC_ilorentz // from run_module
#define ilorentz AC_ilorentz
run_const int AC_iuu0 // from run_module
#define iuu0 AC_iuu0
run_const int AC_iu0x // from run_module
#define iu0x AC_iu0x
run_const int AC_iu0y // from run_module
#define iu0y AC_iu0y
run_const int AC_iu0z // from run_module
#define iu0z AC_iu0z
run_const int AC_ioo // from run_module
#define ioo AC_ioo
run_const int AC_iox // from run_module
#define iox AC_iox
run_const int AC_ioy // from run_module
#define ioy AC_ioy
run_const int AC_ioz // from run_module
#define ioz AC_ioz
run_const int AC_ivv // from run_module
#define ivv AC_ivv
run_const int AC_ivx // from run_module
#define ivx AC_ivx
run_const int AC_ivy // from run_module
#define ivy AC_ivy
run_const int AC_ivz // from run_module
#define ivz AC_ivz
run_const int AC_igradu11 // from run_module
#define igradu11 AC_igradu11
run_const int AC_igradu12 // from run_module
#define igradu12 AC_igradu12
run_const int AC_igradu13 // from run_module
#define igradu13 AC_igradu13
run_const int AC_igradu21 // from run_module
#define igradu21 AC_igradu21
run_const int AC_igradu22 // from run_module
#define igradu22 AC_igradu22
run_const int AC_igradu23 // from run_module
#define igradu23 AC_igradu23
run_const int AC_igradu31 // from run_module
#define igradu31 AC_igradu31
run_const int AC_igradu32 // from run_module
#define igradu32 AC_igradu32
run_const int AC_igradu33 // from run_module
#define igradu33 AC_igradu33
run_const int AC_ispecialvar // from run_module
#define ispecialvar AC_ispecialvar
run_const int AC_ispecialvar2 // from run_module
#define ispecialvar2 AC_ispecialvar2
run_const int AC_iuut // from run_module
#define iuut AC_iuut
run_const int AC_iuxt // from run_module
#define iuxt AC_iuxt
run_const int AC_iuyt // from run_module
#define iuyt AC_iuyt
run_const int AC_iuzt // from run_module
#define iuzt AC_iuzt
run_const int AC_ioot // from run_module
#define ioot AC_ioot
run_const int AC_ioxt // from run_module
#define ioxt AC_ioxt
run_const int AC_ioyt // from run_module
#define ioyt AC_ioyt
run_const int AC_iozt // from run_module
#define iozt AC_iozt
run_const int AC_iuust // from run_module
#define iuust AC_iuust
run_const int AC_iuxst // from run_module
#define iuxst AC_iuxst
run_const int AC_iuyst // from run_module
#define iuyst AC_iuyst
run_const int AC_iuzst // from run_module
#define iuzst AC_iuzst
run_const int AC_ioost // from run_module
#define ioost AC_ioost
run_const int AC_ioxst // from run_module
#define ioxst AC_ioxst
run_const int AC_ioyst // from run_module
#define ioyst AC_ioyst
run_const int AC_iozst // from run_module
#define iozst AC_iozst
run_const int AC_ibbt // from run_module
#define ibbt AC_ibbt
run_const int AC_ibxt // from run_module
#define ibxt AC_ibxt
run_const int AC_ibyt // from run_module
#define ibyt AC_ibyt
run_const int AC_ibzt // from run_module
#define ibzt AC_ibzt
run_const int AC_ijjt // from run_module
#define ijjt AC_ijjt
run_const int AC_ijxt // from run_module
#define ijxt AC_ijxt
run_const int AC_ijyt // from run_module
#define ijyt AC_ijyt
run_const int AC_ijzt // from run_module
#define ijzt AC_ijzt
run_const int AC_ijxb // from run_module
#define ijxb AC_ijxb
run_const int AC_ijxbx // from run_module
#define ijxbx AC_ijxbx
run_const int AC_ijxby // from run_module
#define ijxby AC_ijxby
run_const int AC_ijxbz // from run_module
#define ijxbz AC_ijxbz
run_const int AC_iuxb // from run_module
#define iuxb AC_iuxb
run_const int AC_iuxbx // from run_module
#define iuxbx AC_iuxbx
run_const int AC_iuxby // from run_module
#define iuxby AC_iuxby
run_const int AC_iuxbz // from run_module
#define iuxbz AC_iuxbz
run_const int AC_iugb // from run_module
#define iugb AC_iugb
run_const int AC_iugbx // from run_module
#define iugbx AC_iugbx
run_const int AC_iugby // from run_module
#define iugby AC_iugby
run_const int AC_iugbz // from run_module
#define iugbz AC_iugbz
run_const int AC_ibgu // from run_module
#define ibgu AC_ibgu
run_const int AC_ibgux // from run_module
#define ibgux AC_ibgux
run_const int AC_ibguy // from run_module
#define ibguy AC_ibguy
run_const int AC_ibguz // from run_module
#define ibguz AC_ibguz
run_const int AC_ibdivu // from run_module
#define ibdivu AC_ibdivu
run_const int AC_ibdivux // from run_module
#define ibdivux AC_ibdivux
run_const int AC_ibdivuy // from run_module
#define ibdivuy AC_ibdivuy
run_const int AC_ibdivuz // from run_module
#define ibdivuz AC_ibdivuz
run_const int AC_ibxf // from run_module
#define ibxf AC_ibxf
run_const int AC_ibyf // from run_module
#define ibyf AC_ibyf
run_const int AC_ibzf // from run_module
#define ibzf AC_ibzf
run_const int AC_ibbf // from run_module
#define ibbf AC_ibbf
run_const int AC_ipotself // from run_module
#define ipotself AC_ipotself
run_const int AC_iaa // from run_module
#define iaa AC_iaa
run_const int AC_iax // from run_module
#define iax AC_iax
run_const int AC_iay // from run_module
#define iay AC_iay
run_const int AC_iaz // from run_module
#define iaz AC_iaz
run_const int AC_ispx // from run_module
#define ispx AC_ispx
run_const int AC_ispy // from run_module
#define ispy AC_ispy
run_const int AC_ispz // from run_module
#define ispz AC_ispz
run_const int AC_ifcr // from run_module
#define ifcr AC_ifcr
run_const int AC_ifcrx // from run_module
#define ifcrx AC_ifcrx
run_const int AC_ifcry // from run_module
#define ifcry AC_ifcry
run_const int AC_ifcrz // from run_module
#define ifcrz AC_ifcrz
run_const int AC_ihij // from run_module
#define ihij AC_ihij
run_const int AC_igij // from run_module
#define igij AC_igij
run_const int AC_ihht // from run_module
#define ihht AC_ihht
run_const int AC_ihhx // from run_module
#define ihhx AC_ihhx
run_const int AC_iggt // from run_module
#define iggt AC_iggt
run_const int AC_iggx // from run_module
#define iggx AC_iggx
run_const int AC_istresst // from run_module
#define istresst AC_istresst
run_const int AC_istressx // from run_module
#define istressx AC_istressx
run_const int AC_istress_ij // from run_module
#define istress_ij AC_istress_ij
run_const int AC_ihhtim // from run_module
#define ihhtim AC_ihhtim
run_const int AC_ihhxim // from run_module
#define ihhxim AC_ihhxim
run_const int AC_iggtim // from run_module
#define iggtim AC_iggtim
run_const int AC_iggxim // from run_module
#define iggxim AC_iggxim
run_const int AC_istresstim // from run_module
#define istresstim AC_istresstim
run_const int AC_istressxim // from run_module
#define istressxim AC_istressxim
run_const int AC_iaatest // from run_module
#define iaatest AC_iaatest
run_const int AC_iaztestpq // from run_module
#define iaztestpq AC_iaztestpq
run_const int AC_iaxtest // from run_module
#define iaxtest AC_iaxtest
run_const int AC_iaytest // from run_module
#define iaytest AC_iaytest
run_const int AC_iaztest // from run_module
#define iaztest AC_iaztest
run_const int AC_iuutest // from run_module
#define iuutest AC_iuutest
run_const int AC_iuztestpq // from run_module
#define iuztestpq AC_iuztestpq
run_const int AC_ihhtestpq // from run_module
#define ihhtestpq AC_ihhtestpq
run_const int AC_iqx // from run_module
#define iqx AC_iqx
run_const int AC_iqy // from run_module
#define iqy AC_iqy
run_const int AC_iqz // from run_module
#define iqz AC_iqz
run_const int AC_iqq // from run_module
#define iqq AC_iqq
run_const int AC_ntestscalar // from run_module
#define ntestscalar AC_ntestscalar
run_const int AC_ntestfield // from run_module
#define ntestfield AC_ntestfield
run_const int AC_ntestflow // from run_module
#define ntestflow AC_ntestflow
run_const int AC_ntestlnrho // from run_module
#define ntestlnrho AC_ntestlnrho
run_const int AC_icctest // from run_module
#define icctest AC_icctest
run_const int AC_icctestpq // from run_module
#define icctestpq AC_icctestpq
run_const int AC_iug // from run_module
#define iug AC_iug
run_const int AC_iam // from run_module
#define iam AC_iam
run_const int AC_iamx // from run_module
#define iamx AC_iamx
run_const int AC_iamy // from run_module
#define iamy AC_iamy
run_const int AC_iamz // from run_module
#define iamz AC_iamz
run_const int AC_ivisc_heat // from run_module
#define ivisc_heat AC_ivisc_heat
run_const int AC_ibb // from run_module
#define ibb AC_ibb
run_const int AC_ibx // from run_module
#define ibx AC_ibx
run_const int AC_iby // from run_module
#define iby AC_iby
run_const int AC_ibz // from run_module
#define ibz AC_ibz
run_const int AC_ijj // from run_module
#define ijj AC_ijj
run_const int AC_ijx // from run_module
#define ijx AC_ijx
run_const int AC_ijy // from run_module
#define ijy AC_ijy
run_const int AC_ijz // from run_module
#define ijz AC_ijz
run_const int AC_ibb_sph // from run_module
#define ibb_sph AC_ibb_sph
run_const int AC_ibb_sphr // from run_module
#define ibb_sphr AC_ibb_sphr
run_const int AC_ibb_spht // from run_module
#define ibb_spht AC_ibb_spht
run_const int AC_ibb_sphp // from run_module
#define ibb_sphp AC_ibb_sphp
run_const int AC_inusmag // from run_module
#define inusmag AC_inusmag
run_const int AC_ietasmag // from run_module
#define ietasmag AC_ietasmag
run_const int AC_iaak // from run_module
#define iaak AC_iaak
run_const int AC_iaakim // from run_module
#define iaakim AC_iaakim
run_const int AC_ieek // from run_module
#define ieek AC_ieek
run_const int AC_ieekim // from run_module
#define ieekim AC_ieekim
run_const int AC_iee // from run_module
#define iee AC_iee
run_const int AC_iex // from run_module
#define iex AC_iex
run_const int AC_iey // from run_module
#define iey AC_iey
run_const int AC_iez // from run_module
#define iez AC_iez
run_const int AC_ialfven // from run_module
#define ialfven AC_ialfven
run_const int AC_iff_diff // from run_module
#define iff_diff AC_iff_diff
run_const int AC_iff_diff1 // from run_module
#define iff_diff1 AC_iff_diff1
run_const int AC_iff_diff2 // from run_module
#define iff_diff2 AC_iff_diff2
run_const int AC_iff_div_uu // from run_module
#define iff_div_uu AC_iff_div_uu
run_const int AC_iff_div_aa // from run_module
#define iff_div_aa AC_iff_div_aa
run_const int AC_iff_div_ss // from run_module
#define iff_div_ss AC_iff_div_ss
run_const int AC_iff_div_rho // from run_module
#define iff_div_rho AC_iff_div_rho
run_const int AC_iff_char_c // from run_module
#define iff_char_c AC_iff_char_c
run_const int AC_iff_heat // from run_module
#define iff_heat AC_iff_heat
run_const int AC_isld_char // from run_module
#define isld_char AC_isld_char
run_const int AC_ivisc_forc // from run_module
#define ivisc_forc AC_ivisc_forc
run_const int AC_ivisc_forcx // from run_module
#define ivisc_forcx AC_ivisc_forcx
run_const int AC_ivisc_forcy // from run_module
#define ivisc_forcy AC_ivisc_forcy
run_const int AC_ivisc_forcz // from run_module
#define ivisc_forcz AC_ivisc_forcz
run_const int AC_i_adv_der // from run_module
#define i_adv_der AC_i_adv_der
run_const int AC_i_adv_derx // from run_module
#define i_adv_derx AC_i_adv_derx
run_const int AC_i_adv_dery // from run_module
#define i_adv_dery AC_i_adv_dery
run_const int AC_i_adv_derz // from run_module
#define i_adv_derz AC_i_adv_derz
run_const int AC_iuxbtest // from run_module
#define iuxbtest AC_iuxbtest
run_const int AC_ijxbtest // from run_module
#define ijxbtest AC_ijxbtest
run_const int AC_iugutest // from run_module
#define iugutest AC_iugutest
run_const int AC_iughtest // from run_module
#define iughtest AC_iughtest
run_const int AC_isghtest // from run_module
#define isghtest AC_isghtest
run_const int AC_ishock // from run_module
#define ishock AC_ishock
run_const int AC_ishock_perp // from run_module
#define ishock_perp AC_ishock_perp
run_const int AC_iyh // from run_module
#define iyh AC_iyh
run_const int AC_ihypvis // from run_module
#define ihypvis AC_ihypvis
run_const int AC_ihypres // from run_module
#define ihypres AC_ihypres
run_const int AC_iecr // from run_module
#define iecr AC_iecr
run_const int AC_ismagorinsky // from run_module
#define ismagorinsky AC_ismagorinsky
run_const int AC_iviscosity // from run_module
#define iviscosity AC_iviscosity
run_const int AC_iqrad // from run_module
#define iqrad AC_iqrad
run_const int AC_israd // from run_module
#define israd AC_israd
run_const int AC_ilntt // from run_module
#define ilntt AC_ilntt
run_const int AC_itt // from run_module
#define itt AC_itt
run_const int AC_ikapparho // from run_module
#define ikapparho AC_ikapparho
run_const int AC_ikr_frad // from run_module
#define ikr_frad AC_ikr_frad
run_const int AC_ikr_fradx // from run_module
#define ikr_fradx AC_ikr_fradx
run_const int AC_ikr_frady // from run_module
#define ikr_frady AC_ikr_frady
run_const int AC_ikr_fradz // from run_module
#define ikr_fradz AC_ikr_fradz
run_const int AC_igpotselfx // from run_module
#define igpotselfx AC_igpotselfx
run_const int AC_igpotselfy // from run_module
#define igpotselfy AC_igpotselfy
run_const int AC_igpotselfz // from run_module
#define igpotselfz AC_igpotselfz
run_const int AC_icc // from run_module
#define icc AC_icc
run_const int AC_ilncc // from run_module
#define ilncc AC_ilncc
run_const int AC_ialpm // from run_module
#define ialpm AC_ialpm
run_const int AC_ietat // from run_module
#define ietat AC_ietat
run_const int AC_iacc // from run_module
#define iacc AC_iacc
run_const int AC_issat // from run_module
#define issat AC_issat
run_const int AC_ittc // from run_module
#define ittc AC_ittc
run_const int AC_itauascalar // from run_module
#define itauascalar AC_itauascalar
run_const int AC_iaphi // from run_module
#define iaphi AC_iaphi
run_const int AC_ibphi // from run_module
#define ibphi AC_ibphi
run_const int AC_ieth // from run_module
#define ieth AC_ieth
run_const int AC_idet // from run_module
#define idet AC_idet
run_const int AC_iinvgrid // from run_module
#define iinvgrid AC_iinvgrid
run_const int AC_iguij // from run_module
#define iguij AC_iguij
run_const int AC_igu11 // from run_module
#define igu11 AC_igu11
run_const int AC_igu12 // from run_module
#define igu12 AC_igu12
run_const int AC_igu13 // from run_module
#define igu13 AC_igu13
run_const int AC_igu21 // from run_module
#define igu21 AC_igu21
run_const int AC_igu22 // from run_module
#define igu22 AC_igu22
run_const int AC_igu23 // from run_module
#define igu23 AC_igu23
run_const int AC_igu31 // from run_module
#define igu31 AC_igu31
run_const int AC_igu32 // from run_module
#define igu32 AC_igu32
run_const int AC_igu33 // from run_module
#define igu33 AC_igu33
run_const int AC_icooling // from run_module
#define icooling AC_icooling
run_const int AC_inetheat // from run_module
#define inetheat AC_inetheat
run_const int AC_ilnrhon // from run_module
#define ilnrhon AC_ilnrhon
run_const int AC_irhon // from run_module
#define irhon AC_irhon
run_const int AC_irhoe // from run_module
#define irhoe AC_irhoe
run_const int AC_iuun // from run_module
#define iuun AC_iuun
run_const int AC_iunx // from run_module
#define iunx AC_iunx
run_const int AC_iuny // from run_module
#define iuny AC_iuny
run_const int AC_iunz // from run_module
#define iunz AC_iunz
run_const int AC_iglobal_bx_ext // from run_module
#define iglobal_bx_ext AC_iglobal_bx_ext
run_const int AC_iglobal_by_ext // from run_module
#define iglobal_by_ext AC_iglobal_by_ext
run_const int AC_iglobal_bz_ext // from run_module
#define iglobal_bz_ext AC_iglobal_bz_ext
run_const int AC_iglobal_ax_ext // from run_module
#define iglobal_ax_ext AC_iglobal_ax_ext
run_const int AC_iglobal_ay_ext // from run_module
#define iglobal_ay_ext AC_iglobal_ay_ext
run_const int AC_iglobal_az_ext // from run_module
#define iglobal_az_ext AC_iglobal_az_ext
run_const int AC_iglobal_lnrho0 // from run_module
#define iglobal_lnrho0 AC_iglobal_lnrho0
run_const int AC_iglobal_ss0 // from run_module
#define iglobal_ss0 AC_iglobal_ss0
run_const int AC_icp // from run_module
#define icp AC_icp
run_const int AC_igpx // from run_module
#define igpx AC_igpx
run_const int AC_igpy // from run_module
#define igpy AC_igpy
run_const int AC_irr // from run_module
#define irr AC_irr
run_const int AC_iss_run_aver // from run_module
#define iss_run_aver AC_iss_run_aver
run_const int AC_ifenth // from run_module
#define ifenth AC_ifenth
run_const int AC_iss_flucz // from run_module
#define iss_flucz AC_iss_flucz
run_const int AC_itt_flucz // from run_module
#define itt_flucz AC_itt_flucz
run_const int AC_irho_flucz // from run_module
#define irho_flucz AC_irho_flucz
run_const int AC_iuu_fluc // from run_module
#define iuu_fluc AC_iuu_fluc
run_const int AC_iuu_flucx // from run_module
#define iuu_flucx AC_iuu_flucx
run_const int AC_iuu_flucy // from run_module
#define iuu_flucy AC_iuu_flucy
run_const int AC_iuu_flucz // from run_module
#define iuu_flucz AC_iuu_flucz
run_const int AC_iuu_sph // from run_module
#define iuu_sph AC_iuu_sph
run_const int AC_iuu_sphr // from run_module
#define iuu_sphr AC_iuu_sphr
run_const int AC_iuu_spht // from run_module
#define iuu_spht AC_iuu_spht
run_const int AC_iuu_sphp // from run_module
#define iuu_sphp AC_iuu_sphp
run_const int AC_ics // from run_module
#define ics AC_ics
run_const int AC_imn // from run_module
#define imn AC_imn
run_const int AC_lglob // from run_module
#define lglob AC_lglob
run_const int AC_necessary_imn // from run_module
#define necessary_imn AC_necessary_imn
run_const real AC_penc0 // from run_module
#define penc0 AC_penc0
run_const int AC_lpencil_check // from run_module
#define lpencil_check AC_lpencil_check
run_const int AC_lpencil_check_small // from run_module
#define lpencil_check_small AC_lpencil_check_small
run_const int AC_lpencil_check_no_zeros // from run_module
#define lpencil_check_no_zeros AC_lpencil_check_no_zeros
run_const int AC_lpencil_init // from run_module
#define lpencil_init AC_lpencil_init
run_const int AC_lpencil_requested_swap // from run_module
#define lpencil_requested_swap AC_lpencil_requested_swap
run_const int AC_lpencil_diagnos_swap // from run_module
#define lpencil_diagnos_swap AC_lpencil_diagnos_swap
run_const int AC_lpencil_check_diagnos_opti // from run_module
#define lpencil_check_diagnos_opti AC_lpencil_check_diagnos_opti
run_const int AC_lpencil_check_at_work // from run_module
#define lpencil_check_at_work AC_lpencil_check_at_work
run_const int AC_ipencil_swap // from run_module
#define ipencil_swap AC_ipencil_swap
run_const int AC_it1 // from run_module
#define it1 AC_it1
run_const int AC_it1start // from run_module
#define it1start AC_it1start
run_const int AC_it1d // from run_module
#define it1d AC_it1d
run_const int AC_itspec // from run_module
#define itspec AC_itspec
run_const int AC_nname // from run_module
#define nname AC_nname
run_const int AC_nnamev // from run_module
#define nnamev AC_nnamev
run_const int AC_nnamexy // from run_module
#define nnamexy AC_nnamexy
run_const int AC_nnamexz // from run_module
#define nnamexz AC_nnamexz
run_const int AC_nnamerz // from run_module
#define nnamerz AC_nnamerz
run_const int AC_nnamez // from run_module
#define nnamez AC_nnamez
run_const int AC_nnamey // from run_module
#define nnamey AC_nnamey
run_const int AC_nnamex // from run_module
#define nnamex AC_nnamex
run_const int AC_nnamer // from run_module
#define nnamer AC_nnamer
run_const int AC_nname_sound // from run_module
#define nname_sound AC_nname_sound
run_const int AC_ncoords_sound // from run_module
#define ncoords_sound AC_ncoords_sound
run_const int AC_nr_directions // from run_module
#define nr_directions AC_nr_directions
run_const int AC_itdiagnos // from run_module
#define itdiagnos AC_itdiagnos
run_const real AC_tdiagnos // from run_module
#define tdiagnos AC_tdiagnos
run_const real AC_dtdiagnos // from run_module
#define dtdiagnos AC_dtdiagnos
run_const real AC_t1ddiagnos // from run_module
#define t1ddiagnos AC_t1ddiagnos
run_const real AC_t2davgfirst // from run_module
#define t2davgfirst AC_t2davgfirst
run_const real AC_eps_rkf_diagnos // from run_module
#define eps_rkf_diagnos AC_eps_rkf_diagnos
run_const real AC_fweight[mname] // from run_module
#define fweight AC_fweight
run_const int AC_lout // from run_module
#define lout AC_lout
run_const int AC_headt // from run_module
#define headt AC_headt
run_const int AC_headtt // from run_module
#define headtt AC_headtt
run_const int AC_lrmv // from run_module
#define lrmv AC_lrmv
run_const int AC_ldiagnos // from run_module
#define ldiagnos AC_ldiagnos
run_const int AC_lvideo // from run_module
#define lvideo AC_lvideo
run_const int AC_lwrite_prof // from run_module
#define lwrite_prof AC_lwrite_prof
run_const int AC_lout_sound // from run_module
#define lout_sound AC_lout_sound
run_const int AC_ltracers // from run_module
#define ltracers AC_ltracers
run_const int AC_lfixed_points // from run_module
#define lfixed_points AC_lfixed_points
run_const int AC_l2davg // from run_module
#define l2davg AC_l2davg
run_const int AC_l2davgfirst // from run_module
#define l2davgfirst AC_l2davgfirst
run_const int AC_l1davg // from run_module
#define l1davg AC_l1davg
run_const int AC_l1davgfirst // from run_module
#define l1davgfirst AC_l1davgfirst
run_const int AC_l1dphiavg // from run_module
#define l1dphiavg AC_l1dphiavg
run_const int AC_lwrite_xyaverages // from run_module
#define lwrite_xyaverages AC_lwrite_xyaverages
run_const int AC_lwrite_xzaverages // from run_module
#define lwrite_xzaverages AC_lwrite_xzaverages
run_const int AC_lwrite_yzaverages // from run_module
#define lwrite_yzaverages AC_lwrite_yzaverages
run_const int AC_lwrite_phizaverages // from run_module
#define lwrite_phizaverages AC_lwrite_phizaverages
run_const int AC_lwrite_yaverages // from run_module
#define lwrite_yaverages AC_lwrite_yaverages
run_const int AC_lwrite_zaverages // from run_module
#define lwrite_zaverages AC_lwrite_zaverages
run_const int AC_lwrite_phiaverages // from run_module
#define lwrite_phiaverages AC_lwrite_phiaverages
run_const int AC_ldiagnos_need_zaverages // from run_module
#define ldiagnos_need_zaverages AC_ldiagnos_need_zaverages
run_const int AC_ltime_integrals // from run_module
#define ltime_integrals AC_ltime_integrals
run_const int AC_lreset_seed // from run_module
#define lreset_seed AC_lreset_seed
run_const int AC_lproper_averages // from run_module
#define lproper_averages AC_lproper_averages
run_const int AC_lav_smallx // from run_module
#define lav_smallx AC_lav_smallx
run_const int AC_loutside_avg // from run_module
#define loutside_avg AC_loutside_avg
run_const real AC_xav_max // from run_module
#define xav_max AC_xav_max
run_const real AC_nvol // from run_module
#define nvol AC_nvol
run_const real AC_nvol1 // from run_module
#define nvol1 AC_nvol1
run_const int AC_nseed // from run_module
#define nseed AC_nseed
run_const int AC_seed0 // from run_module
#define seed0 AC_seed0
run_const int AC_ichannel1 // from run_module
#define ichannel1 AC_ichannel1
run_const int AC_ichannel2 // from run_module
#define ichannel2 AC_ichannel2
run_const real AC_fran1[2] // from run_module
#define fran1 AC_fran1
run_const real AC_fran2[2] // from run_module
#define fran2 AC_fran2
run_const int AC_lseed_global // from run_module
#define lseed_global AC_lseed_global
run_const int AC_lseed_procdependent // from run_module
#define lseed_procdependent AC_lseed_procdependent
run_const real AC_yequator // from run_module
#define yequator AC_yequator
run_const real AC_zequator // from run_module
#define zequator AC_zequator
run_const int AC_lequatory // from run_module
#define lequatory AC_lequatory
run_const int AC_lequatorz // from run_module
#define lequatorz AC_lequatorz
run_const int AC_name_half_max // from run_module
#define name_half_max AC_name_half_max
run_const real AC_radius_diag // from run_module
#define radius_diag AC_radius_diag
run_const int AC_lpoint // from run_module
#define lpoint AC_lpoint
run_const int AC_mpoint // from run_module
#define mpoint AC_mpoint
run_const int AC_npoint // from run_module
#define npoint AC_npoint
run_const int AC_lpoint2 // from run_module
#define lpoint2 AC_lpoint2
run_const int AC_mpoint2 // from run_module
#define mpoint2 AC_mpoint2
run_const int AC_npoint2 // from run_module
#define npoint2 AC_npoint2
run_const int AC_iproc_pt // from run_module
#define iproc_pt AC_iproc_pt
run_const int AC_iproc_p2 // from run_module
#define iproc_p2 AC_iproc_p2
run_const int AC_idiag_it // from run_module
#define idiag_it AC_idiag_it
run_const int AC_idiag_t // from run_module
#define idiag_t AC_idiag_t
run_const int AC_idiag_dt // from run_module
#define idiag_dt AC_idiag_dt
run_const int AC_idiag_walltime // from run_module
#define idiag_walltime AC_idiag_walltime
run_const int AC_idiag_timeperstep // from run_module
#define idiag_timeperstep AC_idiag_timeperstep
run_const int AC_idiag_rcylmphi // from run_module
#define idiag_rcylmphi AC_idiag_rcylmphi
run_const int AC_idiag_phimphi // from run_module
#define idiag_phimphi AC_idiag_phimphi
run_const int AC_idiag_zmphi // from run_module
#define idiag_zmphi AC_idiag_zmphi
run_const int AC_idiag_rmphi // from run_module
#define idiag_rmphi AC_idiag_rmphi
run_const int AC_idiag_dtv // from run_module
#define idiag_dtv AC_idiag_dtv
run_const int AC_idiag_dtdiffus // from run_module
#define idiag_dtdiffus AC_idiag_dtdiffus
run_const int AC_idiag_dtdiffus2 // from run_module
#define idiag_dtdiffus2 AC_idiag_dtdiffus2
run_const int AC_idiag_dtdiffus3 // from run_module
#define idiag_dtdiffus3 AC_idiag_dtdiffus3
run_const int AC_idiag_rmesh // from run_module
#define idiag_rmesh AC_idiag_rmesh
run_const int AC_idiag_rmesh3 // from run_module
#define idiag_rmesh3 AC_idiag_rmesh3
run_const int AC_idiag_maxadvec // from run_module
#define idiag_maxadvec AC_idiag_maxadvec
run_const int AC_idiag_eps_rkf // from run_module
#define idiag_eps_rkf AC_idiag_eps_rkf
run_const int AC_lemergency_brake // from run_module
#define lemergency_brake AC_lemergency_brake
run_const int AC_lcopysnapshots_exp // from run_module
#define lcopysnapshots_exp AC_lcopysnapshots_exp
run_const int AC_lwrite_2d // from run_module
#define lwrite_2d AC_lwrite_2d
run_const int AC_lbidiagonal_derij // from run_module
#define lbidiagonal_derij AC_lbidiagonal_derij
run_const int AC_vel_spec // from run_module
#define vel_spec AC_vel_spec
run_const int AC_mag_spec // from run_module
#define mag_spec AC_mag_spec
run_const int AC_uxj_spec // from run_module
#define uxj_spec AC_uxj_spec
run_const int AC_vec_spec // from run_module
#define vec_spec AC_vec_spec
run_const int AC_j_spec // from run_module
#define j_spec AC_j_spec
run_const int AC_jb_spec // from run_module
#define jb_spec AC_jb_spec
run_const int AC_ja_spec // from run_module
#define ja_spec AC_ja_spec
run_const int AC_oo_spec // from run_module
#define oo_spec AC_oo_spec
run_const int AC_relvel_spec // from run_module
#define relvel_spec AC_relvel_spec
run_const int AC_vel_phispec // from run_module
#define vel_phispec AC_vel_phispec
run_const int AC_mag_phispec // from run_module
#define mag_phispec AC_mag_phispec
run_const int AC_uxj_phispec // from run_module
#define uxj_phispec AC_uxj_phispec
run_const int AC_vec_phispec // from run_module
#define vec_phispec AC_vec_phispec
run_const int AC_uxy_spec // from run_module
#define uxy_spec AC_uxy_spec
run_const int AC_bxy_spec // from run_module
#define bxy_spec AC_bxy_spec
run_const int AC_jxbxy_spec // from run_module
#define jxbxy_spec AC_jxbxy_spec
run_const int AC_ep_spec // from run_module
#define ep_spec AC_ep_spec
run_const int AC_nd_spec // from run_module
#define nd_spec AC_nd_spec
run_const int AC_ud_spec // from run_module
#define ud_spec AC_ud_spec
run_const int AC_abs_u_spec // from run_module
#define abs_u_spec AC_abs_u_spec
run_const int AC_ro_spec // from run_module
#define ro_spec AC_ro_spec
run_const int AC_tt_spec // from run_module
#define tt_spec AC_tt_spec
run_const int AC_ss_spec // from run_module
#define ss_spec AC_ss_spec
run_const int AC_cc_spec // from run_module
#define cc_spec AC_cc_spec
run_const int AC_cr_spec // from run_module
#define cr_spec AC_cr_spec
run_const int AC_sp_spec // from run_module
#define sp_spec AC_sp_spec
run_const int AC_ssp_spec // from run_module
#define ssp_spec AC_ssp_spec
run_const int AC_sssp_spec // from run_module
#define sssp_spec AC_sssp_spec
run_const int AC_mu_spec // from run_module
#define mu_spec AC_mu_spec
run_const int AC_lr_spec // from run_module
#define lr_spec AC_lr_spec
run_const int AC_r2u_spec // from run_module
#define r2u_spec AC_r2u_spec
run_const int AC_r3u_spec // from run_module
#define r3u_spec AC_r3u_spec
run_const int AC_oun_spec // from run_module
#define oun_spec AC_oun_spec
run_const int AC_np_spec // from run_module
#define np_spec AC_np_spec
run_const int AC_np_ap_spec // from run_module
#define np_ap_spec AC_np_ap_spec
run_const int AC_rhop_spec // from run_module
#define rhop_spec AC_rhop_spec
run_const int AC_ele_spec // from run_module
#define ele_spec AC_ele_spec
run_const int AC_pot_spec // from run_module
#define pot_spec AC_pot_spec
run_const int AC_ux_spec // from run_module
#define ux_spec AC_ux_spec
run_const int AC_uy_spec // from run_module
#define uy_spec AC_uy_spec
run_const int AC_uz_spec // from run_module
#define uz_spec AC_uz_spec
run_const int AC_a0_spec // from run_module
#define a0_spec AC_a0_spec
run_const int AC_ucp_spec // from run_module
#define ucp_spec AC_ucp_spec
run_const int AC_ou_spec // from run_module
#define ou_spec AC_ou_spec
run_const int AC_ab_spec // from run_module
#define ab_spec AC_ab_spec
run_const int AC_azbz_spec // from run_module
#define azbz_spec AC_azbz_spec
run_const int AC_uzs_spec // from run_module
#define uzs_spec AC_uzs_spec
run_const int AC_ub_spec // from run_module
#define ub_spec AC_ub_spec
run_const int AC_lor_spec // from run_module
#define lor_spec AC_lor_spec
run_const int AC_emf_spec // from run_module
#define emf_spec AC_emf_spec
run_const int AC_tra_spec // from run_module
#define tra_spec AC_tra_spec
run_const int AC_gws_spec // from run_module
#define gws_spec AC_gws_spec
run_const int AC_gwh_spec // from run_module
#define gwh_spec AC_gwh_spec
run_const int AC_gwm_spec // from run_module
#define gwm_spec AC_gwm_spec
run_const int AC_str_spec // from run_module
#define str_spec AC_str_spec
run_const int AC_stg_spec // from run_module
#define stg_spec AC_stg_spec
run_const int AC_gws_spec_boost // from run_module
#define gws_spec_boost AC_gws_spec_boost
run_const int AC_gwh_spec_boost // from run_module
#define gwh_spec_boost AC_gwh_spec_boost
run_const int AC_stt_spec // from run_module
#define stt_spec AC_stt_spec
run_const int AC_stx_spec // from run_module
#define stx_spec AC_stx_spec
run_const int AC_gwd_spec // from run_module
#define gwd_spec AC_gwd_spec
run_const int AC_gwe_spec // from run_module
#define gwe_spec AC_gwe_spec
run_const int AC_gwf_spec // from run_module
#define gwf_spec AC_gwf_spec
run_const int AC_gwg_spec // from run_module
#define gwg_spec AC_gwg_spec
run_const int AC_scl_spec // from run_module
#define scl_spec AC_scl_spec
run_const int AC_vct_spec // from run_module
#define vct_spec AC_vct_spec
run_const int AC_tpq_spec // from run_module
#define tpq_spec AC_tpq_spec
run_const int AC_tgw_spec // from run_module
#define tgw_spec AC_tgw_spec
run_const int AC_scl_spec_boost // from run_module
#define scl_spec_boost AC_scl_spec_boost
run_const int AC_vct_spec_boost // from run_module
#define vct_spec_boost AC_vct_spec_boost
run_const int AC_har_spec // from run_module
#define har_spec AC_har_spec
run_const int AC_hav_spec // from run_module
#define hav_spec AC_hav_spec
run_const int AC_bb2_spec // from run_module
#define bb2_spec AC_bb2_spec
run_const int AC_jj2_spec // from run_module
#define jj2_spec AC_jj2_spec
run_const int AC_b2_spec // from run_module
#define b2_spec AC_b2_spec
run_const int AC_oned // from run_module
#define oned AC_oned
run_const int AC_twod // from run_module
#define twod AC_twod
run_const int AC_ab_phispec // from run_module
#define ab_phispec AC_ab_phispec
run_const int AC_ou_phispec // from run_module
#define ou_phispec AC_ou_phispec
run_const int AC_rhocc_pdf // from run_module
#define rhocc_pdf AC_rhocc_pdf
run_const int AC_cc_pdf // from run_module
#define cc_pdf AC_cc_pdf
run_const int AC_lncc_pdf // from run_module
#define lncc_pdf AC_lncc_pdf
run_const int AC_gcc_pdf // from run_module
#define gcc_pdf AC_gcc_pdf
run_const int AC_lngcc_pdf // from run_module
#define lngcc_pdf AC_lngcc_pdf
run_const int AC_lnspecial_pdf // from run_module
#define lnspecial_pdf AC_lnspecial_pdf
run_const int AC_special_pdf // from run_module
#define special_pdf AC_special_pdf
run_const int AC_ang_jb_pdf1d // from run_module
#define ang_jb_pdf1d AC_ang_jb_pdf1d
run_const int AC_ang_ub_pdf1d // from run_module
#define ang_ub_pdf1d AC_ang_ub_pdf1d
run_const int AC_ang_ou_pdf1d // from run_module
#define ang_ou_pdf1d AC_ang_ou_pdf1d
run_const int AC_test_nonblocking // from run_module
#define test_nonblocking AC_test_nonblocking
run_const int AC_onedall // from run_module
#define onedall AC_onedall
run_const int AC_lsfu // from run_module
#define lsfu AC_lsfu
run_const int AC_lsfb // from run_module
#define lsfb AC_lsfb
run_const int AC_lsfz1 // from run_module
#define lsfz1 AC_lsfz1
run_const int AC_lsfz2 // from run_module
#define lsfz2 AC_lsfz2
run_const int AC_lsfflux // from run_module
#define lsfflux AC_lsfflux
run_const int AC_lpdfu // from run_module
#define lpdfu AC_lpdfu
run_const int AC_lpdfb // from run_module
#define lpdfb AC_lpdfb
run_const int AC_lpdfz1 // from run_module
#define lpdfz1 AC_lpdfz1
run_const int AC_lpdfz2 // from run_module
#define lpdfz2 AC_lpdfz2
run_const int AC_ou_omega // from run_module
#define ou_omega AC_ou_omega
run_const int AC_cor_uu // from run_module
#define cor_uu AC_cor_uu
run_const int AC_ab_kzspec // from run_module
#define ab_kzspec AC_ab_kzspec
run_const int AC_ou_kzspec // from run_module
#define ou_kzspec AC_ou_kzspec
run_const int AC_ou_polar // from run_module
#define ou_polar AC_ou_polar
run_const int AC_ab_polar // from run_module
#define ab_polar AC_ab_polar
run_const int AC_jb_polar // from run_module
#define jb_polar AC_jb_polar
run_const int AC_uut_spec // from run_module
#define uut_spec AC_uut_spec
run_const int AC_uut_polar // from run_module
#define uut_polar AC_uut_polar
run_const int AC_ouout_spec // from run_module
#define ouout_spec AC_ouout_spec
run_const int AC_ouout2_spec // from run_module
#define ouout2_spec AC_ouout2_spec
run_const int AC_ouout_polar // from run_module
#define ouout_polar AC_ouout_polar
run_const int AC_out_spec // from run_module
#define out_spec AC_out_spec
run_const int AC_uot_spec // from run_module
#define uot_spec AC_uot_spec
run_const int AC_saffman_ub // from run_module
#define saffman_ub AC_saffman_ub
run_const int AC_saffman_mag // from run_module
#define saffman_mag AC_saffman_mag
run_const int AC_saffman_mag_c // from run_module
#define saffman_mag_c AC_saffman_mag_c
run_const int AC_saffman_aa // from run_module
#define saffman_aa AC_saffman_aa
run_const int AC_saffman_aa_c // from run_module
#define saffman_aa_c AC_saffman_aa_c
run_const int AC_saffman_bb // from run_module
#define saffman_bb AC_saffman_bb
run_const int AC_uu_fft3d // from run_module
#define uu_fft3d AC_uu_fft3d
run_const int AC_oo_fft3d // from run_module
#define oo_fft3d AC_oo_fft3d
run_const int AC_bb_fft3d // from run_module
#define bb_fft3d AC_bb_fft3d
run_const int AC_jj_fft3d // from run_module
#define jj_fft3d AC_jj_fft3d
run_const int AC_uu_xkyz // from run_module
#define uu_xkyz AC_uu_xkyz
run_const int AC_oo_xkyz // from run_module
#define oo_xkyz AC_oo_xkyz
run_const int AC_bb_xkyz // from run_module
#define bb_xkyz AC_bb_xkyz
run_const int AC_jj_xkyz // from run_module
#define jj_xkyz AC_jj_xkyz
run_const int AC_uu_kx0z // from run_module
#define uu_kx0z AC_uu_kx0z
run_const int AC_oo_kx0z // from run_module
#define oo_kx0z AC_oo_kx0z
run_const int AC_bb_kx0z // from run_module
#define bb_kx0z AC_bb_kx0z
run_const int AC_jj_kx0z // from run_module
#define jj_kx0z AC_jj_kx0z
run_const int AC_bb_k00z // from run_module
#define bb_k00z AC_bb_k00z
run_const int AC_ee_k00z // from run_module
#define ee_k00z AC_ee_k00z
run_const int AC_gwt_fft3d // from run_module
#define gwt_fft3d AC_gwt_fft3d
run_const int AC_em_specflux // from run_module
#define em_specflux AC_em_specflux
run_const int AC_hm_specflux // from run_module
#define hm_specflux AC_hm_specflux
run_const int AC_hc_specflux // from run_module
#define hc_specflux AC_hc_specflux
run_const real AC_fbcx_bot[mcom] // from run_module
#define fbcx_bot AC_fbcx_bot
run_const real AC_fbcx_top[mcom] // from run_module
#define fbcx_top AC_fbcx_top
run_const real AC_fbcy_bot[mcom] // from run_module
#define fbcy_bot AC_fbcy_bot
run_const real AC_fbcy_top[mcom] // from run_module
#define fbcy_top AC_fbcy_top
run_const real AC_fbcz_bot[mcom] // from run_module
#define fbcz_bot AC_fbcz_bot
run_const real AC_fbcz_top[mcom] // from run_module
#define fbcz_top AC_fbcz_top
run_const int AC_lreset_boundary_values // from run_module
#define lreset_boundary_values AC_lreset_boundary_values
run_const real AC_udrift_bc // from run_module
#define udrift_bc AC_udrift_bc
run_const real AC_xfreeze_square // from run_module
#define xfreeze_square AC_xfreeze_square
run_const real AC_yfreeze_square // from run_module
#define yfreeze_square AC_yfreeze_square
run_const real AC_rfreeze_int // from run_module
#define rfreeze_int AC_rfreeze_int
run_const real AC_rfreeze_ext // from run_module
#define rfreeze_ext AC_rfreeze_ext
run_const real AC_wfreeze // from run_module
#define wfreeze AC_wfreeze
run_const real AC_wfreeze_int // from run_module
#define wfreeze_int AC_wfreeze_int
run_const real AC_wfreeze_ext // from run_module
#define wfreeze_ext AC_wfreeze_ext
run_const real AC_wborder // from run_module
#define wborder AC_wborder
run_const real AC_wborder_int // from run_module
#define wborder_int AC_wborder_int
run_const real AC_wborder_ext // from run_module
#define wborder_ext AC_wborder_ext
run_const real AC_tborder // from run_module
#define tborder AC_tborder
run_const real AC_fshift_int // from run_module
#define fshift_int AC_fshift_int
run_const real AC_fshift_ext // from run_module
#define fshift_ext AC_fshift_ext
run_const real AC_theta_lower_border // from run_module
#define theta_lower_border AC_theta_lower_border
run_const real AC_wborder_theta_lower // from run_module
#define wborder_theta_lower AC_wborder_theta_lower
run_const real AC_theta_upper_border // from run_module
#define theta_upper_border AC_theta_upper_border
run_const real AC_wborder_theta_upper // from run_module
#define wborder_theta_upper AC_wborder_theta_upper
run_const real AC_fraction_tborder // from run_module
#define fraction_tborder AC_fraction_tborder
run_const int AC_lmeridional_border_drive // from run_module
#define lmeridional_border_drive AC_lmeridional_border_drive
run_const real AC_border_frac_x[2] // from run_module
#define border_frac_x AC_border_frac_x
run_const real AC_border_frac_y[2] // from run_module
#define border_frac_y AC_border_frac_y
run_const real AC_border_frac_z[2] // from run_module
#define border_frac_z AC_border_frac_z
run_const real AC_border_frac_r[2] // from run_module
#define border_frac_r AC_border_frac_r
run_const int AC_lborder_hyper_diff // from run_module
#define lborder_hyper_diff AC_lborder_hyper_diff
run_const int AC_lfrozen_bcs_x // from run_module
#define lfrozen_bcs_x AC_lfrozen_bcs_x
run_const int AC_lfrozen_bcs_y // from run_module
#define lfrozen_bcs_y AC_lfrozen_bcs_y
run_const int AC_lfrozen_bcs_z // from run_module
#define lfrozen_bcs_z AC_lfrozen_bcs_z
run_const int AC_lstop_on_ioerror // from run_module
#define lstop_on_ioerror AC_lstop_on_ioerror
run_const int AC_aux_count // from run_module
#define aux_count AC_aux_count
run_const int AC_mvar_io // from run_module
#define mvar_io AC_mvar_io
run_const int AC_mvar_down // from run_module
#define mvar_down AC_mvar_down
run_const int AC_maux_down // from run_module
#define maux_down AC_maux_down
run_const int AC_mskipvar // from run_module
#define mskipvar AC_mskipvar
run_const int AC_iwig // from run_module
#define iwig AC_iwig
run_const int AC_nfilter // from run_module
#define nfilter AC_nfilter
run_const real AC_awig // from run_module
#define awig AC_awig
run_const int AC_lrmwig_rho // from run_module
#define lrmwig_rho AC_lrmwig_rho
run_const int AC_lrmwig_full // from run_module
#define lrmwig_full AC_lrmwig_full
run_const int AC_lrmwig_xyaverage // from run_module
#define lrmwig_xyaverage AC_lrmwig_xyaverage
run_const int AC_init_loops // from run_module
#define init_loops AC_init_loops
run_const int AC_lfold_df // from run_module
#define lfold_df AC_lfold_df
run_const int AC_lfold_df_3points // from run_module
#define lfold_df_3points AC_lfold_df_3points
run_const int AC_lshift_datacube_x // from run_module
#define lshift_datacube_x AC_lshift_datacube_x
run_const int AC_lkinflow_as_aux // from run_module
#define lkinflow_as_aux AC_lkinflow_as_aux
run_const real AC_ampl_kinflow_x // from run_module
#define ampl_kinflow_x AC_ampl_kinflow_x
run_const real AC_ampl_kinflow_y // from run_module
#define ampl_kinflow_y AC_ampl_kinflow_y
run_const real AC_ampl_kinflow_z // from run_module
#define ampl_kinflow_z AC_ampl_kinflow_z
run_const real AC_kx_kinflow // from run_module
#define kx_kinflow AC_kx_kinflow
run_const real AC_ky_kinflow // from run_module
#define ky_kinflow AC_ky_kinflow
run_const real AC_kz_kinflow // from run_module
#define kz_kinflow AC_kz_kinflow
run_const real AC_dtphase_kinflow // from run_module
#define dtphase_kinflow AC_dtphase_kinflow
run_const int AC_lfargo_advection // from run_module
#define lfargo_advection AC_lfargo_advection
run_const int AC_lcorotational_frame // from run_module
#define lcorotational_frame AC_lcorotational_frame
run_const real AC_rcorot // from run_module
#define rcorot AC_rcorot
run_const real AC_omega_corot // from run_module
#define omega_corot AC_omega_corot
run_const int AC_llocal_iso // from run_module
#define llocal_iso AC_llocal_iso
run_const int AC_lisotropic_advection // from run_module
#define lisotropic_advection AC_lisotropic_advection
run_const int AC_lreport_undefined_diagnostics // from run_module
#define lreport_undefined_diagnostics AC_lreport_undefined_diagnostics
run_const real AC_ttransient // from run_module
#define ttransient AC_ttransient
run_const real AC_b_ell // from run_module
#define b_ell AC_b_ell
run_const real AC_rbound // from run_module
#define rbound AC_rbound
run_const real AC_grads0 // from run_module
#define grads0 AC_grads0
run_const int AC_lmonolithic_io // from run_module
#define lmonolithic_io AC_lmonolithic_io
run_const int AC_lrescaling_magnetic // from run_module
#define lrescaling_magnetic AC_lrescaling_magnetic
run_const int AC_lrescaling_testscalar // from run_module
#define lrescaling_testscalar AC_lrescaling_testscalar
run_const int AC_lrescaling_testfield // from run_module
#define lrescaling_testfield AC_lrescaling_testfield
run_const real AC_re_mesh // from run_module
#define re_mesh AC_re_mesh
run_const int AC_ldynamical_diffusion // from run_module
#define ldynamical_diffusion AC_ldynamical_diffusion
run_const int AC_ldyndiff_useumax // from run_module
#define ldyndiff_useumax AC_ldyndiff_useumax
run_const int AC_lstratz // from run_module
#define lstratz AC_lstratz
run_const int AC_lnoghost_strati // from run_module
#define lnoghost_strati AC_lnoghost_strati
run_const real AC_tau_aver1 // from run_module
#define tau_aver1 AC_tau_aver1
run_const real AC_lambda5 // from run_module
#define lambda5 AC_lambda5
run_const int AC_lmultithread // from run_module
#define lmultithread AC_lmultithread
run_const int AC_l1dphiavg_save // from run_module
#define l1dphiavg_save AC_l1dphiavg_save
run_const int AC_l1davgfirst_save // from run_module
#define l1davgfirst_save AC_l1davgfirst_save
run_const int AC_ldiagnos_save // from run_module
#define ldiagnos_save AC_ldiagnos_save
run_const int AC_l2davgfirst_save // from run_module
#define l2davgfirst_save AC_l2davgfirst_save
run_const int AC_lout_save // from run_module
#define lout_save AC_lout_save
run_const int AC_l1davg_save // from run_module
#define l1davg_save AC_l1davg_save
run_const int AC_l2davg_save // from run_module
#define l2davg_save AC_l2davg_save
run_const int AC_lout_sound_save // from run_module
#define lout_sound_save AC_lout_sound_save
run_const int AC_lvideo_save // from run_module
#define lvideo_save AC_lvideo_save
run_const int AC_lchemistry_diag_save // from run_module
#define lchemistry_diag_save AC_lchemistry_diag_save
run_const real AC_t1ddiagnos_save // from run_module
#define t1ddiagnos_save AC_t1ddiagnos_save
run_const real AC_t2davgfirst_save // from run_module
#define t2davgfirst_save AC_t2davgfirst_save
run_const real AC_tslice_save // from run_module
#define tslice_save AC_tslice_save
run_const real AC_tsound_save // from run_module
#define tsound_save AC_tsound_save
run_const int AC_num_helper_threads // from run_module
#define num_helper_threads AC_num_helper_threads
run_const int AC_thread_id // from run_module
#define thread_id AC_thread_id
// from run_module

run_const int AC_l2 // from run_module
#define l2 AC_l2
run_const int AC_m2 // from run_module
#define m2 AC_m2
run_const int AC_n2 // from run_module
#define n2 AC_n2
run_const int AC_l2i // from run_module
#define l2i AC_l2i
run_const int AC_m2i // from run_module
#define m2i AC_m2i
run_const int AC_n2i // from run_module
#define n2i AC_n2i
run_const bool AC_ltest_bcs // from run_module
#define ltest_bcs AC_ltest_bcs
run_const real AC_fbcx[mcom][2] // from run_module
#define fbcx AC_fbcx
run_const real AC_fbcy[mcom][2] // from run_module
#define fbcy AC_fbcy
run_const real AC_fbcz[mcom][2] // from run_module
#define fbcz AC_fbcz
run_const real AC_fbcy_1[mcom][2] // from run_module
#define fbcy_1 AC_fbcy_1
run_const real AC_fbcz_1[mcom][2] // from run_module
#define fbcz_1 AC_fbcz_1
run_const real AC_fbcx_2[mcom][2] // from run_module
#define fbcx_2 AC_fbcx_2
run_const real AC_fbcy_2[mcom][2] // from run_module
#define fbcy_2 AC_fbcy_2
run_const real AC_fbcz_2[mcom][2] // from run_module
#define fbcz_2 AC_fbcz_2
run_const real AC_dx2_bound[2*nghost+1] // from run_module
#define dx2_bound AC_dx2_bound
run_const real AC_dy2_bound[2*nghost+1] // from run_module
#define dy2_bound AC_dy2_bound
run_const real AC_dz2_bound[2*nghost+1] // from run_module
#define dz2_bound AC_dz2_bound
