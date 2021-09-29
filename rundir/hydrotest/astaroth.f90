
!  -*-f90-*-  (for emacs)    vim:set filetype=fortran:  (for vim)

! Utils (see astaroth_fortran.cc for definitions)
external achostupdatebuiltinparams
external acgetdevicecount

! Device interface (see astaroth_fortran.cc for definitions)
external acdevicecreate, acdevicedestroy
external acdeviceprintinfo
external acdeviceloadmeshinfo
external acdeviceloadmesh, acdevicestoremesh
external acdeviceintegratesubstep
external acdeviceperiodicboundconds
external acdeviceswapbuffers
external acdevicereducescal, acdevicereducevec
external acdevicesynchronizestream
  
integer(c_int), parameter :: AC_nx = 0
integer(c_int), parameter :: AC_ny = 1
integer(c_int), parameter :: AC_nz = 2
integer(c_int), parameter :: AC_mx = 3
integer(c_int), parameter :: AC_my = 4
integer(c_int), parameter :: AC_mz = 5
integer(c_int), parameter :: AC_nx_min = 6
integer(c_int), parameter :: AC_ny_min = 7
integer(c_int), parameter :: AC_nz_min = 8
integer(c_int), parameter :: AC_nx_max = 9
integer(c_int), parameter :: AC_ny_max = 10
integer(c_int), parameter :: AC_nz_max = 11
integer(c_int), parameter :: AC_mxy = 12
integer(c_int), parameter :: AC_nxy = 13
integer(c_int), parameter :: AC_nxyz = 14
integer(c_int), parameter :: AC_bc_type_bot_x = 15
integer(c_int), parameter :: AC_bc_type_bot_y = 16
integer(c_int), parameter :: AC_bc_type_bot_z = 17
integer(c_int), parameter :: AC_bc_type_top_x = 18
integer(c_int), parameter :: AC_bc_type_top_y = 19
integer(c_int), parameter :: AC_bc_type_top_z = 20
integer(c_int), parameter :: AC_max_steps = 21
integer(c_int), parameter :: AC_save_steps = 22
integer(c_int), parameter :: AC_bin_steps = 23
integer(c_int), parameter :: AC_start_step = 24
integer(c_int), parameter :: AC_NUM_INT_PARAMS = 25

integer(c_int), parameter :: AC_global_grid_n = 0
integer(c_int), parameter :: AC_multigpu_offset = 1
integer(c_int), parameter :: AC_NUM_INT3_PARAMS = 2

integer(c_int), parameter :: AC_dt = 0
integer(c_int), parameter :: AC_dsx = 1
integer(c_int), parameter :: AC_dsy = 2
integer(c_int), parameter :: AC_dsz = 3
integer(c_int), parameter :: AC_inv_dsx = 4
integer(c_int), parameter :: AC_inv_dsy = 5
integer(c_int), parameter :: AC_inv_dsz = 6
integer(c_int), parameter :: AC_max_time = 7
integer(c_int), parameter :: AC_dsmin = 8
integer(c_int), parameter :: AC_xlen = 9
integer(c_int), parameter :: AC_ylen = 10
integer(c_int), parameter :: AC_zlen = 11
integer(c_int), parameter :: AC_xorig = 12
integer(c_int), parameter :: AC_yorig = 13
integer(c_int), parameter :: AC_zorig = 14
integer(c_int), parameter :: AC_unit_density = 15
integer(c_int), parameter :: AC_unit_velocity = 16
integer(c_int), parameter :: AC_unit_length = 17
integer(c_int), parameter :: AC_unit_magnetic = 18
integer(c_int), parameter :: AC_star_pos_x = 19
integer(c_int), parameter :: AC_star_pos_y = 20
integer(c_int), parameter :: AC_star_pos_z = 21
integer(c_int), parameter :: AC_M_star = 22
integer(c_int), parameter :: AC_sink_pos_x = 23
integer(c_int), parameter :: AC_sink_pos_y = 24
integer(c_int), parameter :: AC_sink_pos_z = 25
integer(c_int), parameter :: AC_M_sink = 26
integer(c_int), parameter :: AC_M_sink_init = 27
integer(c_int), parameter :: AC_M_sink_Msun = 28
integer(c_int), parameter :: AC_soft = 29
integer(c_int), parameter :: AC_accretion_range = 30
integer(c_int), parameter :: AC_switch_accretion = 31
integer(c_int), parameter :: AC_cdt = 32
integer(c_int), parameter :: AC_cdtv = 33
integer(c_int), parameter :: AC_cdts = 34
integer(c_int), parameter :: AC_nu_visc = 35
integer(c_int), parameter :: AC_cs_sound = 36
integer(c_int), parameter :: AC_eta = 37
integer(c_int), parameter :: AC_mu0 = 38
integer(c_int), parameter :: AC_cp_sound = 39
integer(c_int), parameter :: AC_gamma = 40
integer(c_int), parameter :: AC_cv_sound = 41
integer(c_int), parameter :: AC_lnT0 = 42
integer(c_int), parameter :: AC_lnrho0 = 43
integer(c_int), parameter :: AC_zeta = 44
integer(c_int), parameter :: AC_trans = 45
integer(c_int), parameter :: AC_nu_shock = 46
integer(c_int), parameter :: AC_bin_save_t = 47
integer(c_int), parameter :: AC_ampl_lnrho = 48
integer(c_int), parameter :: AC_ampl_uu = 49
integer(c_int), parameter :: AC_angl_uu = 50
integer(c_int), parameter :: AC_lnrho_edge = 51
integer(c_int), parameter :: AC_lnrho_out = 52
integer(c_int), parameter :: AC_ampl_aa = 53
integer(c_int), parameter :: AC_init_k_wave = 54
integer(c_int), parameter :: AC_init_sigma_hel = 55
integer(c_int), parameter :: AC_forcing_magnitude = 56
integer(c_int), parameter :: AC_relhel = 57
integer(c_int), parameter :: AC_kmin = 58
integer(c_int), parameter :: AC_kmax = 59
integer(c_int), parameter :: AC_forcing_phase = 60
integer(c_int), parameter :: AC_k_forcex = 61
integer(c_int), parameter :: AC_k_forcey = 62
integer(c_int), parameter :: AC_k_forcez = 63
integer(c_int), parameter :: AC_kaver = 64
integer(c_int), parameter :: AC_ff_hel_rex = 65
integer(c_int), parameter :: AC_ff_hel_rey = 66
integer(c_int), parameter :: AC_ff_hel_rez = 67
integer(c_int), parameter :: AC_ff_hel_imx = 68
integer(c_int), parameter :: AC_ff_hel_imy = 69
integer(c_int), parameter :: AC_ff_hel_imz = 70
integer(c_int), parameter :: AC_G_const = 71
integer(c_int), parameter :: AC_GM_star = 72
integer(c_int), parameter :: AC_unit_mass = 73
integer(c_int), parameter :: AC_sq2GM_star = 74
integer(c_int), parameter :: AC_cs2_sound = 75
integer(c_int), parameter :: AC_NUM_REAL_PARAMS = 76

integer(c_int), parameter :: AC_NUM_REAL3_PARAMS = 0

integer(c_int), parameter :: VTXBUF_LNRHO = 0
integer(c_int), parameter :: VTXBUF_UUX = 1
integer(c_int), parameter :: VTXBUF_UUY = 2
integer(c_int), parameter :: VTXBUF_UUZ = 3
integer(c_int), parameter :: AC_NUM_VTXBUF_HANDLES = 4

integer(c_int), parameter :: AC_NUM_SCALARRAY_HANDLES = 0

integer(c_int), parameter :: STREAM_0 = 0
integer(c_int), parameter :: STREAM_1 = 1
integer(c_int), parameter :: STREAM_2 = 2
integer(c_int), parameter :: STREAM_3 = 3
integer(c_int), parameter :: STREAM_4 = 4
integer(c_int), parameter :: STREAM_5 = 5
integer(c_int), parameter :: STREAM_6 = 6
integer(c_int), parameter :: STREAM_7 = 7
integer(c_int), parameter :: STREAM_8 = 8
integer(c_int), parameter :: STREAM_9 = 9
integer(c_int), parameter :: STREAM_10 = 10
integer(c_int), parameter :: STREAM_11 = 11
integer(c_int), parameter :: STREAM_12 = 12
integer(c_int), parameter :: STREAM_13 = 13
integer(c_int), parameter :: STREAM_14 = 14
integer(c_int), parameter :: STREAM_15 = 15
integer(c_int), parameter :: STREAM_16 = 16
integer(c_int), parameter :: STREAM_17 = 17
integer(c_int), parameter :: STREAM_18 = 18
integer(c_int), parameter :: STREAM_19 = 19
integer(c_int), parameter :: STREAM_20 = 20
integer(c_int), parameter :: STREAM_21 = 21
integer(c_int), parameter :: STREAM_22 = 22
integer(c_int), parameter :: STREAM_23 = 23
integer(c_int), parameter :: STREAM_24 = 24
integer(c_int), parameter :: STREAM_25 = 25
integer(c_int), parameter :: STREAM_26 = 26
integer(c_int), parameter :: STREAM_27 = 27
integer(c_int), parameter :: STREAM_28 = 28
integer(c_int), parameter :: STREAM_29 = 29
integer(c_int), parameter :: STREAM_30 = 30
integer(c_int), parameter :: STREAM_31 = 31
integer(c_int), parameter :: NUM_STREAMS = 32
integer(c_int), parameter :: STREAM_DEFAULT = STREAM_0
integer(c_int), parameter :: STREAM_ALL = NUM_STREAMS
integer(c_int), parameter :: RTYPE_MAX = 0
integer(c_int), parameter :: RTYPE_MIN = 1
integer(c_int), parameter :: RTYPE_RMS = 2
integer(c_int), parameter :: RTYPE_RMS_EXP = 3
integer(c_int), parameter :: RTYPE_ALFVEN_MAX = 4
integer(c_int), parameter :: RTYPE_ALFVEN_MIN = 5
integer(c_int), parameter :: RTYPE_ALFVEN_RMS = 6
integer(c_int), parameter :: RTYPE_SUM = 7
integer(c_int), parameter :: NUM_REDUCTION_TYPES = 8
integer(c_int), parameter :: AC_BOUNDCOND_PERIODIC = 0
integer(c_int), parameter :: AC_BOUNDCOND_SYMMETRIC = 1
integer(c_int), parameter :: AC_BOUNDCOND_ANTISYMMETRIC = 2

type, bind(C) :: AcMeshInfo
  integer(c_int), dimension(AC_NUM_INT_PARAMS)      :: int_params
  integer(c_int), dimension(AC_NUM_INT3_PARAMS, 3)  :: int3_params
  real, dimension(AC_NUM_REAL_PARAMS)               :: real_params
  real, dimension(AC_NUM_REAL3_PARAMS, 3)           :: real3_params
end type AcMeshInfo
  
