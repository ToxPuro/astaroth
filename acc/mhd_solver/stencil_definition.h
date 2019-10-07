#pragma once
#define LDENSITY (1)
#define LHYDRO (1)
#define LMAGNETIC (1)
#define LENTROPY (1)
#define LTEMPERATURE (0)
#define LFORCING (1)
#define LUPWD (1)
#define LSINK (0)

#define AC_THERMAL_CONDUCTIVITY (AcReal(0.001)) // TODO: make an actual config parameter
#define H_CONST (0) // TODO: make an actual config parameter
#define C_CONST (0) // TODO: make an actual config parameter

// Int params
uniform int AC_max_steps;
uniform int AC_save_steps;
uniform int AC_bin_steps;
uniform int AC_bc_type;
uniform int AC_start_step;

// Real params
uniform Scalar AC_dt;
uniform Scalar AC_max_time;
// Spacing
uniform Scalar AC_dsmin;
// physical grid
uniform Scalar AC_xlen;
uniform Scalar AC_ylen;
uniform Scalar AC_zlen;
uniform Scalar AC_xorig;
uniform Scalar AC_yorig;
uniform Scalar AC_zorig;
// Physical units
uniform Scalar AC_unit_density;
uniform Scalar AC_unit_velocity;
uniform Scalar AC_unit_length;
// properties of gravitating star
uniform Scalar AC_star_pos_x;
uniform Scalar AC_star_pos_y;
uniform Scalar AC_star_pos_z;
uniform Scalar AC_M_star;
// properties of sink particle
uniform Scalar AC_sink_pos_x;
uniform Scalar AC_sink_pos_y;
uniform Scalar AC_sink_pos_z;
uniform Scalar AC_M_sink;
uniform Scalar AC_M_sink_init;
uniform Scalar AC_M_sink_Msun;
uniform Scalar AC_soft;
uniform Scalar AC_accretion_range;
uniform Scalar AC_switch_accretion;
//  Run params
uniform Scalar AC_cdt;
uniform Scalar AC_cdtv;
uniform Scalar AC_cdts;
uniform Scalar AC_nu_visc;
uniform Scalar AC_cs_sound;
uniform Scalar AC_eta;
uniform Scalar AC_mu0;
uniform Scalar AC_cp_sound;
uniform Scalar AC_gamma;
uniform Scalar AC_cv_sound;
uniform Scalar AC_lnT0;
uniform Scalar AC_lnrho0;
uniform Scalar AC_zeta;
uniform Scalar AC_trans;
//  Other
uniform Scalar AC_bin_save_t;
//  Initial condition params
uniform Scalar AC_ampl_lnrho;
uniform Scalar AC_ampl_uu;
uniform Scalar AC_angl_uu;
uniform Scalar AC_lnrho_edge;
uniform Scalar AC_lnrho_out;
//  Forcing parameters. User configured.
uniform Scalar AC_forcing_magnitude;
uniform Scalar AC_relhel;
uniform Scalar AC_kmin;
uniform Scalar AC_kmax;
//  Forcing parameters. Set by the generator.
uniform Scalar AC_forcing_phase;
uniform Scalar AC_k_forcex;
uniform Scalar AC_k_forcey;
uniform Scalar AC_k_forcez;
uniform Scalar AC_kaver;
uniform Scalar AC_ff_hel_rex;
uniform Scalar AC_ff_hel_rey;
uniform Scalar AC_ff_hel_rez;
uniform Scalar AC_ff_hel_imx;
uniform Scalar AC_ff_hel_imy;
uniform Scalar AC_ff_hel_imz;
//  Additional helper params  //  (deduced from other params do not set these directly!)
uniform Scalar AC_G_const;
uniform Scalar AC_GM_star;
uniform Scalar AC_unit_mass;
uniform Scalar AC_sq2GM_star;
uniform Scalar AC_cs2_sound;

/*
 * =============================================================================
 * User-defined vertex buffers
 * =============================================================================
 */
#if LENTROPY
uniform ScalarField VTXBUF_LNRHO;
uniform ScalarField VTXBUF_UUX;
uniform ScalarField VTXBUF_UUY;
uniform ScalarField VTXBUF_UUZ;
uniform ScalarField VTXBUF_AX;
uniform ScalarField VTXBUF_AY;
uniform ScalarField VTXBUF_AZ;
uniform ScalarField VTXBUF_ENTROPY;
#elif LMAGNETIC
uniform ScalarField VTXBUF_LNRHO;
uniform ScalarField VTXBUF_UUX;
uniform ScalarField VTXBUF_UUY;
uniform ScalarField VTXBUF_UUZ;
uniform ScalarField VTXBUF_AX;
uniform ScalarField VTXBUF_AY;
uniform ScalarField VTXBUF_AZ;
#elif LHYDRO
uniform ScalarField VTXBUF_LNRHO;
uniform ScalarField VTXBUF_UUX;
uniform ScalarField VTXBUF_UUY;
uniform ScalarField VTXBUF_UUZ;
#else
uniform ScalarField VTXBUF_LNRHO;
#endif

#if LSINK
uniform ScalarField VTXBUF_ACCRETION;
#endif
