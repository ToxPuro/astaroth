#!/usr/bin/env python3
# %%
import numpy as np

# AcReal
# calc_timestep(const AcReal uumax, const AcReal vAmax, const AcReal shock_max, const AcMeshInfo info)
# {
#     static bool warning_shown = false;
#     if (!warning_shown) {
#         WARNING("Note: shock not used in timestep calculation");
#         warning_shown = true;
#     }

#     const long double cdt  = (long double)info.real_params[AC_cdt];
#     const long double cdtv = (long double)info.real_params[AC_cdtv];
#     // const long double cdts     = (long double)info.real_params[AC_cdts];
#     const long double cs2_sound = (long double)info.real_params[AC_cs2_sound];
#     const long double nu_visc   = (long double)info.real_params[AC_nu_visc];
#     const long double eta       = (long double)info.real_params[AC_eta];
#     const long double eta_tfm   = (long double)info.real_params[AC_eta_tfm];
#     const long double chi   = 0; // (long double)info.real_params[AC_chi]; // TODO not calculated
#     const long double gamma = (long double)info.real_params[AC_gamma];
#     const long double dsmin = (long double)min((long double)info.real_params[AC_dsx],
#                                                min((long double)info.real_params[AC_dsy],
#                                                    (long double)info.real_params[AC_dsz]));
#     // const long double nu_shock = (long double)info.real_params[AC_nu_shock];

#     // Old ones from legacy Astaroth
#     // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
#     // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

#     // New, closer to the actual Courant timestep
#     // See Pencil Code user manual p. 38 (timestep section)
#     const long double uu_dt = cdt * dsmin /
#                               (fabsl((long double)uumax) +
#                                sqrtl(cs2_sound + (long double)vAmax * (long double)vAmax));
#     const long double visc_dt = cdtv * dsmin * dsmin /
#                                 (max(max(nu_visc, max(eta, eta_tfm)), gamma * chi));
#     //+ nu_shock * (long double)shock_max);

#     const long double dt = min(uu_dt, visc_dt);
#     ERRCHK_ALWAYS(!isnan(dt) && !isinf(dt));
#     return (AcReal)dt;
# }


umax = 1.407
dt_model = 0.0344 
cdt = 0.4 # 0.4 in Astaroth
cdtv = 0.25 # 0.3 in Astroth
dx = dy = dz = 0.19634955
dsmin = min(dx, min(dy, dz))

cs_sound = 1 # 1 or 0
cs2_sound = cs_sound * cs_sound

vAmax = 10

nu_visc = 5e-2

eta = 1e-2
eta_tfm = 5e-2

gamma = 0.002 # 0.5 in astaroth
chi = 45

uu_dt = (cdt * dsmin) / (umax + np.sqrt(cs2_sound + vAmax * vAmax))
visc_dt = (cdtv * dsmin * dsmin) / (max(max(nu_visc, max(eta, eta_tfm)), gamma * chi))

print(f'uu_dt: {uu_dt}')
print(f'visc_dt: {visc_dt}')
print(f'dt: {min(uu_dt, visc_dt)}')
