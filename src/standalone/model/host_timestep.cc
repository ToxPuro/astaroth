/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "host_timestep.h"

#include "math_utils.h"

static AcReal timescale = AcReal(1.0);

AcReal
host_timestep(const AcReal& umax, const AcMeshInfo& mesh_info)
{
    const long double cdt  = mesh_info.real_params[AC_cdt];
    const long double cdtv = mesh_info.real_params[AC_cdtv];
    // const long double cdts     = mesh_info.real_params[AC_cdts];
    const long double cs2_sound = mesh_info.real_params[AC_cs2_sound];
    const long double nu_visc   = mesh_info.real_params[AC_nu_visc];
    const long double eta       = mesh_info.real_params[AC_eta];
    const long double chi       = 0; // mesh_info.real_params[AC_chi]; // TODO not calculated
    const long double gamma     = mesh_info.real_params[AC_gamma];
    const long double dsmin     = mesh_info.real_params[AC_dsmin];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (umax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt   = cdt * dsmin / (fabsl(umax) + sqrtl(cs2_sound + 0.0l));
    const long double visc_dt = cdtv * dsmin * dsmin /
                                max(max(nu_visc, eta),
                                    max(gamma, chi)); // + 1; // TODO NOTE: comment the +1 out to
                                                      // get scientifically accurate results
    // MV: White the +1? It was messing up my computations!

    const long double dt = min(uu_dt, visc_dt);
    return AcReal(timescale) * AcReal(dt);
}

void
set_timescale(const AcReal scale)
{
    timescale = scale;
}
