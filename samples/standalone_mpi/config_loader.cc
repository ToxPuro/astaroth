/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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
#include "config_loader.h"

#include <stdint.h> // uint8_t, uint32_t
#include <stdio.h>  // print
#include <string.h> // memset

#include "errchk.h"
#include "math_utils.h"

void
set_extra_config_params(AcMeshInfo* config_ptr)
{

    AcMeshInfo& config = *config_ptr;
    // Spacing
    /*
    // %JP: AC_inv_ds[xyz] now calculated inside the mhd kernel
    // TP:  AC_inv_ds      now calculated when updating built-in variables
    config->real_params[AC_inv_dsx] = AcReal(1.) / config->real_params[AC_dsx];
    config->real_params[AC_inv_dsy] = AcReal(1.) / config->real_params[AC_dsy];
    config->real_params[AC_inv_dsz] = AcReal(1.) / config->real_params[AC_dsz];
    */
    //TP: builtin-variable
    //config[AC_dsmin] = min(config[AC_dsx],
    //                                    min(config[AC_dsy],
    //                                        config[AC_dsz]));

    // Real grid coordanates (DEFINE FOR GRID WITH THE GHOST ZONES)
    config[AC_len]  = config[AC_ds]*config[AC_mgrid];
    config[AC_origin] = AcReal(.5) * config[AC_len];

    // Real helpers
    config[AC_cs2_sound] = config[AC_cs_sound] *
                                        config[AC_cs_sound];

    config[AC_cv_sound] = config[AC_cp_sound] /
                                       config[AC_gamma];

    AcReal G_CONST_CGS = AcReal(
        6.674e-8); // cm^3/(g*s^2) GGS definition //TODO define in a separate module
    AcReal M_sun = AcReal(1.989e33); // g solar mass

    config[AC_unit_mass] = (config[AC_unit_length] *
                                         config[AC_unit_length] *
                                         config[AC_unit_length]) *
                                        config[AC_unit_density];

    config[AC_M_sink] = config[AC_M_sink_Msun] * M_sun /
                                     config[AC_unit_mass];
    config[AC_M_sink_init] = config[AC_M_sink_Msun] * M_sun /
                                          config[AC_unit_mass];

    config[AC_G_const] = G_CONST_CGS / ((config[AC_unit_velocity] *
                                                      config[AC_unit_velocity]) /
                                                     (config[AC_unit_density] *
                                                      config[AC_unit_length] *
                                                      config[AC_unit_length]));

    config[AC_sq2GM_star] = AcReal(sqrt(AcReal(2) * config[AC_GM_star]));

#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif
}
