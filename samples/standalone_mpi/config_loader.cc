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
    acPushToConfig(config,AC_len,config[AC_ds]*config[AC_mgrid]);
    acPushToConfig(config,AC_origin,AcReal(.5) * config[AC_len]);


#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif
}
