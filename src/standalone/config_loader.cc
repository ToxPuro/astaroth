/*
    Copyright (C) 2014-2018, Johannes Pekkilae, Miikka Vaeisalae.

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

#include <limits.h> // UINT_MAX
#include <stdint.h> // uint8_t, uint32_t
#include <stdio.h>  // print
#include <string.h> // memset

#include "core/errchk.h"
#include "core/math_utils.h"

static inline void
print(const AcMeshInfo& config)
{
    for (int i = 0; i < NUM_INT_PARAM_TYPES; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_REAL_PARAM_TYPES; ++i)
        printf("[%s]: %g\n", realparam_names[i], double(config.real_params[i]));
}

/**
 \brief Find the index of the keyword in names
 \return Index in range 0...n if the keyword is in names. -1 if the keyword was
 not found.
 */
static int
find_str(const char keyword[], const char* names[], const int& n)
{
    for (int i = 0; i < n; ++i)
        if (!strcmp(keyword, names[i]))
            return i;

    return -1;
}

static void
parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    printf("Config file path: \n %s \n ", path);
    ERRCHK(fp != NULL);

    const size_t BUF_SIZE = 128;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %s", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;

        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAM_TYPES)) >= 0)
            config->int_params[idx] = atoi(value);
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAM_TYPES)) >= 0)
            config->real_params[idx] = AcReal(atof(value));
    }

    fclose(fp);
}

void
update_config(AcMeshInfo* config)
{
    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_max] = config->int_params[AC_ny] + STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_max] = config->int_params[AC_nz] + STENCIL_ORDER / 2;

    // Spacing
    config->real_params[AC_inv_dsx] = AcReal(1.) / config->real_params[AC_dsx];
    config->real_params[AC_inv_dsy] = AcReal(1.) / config->real_params[AC_dsy];
    config->real_params[AC_inv_dsz] = AcReal(1.) / config->real_params[AC_dsz];
    config->real_params[AC_dsmin]   = min(
        config->real_params[AC_dsx], min(config->real_params[AC_dsy], config->real_params[AC_dsz]));

    // Real grid coordanates (DEFINE FOR GRID WITH THE GHOST ZONES)
    config->real_params[AC_xlen] = config->real_params[AC_dsx] * config->int_params[AC_mx];
    config->real_params[AC_ylen] = config->real_params[AC_dsy] * config->int_params[AC_my];
    config->real_params[AC_zlen] = config->real_params[AC_dsz] * config->int_params[AC_mz];

    config->real_params[AC_xorig] = AcReal(.5) * config->real_params[AC_xlen];
    config->real_params[AC_yorig] = AcReal(.5) * config->real_params[AC_ylen];
    config->real_params[AC_zorig] = AcReal(.5) * config->real_params[AC_zlen];

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    // Real helpers
    config->real_params[AC_cs2_sound] = config->real_params[AC_cs_sound] *
                                        config->real_params[AC_cs_sound];

    config->real_params[AC_cv_sound] = config->real_params[AC_cp_sound] /
                                       config->real_params[AC_gamma];

    AcReal G_CONST_CGS = AcReal(
        6.674e-8);                   // g/cm3/s GGS definition //TODO define in a separate module
    AcReal M_sun = AcReal(1.989e33); // g solar mass

    config->real_params[AC_M_star] = config->real_params[AC_M_star] * M_sun /
                                     ((config->real_params[AC_unit_length] *
                                       config->real_params[AC_unit_length] *
                                       config->real_params[AC_unit_length]) *
                                      config->real_params[AC_unit_density]);

    config->real_params[AC_G_CONST] = G_CONST_CGS / ((config->real_params[AC_unit_velocity] *
                                                      config->real_params[AC_unit_velocity]) /
                                                     (config->real_params[AC_unit_density] *
                                                      config->real_params[AC_unit_length]));

    config->real_params[AC_GM_star] = config->real_params[AC_M_star] *
                                      config->real_params[AC_G_CONST];
    config->real_params[AC_sq2GM_star] = AcReal(sqrt(AcReal(2) * config->real_params[AC_GM_star]));

    const bool print_config = true;
    if (print_config) {
        printf("###############################################################"
               "\n");
        printf("Config dimensions recalculated:\n");
        print(*config);
        printf("###############################################################"
               "\n");
    }
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return 0 on success, -1 if there are potentially uninitialized values.
*/
int
load_config(AcMeshInfo* config)
{
    int retval = 0;
    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    parse_config(CONFIG_PATH "astaroth.conf", config);
    update_config(config);

    // sizeof(config) must be a multiple of 4 bytes for this to work
    ERRCHK(sizeof(*config) % sizeof(uint32_t) == 0);
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        if (((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF) {
            WARNING("Some config values may be uninitialized. "
                    "See that all are defined in astaroth.conf\n");
            retval = -1;
        }
    }
    return retval;
}
