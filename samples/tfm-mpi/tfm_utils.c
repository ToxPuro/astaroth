#include "tfm_utils.h"

#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "ini.h"

#include "acc-runtime/api/errchk.h"
#include "astaroth_forcing.h"

int
acParseArguments(const int argc, char* argv[], Arguments* args)
{
    // Default arguments
    args->config_path = NULL; // AC_DEFAULT_CONFIG;

    // Options
    const char short_options[]         = {"c:"};
    const struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"help", required_argument, 0, 'h'},
        {0, 0, 0, 0},
    };
    const char* explanations[] = {
        "Path to the config INI file",
        "Print this message",
    };

    // Parse
    int opt;
    int option_index;
    while ((opt = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
        switch (opt) {
        case 'c':
            args->config_path = optarg;
            break;
        // case 0:
        //     printf("Option with no short options\n");
        //     break;
        case 'h': /* Fallthrough */
        default:
            fprintf(stderr, "Usage: %s <options>\n", argv[0]);
            fprintf(stderr, "Options:\n");
            for (size_t i = 0; long_options[i].name != NULL; ++i) {
                fprintf(stderr, "\t--%s", long_options[i].name);
                if (long_options[i].has_arg == required_argument)
                    fprintf(stderr, " <argument>");
                fprintf(stderr, ": %s", explanations[i]);
                fprintf(stderr, "\n");
            }
            return -1;
        }
    }

    return 0;
}

static int
config_handler(void* user, const char* section, const char* name, const char* value)
{
    AcMeshInfo* info = (AcMeshInfo*)user;

    for (size_t i = 0; i < NUM_INT_PARAMS; ++i) {
        if (strcmp(name, intparam_names[i]) == 0) {
            info->int_params[i] = atoi(value);
            return 1;
        }
    }
    for (size_t i = 0; i < NUM_REAL_PARAMS; ++i) {
        if (strcmp(name, realparam_names[i]) == 0) {
            info->real_params[i] = atof(value);
            return 1;
        }
    }

    fprintf(stderr, "Invalid parameter in section [%s]: %s = %s\n", section, name, value);
    return 0;
}

int
acParseINI(const char* filepath, AcMeshInfo* info)
{
    // Initialize AcMeshInfo to default values
    for (size_t i = 0; i < NUM_INT_PARAMS; ++i)
        info->int_params[i] = INT_MIN;
    for (size_t i = 0; i < NUM_INT3_PARAMS; ++i)
        info->int3_params[i] = (int3){INT_MIN, INT_MIN, INT_MIN};
    for (size_t i = 0; i < NUM_REAL_PARAMS; ++i)
        info->real_params[i] = NAN;
    for (size_t i = 0; i < NUM_REAL3_PARAMS; ++i)
        info->real3_params[i] = (AcReal3){NAN, NAN, NAN};

    // Parse
    int retval = ini_parse(filepath, config_handler, info);

    // Error handling
    if (retval > 0)
        fprintf(stderr, "Error on line %d when parsing config \"%s\"\n", retval, filepath);
    else if (retval == -1)
        fprintf(stderr, "File open error %d when parsing config \"%s\"\n", retval, filepath);
    else if (retval == -2)
        fprintf(stderr,
                "Memory allocation error %d when parsing config \"%s\"\n",
                retval,
                filepath);
    else if (retval < 0)
        fprintf(stderr, "Unknown error %d when parsing config \"%s\"\n", retval, filepath);
    if (retval != 0)
        return -1;

    // Update and verification should be handled elsewhere
    // Update the rest of the parameters
    // acHostUpdateBuiltinParams(info);

    // Check for uninitialized values
    // for (size_t i = 0; i < NUM_INT_PARAMS; ++i)
    //     if (info->int_params[i] == INT_MIN)
    //         fprintf(stderr, "--- Warning: [%s] uninitialized ---\n", intparam_names[i]);
    // for (size_t i = 0; i < NUM_REAL_PARAMS; ++i)
    //     if (info->real_params[i] == (AcReal)NAN)
    //         fprintf(stderr, "--- Warning: [%s] uninitialized ---\n", realparam_names[i]);
    // return 0;
}

int
acPrintArguments(const Arguments args)
{
    printf("[config_path]: %s\n", args.config_path);
    return 0;
}

int
acHostUpdateLocalBuiltinParams(AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config->int_params[AC_nx] > 0);
    ERRCHK_ALWAYS(config->int_params[AC_ny] > 0);
    ERRCHK_ALWAYS(config->int_params[AC_nz] > 0);

    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;

    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_max] = config->int_params[AC_ny_min] + config->int_params[AC_ny];
    config->int_params[AC_nz_max] = config->int_params[AC_nz_min] + config->int_params[AC_nz];

    /*
    #ifdef AC_dsx
        printf("HELLO!\n");
        ERRCHK_ALWAYS(config->real_params[AC_dsx] > 0);
        config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsx]));
    #endif
    #ifdef AC_dsy
        ERRCHK_ALWAYS(config->real_params[AC_dsy] > 0);
        config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsy]));
    #endif
    #ifdef AC_dsz
        ERRCHK_ALWAYS(config->real_params[AC_dsz] > 0);
        config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsz]));
    #endif
    */

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    /* Multi-GPU params */
    // config->int3_params[AC_multigpu_offset] = (int3){0, 0, 0};
    // config->int3_params[AC_global_grid_n]   = (int3){
    //     config->int_params[AC_nx],
    //     config->int_params[AC_ny],
    //     config->int_params[AC_nz],
    // };

    return 0;
}

int
acHostUpdateForcingParams(AcMeshInfo* info)
{
    ForcingParams forcing_params = generateForcingParams(info->real_params[AC_relhel],
                                                         info->real_params[AC_forcing_magnitude],
                                                         info->real_params[AC_kmin],
                                                         info->real_params[AC_kmax]);
    loadForcingParamsToMeshInfo(forcing_params, info);

    return 0;
}

int
acHostUpdateMHDSpecificParams(AcMeshInfo* info)
{
    // Forcing
    acHostUpdateForcingParams(info);

    // Derived values
    info->real_params[AC_cs2_sound] = info->real_params[AC_cs_sound] *
                                      info->real_params[AC_cs_sound];

    // Other
    info->real_params[AC_center_x] = info->real_params[AC_global_sx] / 2;
    info->real_params[AC_center_y] = info->real_params[AC_global_sy] / 2;
    info->real_params[AC_center_z] = info->real_params[AC_global_sz] / 2;

    return 0;
}

int
acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info)
{
    info->real_params[AC_dsx] = info->real_params[AC_global_sx] /
                                (info->int_params[AC_global_nx] - 1);
    info->real_params[AC_dsy] = info->real_params[AC_global_sy] /
                                (info->int_params[AC_global_ny] - 1);
    info->real_params[AC_dsz] = info->real_params[AC_global_sz] /
                                (info->int_params[AC_global_nz] - 1);

    return 0;
}

int
acPrintMeshInfoTFM(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("[%s]: (%d, %d, %d)\n",
               int3param_names[i],
               config.int3_params[i].x,
               config.int3_params[i].y,
               config.int3_params[i].z);
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("[%s]: %g\n", realparam_names[i], (double)(config.real_params[i]));
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("[%s]: (%g, %g, %g)\n",
               real3param_names[i],
               (double)(config.real3_params[i].x),
               (double)(config.real3_params[i].y),
               (double)(config.real3_params[i].z));

    return 0;
}

static AcReal
max(const AcReal a, const AcReal b)
{
    return a >= b ? a : b;
}

static AcReal
min(const AcReal a, const AcReal b)
{
    return a < b ? a : b;
}

AcReal
calc_timestep(const AcReal uumax, const AcReal vAmax, const AcReal shock_max, const AcMeshInfo info)
{
    static bool warning_shown = false;
    if (!warning_shown) {
        WARNING("Note: shock not used in timestep calculation");
        warning_shown = true;
    }

    const long double cdt  = (long double)info.real_params[AC_cdt];
    const long double cdtv = (long double)info.real_params[AC_cdtv];
    // const long double cdts     = (long double)info.real_params[AC_cdts];
    const long double cs2_sound = (long double)info.real_params[AC_cs2_sound];
    const long double nu_visc   = (long double)info.real_params[AC_nu_visc];
    const long double eta       = (long double)info.real_params[AC_eta];
    const long double chi   = 0; // (long double)info.real_params[AC_chi]; // TODO not calculated
    const long double gamma = (long double)info.real_params[AC_gamma];
    const long double dsmin = (long double)min(info.real_params[AC_dsx],
                                               min(info.real_params[AC_dsy],
                                                   info.real_params[AC_dsz]));
    // const long double nu_shock = (long double)info.real_params[AC_nu_shock];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt = cdt * dsmin /
                              (fabsl((long double)uumax) +
                               sqrtl(cs2_sound + (long double)vAmax * (long double)vAmax));
    const long double visc_dt = cdtv * dsmin * dsmin / (max(max(nu_visc, eta), gamma * chi));
    //+ nu_shock * (long double)shock_max);

    const long double dt = min(uu_dt, visc_dt);
    ERRCHK_ALWAYS(!isnan(dt) && !isinf(dt));
    return (AcReal)dt;
}
