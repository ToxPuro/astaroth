#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "astaroth.h"
#include "astaroth_forcing.h"

#include "ini.h"
#include "stencil_loader.h"

static AcReal current_time = 0;

/** Takes the global info as parameter */
static int
acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info)
{
    info->real_params[AC_dsx] = info->real_params[AC_box_size_x] / (info->int_params[AC_nx] - 1);
    info->real_params[AC_dsy] = info->real_params[AC_box_size_y] / (info->int_params[AC_ny] - 1);
    info->real_params[AC_dsz] = info->real_params[AC_box_size_z] / (info->int_params[AC_nz] - 1);

    return EXIT_SUCCESS;
}

static void
loadForcingParamsToMeshInfo(const ForcingParams forcing_params, AcMeshInfo* info)
{
    info->real_params[AC_forcing_magnitude] = forcing_params.magnitude;
    info->real_params[AC_forcing_phase]     = forcing_params.phase;

    info->real_params[AC_k_forcex] = forcing_params.k_force.x;
    info->real_params[AC_k_forcey] = forcing_params.k_force.y;
    info->real_params[AC_k_forcez] = forcing_params.k_force.z;

    info->real_params[AC_ff_hel_rex] = forcing_params.ff_hel_re.x;
    info->real_params[AC_ff_hel_rey] = forcing_params.ff_hel_re.y;
    info->real_params[AC_ff_hel_rez] = forcing_params.ff_hel_re.z;

    info->real_params[AC_ff_hel_imx] = forcing_params.ff_hel_im.x;
    info->real_params[AC_ff_hel_imy] = forcing_params.ff_hel_im.y;
    info->real_params[AC_ff_hel_imz] = forcing_params.ff_hel_im.z;

    info->real_params[AC_kaver] = forcing_params.kaver;
}

static void
loadForcingParamsToDevice(const Device device, const ForcingParams forcing_params)
{
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_forcing_magnitude,
                              forcing_params.magnitude);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_forcing_phase, forcing_params.phase);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_k_forcex, forcing_params.k_force.x);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_k_forcey, forcing_params.k_force.y);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_k_forcez, forcing_params.k_force.z);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_rex, forcing_params.ff_hel_re.x);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_rey, forcing_params.ff_hel_re.y);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_rez, forcing_params.ff_hel_re.z);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_imx, forcing_params.ff_hel_im.x);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_imy, forcing_params.ff_hel_im.y);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_ff_hel_imz, forcing_params.ff_hel_im.z);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_kaver, forcing_params.kaver);
    acDeviceSynchronizeStream(device, STREAM_ALL);
}

static void
printForcingParams(const ForcingParams forcing_params)
{
    printf("Forcing parameters:\n"
           " magnitude: %lf\n"
           " phase: %lf\n"
           " k force: %lf\n"
           "          %lf\n"
           "          %lf\n"
           " ff hel real: %lf\n"
           "            : %lf\n"
           "            : %lf\n"
           " ff hel imag: %lf\n"
           "            : %lf\n"
           "            : %lf\n"
           " k aver: %lf\n"
           "\n",
           forcing_params.magnitude, forcing_params.phase, forcing_params.k_force.x,
           forcing_params.k_force.y, forcing_params.k_force.z, forcing_params.ff_hel_re.x,
           forcing_params.ff_hel_re.y, forcing_params.ff_hel_re.z, forcing_params.ff_hel_im.x,
           forcing_params.ff_hel_im.y, forcing_params.ff_hel_im.z, forcing_params.kaver);
}

static int
acHostUpdateMHDSpecificParams(AcMeshInfo* info)
{
    // Forcing
    ForcingParams forcing_params = generateForcingParams(info->real_params[AC_relhel],
                                                         info->real_params[AC_forcing_magnitude],
                                                         info->real_params[AC_kmin],
                                                         info->real_params[AC_kmax]);
    loadForcingParamsToMeshInfo(forcing_params, info);

    // Derived values
    info->real_params[AC_cs2_sound] = info->real_params[AC_cs_sound] *
                                      info->real_params[AC_cs_sound];

    // Other
    info->real_params[AC_center_x] = info->real_params[AC_box_size_x] / 2;
    info->real_params[AC_center_y] = info->real_params[AC_box_size_y] / 2;
    info->real_params[AC_center_z] = info->real_params[AC_box_size_z] / 2;

    return EXIT_SUCCESS;
}

typedef struct {
    char* config_path;
} Arguments;

static void
acPrintArguments(const Arguments args)
{
    printf("[config_path]: %s\n", args.config_path);
}

static int
acParseArguments(const int argc, char* argv[], Arguments* args)
{
    // Default arguments
    args->config_path = AC_DEFAULT_CONFIG;

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
                printf("\t--%s", long_options[i].name);
                if (long_options[i].has_arg == required_argument)
                    printf(" <argument>");
                printf(": %s", explanations[i]);
                printf("\n");
            }
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
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

static int
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
        fprintf(stderr, "Memory allocation error %d when parsing config \"%s\"\n", retval,
                filepath);
    else if (retval < 0)
        fprintf(stderr, "Unknown error %d when parsing config \"%s\"\n", retval, filepath);
    if (retval != 0)
        return EXIT_FAILURE;

    // Update the rest of the parameters
    acHostUpdateBuiltinParams(info);

    // Check for uninitialized values
    for (size_t i = 0; i < NUM_INT_PARAMS; ++i)
        if (info->int_params[i] == INT_MIN)
            fprintf(stderr, "--- Warning: [%s] uninitialized ---\n", intparam_names[i]);
    for (size_t i = 0; i < NUM_REAL_PARAMS; ++i)
        if (info->real_params[i] == (AcReal)NAN)
            fprintf(stderr, "--- Warning: [%s] uninitialized ---\n", realparam_names[i]);
    return EXIT_SUCCESS;
}

static int
acDeviceWriteProfileToDisk(const Device device, const Profile profile, const char* filepath)
{
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const size_t mz       = as_size_t(info.int3_params[AC_global_grid_n].z +
                                2 * ((STENCIL_DEPTH - 1) / 2));

    AcBuffer host_profile = acBufferCreate(mz, false);
    acDeviceStoreProfile(device, profile, host_profile.data, host_profile.count);
    acHostWriteProfileToFile(filepath, host_profile.data, host_profile.count);
    acBufferDestroy(&host_profile);
    return EXIT_SUCCESS;
}

static int
tfm_init_profiles(const Device device)
{
    const AcMeshInfo info  = acDeviceGetLocalConfig(device);
    const AcReal global_lz = info.real_params[AC_box_size_z];
    const size_t global_nz = as_size_t(info.int3_params[AC_global_grid_n].z);
    const long offset      = -info.int_params[AC_nz_min]; // TODO take multigpu into account
    const size_t local_mz  = as_size_t(info.int_params[AC_mz]);

    const AcReal amplitude  = info.real_params[AC_profile_amplitude];
    const AcReal wavenumber = info.real_params[AC_profile_wavenumber];

    AcReal host_profile[local_mz];

    // All to zero
    acHostInitProfileToValue(0, local_mz, host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_z);

    // B1c (here B11) and B2c (here B21) to cosine
    acHostInitProfileToCosineWave(global_lz, global_nz, offset, amplitude, wavenumber, local_mz,
                                  host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_y);

    // B1s (here B12) and B2s (here B22)
    acHostInitProfileToSineWave(global_lz, global_nz, offset, amplitude, wavenumber, local_mz,
                                host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_y);

    return EXIT_SUCCESS;
}

static AcReal
max(const AcReal a, const AcReal b)
{
    return a > b ? a : b;
}

static AcReal
min(const AcReal a, const AcReal b)
{
    return a < b ? a : b;
}

static long double
maxl(const long double a, const long double b)
{
    return a > b ? a : b;
}

static long double
minl(const long double a, const long double b)
{
    return a < b ? a : b;
}

AcReal
calc_timestep(const Device device, const AcMeshInfo info)
{
    AcReal uumax = 0.0;
    AcReal vAmax = 0.0;
    // AcReal shock_max = 0.0;
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                      &uumax);

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
    const long double visc_dt = cdtv * dsmin * dsmin / (maxl(maxl(nu_visc, eta), gamma * chi));
    //+ nu_shock * (long double)shock_max);

    const long double dt = minl(uu_dt, visc_dt);
    // ERRCHK_ALWAYS(is_valid((AcReal)dt));
    return (AcReal)(dt);
}

int
tfm_run_pipeline_original(const Device device)
{
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const AcMeshDims dims = acGetMeshDims(info);
    const AcReal dt       = calc_timestep(device, info);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, dt);

    for (size_t step_number = 0; step_number < 3; ++step_number) {
        acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, step_number);

        // Compute: hydrodynamics
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1);

        // Boundary conditions: hydrodynamics
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_LNRHO, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUX, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUY, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUZ, dims.m0, dims.m1);

        // Profile averages
        acDeviceReduceXYAverages(device, STREAM_DEFAULT);

        // Compute: test fields
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b11, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b12, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b21, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b22, dims.n0, dims.n1);

        // Boundary conditions: test fields
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_z, dims.m0, dims.m1);
    }
    return EXIT_SUCCESS;
}

int
tfm_run_pipeline(const Device device)
{
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const AcMeshDims dims = acGetMeshDims(info);

    // Current time (temporary hack, TODO better)
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_current_time, current_time);

    // Timestep
    const AcReal dt = calc_timestep(device, info);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, dt);

    // Apply forcing
    // TODO

    for (int step_number = 0; step_number < 3; ++step_number) {
        acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, step_number);

        // Boundary conditions: hydrodynamics
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_LNRHO, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUX, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUY, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUZ, dims.m0, dims.m1);

        // Boundary conditions: test fields
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_z, dims.m0, dims.m1);

        // Boundary conditions: derived test fields
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_z, dims.m0, dims.m1);

        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_x, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_y, dims.m0, dims.m1);
        acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_z, dims.m0, dims.m1);

        // Profile averages
        acDeviceReduceXYAverages(device, STREAM_DEFAULT);

        // Compute: hydrodynamics
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1);

        // Compute: test fields
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b11, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b12, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b21, dims.n0, dims.n1);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve_tfm_b22, dims.n0, dims.n1);

        acDeviceSwapBuffers(device);
    }
    // Boundary conditions: hydrodynamics
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_LNRHO, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUX, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUY, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, VTXBUF_UUZ, dims.m0, dims.m1);

    // Boundary conditions: test fields
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a11_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a12_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a21_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_a22_z, dims.m0, dims.m1);

    // Boundary conditions: derived test fields
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb11_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb12_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb21_z, dims.m0, dims.m1);

    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_x, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_y, dims.m0, dims.m1);
    acDevicePeriodicBoundcondStep(device, STREAM_DEFAULT, TF_uxb22_z, dims.m0, dims.m1);

    // Update current time
    current_time += dt;
    return EXIT_SUCCESS;
}

static int
write_diagnostic_step(const Device device, const size_t step)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.mesh", step, vtxbuf_names[i]);
        printf("Writing %s\n", filepath);
        acDeviceWriteMeshToDisk(device, i, filepath);
    }
    for (int i = 0; i < NUM_PROFILES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.profile", step, profile_names[i]);
        printf("Writing %s\n", filepath);
        acDeviceWriteProfileToDisk(device, (Profile)i, filepath);
    }
    return EXIT_SUCCESS;
}

int
main(int argc, char* argv[])
{
    cudaProfilerStop();
    printf("sizeof(AcReal): %zu\n", sizeof(AcReal));

    // Arguments
    Arguments args;
    acParseArguments(argc, argv, &args);

    printf("Arguments:\n");
    acPrintArguments(args);
    printf("\n");

    // Mesh configuration
    AcMeshInfo info;
    acParseINI(args.config_path, &info);
    acHostUpdateTFMSpecificGlobalParams(&info);
    acHostUpdateMHDSpecificParams(&info);

    printf("MeshInfo:\n");
    acPrintMeshInfo(info);
    printf("\n");

    // Temporary workaround for saving the simulation time
    current_time = info.real_params[AC_current_time];

    // Device
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    // Load stencil coefficients (NOTE global info)
    AcReal* stencils = get_stencil_coeffs(info);
    acDeviceLoadStencils(device, STREAM_DEFAULT, stencils);
    free(stencils);

    AcMeshInfo local_info = acDeviceGetLocalConfig(device);
    const AcMeshDims dims = acGetMeshDims(local_info);
    printf("Local MeshInfo:\n");
    acPrintMeshInfo(local_info);
    printf("\n");

    // Initialize the random number generator
    const size_t seed  = 12345;
    const size_t pid   = 0;
    const size_t count = acVertexBufferCompdomainSize(info);
    acRandInitAlt(seed, count, pid);
    srand(seed);

    // Dryrun
    acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 0, dims.n0, dims.n1, 1e-5);
    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 1, dims.n0, dims.n1, 1e-5);
    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 2, dims.n0, dims.n1, 1e-5);
    tfm_run_pipeline(device);

    // Initialize the mesh and reload all device constants
    // acDeviceResetMesh(device, STREAM_DEFAULT);
    // acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    // tfm_init_profiles(device);
    // acDeviceSynchronizeStream(device, STREAM_ALL);

    // // Integration-----------------------------
    // int retval;
    // AcMesh model, candidate;
    // acHostMeshCreate(info, &model);
    // acHostMeshCreate(info, &candidate);
    // acHostMeshRandomize(&model);
    // acHostMeshRandomize(&candidate);

    // // BC
    // acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    // acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    // acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    // acHostMeshApplyPeriodicBounds(&model);
    // AcResult res = acVerifyMesh("Boundconds", model, candidate);
    // if (res != AC_SUCCESS) {
    //     retval = res;
    //     WARNCHK_ALWAYS(retval);
    // }

    // // INTEG
    // // acHostMeshRandomize(&model);
    // // acHostMeshApplyPeriodicBounds(&model);
    // // acDeviceResetMesh(device, STREAM_DEFAULT);
    // // acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    // // // acDeviceSwapBuffers(device);
    // // // acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    // acDeviceResetMesh(device, STREAM_DEFAULT);
    // acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
    // acDeviceSwapBuffers(device);
    // acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    // acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    // acDeviceStoreMesh(device, STREAM_DEFAULT, &model);
    // acDeviceSynchronizeStream(device, STREAM_ALL);
    // acHostMeshApplyPeriodicBounds(&model);

    // const AcReal dt                    = 1e-5;
    // const size_t NUM_INTEGRATION_STEPS = 10;
    // for (size_t j = 0; j < NUM_INTEGRATION_STEPS; ++j) {
    //     for (int i = 0; i < 3; ++i) {
    //         acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    //         acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, dims.n0, dims.n1, dt);
    //         acDeviceSwapBuffers(device);
    //     }
    //     // tfm_run_pipeline(device); // OK if set dt to same as with host
    // }

    // acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    // acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    // if (pid == 0) {

    //     // Host integrate
    //     for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    //         acHostIntegrateStep(model, dt);

    //     acHostMeshApplyPeriodicBounds(&model);
    //     const AcResult res = acVerifyMesh("Integration", model, candidate);
    //     if (res != AC_SUCCESS) {
    //         retval = res;
    //         WARNCHK_ALWAYS(retval);
    //     }

    //     srand(123567);
    //     acHostMeshRandomize(&model);
    //     // acHostMeshSet((AcReal)1.0, &model);
    //     acHostMeshApplyPeriodicBounds(&model);
    // }
    // //---------------------------------

    // Simulation loop
    acDeviceResetMesh(device, STREAM_DEFAULT);
    // acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1);
    acDeviceSwapBuffers(device);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    tfm_init_profiles(device);

    // Write the initial step out for reference
    write_diagnostic_step(device, 0);

    const size_t nsteps          = info.int_params[AC_simulation_nsteps];
    const size_t output_interval = info.int_params[AC_simulation_output_interval];
    for (size_t step = 1; step <= nsteps; ++step) {
        // Generate forcing
        ForcingParams forcing_params = generateForcingParams(info.real_params[AC_relhel],
                                                             info.real_params[AC_forcing_magnitude],
                                                             info.real_params[AC_kmin],
                                                             info.real_params[AC_kmax]);
        loadForcingParamsToDevice(device, forcing_params);
        printForcingParams(forcing_params);

        // Simulate
        tfm_run_pipeline(device);

        // AcReal urms;
        // acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
        //                   &urms);
        // printf("urms: %.2g, step %zu\n", urms, step);

        // Output
        if ((step % output_interval) == 0)
            write_diagnostic_step(device, step);
    }

    return EXIT_SUCCESS;
}