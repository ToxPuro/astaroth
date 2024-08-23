#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "astaroth.h"
// #include "host_forcing.h"

#include "ini.h"

static acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info)
{
    info->real_params[AC_dsx] = info->real_params[AC_box_size_x] / (info->int_params[AC_nx] - 1);
    info->real_params[AC_dsy] = info->real_params[AC_box_size_y] / (info->int_params[AC_ny] - 1);
    info->real_params[AC_dsz] = info->real_params[AC_box_size_z] / (info->int_params[AC_nz] - 1);
}

static acHostUpdateMHDSpecificParams(AcMeshInfo* info)
{
    // const ForcingParams forcing_params = generateForcingParams(*info);
    // loadForcingParamsToMeshInfo(forcing_params, info);
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
    return EXIT_SUCCESS;
}

static int
setup_tfm(void)
{
    return EXIT_SUCCESS;
}

static int
run_tfm(void)
{
    return EXIT_SUCCESS;
}

int
main(int argc, char* argv[])
{
    cudaProfilerStop();

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

    // Device
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    return EXIT_SUCCESS;
}