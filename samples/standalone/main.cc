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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run.h"
#include "errchk.h"

// Write all errors from stderr to an <errorlog_name> in the current working
// directory
static const bool write_log_to_a_file = false;
static const char* errorlog_name      = "error.log";

static void
errorlog_init(void)
{
    FILE* fp = freopen(errorlog_name, "w", stderr); // Log errors to a file
    if (!fp)
        perror("Error redirecting stderr to a file");
}

static void
errorlog_quit(void)
{
    fclose(stderr);

    // Print contents of the latest errorlog to screen
    FILE* fp = fopen(errorlog_name, "r");
    if (fp) {
        for (int c = getc(fp); c != EOF; c = getc(fp))
            putchar(c);
        fclose(fp);
    }
    else {
        perror("Error opening error log");
    }
}

typedef struct {
    char* key[2];
    char* description;
} Option;

static Option
createOption(const char* key, const char* key_short, const char* description)
{
    Option option;

    option.key[0]      = strdup(key);
    option.key[1]      = strdup(key_short);
    option.description = strdup(description);

    return option;
}

static void
destroyOption(Option* option)
{
    free(option->key[0]);
    free(option->key[1]);
    free(option->description);
}

typedef enum {
    HELP,
    TEST,
    BENCHMARK,
    SIMULATE,
    RENDER,
    CONFIG,
    SEED,
    WRITE_TIMESTEP_FILE,
    READ_TIMESTEP_FILE,
    ANALYZE_STEPS,
    NUM_OPTIONS,
} OptionType;

static int
findOption(const char* str, const Option options[NUM_OPTIONS])
{
    for (int i = 0; i < NUM_OPTIONS; ++i)
        if (!strcmp(options[i].key[0], str) || !strcmp(options[i].key[1], str))
            return i;

    return -1;
}

static void
print_options(const Option options[NUM_OPTIONS])
{
    // Formatting
    int keylen[2] = {0};
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        int len0 = strlen(options[i].key[0]);
        int len1 = strlen(options[i].key[1]);
        if (keylen[0] < len0)
            keylen[0] = len0;
        if (keylen[1] < len1)
            keylen[1] = len1;
    }

    for (int i = 0; i < NUM_OPTIONS; ++i)
        printf("\t%*s | %*s: %s\n", keylen[0], options[i].key[0], keylen[1], options[i].key[1],
               options[i].description);
}

static void
print_help(const Option options[NUM_OPTIONS])
{
    puts("Usage: ./ac_run [options]");
    print_options(options);
    printf("\n");
    puts("For bug reporting, see README.md");
}

int
main(int argc, char* argv[])
{
    if (write_log_to_a_file) {
        errorlog_init();
        atexit(errorlog_quit);
    }

    // Create options
    // clang-format off
    Option options[NUM_OPTIONS];
    options[HELP]               = createOption("--help", "-h", "Prints this help.");
    options[TEST]               = createOption("--test", "-t", "Runs autotests.");
    options[BENCHMARK]          = createOption("--benchmark", "-b", "Runs benchmarks.");
    options[SIMULATE]           = createOption("--simulate", "-s", "Runs the simulation.");
    options[RENDER]             = createOption("--render", "-r", "Runs the real-time renderer.");
    options[CONFIG]             = createOption("--config", "-c", "Uses the config file given after this flag instead of the default.");
    options[SEED]               = createOption("--seed", "-e", "Uses the number given after this flag instead of the default seed for the rng");
    options[WRITE_TIMESTEP_FILE] = createOption("--write_timestep_file", "-w", "record the timesteps used in this file");
    options[READ_TIMESTEP_FILE] = createOption("--read_timestep_file", "-f", "Use timesteps from this file (instead of calculating them on the fly)");
    options[ANALYZE_STEPS]      = createOption("--analyze_steps", "-a", "send every a-th timestep dump to be analyzed by a python server");
    // clang-format on

    int analyze_steps = -1;
    int seed = 312256655;
    char *write_timestep_file = NULL; // NULL is valid, meaning no recording
    char *read_timestep_file = NULL; // NULL is valid, meaning calculate the timesteps on the fly

    printf("there are %d values in argv, they are:\n", argc);
    for (int i=0; i<argc; i++) {
        printf(argv[i]);
        printf("\n");
    }

    if (argc == 1) {
        print_help(options);
    }
    else {
        char* config_path = NULL;
        for (int i = 1; i < argc; ++i) {
            const int option = findOption(argv[i], options);
            printf("option %s in position %d maps to intvalue %d, description is %s\n", argv[i], i, option, options[option].description);
            switch (option) {
            case CONFIG:
                if (i + 1 < argc) {
                    config_path = strdup(argv[i + 1]);
                }
                else {
                    printf("Syntax error. Usage: --config <config path>.\n");
                    return EXIT_FAILURE;
                }
                i++;
                break;
            case SEED:
                if (i + 1 < argc) {
                    seed = atoi(argv[i+1]);
                    // 0 signals failure but is also a legal output for the string "0"
                    if (seed == 0 && ! strcmp(argv[i+1], "0")) {
                        printf("invalid seed argument %s, must be an integer\n", argv[i+1]);
                        return EXIT_FAILURE;
                    }
                }
                else {
                    printf("Syntax error. Usage: --seed <number>\n");
                    return EXIT_FAILURE;
                }
                i++;
                break;
            case WRITE_TIMESTEP_FILE:
                if (i + 1 < argc) {
                    write_timestep_file = strdup(argv[i + 1]);
                }
                else {
                    printf("Syntax error. Usage: --write_timestep_file <write_timestep_file path>.\n");
                    return EXIT_FAILURE;
                }
                i++;
                break;
            case READ_TIMESTEP_FILE:
                if (i + 1 < argc) {
                    read_timestep_file = strdup(argv[i + 1]);
                }
                else {
                    printf("Syntax error. Usage: --read_timestep_file <read_timestep_file path>.\n");
                    return EXIT_FAILURE;
                }
                i++;
                break;
            case ANALYZE_STEPS:
                if (i+1 < argc) {
                    analyze_steps = atoi(argv[i+1]);
                }
                else {
                    printf("Syntax error. Usage: --analyze_steps <n> to analyze every n-th timestep by sending it to python\n");
                }
                i++;
                break;
            default:
                fprintf(stderr, "other option %s\n", argv[i]);// this is not an error
                break; // Do nothing
            }
        }
        if (!config_path)
            config_path = strdup(AC_DEFAULT_CONFIG);

        printf("Config path: %s\n", config_path);
        printf("using seed: %d\n", seed);
        printf("reading timesteps from: %s\n", read_timestep_file);
        printf("recording timesteps in: %s\n", write_timestep_file);
        ERRCHK_ALWAYS(config_path);
        //ERRCHK_ALWAYS((read_timestep_file && !write_timestep_file) || (!read_timestep_file && write_timestep_file))

        for (int i = 1; i < argc; ++i) {
            const int option = findOption(argv[i], options);
            switch (option) {
            case HELP:
                print_help(options);
                break;
            case TEST:
                run_autotest(config_path);
                break;
            case BENCHMARK:
                run_benchmark(config_path);
                break;
            case SIMULATE:
                run_simulation(config_path, seed, read_timestep_file, write_timestep_file, analyze_steps);
                break;
            case RENDER:
                run_renderer(config_path);
                break;
            // these options have an extra parameter that shouldnt go through the switch-case
            case WRITE_TIMESTEP_FILE:
            case READ_TIMESTEP_FILE:
            case ANALYZE_STEPS:
            case CONFIG:
            case SEED:
                ++i;
                break;
            default:
                printf("Invalid option %s\n", argv[i]);
                //break; // Do nothing <-- original astaroth
                return EXIT_FAILURE; //<-- I like this better...
            }
        }

        free(config_path);
    }

    for (int i = 0; i < NUM_OPTIONS; ++i)
        destroyOption(&options[i]);

    return EXIT_SUCCESS;
}
