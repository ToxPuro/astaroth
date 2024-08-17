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
#include "astaroth_utils.h"

#include <stdint.h> // uint8_t, uint32_t
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "errchk.h"

/**
 \brief Find the index of the keyword in names
 \return Index in range 0...n if the keyword is in names. -1 if the keyword was
 not found.
 */
extern "C"
{
static int
find_str(const char keyword[], const char* names[], const int n)
{
    for (int i = 0; i < n; ++i)
        if (!strcmp(keyword, names[i]))
            return i;

    return -1;
}
static int
find_array(const char keyword[], const array_info* info, const int n)
{
	for(int i = 0; i < n; ++i)
        	if (!strcmp(keyword, info[i].name))
			return i;
	return -1;
}

static bool
is_bctype(const int idx)
{
    return idx == AC_bc_type_top_x || idx == AC_bc_type_bot_x || //
           idx == AC_bc_type_top_y || idx == AC_bc_type_bot_y 
#if TWO_D == 0
	   || idx == AC_bc_type_top_z || idx == AC_bc_type_bot_z
#endif
	  ;
}

static bool
is_initcondtype(const int idx)
{
    return idx == AC_init_type;
}

static int
parse_intparam(const size_t idx, const char* value)
{
    if (is_bctype(idx)) {
        int bctype = -1;
        if ((bctype = find_str(value, bctype_names, NUM_BCTYPES)) >= 0)
            return bctype;
        else {
            fprintf(stderr,
                    "ERROR PARSING CONFIG: Invalid BC type: %s, do not know what to do with it.\n",
                    value);
            fprintf(stdout, "Valid BC types:\n");
            acQueryBCtypes();
            ERROR("Invalid boundary condition type found in config");
            return 0;
        }
    }
    else if (is_initcondtype(idx)) {
        int initcondtype = -1;
        if ((initcondtype = find_str(value, initcondtype_names, NUM_INIT_TYPES)) >= 0)
            return initcondtype;
        else {
            fprintf(stderr,
                    "ERROR PARSING CONFIG: Invalid initial condition type: %s, do not know what to "
                    "do with it.\n",
                    value);
            fprintf(stdout, "Valid initial condition types:\n");
            acQueryInitcondtypes();
            ERROR("Invalid initial condition type found in config");
            return 0;
        }
    }
    else {
        return atoi(value);
    }
}
int
get_entries(char** dst, const char* line)
{
      char* line_copy = strdup(line);
      int counter = 0;
      char* token;
      token = strtok(line_copy,",");
      while(token != NULL)
      {
              dst[counter] = strdup(token);
              ++counter;
              token = strtok(NULL,",");
      }
      free(line_copy);
      return counter;
}

static void
parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    printf("Config file path: %s\n", path);
    ERRCHK_ALWAYS(fp != NULL);

    const size_t BUF_SIZE = 128;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %[^\n]", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;

        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAMS)) >= 0) {
            config->int_params[idx] = parse_intparam(idx, value);
        }
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAMS)) >= 0) {
            AcReal real_val = atof(value);
            if (isnan(real_val)) {
                fprintf(stderr,
                        "ERROR PARSING CONFIG: parameter \"%s\" value \"%s\" parsed as NAN\n",
                        keyword, value);
            }
            // OL: should we fail here? Could be dangerous to continue
            config->real_params[idx] = real_val;
        }
        else if ((idx = find_array(keyword, int_array_info, NUM_INT_ARRAYS)) >= 0) {
		if(!int_array_info[idx].is_dconst)
			fprintf(stderr,"ERROR PARSING CONFIG: can't assign to global array: \"%s\": SKIPPING\n",keyword);
		char* array_vals[int_array_info[idx].length];
		const int n_vals = get_entries(array_vals, value);
		if(n_vals != int_array_info[idx].length)
			fprintf(stderr,"ERROR PARSING CONFIG: gave %d values to array %s which of size %d: SKIPPING\n",n_vals,keyword,int_array_info[idx].length);
		else
		{
			config->int_arrays[idx] = (int*)malloc(sizeof(AcReal)*int_array_info[idx].length);
			for(int i = 0; i < int_array_info[idx].length; ++i)
			{
				config->int_arrays[idx][i] =  atoi(array_vals[i]);
				free(array_vals[i]);
			}
		}


	}
        else if ((idx = find_array(keyword, real_array_info, NUM_REAL_ARRAYS)) >= 0) {
		if(!real_array_info[idx].is_dconst)
			fprintf(stderr,"ERROR PARSING CONFIG: can't assign to global array: \"%s\": SKIPPING\n",keyword);
		char* array_vals[real_array_info[idx].length];
		const int n_vals = get_entries(array_vals, value);
		if(n_vals != real_array_info[idx].length)
			fprintf(stderr,"ERROR PARSING CONFIG: gave %d values to array %s which of size %d: SKIPPING\n",n_vals,keyword,real_array_info[idx].length);
		else
		{
			fprintf(stderr,"Reading in real array: %s\n",keyword);
			config->real_arrays[idx] = (AcReal*)malloc(sizeof(AcReal)*real_array_info[idx].length);
			for(int i = 0; i < real_array_info[idx].length; ++i)
			{
				config->real_arrays[idx][i] =  atof(array_vals[i]);
				free(array_vals[i]);
			}
		}


	}
    }

    fclose(fp);
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/
AcResult
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config_path);


    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    //these are set to nullpointers for the users convenience that the user doesn't have to set them to null elsewhere
    //if they are present in the config then they are initialized correctly
    //sticks to the old API since we anyways overwrite the whole config
    memset(config->real_arrays, 0,NUM_REAL_ARRAYS *sizeof(AcReal*));
    memset(config->int_arrays,  0,NUM_INT_ARRAYS  *sizeof(int*));
    memset(config->bool_arrays, 0,NUM_BOOL_ARRAYS *sizeof(bool*));
    memset(config->int3_arrays, 0,NUM_INT3_ARRAYS *sizeof(int*));
    memset(config->real3_arrays,0,NUM_REAL3_ARRAYS*sizeof(int*));

    parse_config(config_path, config);
    acHostUpdateBuiltinParams(config);
#if AC_VERBOSE
    printf("###############################################################\n");
    printf("Config dimensions loaded:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif

    // sizeof(config) must be a multiple of 4 bytes for this to work
    ERRCHK_ALWAYS(sizeof(*config) % sizeof(uint32_t) == 0);

    // Check for uninitialized config values
    bool uninitialized_config_val = false;
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        uninitialized_config_val |= ((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF;
    }

#if AC_VERBOSE
    if (uninitialized_config_val) {
        fprintf(stderr, "Some config values may be uninitialized. "
                        "See that all are defined in astaroth.conf\n");
    }
#endif

    return uninitialized_config_val ? AC_FAILURE : AC_SUCCESS;
}
}
