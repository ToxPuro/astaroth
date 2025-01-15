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
#include "acc_runtime.h"

#include <stdint.h> // uint8_t, uint32_t
#include <string>
#include <math.h>
#include <ctype.h>
#include <vector>
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

//static bool
//is_bctype(const int)
//{
//	return false;
//}
//
//static bool
//is_initcondtype(const int)
//{
//    return false;
//}


static int3
parse_int3param(const char* value)
{
	int x, y, z;
    	sscanf(value,"{%d,%d,%d}", &x, &y, &z);
	return (int3){x,y,z};
}

static AcReal3
parse_real3param(const char* value)
{
	double x, y, z;
    	sscanf(value,"{%lg,%lg,%lg}", &x, &y, &z);
	return (AcReal3){(AcReal)x,(AcReal)y,(AcReal)z};
}

static bool
parse_boolparam(const char* value)
{
	if(!strcmp(value,"true"))
		return true;
	if(!strcmp(value,"True"))
		return true;
	if(!strcmp(value,"T"))
		return true;
	if(!strcmp(value,"false"))
		return false;
	if(!strcmp(value,"False"))
		return false;
	if(!strcmp(value,"F"))
		return false;
	return atoi(value);
}

static void 
extract_between_symbol(const char *str, char *result, const char symbol,const char end_symbol) {
    const char *start = strchr(str, symbol);
    const char *end = strchr(start + 1, end_symbol);

    if (start && end && start < end) {
        size_t len = end - start - 1;
        strncpy(result, start + 1, len);
        result[len] = '\0'; // Null-terminate the result
    } else {
        result[0] = '\0'; // No valid substring found
    }
}

static std::vector<std::string>
get_entries(const char* line, const char symbol, const char end_symbol)
{

      std::vector<std::string> dst{};
      char* line_copy = (char*)malloc(sizeof(char)*strlen(line));
      extract_between_symbol(line,line_copy,symbol,end_symbol);
      char* token;
      token = strtok(line_copy,",");
      while(token != NULL)
      {
	      dst.push_back(token);
              token = strtok(NULL,",");
      }
      free(line_copy);
      return dst;
}

static AcBool3
parse_bool3param(const char* value)
{
	auto entries = get_entries(value,'{','}');
	return (AcBool3){parse_boolparam(entries[0].c_str()),parse_boolparam(entries[1].c_str()),parse_boolparam(entries[2].c_str())};
}

static AcReal
parse_realparam(const char* value)
{
	return atof(value);
}
static int
parse_intparam(const char* value)
{
	return atoi(value);
}


#define LOAD_ARRAY(UPPER_CASE,DATATYPE,LOWER_CASE,UP_NAME) \
        else if ((idx = find_array(keyword, LOWER_CASE##_array_info, NUM_##UPPER_CASE##_ARRAYS+NUM_##UPPER_CASE##_COMP_ARRAYS)) >= 0) { \
		auto array_vals = get_entries(value,'[',']'); \
		const bool is_comp = (idx >= NUM_##UPPER_CASE##_ARRAYS); \
		idx -= NUM_##UPPER_CASE##_ARRAYS*is_comp; \
		const auto size = (is_comp) ? \
				get_array_length((Ac##UP_NAME##CompArrayParam)idx,config->params.scalars) :\
				get_array_length((Ac##UP_NAME##ArrayParam)idx,config->params.scalars); \
		if(array_vals.size() != (size_t)size) \
			fprintf(stderr,"ERROR PARSING CONFIG: gave %zu values to array %s which of size %d: SKIPPING\n",array_vals.size(),keyword,size); \
		else \
		{ \
			DATATYPE* dst = (DATATYPE*)malloc(sizeof(DATATYPE)*size);\
			for(int i = 0; i < size; ++i) \
			{ \
				dst[i] = parse_##LOWER_CASE##param(array_vals[i].c_str()); \
			} \
			if(is_comp) \
			{ \
				config->run_consts.config.LOWER_CASE##_arrays[idx] = dst; \
				config->run_consts.is_loaded.LOWER_CASE##_arrays[idx] = true; \
			} \
			else \
			{ \
				config->params.arrays.LOWER_CASE##_arrays[idx] = dst; \
			} \
		} \
	}

static void
parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    printf("Config file path: %s\n", path);
    ERRCHK_ALWAYS(fp != NULL);

    const size_t BUF_SIZE = 10000;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %[^\n]", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;
        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcIntParam>(idx),parse_intparam(value));
        }
        else if ((idx = find_str(keyword, int_comp_param_names, NUM_INT_COMP_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcIntCompParam>(idx),parse_intparam(value));
        }
        if ((idx = find_str(keyword, boolparam_names, NUM_BOOL_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcBoolParam>(idx),parse_boolparam(value));
        }
        else if ((idx = find_str(keyword, bool_comp_param_names, NUM_BOOL_COMP_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcBoolCompParam>(idx),parse_boolparam(value));
        }
        else if ((idx = find_str(keyword, int3param_names, NUM_INT3_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcInt3Param>(idx),parse_int3param(value));
        }
        else if ((idx = find_str(keyword, int3_comp_param_names, NUM_INT3_COMP_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcInt3CompParam>(idx),parse_int3param(value));
        }
        else if ((idx = find_str(keyword, bool3param_names, NUM_BOOL3_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcBool3Param>(idx),parse_bool3param(value));
        }
        else if ((idx = find_str(keyword, bool3_comp_param_names, NUM_BOOL3_COMP_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcBool3CompParam>(idx),parse_bool3param(value));
        }
        else if ((idx = find_str(keyword, real3param_names, NUM_REAL3_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcReal3Param>(idx),parse_real3param(value));
        }
        else if ((idx = find_str(keyword, real3_comp_param_names, NUM_REAL3_COMP_PARAMS)) >= 0) {
	    acPushToConfig(*config,static_cast<AcReal3CompParam>(idx),parse_real3param(value));
        }
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAMS)) >= 0) {
            AcReal real_val = parse_realparam(value);
            if (isnan(real_val)) {
                fprintf(stderr,
                        "ERROR PARSING CONFIG: parameter \"%s\" value \"%s\" parsed as NAN\n",
                        keyword, value);
            }
            // OL: should we fail here? Could be dangerous to continue
	    acPushToConfig(*config,static_cast<AcRealParam>(idx),real_val);
        }
        else if ((idx = find_str(keyword, real_comp_param_names, NUM_REAL_COMP_PARAMS)) >= 0) 	      {
            AcReal real_val = atof(value);
            if (isnan(real_val)) {
                fprintf(stderr,
                        "ERROR PARSING CONFIG: parameter \"%s\" value \"%s\" parsed as NAN\n",
                        keyword, value);
            }
            // OL: should we fail here? Could be dangerous to continue
	    acPushToConfig(*config,static_cast<AcRealCompParam>(idx),real_val);
        }

	LOAD_ARRAY(INT,int,int,Int)
	LOAD_ARRAY(BOOL,bool,bool,Bool)
	LOAD_ARRAY(REAL,AcReal,real,Real)
    }

    fclose(fp);
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/

#define _UNUSED __attribute__((unused)) // Does not give a warning if unused

static AcResult UNUSED
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config_path);


    *config = acInitInfo();
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
