// clang-format off
/**
  Code generator for unrolling and reordering memory accesses

  Key structures:
    
      stencils

      int stencils_accessed[kernel][field][stencil]: Set if `stencil` accessed for `field` in `kernel`
      char* stencils[stencil][depth][height][width]: contains the expression to compute the stencil coefficient

      char* stencil_unary_ops[stencil]: contains the function name of the unary operation used to process `stencil`
      char* stencil_binary_ops[stencil]: contains the function name of the binary operation used to process `stencil`

      A stencil is defined (formally) as

        f: R -> R       | Map operator
        g: R^{|s|} -> R | Reduce operator
        p: stencil points
        w: stencil element weights (coefficients)

        f(p_i) = ...
        s_i = w_i f(p_i)
        g(s) = ...

        For example for an ordinary stencil
        f(p_i) = p_i
        g(s) = sum_{i=1}^{|s|} s_i = sum s_i

      Alternatively by recursion
        G(p_0) = w_i f(p_0)
        G(p_i) = g(w_i f(p_i), G(p_{i-1}))

      Could also simplify notation by incorporating w into f

      CS view:
        res = f(p[0])
        for i in 1,len(p):
          res = g(f(p[i]), res)
*/
// clang-format on
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "user_defines.h"

#include "stencil_accesses.h"
#include "stencilgen.h"

#include "implementation.h"


void
raise_error(const char* str)
{
  // Make sure the error does not go unnoticed
  //
  // It is not clear how the CMake building process
  // could be stopped if a part of code generation
  // fails but an infinite loop is an easy and
  // effective way to inform the user something went wrong
  while (1)
    fprintf(stderr, "FATAL ERROR: %s\n", str);
  exit(EXIT_FAILURE);
}

void
gen_stencil_definitions(void)
{
  if (!NUM_ALL_FIELDS)
    raise_error("Must declare at least one Field in the DSL code!");

  if (!NUM_STENCILS)
    raise_error("Must declare at least one Stencil in the DSL code!");

#if TWO_D == 0
    printf(
        "static __device__ /*const*/ AcReal /*__restrict__*/ "
        "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]={");
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      printf("{");
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        printf("{");
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          printf("{");
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            const char* coeff = stencils[stencil][depth][height][width];
	    printf("%s,",coeff && !strstr(coeff,"DCONST") ? coeff : "AcReal(NAN)");
          }
          printf("},");
        }
        printf("},");
      }
      printf("},");
    }
    printf("};");
#else
    printf(
        "static __device__ /*const*/ AcReal /*__restrict__*/ "
        "stencils[NUM_STENCILS][STENCIL_HEIGHT][STENCIL_WIDTH]={");
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      printf("{");
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        printf("{");
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
            const char* coeff = stencils[stencil][depth][height][width];
	    printf("%s,",coeff && !strstr(coeff,"DCONST") ? coeff : "AcReal(NAN)");
        }
        printf("},");
      }
      printf("},");
    }
    printf("};");
#endif
  FILE* stencil_coeffs_file= fopen("coeffs.h", "w");
#if TWO_D == 0
  fprintf(stencil_coeffs_file,
      "AcReal "
      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]={");
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    fprintf(stencil_coeffs_file,"{");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      fprintf(stencil_coeffs_file,"{");
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        fprintf(stencil_coeffs_file,"{");
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          fprintf(stencil_coeffs_file,"%s,", dynamic_coeffs[stencil][depth][height][width]
                            ? dynamic_coeffs[stencil][depth][height][width]
                            : "0");
        }
        fprintf(stencil_coeffs_file,"},");
      }
      fprintf(stencil_coeffs_file,"},");
    }
    fprintf(stencil_coeffs_file,"},");
  }
  fprintf(stencil_coeffs_file,"};");
#else
  fprintf(stencil_coeffs_file,
      "AcReal "
      "stencils[NUM_STENCILS][STENCIL_HEIGHT][STENCIL_WIDTH]={");
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    fprintf(stencil_coeffs_file,"{");
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      fprintf(stencil_coeffs_file,"{");
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        fprintf(stencil_coeffs_file,"%s,", dynamic_coeffs[stencil][height][width]
                          ? dynamic_coeffs[stencil][height][width]
                          : "0");
      }
      fprintf(stencil_coeffs_file,"},");
    }
    fprintf(stencil_coeffs_file,"},");
  }
  fprintf(stencil_coeffs_file,"};");
#endif
  fclose(stencil_coeffs_file);
}

#include "kernel_reduce_info.h"

void
gen_kernel_common_prefix()
{
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * blockDim.x + start.x,"
         "threadIdx.y + blockIdx.y * blockDim.y + start.y,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z,"
         "};");
  printf("const int3 globalVertexIdx = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
#if TWO_D == 0
  printf("const int3 globalGridN = (int3){VAL(AC_nxgrid),VAL(AC_nygrid), VAL(AC_nzgrid)};");
#else
  printf("const int3 globalGridN = (int3){VAL(AC_nxgrid), VAL(AC_nygrid), 1};");
#endif
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");

#if TWO_D == 0
    printf(
        "const int3 localCompdomainVertexIdx = (int3){"
        "threadIdx.x + blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
        "threadIdx.y + blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
        "threadIdx.z + blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2,"
        "};");
#else
    printf(
        "const int3 localCompdomainVertexIdx = (int3){"
        "threadIdx.x + blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
        "threadIdx.y + blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
        "0,"
        "};");
#endif
  printf("const int local_compdomain_idx = "
         "LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning
  printf("(void)local_compdomain_idx;");     // Silence unused warning
					     //
// Write vba.out
#if 1
  // Original
  printf("const auto write_base __attribute__((unused)) = [&](const Field& handle, const AcReal& value) {");
  printf("vba.out[handle][idx] = value;");
  printf("};");

  //  Non-temporal store intrinsic could reduce L2 pressure on AMD but no effect
  //  in practice (no effect on the first pass, a slight slowdown in the second
  //  pass 4.6 ms vs 4.3 ms)
  // printf("const auto write=[&](const Field& field, const AcReal value)"
  //  "{ __builtin_nontemporal_store(value, &vba.out[field][idx]); };");
#else
  // Buffered, no effect on performance
  // !Remember to emit write insructions in ac.y if this is enabled!
  printf("AcReal out_buffer[NUM_ALL_FIELDS];");
  for (int field = 0; field < NUM_ALL_FIELDS; ++field)
    printf("out_buffer[%d] = (AcReal)NAN;", field);

  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ out_buffer[field] = value; };");
/*
for (int field = 0; field < NUM_ALL_FIELDS; ++field)
printf("vba.out[%d][idx] = out_buffer[%d];", field, field);
*/
#endif

  //TP: for now profile reads are not cached since they are usually read in only once and anyways since they are smaller can fit more easily to cache.
  //TP: if in the future a use case uses profiles a lot reconsider this
  if(!NUM_PROFILES) return;
  printf("const auto value_profile_x __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.x];");
  printf("};");

  printf("const auto value_profile_y __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.y];");
  printf("};");

  printf("const auto value_profile_z __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.z];");
  printf("};");

  printf("const auto value_profile_xy __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mx)*vertexIdx.y];");
  printf("};");

  printf("const auto value_profile_xz __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mx)*vertexIdx.z];");
  printf("};");

  printf("const auto value_profile_yx __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_my)*vertexIdx.x];");
  printf("};");

  printf("const auto value_profile_yz __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_my)*vertexIdx.z];");
  printf("};");

  printf("const auto value_profile_zx __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mz)*vertexIdx.x];");
  printf("};");

  printf("const auto value_profile_zy __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mz)*vertexIdx.y];");
  printf("};");

    printf("const auto reduce_sum_real_x __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_y __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_z __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_xy __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_xz __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_yx __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_yz __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_zx __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");
    printf("const auto reduce_sum_real_zy __attribute__((unused)) = [&](const bool& , const AcReal& val, const Profile& output)"
          	  "{ vba.reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][0][idx] = val; };");

}

void
gen_kernel_prefix(const int curr_kernel)
{
  gen_kernel_common_prefix();

  if(kernel_calls_reduce[curr_kernel] )
  {
    if(NUM_REAL_OUTPUTS)
    {
    	printf("AcReal reduce_sum_res_real[NUM_REAL_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_REAL_OUTPUTS;  ++i)
    	        printf("0.0,");
    	printf("};\n");
    	printf("AcReal reduce_max_res_real[NUM_REAL_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_REAL_OUTPUTS;  ++i)
    	        printf("-1000000.0,");
    	printf("};\n");
    	printf("AcReal reduce_min_res_real[NUM_REAL_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_REAL_OUTPUTS;  ++i)
    	        printf("1000000.0,");
    	printf("};\n");
    	printf("bool should_reduce_real[NUM_REAL_OUTPUTS] = {\n");
    		printf("false,");
    	printf("};\n");
        printf("(void)reduce_sum_res_real;");
        printf("(void)reduce_min_res_real;");
        printf("(void)reduce_max_res_real;");
        printf("const auto reduce_sum_real __attribute__((unused)) = [&](const bool& condition, const AcReal& val, const AcRealOutputParam& output)"
              	  "{ should_reduce_real[(int)output] = condition; reduce_sum_res_real[(int)output] = val; };");
        printf("const auto reduce_min_real __attribute__((unused)) = [&](const bool& condition, const AcReal& val, const AcRealOutputParam& output)"
              	  "{ should_reduce_real[(int)output] = condition; reduce_min_res_real[(int)output] = val; };");
        printf("const auto reduce_max_real __attribute__((unused)) = [&](const bool& condition, const AcReal& val, const AcRealOutputParam& output)"
    		  "{ should_reduce_real[(int)output] = condition; reduce_max_res_real[(int)output] = val; };");
    }
    if(NUM_INT_OUTPUTS)
    {
    	printf("int reduce_sum_res_int[NUM_INT_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_INT_OUTPUTS;  ++i)
    	        printf("0,");
    	printf("};\n");
    	printf("int reduce_max_res_int[NUM_INT_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_INT_OUTPUTS;  ++i)
    	        printf("-10000000,");
    	printf("};\n");
    	printf("int reduce_min_res_int[NUM_INT_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_INT_OUTPUTS;  ++i)
    	        printf("10000000,");
    	printf("};\n");
    	printf("bool should_reduce_int[NUM_INT_OUTPUTS] = {\n");
    	for(int i = 0; i< NUM_INT_OUTPUTS;  ++i)
    		printf("false,");
    	printf("};\n");
    	

    	printf("(void)reduce_sum_res_int;");
    	printf("(void)reduce_min_res_int;");
    	printf("(void)reduce_max_res_int;");
        printf("const auto reduce_sum_int __attribute__((unused)) = [&](const bool& condition, const int& val, const AcIntOutputParam& output)"
              	  "{ should_reduce_int[(int)output] = condition; reduce_sum_res_int[(int)output] = val; };");
        printf("const auto reduce_min_int __attribute__((unused)) = [&](const bool& condition, const int& val, const AcIntOutputParam& output)"
              	  "{ should_reduce_int[(int)output] = condition; reduce_min_res_int[(int)output] = val; };");
        printf("const auto reduce_max_int __attribute__((unused)) = [&](const bool& condition, const int& val, const AcIntOutputParam& output)"
   		  "{ should_reduce_int[(int)output] = condition; reduce_max_res_int[(int)output] = val; };");
    }


  }
}

static void
gen_return_if_oob(const int curr_kernel)
{
       printf("const bool out_of_bounds = vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z;\n");
       if(kernel_calls_reduce[curr_kernel] )
       {
	 if(NUM_REAL_OUTPUTS)
	 	printf("for(int i = 0; i < NUM_REAL_OUTPUTS; ++i) should_reduce_real[i] = out_of_bounds;\n");
	 if(NUM_INT_OUTPUTS)
	 	printf("for(int i = 0; i < NUM_INT_OUTPUTS; ++i)   should_reduce_int[i] = out_of_bounds;\n");
       }
       printf("if(!out_of_bounds){\n#include \"user_non_scalar_constants.h\"\n");
}
static int
get_original_index(const int* mappings, const int field)
{
	for(int i = 0; i <= NUM_ALL_FIELDS; ++i)
		if (mappings[i] == field) return i;
	return -1;
}

static void
prefetch_output_elements_and_gen_prev_function(const bool gen_mem_accesses, const int cur_kernel)
{
  if(gen_mem_accesses) return;
  // Read vba.out
#if 0
  // Original (compute when needed)
  // SINGLEPASS_INTEGRATION=ON, 4.97 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 6.09 ms (full step, 128^3)
  printf("const auto previous __attribute__((unused)) =[&](const Field& field)"
         "{ return vba.out[field][idx]; };");
#else
  // Prefetch output fields
  // SINGLEPASS_INTEGRATION=ON, 4.18 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 4.77 ms (full step, 128^3)
  //TP: don't gen previous at all if no fields use it. Done to declutter the resulting code and to speedup compilation
  // Note: previous() not enabled for profiles to avoid overhead.
  // Can reconsider if there would be use for it.
  bool gen_previous = false;
  for(int field = 0;  field < NUM_ALL_FIELDS; ++field) gen_previous |= previous_accessed[cur_kernel][field];
  if(!gen_previous) 
  {
  	printf("const auto previous_base __attribute__((unused)) = [&](const Field& field) {(void)field; return (AcReal)NAN;};");
	return;
  }
  for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
  {
    if(previous_accessed[cur_kernel][original_field])
    {
      const int field = get_original_index(field_remappings,original_field);
      printf("const auto f%s_prev = vba.out[%s][idx];", field_names[field], field_names[field]);
    }
  }

  printf("const auto previous_base __attribute__((unused)) = [&](const Field& field)"
         "{ switch (field) {");
  for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
  {
    if(previous_accessed[cur_kernel][original_field])
    {
      const int field = get_original_index(field_remappings,original_field);
      printf("case %s: { return f%s_prev; }", field_names[field], field_names[field]);
    }
  }

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
#endif
}


void
gen_analysis_stencils(FILE* stream)
{
  for (size_t i = 0; i < NUM_STENCILS; ++i)
    fprintf(stream,"const auto %s=[&](const auto& field_in)"
           "{stencils_accessed[field_in][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
}

void
gen_stencil_accesses()
{
  gen_return_if_oob(0);
}

/** ct_const_weights: Compile-time constant weights
  If ct_const_weights = false, the stencil coeffs are fetched from constant
  memory at runtime
*/
/*
static void
prefetch_stencil_coeffs(const int curr_kernel, const bool ct_const_weights)
{
  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_ALL_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            if (ct_const_weights)
              printf("%s;", stencils[stencil][depth][height][width]);
            else
              printf("stencils[%d][%d][%d][%d];", stencil, depth, height,
                     width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }
}

static void
prefetch_stencil_elements(const int curr_kernel)
{
  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_ALL_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_ALL_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("__ldg(&");
              printf("vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              printf(")");
              printf(";");
              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
}

static void
compute_stencil_ops(const int curr_kernel)
{
  int stencil_initialized[NUM_ALL_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_ALL_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("s%d_%d_%d_%d * %s(f%d_%d_%d_%d);", //
                       stencil, depth, height, width,
                       stencil_unary_ops[stencil], field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,"
                       "s%d_%d_%d_%d * %s(f%d_%d_%d_%d)"
                       ");",
                       stencil_binary_ops[stencil], field, stencil, //
                       stencil, depth, height, width,               //
                       stencil_unary_ops[stencil], field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int field = 0; field < NUM_ALL_FIELDS; ++field)
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil)
      if (stencil_initialized[field][stencil] !=
          stencils_accessed[curr_kernel][field][stencil])
        raise_error("stencil_initialized != stencil_accessed, this affects "
                    "gen_stencil_functions (stencil_accessed should be "
                    "replaced with stencil_initialized)");
}
*/

static void
get_stencils_valid_for_profiles(bool stencil_valid_for_profiles[NUM_STENCILS])
{
  // Check which stencils are invalid for profiles
  // (computed in a new array to avoid side effects).
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
#if TWO_D == 1
    stencil_valid_for_profiles[stencil] = false;
#else
    stencil_valid_for_profiles[stencil] = true;
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          const bool hmid = (height == (STENCIL_HEIGHT - 1) / 2);
          const bool wmid = (width == (STENCIL_WIDTH - 1) / 2);
          if (hmid && wmid)
            continue;

          if (stencils[stencil][depth][height][width])
            stencil_valid_for_profiles[stencil] = false;
        }
      }
    }
#endif
  }
}

static void
gen_stencil_functions(const int curr_kernel)
{
  bool stencil_valid_for_profiles[NUM_STENCILS];
  get_stencils_valid_for_profiles(stencil_valid_for_profiles);
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    //TP: don't gen stencil function at all if no fields use it. Done to declutter the resulting code and to speedup compilation
    bool gen_stencil = false;
    for (int field = 0; field < NUM_ALL_FIELDS; ++field) gen_stencil |= stencils_accessed[curr_kernel][field][stencil];
    if(!gen_stencil)
    {
	    printf("const auto %s __attribute__((unused)) = [&](const auto& field) { (void) field; return (AcReal)NAN;};",stencil_names[stencil]);
	    continue;
    }
    printf("const auto %s __attribute__((unused)) = [&](const auto& field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
    {
      if (stencils_accessed[curr_kernel][original_field][stencil])
      {
        const int field = get_original_index(field_remappings,original_field);
        printf("case %s: return f%s_s%s;", field_names[field], field_names[field], stencil_names[stencil]);
      }
    }

    for (int profile = 0; profile < NUM_PROFILES; ++profile) {
      if (stencil_valid_for_profiles[stencil])
        if (stencils_accessed[curr_kernel][NUM_FIELDS + profile][stencil])
          printf("case %d: return p%d_s%d;", NUM_FIELDS + profile, profile,
                 stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}

#if TWO_D == 0
#include "3d_caching_implementations.h"
#endif


#include <stdint.h>
typedef struct {
  uint64_t x, y;
} uint2_64;

typedef struct {
  uint64_t x, y, z;
} uint3_64;

static inline uint64_t
nearest_power_of_to_above(const uint64_t i)
{
  uint64_t power = 1;
  while (power < i)
    power *= 2;
  return power;
}

static inline uint2_64
morton2(const uint64_t pid)
{
  uint64_t i, j;
  i = j = 0;

  for (int bit = 0; bit <= 32; ++bit) {
    const uint64_t mask = 0x1l << 2 * bit;
    i |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
    j |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
  }
  return (uint2_64){i, j};
}

static inline uint3_64
morton3(const uint64_t pid)
{
  uint64_t i, j, k;
  i = j = k = 0;

  for (int bit = 0; bit <= 21; ++bit) {
    const uint64_t mask = 0x1l << 3 * bit;
    i |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
    j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
    k |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
  }
  return (uint3_64){i, j, k};
}

uint64_t
max(const uint64_t a, const uint64_t b)
{
  return a > b ? a : b;
}
void
printf_stencil_point(const int stencil, const int depth, const int height, const int width)
{
	const char* coeff = stencils[stencil][depth][height][width];
	if(!strcmp(coeff,"1")) return;
	if(!strstr(coeff,"DCONST"))
	    printf("%s *",stencils[stencil][depth][height][width]);
	else
	    printf("stencils[%d][%d][%d][%d] *",stencil,depth,height,width);
}

void
gen_kernel_body(const int curr_kernel)
{
  const bool gen_mem_accesses = false;
  if (IMPLEMENTATION != IMPLICIT_CACHING) {
    if (NUM_PROFILES > 0) {
      fprintf(stderr, "Fatal error: NUM_PROFILES > 0 not supported with other "
                      "than IMPLEMENTATION=IMPLICIT_CACHING\n");
      return;
    }
  }
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING: {
    gen_kernel_prefix(curr_kernel);
    gen_return_if_oob(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);

    int stencil_initialized[NUM_ALL_FIELDS + NUM_PROFILES][NUM_STENCILS] = {0};
    // const size_t nbx  = nearest_power_of_to_above(STENCIL_WIDTH);
    // const size_t nby  = nearest_power_of_to_above(STENCIL_HEIGHT);
    // const size_t nbz  = nearest_power_of_to_above(STENCIL_DEPTH);
    // const size_t nall = max(nbz, max(nby, nbx));
    // for (size_t i = 0; i < nall * nall * nall; ++i) {
    //   const uint3_64 spatial = morton3(i);
    //   const int width        = spatial.x;
    //   const int height       = spatial.y;
    //   const int depth        = spatial.z;
    //   if (width >= STENCIL_WIDTH)
    //     continue;
    //   if (height >= STENCIL_HEIGHT)
    //     continue;
    //   if (depth >= STENCIL_DEPTH)
    //     continue;

    // for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    //   const size_t nbx  = nearest_power_of_to_above(STENCIL_WIDTH);
    //   const size_t nby  = nearest_power_of_to_above(STENCIL_HEIGHT);
    //   const size_t nall = max(nbx, nby);
    //   for (size_t i = 0; i < nall * nall; ++i) {
    //     const uint2_64 spatial = morton2(i);
    //     const int width        = spatial.x;
    //     const int height       = spatial.y;
    //     if (width >= STENCIL_WIDTH)
    //       continue;
    //     if (height >= STENCIL_HEIGHT)
    //       continue;

    // BLOCK_SIZE is the number fields processed in sequence
    // f.ex. BLOCK_SIZE = 2 results in the following evaluation order:
    //
    //    f0_s0 += ...
    //    f1_s0 += ...
    //    f0_s1 += ...
    //    f1_s1 += ...
    //
    // BLOCK_SIZE=NUM_ALL_FIELDS by default (the original implementation)
    // tradeoff:
    //  A) larger BLOCK_SIZE
    //    + deeper instruction pipeline (instruction-level parallelism)
    //    - larger working set (can cause cache thrashing)
    //  B) smaller BLOCK_SIZE =
    //    + smaller working set (better cache locality)
    //    - shallower instruction pipeline (more stalling due to data
    //    dependencies)
    const int BLOCK_SIZE = NUM_ALL_FIELDS;
    const int NUM_BLOCKS = (NUM_ALL_FIELDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (BLOCK_SIZE * NUM_BLOCKS < NUM_ALL_FIELDS)
      raise_error("Invalid BLOCK_SIZE * NUM_BLOCKS, was smaller than "
                  "NUM_ALL_FIELDS in stencilgen.c\n");

#if TWO_D == 0
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            for (int field_block = 0; field_block < NUM_BLOCKS; ++field_block) {
              for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                for (int foffset = 0; foffset < BLOCK_SIZE; ++foffset) {
                  const int field = foffset + field_block * BLOCK_SIZE;
                  if (field >= NUM_ALL_FIELDS)
                    break;

                  // Skip if the stencil is not used
                  if (!stencils_accessed[curr_kernel][field][stencil])
                    continue;

                  if (stencils[stencil][depth][height][width]) {
                    if (!stencil_initialized[field][stencil]) {
                      printf("auto f%s_s%s = ", field_names[get_original_index(field_remappings,field)], stencil_names[stencil]);
		      printf_stencil_point(stencil,depth,height,width);
                      printf("%s(", stencil_unary_ops[stencil]);
                      printf("__ldg(&");
                      printf("vba.in[%s]"
                             "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                             "vertexIdx.z+(%d))])",
                             field_names[get_original_index(field_remappings,field)], -STENCIL_ORDER / 2 + width,
                             -STENCIL_ORDER / 2 + height,
                             -STENCIL_ORDER / 2 + depth);
                      printf(")");
                      printf(";");

                      stencil_initialized[field][stencil] = 1;
                    }
                    else {
                      printf("f%s_s%s = ", field_names[get_original_index(field_remappings,field)], stencil_names[stencil]);
                      printf("%s(f%s_s%s, ", stencil_binary_ops[stencil], field_names[get_original_index(field_remappings,field)],
                             stencil_names[stencil]);
		      printf_stencil_point(stencil,depth,height,width);
                      printf("%s(", stencil_unary_ops[stencil]);
                      printf("__ldg(&");
                      printf("vba.in[%s]"
                             "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                             "vertexIdx.z+(%d))])",
                             field_names[get_original_index(field_remappings,field)], -STENCIL_ORDER / 2 + width,
                             -STENCIL_ORDER / 2 + height,
                             -STENCIL_ORDER / 2 + depth);
                      printf(")");
                      printf(");");
                    }
                  }
                }
              }
            }
          }
        }
      }
#else
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            for (int field_block = 0; field_block < NUM_BLOCKS; ++field_block) {
              for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                for (int foffset = 0; foffset < BLOCK_SIZE; ++foffset) {
                  const int field = foffset + field_block * BLOCK_SIZE;
                  if (field >= NUM_ALL_FIELDS)
                    break;

                  // Skip if the stencil is not used
                  if (!stencils_accessed[curr_kernel][field][stencil])
                    continue;

                  if (stencils[stencil][height][width]) {
                    if (!stencil_initialized[field][stencil]) {
                      printf("auto f%s_s%s = ", field_names[get_original_index(field_remappings,field)], stencil_names[stencil]);
		      printf_stencil_point(stencil,depth,height,width);
                      printf("%s(", stencil_unary_ops[stencil]);
                      printf("__ldg(&");
                      printf("vba.in[%s]"
                             "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                             "0)])",
                             field_names[get_original_index(field_remappings,field)], -STENCIL_ORDER / 2 + width,
                             -STENCIL_ORDER / 2 + height);
                      printf(")");
                      printf(";");

                      stencil_initialized[field][stencil] = 1;
                    }
                    else {
                      printf("f%s_s%s = ", field_names[get_original_index(field_remappings,field)], stencil_names[stencil]);
                      printf("%s(f%s_s%s, ", stencil_binary_ops[stencil], field_names[get_original_index(field_remappings,field)],
                             stencil_names[stencil]);
		      printf_stencil_point(stencil,depth,height,width);
                      printf("%s(", stencil_unary_ops[stencil]);
                      printf("__ldg(&");
                      printf("vba.in[%s]"
                             "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                             "0)])",
                             field_names[get_original_index(field_remappings,field)], -STENCIL_ORDER / 2 + width,
                             -STENCIL_ORDER / 2 + height);
                      printf(")");
                      printf(");");
                    }
                  }
                }
              }
            }
          }
        }
#endif

    bool stencil_valid_for_profiles[NUM_STENCILS];
    get_stencils_valid_for_profiles(stencil_valid_for_profiles);

    // Uncomment to print valid stencils
    // for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    //   fprintf(stderr, "Stencil %s (%du): %d\n", stencil_names[stencil],
    //   stencil,
    //           stencil_valid_for_profiles[stencil]);
    // }

    // Profiles
#if TWO_D == 0
    const int PROFILE_BLOCK_SIZE = NUM_PROFILES;
    const int NUM_PROFILE_BLOCKS = NUM_PROFILES ? (NUM_PROFILES +
                                                   PROFILE_BLOCK_SIZE - 1) /
                                                      PROFILE_BLOCK_SIZE
                                                : 0;
    if (PROFILE_BLOCK_SIZE * NUM_PROFILE_BLOCKS < NUM_PROFILES)
      raise_error(
          "Invalid PROFILE_BLOCK_SIZE * NUM_PROFILE_BLOCKS, was smaller than "
          "NUM_PROFILES in stencilgen.c\n");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int profile_block = 0; profile_block < NUM_PROFILE_BLOCKS;
               ++profile_block) {
            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
              for (int foffset = 0; foffset < PROFILE_BLOCK_SIZE; ++foffset) {
                const int profile = foffset +
                                    profile_block * PROFILE_BLOCK_SIZE;
                if (profile >= NUM_PROFILES)
                  break;

                // Skip if the stencil is not used
                if (!stencils_accessed[curr_kernel][NUM_FIELDS + profile]
                                      [stencil])
                  continue;

                // Skip if the stencil is invalid for profiles
                if (!stencil_valid_for_profiles[stencil])
                  continue;

                if (stencils[stencil][depth][height][width]) {
                  if (!stencil_initialized[NUM_FIELDS + profile][stencil]) {
                    printf("auto p%d_s%d = ", profile, stencil);
		    printf_stencil_point(stencil,depth,height,width);
                    printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                    printf("__ldg(&");
#endif
                    printf("vba.profiles.in[%d]"
                           "[vertexIdx.z+(%d)])",
                           profile, -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
                    printf(")");
#endif
                    printf(";");

                    stencil_initialized[NUM_FIELDS + profile][stencil] = 1;
                  }
                  else {
                    printf("p%d_s%d = ", profile, stencil);
                    printf("%s(p%d_s%d, ", stencil_binary_ops[stencil], profile,
                           stencil);
		    printf_stencil_point(stencil,depth,height,width);
                    printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                    printf("__ldg(&");
#endif
                    printf("vba.profiles.in[%d]"
                           "[vertexIdx.z+(%d)])",
                           profile, -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
                    printf(")");
#endif
                    printf(");");
                  }
                }
              }
            }
          }
        }
      }
    }
#endif

    gen_stencil_functions(curr_kernel);
    /*
    gen_kernel_prefix(gen_mem_accesses);
    gen_return_if_oob();

    prefetch_output_elements_and_gen_prev_function(false,curr_kernel);
    prefetch_stencil_elements(curr_kernel);
    prefetch_stencil_coeffs(curr_kernel, false);

    compute_stencil_ops(curr_kernel);
    gen_stencil_functions(curr_kernel);
    */

    return;
  }
#if TWO_D == 0
  case EXPLICIT_CACHING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_CACHING_3D_BLOCKING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_CACHING_4D_BLOCKING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_4d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_PINGPONG_txw: {
    #if IMPLEMENTATION == EXPLICIT_PINGPONG_txw
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_pingpong_txw_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    #endif
    return;
  }
  case EXPLICIT_PINGPONG_txy: {
    #if IMPLEMENTATION == EXPLICIT_PINGPONG_txy
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_pingpong_txy_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    #endif
    return;
  }
  case EXPLICIT_PINGPONG_txyblocked: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    // prefetch_stencil_elems_to_smem_pingpong_txyblocked_and_compute_stencil_ops(
    //     curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_ROLLING_PINGPONG: {
    #if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    #endif
    return;
  }
#endif
  default: {
    fprintf(stderr,
            "Fatal error: invalid IMPLEMENTATION passed to stencilgen.c");
    return;
  }
  }
}

int
main(int argc, char** argv)
{
  (void)vtxbuf_names; // Unused

  // Generate stencil definitions
  if (argc == 2 && !strcmp(argv[1], "-definitions")) {
    gen_stencil_definitions();
  }
  // Generate memory accesses for the DSL kernels
  else if (argc == 2 && !strcmp(argv[1], "-mem-accesses")) {
    gen_stencil_accesses(true);
  }
  else if (argc == 3) {
    const int curr_kernel = atoi(argv[2]);
    gen_kernel_body(curr_kernel);
  }
  else {
    fprintf(stderr, "Fatal error: invalid arguments passed to stencilgen.c");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

