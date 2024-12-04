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

#include "builtin_enums.h"
#include "user_defines.h"

#include "stencil_accesses.h"
#include "stencilgen.h"

#include "implementation.h"

#if AC_USE_HIP
const char* ffs_string = "__ffsll";
#else
const char* ffs_string = "__ffs";
#endif


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
  {
    raise_error("Must declare at least one Field in the DSL code!");
  }

  if (!NUM_STENCILS)
  {
    raise_error("Must declare at least one Stencil in the DSL code!");
  }

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
  FILE* stencil_coeffs_file= fopen("coeffs.h", "w");
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
  fclose(stencil_coeffs_file);
}

#include "kernel_reduce_info.h"

void
gen_kernel_common_prefix()
{
  printf("const int3 tid = (int3){"
         "threadIdx.x + blockIdx.x * blockDim.x,"
         "threadIdx.y + blockIdx.y * blockDim.y,"
         "threadIdx.z + blockIdx.z * blockDim.z,"
         "};");
  printf("const int3 vertexIdx = (int3){"
         "tid.x + start.x,"
         "tid.y + start.y,"
         "tid.z + start.z,"
         "};");
  printf("const int3 globalVertexIdx = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
  printf("[[maybe_unused]] const int idx = DEVICE_VTXBUF_IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");
  printf("(void)vba;");

    printf(
        "const int3 localCompdomainVertexIdx = (int3){"
        "threadIdx.x + blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
        "threadIdx.y + blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
        "threadIdx.z + blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2,"
        "};");
  printf("const int local_compdomain_idx = "
         "LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)local_compdomain_idx;");     // Silence unused warning
					     //
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
  printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.y];");
  printf("};");

  printf("const auto value_profile_xz __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.z];");
  printf("};");

  printf("const auto value_profile_yx __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.x];");
  printf("};");

  printf("const auto value_profile_yz __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z];");
  printf("};");

  printf("const auto value_profile_zx __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.x];");
  printf("};");

  printf("const auto value_profile_zy __attribute__((unused)) = [&](const Profile& handle) {");
  printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.y];");
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

bool
has_buffered_writes(const char* kernel_name)
{
	return strstr(kernel_name,"FUSED") != NULL;
}

static int
get_original_index(const int* mappings, const int field)
{
	for(int i = 0; i <= NUM_ALL_FIELDS; ++i)
		if (mappings[i] == field) return i;
	return -1;
}

void
gen_kernel_write_funcs(const int curr_kernel)
{
  // Write vba.out
  #if 1
    // Original
    bool written_something = false;
    for(int field = 0; field < NUM_ALL_FIELDS; ++field)
	    written_something |= write_called[curr_kernel][field];
    if(!written_something) return;
    if(has_buffered_writes(kernel_names[curr_kernel]))
    {
  	    for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
  	    {
      	      if (stencils_accessed[curr_kernel][original_field][0]) continue;
  	      if(!write_called[curr_kernel][original_field]) continue;
	      //TP: field is buffered written but not read, for now simply set value to 0
  	      const int field = get_original_index(field_remappings,original_field);
	      printf("f%s_value_stencil = 0.0;",field_names[field]);
	    }
	    printf("const auto write_base __attribute__((unused)) = [&](const Field& handle, const AcReal& value) {");
            printf("switch (handle) {");
  	    for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
  	    {
  	      if(!write_called[curr_kernel][original_field]) continue;
  	      const int field = get_original_index(field_remappings,original_field);
      	      printf("case %s: { f%s_svalue_stencil = value; break;}", field_names[field], field_names[field]);
  	    }
      	    printf("default: { break;}");
	    printf("}};");
	    return;
    }

    const bool AC_NON_TEMPORAL_WRITES = false;
    if(!AC_NON_TEMPORAL_WRITES)
    {
    	printf("const auto write_base __attribute__((unused)) = [&](const Field& handle, const AcReal& value) {");
    	printf("vba.out[handle][idx] = value;");
    	printf("};");
    }
    else
    {
    //  Non-temporal store intrinsic could reduce L2 pressure on AMD but no effect
    //  in practice (no effect on the first pass, a slight slowdown in the second
    //  pass 4.6 ms vs 4.3 ms)
     printf("const auto write_base __attribute__((unused)) =[&](const Field& field, const AcReal value)"
      "{ __builtin_nontemporal_store(value, &vba.out[field][idx]); };");
    }
  
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
}

void
gen_kernel_prefix(const int curr_kernel)
{
  (void)curr_kernel;
  gen_kernel_common_prefix();
}
typedef enum ReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} ReduceOp;
void
print_reduce_ops(const ReduceOp op, const char* define_name)
{
	printf("if (!condition) return;");
#if AC_USE_HIP
	const char* shuffle_instruction = "rocprim::warp_shuffle(val,target_tid)";
#else
	const char* shuffle_instruction = "__shfl_sync(AC_INTERNAL_active_threads,val,target_tid)";
#endif
	const char* op_instruction = 
		op == REDUCE_SUM ? "val += shuffle_tmp" :
		op == REDUCE_MIN ? "val = val > shuffle_tmp ? shuffle_tmp : val;" :
		op == REDUCE_MAX ? "val = val > shuffle_tmp ? val : shuffle_tmp;" :
		NULL;
	if(op == NO_REDUCE || op_instruction == NULL) 
		printf("Incorrect reduction for %s\n",define_name);

	//TP: the idea behind the algorithm is that if the target tid is inactive 
	//we access the value the target tid was responsible for reducing
	//
	//The target tid should be the smallest **active** thread index between the normal target (lane_id + offset) 
	//and those that the target tid is responsible for reducing.
	//since the tid we access is active it has reduced all values it was responsible for
	
	//The smallest active thread index can be efficiently calculated with bit operations and ctz (Count Trailing Zeroes)
	//Unfortunately CUDA does not support ctz so we calculate ctz with ffs-1 (Find First Set)
	//Could use ctz on HIP but for not use ffs on both
	printf("if constexpr (warp_size == 32) {"
			"if (AC_INTERNAL_all_threads_active) {"
				"size_t target_tid = lane_id + 16;"
				"auto shuffle_tmp = %s;"
		        	"%s;"

				"target_tid = lane_id + 8;"
				"shuffle_tmp = %s;"
		        	"%s;"

				"target_tid = lane_id + 4;"
				"shuffle_tmp = %s;"
		        	"%s;"

				"target_tid = lane_id + 2;"
				"shuffle_tmp = %s;"
		        	"%s;"

				"target_tid = lane_id + 1;"
				"shuffle_tmp = %s;"
		        	"%s;"
			"}"
			"else if (AC_INTERNAL_lower_active_threads_are_contiguos ) {"
				"size_t target_tid = lane_id + 16;"
				"auto shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 8;"
				"shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 4;"
				"shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 2;"
				"shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 1;"
				"shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

			"}"
			"else {"
				"size_t target_tid = lane_id + 16;"
				"auto shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"size_t target_mask = (16777472 << lane_id);" //2^8 + 2^24
				"auto active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"target_mask = (268439568 << lane_id);" //2^4+2^12+2^28
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"target_mask = (1073758276 << lane_id);" //2^2+2^6+2^14+2^30
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"target_mask = (2147516554<< lane_id);" //2^1+2^3+2^7+2^15+2^31
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"
			"}"
		"}"
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction

	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction

	      ,shuffle_instruction,op_instruction

	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	  );

#if AC_USE_HIP
//TP: if we use CUDA we get compiler warnings about too large shifts since active threads is unsigned long instead of unsigned long long
	printf("else if constexpr (warp_size == 64) {"
			"if (AC_INTERNAL_all_threads_active) {"
				"unsigned long long target_tid = lane_id + 32;"
				"auto shuffle_tmp = %s;"
				"%s;"

				"target_tid = lane_id + 16;"
				"shuffle_tmp = %s;"
				"%s;"

				"target_tid = lane_id + 8;"
				"shuffle_tmp = %s;"
				"%s;"

				"target_tid = lane_id + 4;"
				"shuffle_tmp = %s;"
				"%s;"

				"target_tid = lane_id + 2;"
				"shuffle_tmp = %s;"
				"%s;"

				"target_tid = lane_id + 1;"
				"shuffle_tmp = %s;"
				"%s;"
			"}"
			"else if (AC_INTERNAL_lower_active_threads_are_contiguos) {"
				"unsigned long long target_tid = lane_id + 32;"
				"auto shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 16;"
				"shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 8;"
				"shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 4;"
				"shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 2;"
				"shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"target_tid = lane_id + 1;"
				"shuffle_tmp = %s;"
				"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"
			"}"
			"else {"
				"unsigned long long target_tid = lane_id + 32;"
				"auto shuffle_tmp = %s;"
		        	"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"

				"constexpr unsigned long long possible_tids_1 = (AC_one << 16) + (AC_one << 48);"
				"unsigned long long target_mask = (possible_tids_1 << lane_id);"
				"auto active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"const unsigned long long possible_tids_2 = (AC_one << 8) + (AC_one << 24) + (AC_one << 56);"
				"target_mask = (possible_tids_2 << lane_id);"
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"const unsigned long long possible_tids_3 = (AC_one << 4) + (AC_one << 12) + (AC_one << 28) + (AC_one << 60);"
				"target_mask = (possible_tids_3 << lane_id);"
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"const unsigned long long possible_tids_4 = (AC_one << 2) + (AC_one << 6) + (AC_one << 14) + (AC_one << 30) + (AC_one << 62);"
				"target_mask = (possible_tids_4 << lane_id);"
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"

				"const unsigned long long possible_tids_5 = (AC_one << 1) + (AC_one << 3) + (AC_one << 7) + (AC_one << 15) + (AC_one << 31) + (AC_one << 63);"
				"target_mask = (possible_tids_5 << lane_id);"
				"active = (AC_INTERNAL_active_threads & target_mask);"
				"target_tid = !active ? 0 : %s(active)-1;"
				"shuffle_tmp = %s;"
		        	"if(target_tid) %s;"
			"}"
		"}"
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction

	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction
	      ,shuffle_instruction,op_instruction


	      ,shuffle_instruction,op_instruction

	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	      ,ffs_string,shuffle_instruction,op_instruction
	  );
#endif
	printf(
		  "if(lane_id == warp_leader_id) {vba.reduce_scratchpads_%s[(int)output][0][warp_out_index] = val;}"
	      ,define_name);
}
void
printf_reduce_funcs(const char* datatype, const char* define_name, const char* enum_name)
{
        printf("const auto reduce_sum_%s __attribute__((unused)) = [&](const bool& condition, %s val, const %sOutputParam& output) {",define_name,datatype,enum_name);
	print_reduce_ops(REDUCE_SUM, define_name);
	printf("};");

        printf("const auto reduce_min_%s __attribute__((unused)) = [&](const bool& condition, %s val, const %sOutputParam& output) {",define_name,datatype,enum_name);
	print_reduce_ops(REDUCE_MIN, define_name);
	printf("};");

        printf("const auto reduce_max_%s __attribute__((unused)) = [&](const bool& condition, %s val, const %sOutputParam& output) {",define_name,datatype,enum_name);
	print_reduce_ops(REDUCE_MAX, define_name);
	printf("};");

}

void
gen_kernel_reduce_funcs(const int curr_kernel)
{
  if(kernel_calls_reduce[curr_kernel] )
  {
#if AC_USE_HIP
	printf("constexpr size_t warp_size = warpSize;");
        printf("const size_t warp_id = rocprim::warp_id();");
	printf("constexpr unsigned long long AC_one = 1;");
        printf("const unsigned long long AC_INTERNAL_lower_warp_mask = (AC_one << (warp_size/2)) - 1;");
#else
	printf("constexpr size_t warp_size = 32;");
	printf("const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size;");
        printf("constexpr size_t AC_INTERNAL_lower_warp_mask = (1 << 16) - 1;");
#endif
    printf("const size_t AC_INTERNAL_lower_active_warp_mask = AC_INTERNAL_active_threads & AC_INTERNAL_lower_warp_mask;");
    //TP: if lower (lane_id < warp_size/2) active threads are contiguous i.e. there are no inactive threads between active threads
    //then can perform the reductions without calculating tids
    printf("const bool AC_INTERNAL_lower_active_threads_are_contiguos = !(AC_INTERNAL_lower_active_warp_mask & (AC_INTERNAL_lower_active_warp_mask+1));");
    //TP: if all threads are active can skip checks checking if target tid is active in reductions
    printf("const bool AC_INTERNAL_all_threads_active = AC_INTERNAL_active_threads+1 == 0;");
    printf("const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) %% warp_size;");
    printf("const size_t warp_leader_id  = %s(AC_INTERNAL_active_threads)-1;",ffs_string);
    printf("const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;");
    printf("const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;");
    printf("const int warp_out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;");
    if(NUM_REAL_OUTPUTS)
	printf_reduce_funcs("AcReal","real","AcReal");
    if(NUM_INT_OUTPUTS)
	printf_reduce_funcs("int","int","AcInt");


  }
}
static void
gen_return_if_oob(const int curr_kernel)
{
       printf("const bool out_of_bounds = vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z;\n");
       if(kernel_calls_reduce[curr_kernel] )
       {
#if AC_USE_HIP
	       printf("const auto AC_INTERNAL_active_threads = __ballot(!out_of_bounds);");
#else
	       printf("const auto AC_INTERNAL_active_threads = __ballot_sync(0xffffffff,!out_of_bounds);");
#endif
       }
       printf("if(out_of_bounds) return;");
       printf("{\n"
			"#include \"user_non_scalar_constants.h\"\n"
			"#include \"user_builtin_non_scalar_constants.h\"\n"
	     );
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
              printf("vba.in[%d][DEVICE_VTXBUF_IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
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

#include "3d_caching_implementations.h"


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
	const bool no_load = false;
	const char* coeff = stencils[stencil][depth][height][width];
	if(!strcmp(coeff,"1")) return;
	if(!strstr(coeff,"DCONST") && no_load)
	    printf("%s *",coeff);
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
    gen_kernel_reduce_funcs(curr_kernel);
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
    {
      raise_error("Invalid BLOCK_SIZE * NUM_BLOCKS, was smaller than "
                  "NUM_ALL_FIELDS in stencilgen.c\n");
    }

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
                             "[DEVICE_VTXBUF_IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
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
                             "[DEVICE_VTXBUF_IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
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

    bool stencil_valid_for_profiles[NUM_STENCILS];
    get_stencils_valid_for_profiles(stencil_valid_for_profiles);

    // Uncomment to print valid stencils
    // for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    //   fprintf(stderr, "Stencil %s (%du): %d\n", stencil_names[stencil],
    //   stencil,
    //           stencil_valid_for_profiles[stencil]);
    // }

    // Profiles
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
                    printf("__ldg(&");
                    printf("vba.profiles.in[%d]"
                           "[vertexIdx.z+(%d)])",
                           profile, -STENCIL_ORDER / 2 + depth);
                    printf(")");
                    printf(";");

                    stencil_initialized[NUM_FIELDS + profile][stencil] = 1;
                  }
                  else {
                    printf("p%d_s%d = ", profile, stencil);
                    printf("%s(p%d_s%d, ", stencil_binary_ops[stencil], profile,
                           stencil);
		    printf_stencil_point(stencil,depth,height,width);
                    printf("%s(", stencil_unary_ops[stencil]);
                    printf("__ldg(&");
                    printf("vba.profiles.in[%d]"
                           "[vertexIdx.z+(%d)])",
                           profile, -STENCIL_ORDER / 2 + depth);
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
  case EXPLICIT_CACHING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);
    gen_kernel_reduce_funcs(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_CACHING_3D_BLOCKING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);
    gen_kernel_reduce_funcs(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    return;
  }
  case EXPLICIT_CACHING_4D_BLOCKING: {
    gen_kernel_prefix(curr_kernel); // Note no bounds check

    prefetch_stencil_elems_to_smem_4d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob(curr_kernel);
    gen_kernel_reduce_funcs(curr_kernel);

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
    gen_kernel_reduce_funcs(curr_kernel);

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
    gen_kernel_reduce_funcs(curr_kernel);

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
    gen_kernel_reduce_funcs(curr_kernel);

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
    gen_kernel_reduce_funcs(curr_kernel);

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(gen_mem_accesses,curr_kernel);
    #endif
    return;
  }
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
    gen_kernel_write_funcs(curr_kernel);
  }
  else {
    fprintf(stderr, "Fatal error: invalid arguments passed to stencilgen.c");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

