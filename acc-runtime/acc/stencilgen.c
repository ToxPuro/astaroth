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
typedef enum ReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} ReduceOp;

void
print_warp_reduce_func(const char* datatype, const char* define_name, const ReduceOp op);

#include "stencilgen.h"

#include "implementation.h"
#define ONE_DIMENSIONAL_PROFILE (1 << 20)
#define TWO_DIMENSIONAL_PROFILE (1 << 21)
typedef enum {
	PROFILE_X  = (1 << 0) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Y  = (1 << 1) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Z  = (1 << 2) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_XY = (1 << 3) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_XZ = (1 << 4) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YX = (1 << 5) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YZ = (1 << 6) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZX = (1 << 7) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZY = (1 << 8) | TWO_DIMENSIONAL_PROFILE,
} AcProfileType;
#include "profiles_info.h"

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
            //const char* coeff = stencils[stencil][depth][height][width];
	    //printf("%s,",coeff && !strstr(coeff,"DCONST") ? coeff : "AcReal(NAN)");
	    //TP: always have initially nans in the stencils to ensure stencils are always loaded at runtime
	    printf("AcReal(NAN),");
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
#include "mem_access_helper_funcs.h"

bool
z_block_loop_before_y(const int curr_kernel)
{
	int reductions_across_y = 0;
	int reductions_across_z = 0;
	for(int i = 0; i < NUM_PROFILES; ++i)
	{
		if(!reduced_profiles[curr_kernel][i]) continue;
		reductions_across_y += (prof_types[i] == PROFILE_XZ || prof_types[i] == PROFILE_ZX || prof_types[i] == PROFILE_Z);
		reductions_across_z += (prof_types[i] == PROFILE_YX || prof_types[i] == PROFILE_XY || prof_types[i] == PROFILE_Y);
	}
	return reductions_across_y > reductions_across_z;

}

int
get_num_reduced_profiles(const AcProfileType prof_type, const int kernel)
{
	int res = 0;
    	for(int i = 0; i < NUM_PROFILES; ++i)
	    res += (reduced_profiles[kernel][i] && prof_types[i] == prof_type);
	return res;
}

int
get_num_reduced_vars(const int N, const int* arr, const ReduceOp op)
{
	int res = 0;
	for(int i = 0; i < N; ++i) res += ((ReduceOp)arr[i] == op);
	return res;
}

void
gen_kernel_block_loops(const int curr_kernel)
{
  printf(
	"#include \"user_non_scalar_constants.h\"\n"
	"#include \"user_builtin_non_scalar_constants.h\"\n"
	 );
#if AC_USE_HIP
   printf("[[maybe_unused]] constexpr size_t warp_size = rocprim__warpSize();");
#else
   printf("[[maybe_unused]] constexpr size_t warp_size = 32;");
#endif
  if(kernel_calls_reduce[curr_kernel] && (kernel_has_block_loops(curr_kernel) || BUFFERED_REDUCTIONS))
  {
		for(int i = 0; i < NUM_REAL_OUTPUTS; ++i)
		{
			if(reduced_reals[curr_kernel][i] == REDUCE_SUM)
				printf("AcReal %s_reduce_output = 0.0;",real_output_names[i]);
			if(reduced_reals[curr_kernel][i] == REDUCE_MIN)
				printf("AcReal %s_reduce_output = AC_REAL_MAX;",real_output_names[i]);
			if(reduced_reals[curr_kernel][i] == REDUCE_MAX)
				printf("AcReal %s_reduce_output = -AC_REAL_MAX;",real_output_names[i]);
		}
		for(int i = 0; i < NUM_INT_OUTPUTS; ++i)
		{
			if(reduced_ints[curr_kernel][i] == REDUCE_SUM)
				printf("int %s_reduce_output = 0.0;",int_output_names[i]);
			if(reduced_ints[curr_kernel][i] == REDUCE_MIN)
				printf("int %s_reduce_output = INT_MAX;",int_output_names[i]);
			if(reduced_ints[curr_kernel][i] == REDUCE_MAX)
				printf("int %s_reduce_output = -INT_MAX;",int_output_names[i]);
		}
#if AC_DOUBLE_PRECISION
		for(int i = 0; i < NUM_FLOAT_OUTPUTS; ++i)
		{
			if(reduced_floats[curr_kernel][i] == REDUCE_SUM)
				printf("float %s_reduce_output = 0.0;",float_output_names[i]);
			if(reduced_floats[curr_kernel][i] == REDUCE_MIN)
				printf("float %s_reduce_output = FLT_MAX;",float_output_names[i]);
			if(reduced_floats[curr_kernel][i] == REDUCE_MAX)
				printf("float %s_reduce_output = -FLT_MAX;",float_output_names[i]);
		}
#endif
	#if AC_USE_HIP
        	printf("const size_t warp_id = rocprim__warpId();");
	#else
		printf("const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size;");
	#endif

	{
		if(kernel_has_block_loops(curr_kernel))
	    		printf("[[maybe_unused]] constexpr size_t warp_leader_id  = 0;");
	    	printf("const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) %% warp_size;");
	    	printf("const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;");
	    	printf("const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;");
	    	printf("[[maybe_unused]] const int warp_out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;");
       #if AC_USE_HIP
	        printf("[[maybe_unused]] auto AC_INTERNAL_active_threads = __ballot(1);");
       #else
	        printf("[[maybe_unused]] auto AC_INTERNAL_active_threads = __ballot_sync(0xffffffff,1);");
       #endif
    		printf("[[maybe_unused]] bool AC_INTERNAL_active_threads_are_contiguos = !(AC_INTERNAL_active_threads & (AC_INTERNAL_active_threads+1));");
    		//TP: if all threads are active can skip checks checking if target tid is active in reductions
    		printf("[[maybe_unused]] bool AC_INTERNAL_all_threads_active = AC_INTERNAL_active_threads+1 == 0;");
  		print_warp_reduce_func("AcReal", "real", REDUCE_SUM);
		if(get_num_reduced_vars(NUM_REAL_OUTPUTS,reduced_reals[curr_kernel],REDUCE_MIN)) print_warp_reduce_func("AcReal", "real", REDUCE_MIN);
		if(get_num_reduced_vars(NUM_REAL_OUTPUTS,reduced_reals[curr_kernel],REDUCE_MAX)) print_warp_reduce_func("AcReal", "real", REDUCE_MAX);

		if(get_num_reduced_vars(NUM_INT_OUTPUTS,reduced_ints[curr_kernel],REDUCE_SUM))   print_warp_reduce_func("int", "int", REDUCE_SUM);
		if(get_num_reduced_vars(NUM_INT_OUTPUTS,reduced_ints[curr_kernel],REDUCE_MIN))   print_warp_reduce_func("int", "int", REDUCE_MIN);
		if(get_num_reduced_vars(NUM_INT_OUTPUTS,reduced_ints[curr_kernel],REDUCE_MAX))   print_warp_reduce_func("int", "int", REDUCE_MAX);
#if AC_DOUBLE_PRECISION
		if(get_num_reduced_vars(NUM_FLOAT_OUTPUTS,reduced_floats[curr_kernel], REDUCE_SUM)) print_warp_reduce_func("float", "float", REDUCE_SUM);
		if(get_num_reduced_vars(NUM_FLOAT_OUTPUTS,reduced_floats[curr_kernel], REDUCE_MIN)) print_warp_reduce_func("float", "float", REDUCE_MIN);
		if(get_num_reduced_vars(NUM_FLOAT_OUTPUTS,reduced_floats[curr_kernel], REDUCE_MAX)) print_warp_reduce_func("float", "float", REDUCE_MAX);
#endif
	}
  }
  if(kernel_has_block_loops(curr_kernel))
  {

	const bool z_before = z_block_loop_before_y(curr_kernel);




	printf("const dim3 step_size = {"
			"gridDim.x*blockDim.x,"
			"gridDim.y*blockDim.y,"
			"gridDim.z*blockDim.z"
	       "};");
	printf("const int should_not_be_more_y = end.y-start.y-threadIdx.y-(blockIdx.y*blockDim.y);");
	printf("const int should_not_be_more_z = end.z-start.z-threadIdx.z-(blockIdx.z*blockDim.z);");

	printf("const int calculated_end_y = ((should_not_be_more_y + step_size.y - 1)/(step_size.y));");
	printf("const int calculated_end_z = ((should_not_be_more_z + step_size.z - 1)/(step_size.z));");

	printf("[[maybe_unused]] const int last_block_idx_y = (vba.block_factor.y < calculated_end_y ? vba.block_factor.y : calculated_end_y) -1;");
	printf("[[maybe_unused]] const int last_block_idx_z = (vba.block_factor.z < calculated_end_z ? vba.block_factor.z : calculated_end_z) -1;");

  	printf("for(int current_block_idx_x = 0; current_block_idx_x < vba.block_factor.x; ++current_block_idx_x) {");
	for(int i = 0; i < NUM_PROFILES; ++i)
	{
		if(!reduced_profiles[curr_kernel][i]) continue;
		if(prof_types[i] == PROFILE_X)
			printf("AcReal %s_reduce_output = 0.0;",profile_names[i]);
	}
	for(int i = 0; i < NUM_PROFILES; ++i)
	{
		if(!reduced_profiles[curr_kernel][i]) continue;
		if(!z_before && (prof_types[i] == PROFILE_XZ || prof_types[i] == PROFILE_ZX || prof_types[i] == PROFILE_Z))
			printf("AcReal %s_reduce_output[16]{};",profile_names[i]);
		if(z_before && (prof_types[i] == PROFILE_XY || prof_types[i] == PROFILE_YX || prof_types[i] == PROFILE_Y))
			printf("AcReal %s_reduce_output[16]{};",profile_names[i]);
	}

	if(z_before)
  		printf("for(int current_block_idx_z = 0; current_block_idx_z < vba.block_factor.z; ++current_block_idx_z) {");
	else
  		printf("for(int current_block_idx_y = 0; current_block_idx_y < vba.block_factor.y; ++current_block_idx_y) {");

	for(int i = 0; i < NUM_PROFILES; ++i)
	{
		if(!reduced_profiles[curr_kernel][i]) continue;
		if(z_before && (prof_types[i] == PROFILE_XZ || prof_types[i] == PROFILE_ZX || prof_types[i] == PROFILE_Z))
			printf("AcReal %s_reduce_output{0.0};",profile_names[i]);
		if(!z_before && (prof_types[i] == PROFILE_XY || prof_types[i] == PROFILE_YX || prof_types[i] == PROFILE_Y))
			printf("AcReal %s_reduce_output{0.0};",profile_names[i]);
	}

	if(!z_before)
  		printf("for(int current_block_idx_z = 0; current_block_idx_z < vba.block_factor.z; ++current_block_idx_z) {");
	else
  		printf("for(int current_block_idx_y = 0; current_block_idx_y < vba.block_factor.y; ++current_block_idx_y) {");
  	printf("const dim3 current_block_idx = {"
			"blockIdx.x + current_block_idx_x*gridDim.x," 
			"blockIdx.y + current_block_idx_y*gridDim.y," 
			"blockIdx.z + current_block_idx_z*gridDim.z," 
			"};");
	return;
  }
  else
  {
	  printf("{"
		 "{"
		 "{"
		);
  }
  printf("const dim3 current_block_idx = blockIdx;");
}
void
gen_kernel_common_prefix()
{ 
  printf("const int3 tid = (int3){"
         "threadIdx.x + current_block_idx.x * blockDim.x,"
         "threadIdx.y + current_block_idx.y * blockDim.y,"
         "threadIdx.z + current_block_idx.z * blockDim.z,"
         "};");
  printf("const int3 vertexIdx = (int3){"
         "tid.x + start.x,"
         "tid.y + start.y,"
         "tid.z + start.z,"
         "};");
  printf("const int3 globalVertexIdx __attribute__((unused)) = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
  printf("[[maybe_unused]] const int idx = DEVICE_VTXBUF_IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");
  printf("(void)vba;");

    printf(
        "const int3 localCompdomainVertexIdx = (int3){"
        "threadIdx.x + current_block_idx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
        "threadIdx.y + current_block_idx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
        "threadIdx.z + current_block_idx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2,"
        "};");
  printf("const int local_compdomain_idx = "
         "LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)local_compdomain_idx;");     // Silence unused warning
					     //
}
void 
gen_profile_funcs(const int kernel)
{
  //TP: for now profile reads are not cached since they are usually read in only once and anyways since they are smaller can fit more easily to cache.
  //TP: if in the future a use case uses profiles a lot reconsider this
  if(kernel_reads_profile(kernel))
  {
  	printf("[[maybe_unused]] const auto value_profile_x __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_y __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_z __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_xy __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.y];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_xz __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_yx __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_yz __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_zx __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_zy __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.y];");
  	printf("};");
  }
  else
  {
  	printf("[[maybe_unused]] const auto value_profile_x __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_y __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_z __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_xy __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.y];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_xz __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_yx __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_yz __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_zx __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.x];");
  	printf("};");

  	printf("[[maybe_unused]] const auto value_profile_zy __attribute__((unused)) = [&](const Profile& handle) {");
  	printf("return vba.profiles.in[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.y];");
  	printf("};");
  }
  {
    if(kernel_has_block_loops(kernel))
    	printf("[[maybe_unused]] const int3 profileReduceOutputVertexIdx= (int3){"
    	       "threadIdx.x + blockIdx.x * blockDim.x,"
    	       "threadIdx.y + blockIdx.y * blockDim.y,"
    	       "threadIdx.z + blockIdx.z * blockDim.z,"
    	       "};");    
    else
    	printf("[[maybe_unused]] const int3& profileReduceOutputVertexIdx = vertexIdx;");


    if(!get_num_reduced_profiles(PROFILE_X,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_x __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_x __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_X) continue;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output += val;",profile_names[i]);
    	    	printf("if(current_block_idx_y == last_block_idx_y && current_block_idx_z == last_block_idx_z) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][vertexIdx.x + VAL(AC_mlocal).x*(start.y - NGHOST + profileReduceOutputVertexIdx.y + VAL(AC_reduction_tile_dimensions).y*(start.z - NGHOST + profileReduceOutputVertexIdx.z))] = %s_reduce_output;",profile_names[i]);
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }

    //!!TP: NOTE this only works as long as blockfactor.x == 1!!
    if(!get_num_reduced_profiles(PROFILE_Y,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_y __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_y __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? "[current_block_idx_y]"
    	    	    				  : ""
    	    					  ;
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_Y) continue;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_z == last_block_idx_z) {"); 
    	    	printf("if(blockDim.x %% warp_size != 0) {");
    	    		printf("d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][start.x - NGHOST + profileReduceOutputVertexIdx.x + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*(start.z - NGHOST + profileReduceOutputVertexIdx.z))] = %s_reduce_output%s;",profile_names[i],access);
    	    	printf("}");
    	    	printf("else {");
    	    	   	printf("const AcReal reduce_res = warp_reduce_sum_real(%s_reduce_output%s);",profile_names[i],access);
    	    		printf("if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][localCompdomainVertexIdx.x/warp_size + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*(start.z - NGHOST + profileReduceOutputVertexIdx.z))] = reduce_res;");
    	    	printf("}");
    	    	printf("}");
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }

    
    if(!get_num_reduced_profiles(PROFILE_Z,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_z __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	//!!TP: NOTE this only works as long as blockfactor.x == 1!!
    	printf("const auto reduce_sum_real_z __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_Z) continue;
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? ""
    	    	    				  : "[current_block_idx_z]"
    	    					  ;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_y == last_block_idx_y) {"); 
    	    		printf("if(blockDim.x %% warp_size != 0) {");
    	    			printf("d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][start.x - NGHOST + profileReduceOutputVertexIdx.x + VAL(AC_reduction_tile_dimensions).x*(start.y - NGHOST + profileReduceOutputVertexIdx.y + VAL(AC_reduction_tile_dimensions).y*vertexIdx.z)] = %s_reduce_output%s;",profile_names[i],access);
    	    		printf("}");
    	    		printf("else {");
    	    	   		printf("const AcReal reduce_res = warp_reduce_sum_real(%s_reduce_output%s);",profile_names[i],access);
    	    			printf("if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][localCompdomainVertexIdx.x/warp_size + VAL(AC_reduction_tile_dimensions).x*(start.z - NGHOST + profileReduceOutputVertexIdx.y + VAL(AC_reduction_tile_dimensions).y*vertexIdx.z)] = reduce_res;");
    	    		printf("}");
    	    	printf("}");
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }


    if(!get_num_reduced_profiles(PROFILE_XY,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_xy __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_xy __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_XY) continue;
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? "[current_block_idx_y]"
    	    	    				  : ""
    	    					  ;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_z == last_block_idx_z) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][vertexIdx.x + VAL(AC_mlocal).x*(vertexIdx.y + VAL(AC_mlocal).y*(start.z - NGHOST + profileReduceOutputVertexIdx.z))] = %s_reduce_output%s;",profile_names[i],access);
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }

    if(!get_num_reduced_profiles(PROFILE_YX,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_yx __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_yx __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? "[current_block_idx_y]"
    	    	    				  : ""
    	    					  ;
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_YX) continue;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_z == last_block_idx_z) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][vertexIdx.x + VAL(AC_mlocal).x*(vertexIdx.y + VAL(AC_mlocal).y*(start.z - NGHOST + profileReduceOutputVertexIdx.z))] = %s_reduce_output%s;",profile_names[i],access);
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }

    if(!get_num_reduced_profiles(PROFILE_XZ,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_xz __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_xz __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_XZ) continue;
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? ""
    	    	    				  : "[current_block_idx_z]"
    	    					  ;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_y == last_block_idx_y) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][vertexIdx.x + VAL(AC_mlocal).x*(start.y - NGHOST + profileReduceOutputVertexIdx.y + VAL(AC_reduction_tile_dimensions).y*vertexIdx.z)] = %s_reduce_output%s;",profile_names[i],access);
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }

    if(!get_num_reduced_profiles(PROFILE_ZX,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_zx __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
    	printf("const auto reduce_sum_real_zx __attribute__((unused)) = [&](const AcReal& val, const Profile& output) {");
    	printf("switch (output) {");
    	for(int i = 0; i < NUM_PROFILES; ++i)
    	{
    	        const char* access = 
    	    	    z_block_loop_before_y(kernel) ? ""
    	    	    				  : "[current_block_idx_z]"
    	    					  ;
    	        if(!reduced_profiles[kernel][i] || prof_types[i] != PROFILE_ZX) continue;
    	 	    printf("case %s: { ",profile_names[i]);
    	        	printf("%s_reduce_output%s += val;",profile_names[i],access);
    	    	printf("if(current_block_idx_y == last_block_idx_y) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][vertexIdx.x + VAL(AC_mlocal).x*(start.y - NGHOST + profileReduceOutputVertexIdx.y + VAL(AC_reduction_tile_dimensions).y*vertexIdx.z)] = %s_reduce_output%s;",profile_names[i],access);
    	        printf("break;}");
    	}
    	printf("default: {}");
    	printf("}");
    	printf("};");
    }
    if(!get_num_reduced_profiles(PROFILE_YZ,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_yz __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
	    printf("const auto reduce_sum_real_yz __attribute__((unused)) = [&](const AcReal& val, const Profile& output)"
			   "{"
			   "if(blockDim.x %% warp_size != 0) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][start.x - NGHOST + profileReduceOutputVertexIdx.x + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z)] = val;" 
			   "else {"
			   "const AcReal reduce_res = warp_reduce_sum_real(val);"
	          	   "if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][localCompdomainVertexIdx.x/warp_size + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z)] = reduce_res;"
			   "}"
			   "};");
    }

    if(!get_num_reduced_profiles(PROFILE_ZY,kernel))
    {
    	printf("[[maybe_unused]] const auto reduce_sum_real_zy __attribute__((unused)) = [&](const AcReal&, const Profile&) {};");
    }
    else
    {
	    printf("const auto reduce_sum_real_zy __attribute__((unused)) = [&](const AcReal& val, const Profile& output)"
			   "{"
			   "if(blockDim.x %% warp_size != 0) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][start.x - NGHOST + profileReduceOutputVertexIdx.x + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z)] = val;" 
			   "else {"
			   "const AcReal reduce_res = warp_reduce_sum_real(val);"
	          	   "if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_real[PROF_SCRATCHPAD_INDEX(output)][localCompdomainVertexIdx.x/warp_size + VAL(AC_reduction_tile_dimensions).x*(vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z)] = reduce_res;"
			   "}"
			   "};");
    }
  }
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
    bool written_profile = false;
    for(int profile = 0; profile < NUM_PROFILES; ++profile)
	    written_profile |= write_called_profile[curr_kernel][profile];
    if(written_profile)
    {
    	printf("const auto write_profile_x __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.y == NGHOST && vertexIdx.z == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.x] = value;");
    	printf("}");
    	printf("};");
	
    	printf("const auto write_profile_y __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.x == NGHOST && vertexIdx.z == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.y] = value;");
    	printf("}");
    	printf("};");

    	printf("const auto write_profile_z __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.x == NGHOST && vertexIdx.y == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.z] = value;");
    	printf("}");
    	printf("};");

    	printf("const auto write_profile_xy __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.z == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.y] = value;");
    	printf("}");
    	printf("};");

    	printf("const auto write_profile_xz __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.y == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.z] = value;");
    	printf("}");
    	printf("};");


    	printf("const auto write_profile_yx __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.z == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.x] = value;");
    	printf("}");
    	printf("};");

    	printf("const auto write_profile_yz __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.x == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.y + VAL(AC_mlocal).y*vertexIdx.z] = value;");
    	printf("}");
    	printf("};");
	
    	printf("const auto write_profile_zx __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.y == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.x] = value;");
    	printf("}");
    	printf("};");

    	printf("const auto write_profile_zy __attribute__((unused)) = [&](const Profile& handle, const AcReal& value) {");
	printf("if(vertexIdx.x == NGHOST){");
    	printf("  vba.profiles.out[handle][vertexIdx.z + VAL(AC_mlocal).z*vertexIdx.y] = value;");
    	printf("}");
    	printf("};");

    }
    bool written_something = false;
    for(int field = 0; field < NUM_ALL_FIELDS; ++field)
	    written_something |= write_called[curr_kernel][field];
    if(!written_something) 
    {
    	printf("const auto write_base __attribute__((unused)) = [&](const Field&, const AcReal&) {};");
	return;
    }
    if(has_buffered_writes(kernel_names[curr_kernel]))
    {
  	    for (int original_field = 0; original_field < NUM_ALL_FIELDS; ++original_field)
  	    {
      	      if (stencils_accessed[curr_kernel][original_field][0]) continue;
  	      if(!write_called[curr_kernel][original_field]) continue;
	      //TP: field is buffered written but not read, for now simply set value to 0
  	      const int field = get_original_index(field_remappings,original_field);
	      printf("AcReal f%s_svalue_stencil = 0.0;",field_names[field]);
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
  gen_kernel_block_loops(curr_kernel);
  gen_kernel_common_prefix();
  gen_profile_funcs(curr_kernel);
}

#include "warp_reduce.h"


void
print_butterfly_iteration(FILE* stream, const int iteration, const char* op_instruction, const char* base_shift_type, const char* tid_shift_type, const char* mask_type, const char* smallest_active_type)
{
	//TP: optimization: don't need to calculcate first set bit since only a single bit can be active can do it with simple AND
	if(iteration == 0)
	{
	       	fprintf(stream,"unsigned long target_tid = lane_id ^ 1;"
	       			"auto shuffle_tmp = %s;"
	       			"if((AC_INTERNAL_active_threads >> target_tid) & 1) %s;"
				,shuffle_instruction
				,op_instruction
			);
		return;
	}
	const int base_shift_num                = (1 << iteration);
	const int tid_shift_num                 = (1 << (iteration+1))-1;

	//TP: for some reason produces 0 for shift of 32 without the conditional even though long long should not overflow??
        const unsigned long long mask_num       =
                base_shift_num == 32 ? 4294967295 :
                (1 << base_shift_num)-1;

	fprintf(stream,"%s base_shift = ((lane_id & %d) ? 0 : %d);",base_shift_type,base_shift_num,base_shift_num);
	fprintf(stream,"%s tid_shift  = (lane_id & (~%d));",tid_shift_type,tid_shift_num);
	fprintf(stream,"%s mask       = AC_INTERNAL_active_threads & (%lldULL << (tid_shift + base_shift));",mask_type,mask_num);
	fprintf(stream,"%s smallest_active = %s(mask);",smallest_active_type,ffs_string);
	fprintf(stream,"target_tid = smallest_active-1;");
	fprintf(stream,"shuffle_tmp = %s;",shuffle_instruction);
	fprintf(stream,"if(smallest_active) %s;",op_instruction);
}
void
print_butterfly_warp_reduce(FILE* stream, const int warp_size, const char* op_instruction)
{
	print_butterfly_iteration(stream,0,op_instruction,"int","auto","auto","auto");
	print_butterfly_iteration(stream,1,op_instruction,"int","auto","auto","auto");
	for(int iteration= 2; (1 << iteration) < warp_size; ++iteration)
		print_butterfly_iteration(stream,iteration,op_instruction,"","","","");
}
void
print_reduce_ops(const ReduceOp op, const char* define_name)
{
        const char* op_instruction = get_op_instruction(op);
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

#if AC_USE_HIP
        printf("if constexpr (warp_size == 32) {");
        printf("if (AC_INTERNAL_all_threads_active) {");
        print_warp_reduction(stdout,32,op_instruction,false);
        printf("}");
        printf("else if (AC_INTERNAL_active_threads_are_contiguos ) {");
        print_warp_reduction(stdout,32,op_instruction,true);
        printf("}");
        printf("else {");
        print_butterfly_warp_reduce(stdout, 32,op_instruction);
        printf("}");
        printf("}");
//TP: if we use CUDA we get compiler warnings about too large shifts since active threads is unsigned long instead of unsigned long long
        printf("else if constexpr (warp_size == 64) {");
        printf("if (AC_INTERNAL_all_threads_active) {");
        print_warp_reduction(stdout,64,op_instruction,false);
        printf("}");
        printf("else if (AC_INTERNAL_active_threads_are_contiguos) {");
        print_warp_reduction(stdout,64,op_instruction,true);
        printf("}");
        printf("else {");
        print_butterfly_warp_reduce(stdout,64,op_instruction);
        printf("}");
        printf("}");
#else
        printf("if (AC_INTERNAL_all_threads_active) {");
        print_warp_reduction(stdout,32,op_instruction,false);
        printf("}");
        printf("else if (AC_INTERNAL_active_threads_are_contiguos ) {");
        print_warp_reduction(stdout,32,op_instruction,true);
        printf("}");
        printf("else {");
        print_butterfly_warp_reduce(stdout, 32,op_instruction);
        printf("}");
#endif
}

void
print_output_reduce_res(const char* define_name, const ReduceOp op, const int curr_kernel)
{
	if(kernel_has_block_loops(curr_kernel))
	{
		printf("if(lane_id == warp_leader_id) {");
		printf("if(current_block_idx_x == 0 && current_block_idx_y == 0 && current_block_idx_z == 0) d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index] = val;"
			,define_name);
		printf("else ");
		if(op == REDUCE_SUM)
			printf("d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index] += val;"
			,define_name);
		if(op == REDUCE_MIN)
			printf("d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index] = min(val,d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index]);"
			,define_name,define_name);
		if(op == REDUCE_MAX)
			printf("d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index] = max(val,d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index]);"
			,define_name,define_name);
		printf("}");

		return;
	}
	printf(
		  "if(lane_id == warp_leader_id) {d_symbol_reduce_scratchpads_%s[(int)output][warp_out_index] = val;}"
	      ,define_name);
}
void
print_warp_reduce_func(const char* datatype, const char* define_name, const ReduceOp op)
{
	printf("const auto warp_reduce_%s_%s __attribute__((unused)) = [&](%s val) {",reduce_op_to_name(op), define_name, datatype);
	print_reduce_ops(op, define_name);
	printf("return val;");
	printf("};");
}
void
print_reduce_func(const char* datatype, const char* define_name, const char* enum_name, const int curr_kernel, const char** names, const int* is_reduced, const size_t n_elems, const ReduceOp op)
{
    	if(!get_num_reduced_vars(n_elems,is_reduced,op))
	{
        	printf("[[maybe_unused]] const auto reduce_%s_%s __attribute__((unused)) = [&](%s, const %sOutputParam&) {};",reduce_op_to_name(op), define_name, datatype,enum_name);
		return;
	}
        printf("[[maybe_unused]] const auto reduce_%s_%s __attribute__((unused)) = [&](%s val, const %sOutputParam& output) {",reduce_op_to_name(op),define_name,datatype,enum_name);
	if(kernel_has_block_loops(curr_kernel) || BUFFERED_REDUCTIONS)
	{
		printf("switch(output) {");
		for(size_t i = 0; i < n_elems; ++i)
		{
			if((ReduceOp)is_reduced[i] != op) continue;
			printf("case %s: {",names[i]);
			if(op == REDUCE_SUM)
				printf("%s_reduce_output += val;",names[i]);
			if(op == REDUCE_MIN)
				printf("%s_reduce_output = min(%s_reduce_output,val);",names[i],names[i]);
			if(op == REDUCE_MAX)
				printf("%s_reduce_output = max(%s_reduce_output,val);",names[i],names[i]);
			printf("break;");
			printf("}");
		}
		printf("default: {}");
		printf("}");
	}
	else
	{
		print_reduce_ops(op, define_name);
		print_output_reduce_res(define_name,op,curr_kernel);
	}
	printf("};");
}
void
printf_reduce_funcs(const char* datatype, const char* define_name, const char* enum_name, const int curr_kernel, const char** names, const int* is_reduced, const size_t n_elems)
{
	print_reduce_func(datatype,define_name,enum_name,curr_kernel,names,is_reduced,n_elems,REDUCE_SUM);
	print_reduce_func(datatype,define_name,enum_name,curr_kernel,names,is_reduced,n_elems,REDUCE_MIN);
	print_reduce_func(datatype,define_name,enum_name,curr_kernel,names,is_reduced,n_elems,REDUCE_MAX);
}

void
gen_kernel_reduce_funcs(const int curr_kernel)
{
  if(kernel_calls_reduce[curr_kernel] )
  {
    if(!kernel_has_block_loops(curr_kernel))
    {
	if(!BUFFERED_REDUCTIONS)
	{
#if AC_USE_HIP
        	printf("[[maybe_unused]] const size_t warp_id = rocprim__warpId();");
#else
		printf("[[maybe_unused]] const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size;");
#endif
    		printf("[[maybe_unused]] const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) %% warp_size;");

    		printf("[[maybe_unused]] const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;");

    		printf("[[maybe_unused]] const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;");

    		printf("[[maybe_unused]] const int warp_out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;");
	}
    	printf("%s warp_leader_id  = %s(AC_INTERNAL_active_threads)-1;",false ? "" : "[[maybe_unused]] const size_t" ,ffs_string);
    }
    printf_reduce_funcs("AcReal","real","AcReal",curr_kernel,real_output_names,reduced_reals[curr_kernel],NUM_REAL_OUTPUTS);
    printf_reduce_funcs("int","int","AcInt",curr_kernel,int_output_names,reduced_ints[curr_kernel],NUM_INT_OUTPUTS);
#if AC_DOUBLE_PRECISION
    printf_reduce_funcs("float","float","AcFloat",curr_kernel,float_output_names,reduced_floats[curr_kernel],NUM_FLOAT_OUTPUTS);
#endif
  }
}
static void
gen_return_if_oob(const int curr_kernel)
{

       if(is_boundcond_kernel[curr_kernel])
       {
	       printf("const bool inside_computational_domain = "
			       "vertexIdx.x >= VAL(AC_nmin).x && vertexIdx.x < VAL(AC_nlocal_max).x\n"
			       "&& vertexIdx.y >= VAL(AC_nmin).y && vertexIdx.y < VAL(AC_nlocal_max).y\n"
			       "&& vertexIdx.z >= VAL(AC_nmin).z && vertexIdx.z < VAL(AC_nlocal_max).z;\n"
		     );
	       printf("if(inside_computational_domain) return;");
       }
       printf("const bool out_of_bounds = vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z;\n");
       if(kernel_calls_reduce[curr_kernel] )
       {
		const char* type = kernel_has_block_loops(curr_kernel) || BUFFERED_REDUCTIONS ? "" : "[[maybe_unused]] const auto";
#if AC_USE_HIP
	       printf("%s AC_INTERNAL_active_threads = __ballot(!out_of_bounds);",type);
#else
	       printf("%s AC_INTERNAL_active_threads = __ballot_sync(0xffffffff,!out_of_bounds);",type);
#endif
    	        //TP: if active threads are contigous i.e. not any inactive threads between active threads
    	        //then can perform the reductions without calculating tids
    		printf("%s AC_INTERNAL_active_threads_are_contiguos = !(AC_INTERNAL_active_threads & (AC_INTERNAL_active_threads+1));",type);
    		//TP: if all threads are active can skip checks checking if target tid is active in reductions
    		printf("%s AC_INTERNAL_all_threads_active = AC_INTERNAL_active_threads+1 == 0;",type);
       }
       printf("if(out_of_bounds) %s;",kernel_has_block_loops(curr_kernel) ? "continue" : "return");
       printf("{\n");
  // Enable excluding some internal domain
  printf("\n#if defined(AC_ENABLE_EXCLUDE_INNER)\n");
  printf("if (DCONST(AC_exclude_inner)) {");
  // printf("const int nix_min = (DCONST(AC_nx_min) + (STENCIL_WIDTH-1)/2);");
  // printf("const int nix_max = (DCONST(AC_nx_max) - (STENCIL_WIDTH-1)/2);");
  // printf("const int niy_min = (DCONST(AC_ny_min) + (STENCIL_HEIGHT-1)/2);");
  // printf("const int niy_max = (DCONST(AC_ny_max) - (STENCIL_HEIGHT-1)/2);");
  // printf("const int niz_min = (DCONST(AC_nz_min) + (STENCIL_DEPTH-1)/2);");
  // printf("const int niz_max = (DCONST(AC_nz_max) - (STENCIL_DEPTH-1)/2);");
  printf("const int nix_min = (start.x + (STENCIL_WIDTH-1)/2);");
  printf("const int nix_max = (end.x - (STENCIL_WIDTH-1)/2);");
  printf("const int niy_min = (start.y + (STENCIL_HEIGHT-1)/2);");
  printf("const int niy_max = (end.y - (STENCIL_HEIGHT-1)/2);");
  printf("const int niz_min = (start.z + (STENCIL_DEPTH-1)/2);");
  printf("const int niz_max = (end.z - (STENCIL_DEPTH-1)/2);");
  printf("if (vertexIdx.x >= nix_min && vertexIdx.x < nix_max && ");
  printf("    vertexIdx.y >= niy_min && vertexIdx.y < niy_max && ");
  printf("    vertexIdx.z >= niz_min && vertexIdx.z < niz_max) { return; }");
  printf("}");
  printf("\n#endif\n");
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
  gen_kernel_block_loops(0);
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
static bool
stencil_accesses_z_ghost_zone(const size_t stencil)
{
  // Check which stencils are invalid for profiles
  // (computed in a new array to avoid side effects).
    bool res = false;
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
	  const bool dmid = (depth == (STENCIL_DEPTH-1)/2);
          res |= !dmid && stencils[stencil][depth][height][width];
        }
      }
    }
    return res;
}

static bool
stencil_accesses_y_ghost_zone(const size_t stencil)
{
  // Check which stencils are invalid for profiles
  // (computed in a new array to avoid side effects).
    bool res = false;
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
	  const bool hmid = (height == (STENCIL_HEIGHT-1)/2);
          res |= !hmid && stencils[stencil][depth][height][width];
        }
      }
    }
    return res;
}

static bool
stencil_accesses_x_ghost_zone(const size_t stencil)
{
  // Check which stencils are invalid for profiles
  // (computed in a new array to avoid side effects).
    bool res = false;
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
	  const bool wmid = (width == (STENCIL_WIDTH-1)/2);
          res |= !wmid && stencils[stencil][depth][height][width];
        }
      }
    }
    return res;
}

static bool
stencil_valid_for_profile(const int stencil, const AcProfileType type)
{
	if(type == PROFILE_X) 
		 return !(stencil_accesses_y_ghost_zone(stencil) || stencil_accesses_z_ghost_zone(stencil));
	if(type == PROFILE_Y) 
		 return !(stencil_accesses_x_ghost_zone(stencil) || stencil_accesses_z_ghost_zone(stencil));
	if(type == PROFILE_Z) 
		 return !(stencil_accesses_y_ghost_zone(stencil) || stencil_accesses_x_ghost_zone(stencil));
	//TP: the formulas below are valid but for now derivatives on 2d profiles are not supported
	//if(type == PROFILE_XY || type == PROFILE_YX) 
	//	 return !(stencil_accesses_z_ghost_zone(stencil));
	//if(type == PROFILE_XZ || type == PROFILE_ZX) 
	//	 return !(stencil_accesses_y_ghost_zone(stencil));
	//if(type == PROFILE_YZ || type == PROFILE_ZY) 
	//	 return !(stencil_accesses_x_ghost_zone(stencil));
	return false;
}

static void
gen_stencil_functions(const int curr_kernel)
{
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    //TP: don't gen stencil function at all if no fields use it. Done to declutter the resulting code and to speedup compilation
    bool gen_stencil = false;
    for (int field = 0; field < NUM_ALL_FIELDS; ++field) gen_stencil |= stencils_accessed[curr_kernel][field][stencil];
    if(!gen_stencil)
    {
	    printf("const auto %s __attribute__((unused)) = [&](const Field& field) { (void) field; return (AcReal)NAN;};",stencil_names[stencil]);
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

    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
  {
  	for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
  	  //TP: don't gen stencil function at all if no fields use it. Done to declutter the resulting code and to speedup compilation
  	  bool gen_stencil = false;
  	  for (int profile = 0; profile < NUM_PROFILES; ++profile) gen_stencil |= stencils_accessed[curr_kernel][profile + NUM_ALL_FIELDS][stencil];
  	  if(!gen_stencil)
  	  {
  	          printf("const auto %s_profile __attribute__((unused)) = [&](const Profile& profile) { (void) profile; return (AcReal)NAN;};",stencil_names[stencil]);
  	          continue;
  	  }
  	  printf("const auto %s_profile __attribute__((unused)) = [&](const Profile& profile){",
  	         stencil_names[stencil]);
  	  printf("switch (profile) {");
  	  for (int profile = 0; profile < NUM_PROFILES; ++profile) {
  	    if (stencils_accessed[curr_kernel][NUM_ALL_FIELDS + profile][stencil] && stencil_valid_for_profile(stencil,prof_types[profile]))
	    {
  	      printf("case %s: return p%d_s%d;", profile_names[profile], profile,
  	             stencil);
	    }
  	  }

  	  printf("default: return (AcReal)NAN;");
  	  printf("}");
  	  printf("};");
  	}
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
                if (!stencils_accessed[curr_kernel][NUM_ALL_FIELDS + profile]
                                      [stencil])
                  continue;

                // Skip if the stencil is invalid for profiles
                if (!stencil_valid_for_profile(stencil,prof_types[profile]))
                  continue;

		const char* vertex_idx = prof_types[profile] == PROFILE_X ? "vertexIdx.x" :
					 prof_types[profile] == PROFILE_Y ? "vertexIdx.y" :
					 prof_types[profile] == PROFILE_Z ? "vertexIdx.z" :
					 NULL;
		const int   offset     = prof_types[profile] == PROFILE_X ? -STENCIL_ORDER/2 + width :
					 prof_types[profile] == PROFILE_Y ? -STENCIL_ORDER/2 + height:
					 prof_types[profile] == PROFILE_Z ? -STENCIL_ORDER/2 + depth :
					 0;

                if (stencils[stencil][depth][height][width]) {
                  if (!stencil_initialized[NUM_ALL_FIELDS + profile][stencil]) {
                    printf("auto p%d_s%d = ", profile, stencil);
		    printf_stencil_point(stencil,depth,height,width);
                    printf("%s(", stencil_unary_ops[stencil]);
                    printf("__ldg(&");
                    printf("vba.profiles.in[%d]"
                           "[%s+(%d)])",
                           profile, vertex_idx, offset);
                    printf(")");
                    printf(";");

                    stencil_initialized[NUM_ALL_FIELDS + profile][stencil] = 1;
                  }
                  else {
                    printf("p%d_s%d = ", profile, stencil);
                    printf("%s(p%d_s%d, ", stencil_binary_ops[stencil], profile,
                           stencil);
		    printf_stencil_point(stencil,depth,height,width);
                    printf("%s(", stencil_unary_ops[stencil]);
                    printf("__ldg(&");
                    printf("vba.profiles.in[%d]"
                           "[%s+(%d)])",
                           profile, vertex_idx,offset);
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

