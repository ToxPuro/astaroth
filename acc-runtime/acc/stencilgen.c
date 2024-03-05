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
#include "profilegen.h"

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
  if (!NUM_FIELDS)
    raise_error("Must declare at least one Field in the DSL code!");

  if (!NUM_STENCILS)
    raise_error("Must declare at least one Stencil in the DSL code!");

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
          printf("%s,", stencils[stencil][depth][height][width]
                            ? stencils[stencil][depth][height][width]
                            : "0");
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

void
gen_kernel_prefix(void)
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
  printf("const int3 globalGridN = (int3){d_mesh_info.int_params[AC_nxgrid], d_mesh_info.int_params[AC_nygrid], d_mesh_info.int_params[AC_nzgrid]};");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");

  printf(
      "const int3 localCompdomainVertexIdx = (int3){"
      "threadIdx.x + blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
      "threadIdx.y + blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
      "threadIdx.z + blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2,"
      "};");
  printf("const int local_compdomain_idx = "
         "LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning
  printf("(void)local_compdomain_idx;");     // Silence unused warning

// Write vba.out
#if 1
  // Original
  printf("const auto write __attribute__((unused))  = [&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");
  printf("const auto vecwrite __attribute__((unused)) = [&](const int3 field3, const AcReal3 value)"
         "{ vba.out[field3.x][idx] = value.x; vba.out[field3.y][idx] = value.y; vba.out[field3.z][idx] = value.z;};");

  //  Non-temporal store intrinsic could reduce L2 pressure on AMD but no effect
  //  in practice (no effect on the first pass, a slight slowdown in the second
  //  pass 4.6 ms vs 4.3 ms)
  // printf("const auto write=[&](const Field field, const AcReal value)"
  //  "{ __builtin_nontemporal_store(value, &vba.out[field][idx]); };");
#else
  // Buffered, no effect on performance
  // !Remember to emit write insructions in ac.y if this is enabled!
  printf("AcReal out_buffer[NUM_FIELDS];");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("out_buffer[%d] = (AcReal)NAN;", field);

  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ out_buffer[field] = value; };");
/*
for (int field = 0; field < NUM_FIELDS; ++field)
printf("vba.out[%d][idx] = out_buffer[%d];", field, field);
*/
#endif
}

static void
gen_return_if_oob(void)
{
  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");
}

static void
prefetch_output_elements_and_gen_prev_function(const bool gen_mem_accesses)
{
  // Read vba.out
#if 0
  // Original (compute when needed)
  // SINGLEPASS_INTEGRATION=ON, 4.97 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 6.09 ms (full step, 128^3)
  printf("const auto previous __attribute__((unused)) =[&](const Field field)"
         "{ return vba.out[field][idx]; };");
#else
  // Prefetch output fields
  // SINGLEPASS_INTEGRATION=ON, 4.18 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 4.77 ms (full step, 128^3)
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("const auto f%d_prev = vba.out[%d][idx];", field, field);

  printf("const auto previous __attribute__((unused)) = [&](const Field field)"
         "{ switch (field) {");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("case %d: { return f%d_prev; }", field, field);

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
  printf("const auto vecprevious __attribute__((unused)) = [&](const int3 field3){return (AcReal3) {previous((Field)field3.x), previous((Field)field3.y), previous((Field)field3.z)};};");

  //Note to Johannes, on HIP __ldg is no-op so we can safely use it both for CUDA and HIP
  for (int profile = 0; profile < NUM_PROFILES; ++profile){
    for(int read_index = 0; read_index < profile_read_set_sizes[profile]; ++read_index) 
    {
      //TP: if we are generating mem accesses for some reason reading from vba does not work
      //as a workaround simply set the values to zero
      char* vertex_idx_coordinate = profile_dims[profile] == 1 ? "vertexIdx.x" : profile_dims[profile] == 2 ? "vertexIdx.y" : "vertexIdx.z";
      if(gen_mem_accesses)
          printf("const auto p_%d_%d= 0;", profile, read_index);
      else
          printf("const auto p_%d_%d= __ldg(&vba.profiles[%d][%s+(%d)]);", profile, read_index, profile, vertex_idx_coordinate,profile_read_set[profile][read_index]);
    }
  }

  for (int field = 0; field < NUM_FIELDS; ++field)
    if(gen_mem_accesses)
      printf("const auto f%d_val= (AcReal)0.0;", field);
    else
      printf("const auto f%d_val= __ldg(&vba.in[%d][idx]);", field, field);

  printf("const auto value __attribute__((unused)) = [&](const auto field)"
         "{ switch (field) {");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("case %d: { return f%d_val; }", field, field);

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
  printf("const auto vecvalue __attribute__((unused)) = [&](const int3 field3){return (AcReal3) {value((Field)field3.x), value((Field)field3.y), value((Field)field3.z)};};");
#endif
}

void
gen_stencil_accesses(void)
{
  gen_kernel_prefix();
  gen_return_if_oob();
  prefetch_output_elements_and_gen_prev_function(true);

  printf("AcReal /*__restrict__*/ "
         "processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s=[&](const auto field)"
           "{stencils_accessed[field][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
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
          for (int field = 0; field < NUM_FIELDS; ++field)
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
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
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
#if !AC_USE_HIP
              printf("__ldg(&");
#endif
              printf("vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
              printf(")");
#endif
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
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
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

  for (int field = 0; field < NUM_FIELDS; ++field)
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil)
      if (stencil_initialized[field][stencil] !=
          stencils_accessed[curr_kernel][field][stencil])
        raise_error("stencil_initialized != stencil_accessed, this affects "
                    "gen_stencil_functions (stencil_accessed should be "
                    "replaced with stencil_initialized)");
}
*/

static void
gen_stencil_functions(const int curr_kernel)
{
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencils_accessed[curr_kernel][field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}

/** Supports 2.5D and 2D smem blocking (see `rolling_cache` switch) */
static void
prefetch_stencil_elems_to_smem_and_compute_stencil_ops(const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
         "(STENCIL_DEPTH-1)/2};");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

      const bool rolling_cache = true;
      if (rolling_cache) {
        // 2.5D blocking with smem

        // Fetch from gmem
        printf("if (%d == 0 || threadIdx.z == blockDim.z - 1) {", depth);
        printf("for (int curr = sid; curr < sx * sy;"
               "curr += blockDim.x * blockDim.y) {");
        printf("const int i = curr %% sx;");
        printf("const int j = curr / sx;");
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + (%d) >= end.z + (STENCIL_DEPTH-1)/2){ break; }",
               depth);
        printf("smem[i + j * sx + ((threadIdx.z+%d)%%blockDim.z) * sx * sy] = ",
               depth);
        printf("vba.in[%d]"
               "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
               field, depth);
        printf("}");
        printf("}");
        printf("__syncthreads();");
      }
      else {
        // 2D blocking with smem
        printf("for (int curr = sid; curr < sx * sy;"
               "curr += blockDim.x * blockDim.y) {");
        printf("const int i = curr %% sx;");
        printf("const int j = curr / sx;");
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + (%d) >= end.z + (STENCIL_DEPTH-1)/2){ break; }",
               depth);
        printf("smem[i + j * sx + ((threadIdx.z+%d)%%blockDim.z) * sx * sy] = ",
               depth);
        printf("vba.in[%d]"
               "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
               field, depth);
        printf("}");
        printf("__syncthreads();");
      }

      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z+%d)%%blockDim.z) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z+%d)%%blockDim.z) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, depth);
                printf(");");
              }
            }
          }
        }
      }
      printf("__syncthreads();");
    }
  }
}

/** Supports 3D smem blocking (see `rolling_cache` switch) */
static void
prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {

    printf("for (int curr = sid; curr < sx * sy * sz; curr += tpb) {");
    printf("const int i = curr %% sx;");
    printf("const int j = (curr %% (sx * sy)) / sx;");
    printf("const int k = curr / (sx * sy);");
    printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
    printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
    printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
    printf("smem[i + j * sx + k * sx * sy] = ");
#if !AC_USE_HIP
    printf("__ldg(&");
#endif
    printf("vba.in[%d]", field);
    printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
#if !AC_USE_HIP
    printf(")");
#endif
    printf(";");
    printf("}");
    printf("__syncthreads();");

    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, depth);
                printf(");");
              }
            }
          }
        }
      }
    }
    printf("__syncthreads();");
  }
}

/** Supports 2.5D and 2D smem blocking (see `rolling_cache` switch) */
static void
prefetch_stencil_elems_to_smem_4d_and_compute_stencil_ops(const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = NUM_FIELDS;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
         "(STENCIL_DEPTH-1)/2};");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

    const bool rolling_cache = true;
    if (rolling_cache) {
      // 2.5D blocking with smem

      // Fetch from gmem
      printf("if (%d == 0 || threadIdx.z == blockDim.z - 1) {", depth);
      printf("for (int curr = sid; curr < sx * sy * sz;"
             "curr += blockDim.x * blockDim.y) {");
      printf("const int i = curr %% sx;");
      printf("const int j = (curr %% sy) / sx;");
      printf("const int field = curr / (sx * sy);");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
      printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
      printf("if (field >= NUM_FIELDS){ break; }");
      printf("smem[i + j * sx + (field) * sx * sy] = ");
      printf("vba.in[field]"
             "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
             depth);
      printf("}");
      printf("}");
      printf("__syncthreads();");
    }
    else {
      // 2D blocking with smem
      printf("for (int curr = sid; curr < sx * sy * sz;"
             "curr += blockDim.x * blockDim.y * blockDim.z) {");
      printf("const int i = curr %% sx;");
      printf("const int j = (curr %% sy) / sx;");
      printf("const int field = curr / (sx * sy);");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
      printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
      printf("if (field >= NUM_FIELDS){ break; }");
      printf("smem[i + j * sx + field * sx * sy] = ");
      printf("vba.in[field]"
             "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
             depth);
      printf("}");
      printf("__syncthreads();");
    }

    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
          for (int field = 0; field < NUM_FIELDS; ++field) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(%d) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, field);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(%d) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, field);
                printf(");");
              }
            }
          }
        }
      }
    }
    printf("__syncthreads();");
  }
}
/** Ping-pong 2D txw*/
#if IMPLEMENTATION == EXPLICIT_PINGPONG_txw
 static void
 refetch_stencil_elems_to_smem_pingpong_txw_and_compute_stencil_ops(
    const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sw = NUM_FIELDS;");
  printf("const int sb = 2;");
  printf("const int sid = threadIdx.x;");
  printf(
      "const int3 baseIdx = (int3){"
      "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
      "threadIdx.y + blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
      "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
      "(STENCIL_DEPTH-1)/2};");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  printf("for (int curr=sid; curr < sx*sw; curr += blockDim.x) {");
  printf("const int i = curr %% sx;");
  printf("const int w = curr / sx;");
  printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
  printf("if (w >= NUM_FIELDS) {break;}");
  printf("smem[i + w * sx + 0 * sx*sw] = ");
  printf("vba.in[w]"
         "[IDX(baseIdx.x + i, baseIdx.y + 0, baseIdx.z + 0)];");
  printf("}");
  printf("__syncthreads();");
  for (int fiber = 0; fiber < STENCIL_HEIGHT * STENCIL_DEPTH; ++fiber) {
    const int height = fiber % STENCIL_HEIGHT;
    const int depth  = fiber / STENCIL_HEIGHT;
    if (fiber < STENCIL_HEIGHT * STENCIL_DEPTH - 1) {
      // NOTE THIS IS WRONG, SHOULD USE next_height = (fiber+1) %... and
      // next_depth!
      printf("for (int curr=sid; curr < sx*sw; curr += blockDim.x) {");
      printf("const int i = curr %% sx;");
      printf("const int w = curr / sx;");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
      printf("if (w >= NUM_FIELDS) {break;}");
      printf("smem[i + w * sx + (%d) * sx*sw] = ", (fiber + 1) % 2);
      printf("vba.in[w]"
             "[IDX(baseIdx.x + i, baseIdx.y + (%d), baseIdx.z + (%d))];",
             height, depth);
      printf("}");
    }
    for (int width = 0; width < STENCIL_WIDTH; ++width) {
      for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          // Skip if the stencil is not used
          if (!stencils_accessed[curr_kernel][field][stencil])
            continue;
          if (stencils[stencil][depth][height][width]) {
            if (!stencil_initialized[field][stencil]) {
              printf("auto f%d_s%d = ", field, stencil);
              printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                     width);
              printf("%s(smem[(threadIdx.x + %d) + "
                     "(%d) * sx + "
                     "(%d) * sx * sw]);",
                     stencil_unary_ops[stencil], width, field, fiber % 2);
              stencil_initialized[field][stencil] = 1;
            }
            else {
              printf("f%d_s%d = ", field, stencil);
              printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                     stencil);
              printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                     width);
              printf("%s(smem[(threadIdx.x + %d) + "
                     "(%d) * sx + "
                     "(%d) * sx * sw])",
                     stencil_unary_ops[stencil], width, field, fiber % 2);
              printf(");");
            }
          }
        }
      }
    }
    printf("__syncthreads();");
  }
}
#endif
/** Ping-pong 2D txy*/
#if IMPLEMENTATION == EXPLICIT_PINGPONG_txy
static void
prefetch_stencil_elems_to_smem_pingpong_txy_and_compute_stencil_ops(
    const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sb = 2;");
  printf("const int sid = threadIdx.x + threadIdx.y * blockDim.x;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
         "(STENCIL_DEPTH-1)/2};");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  printf("for (int curr=sid; curr < sx*sy; curr += blockDim.x * blockDim.y) {");
  printf("const int i = curr %% sx;");
  printf("const int j = curr / sx;");
  printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
  printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
  printf("smem[i + j * sx + 0 * sx*sy] = ");
  printf("vba.in[0]"
         "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + 0)];");
  printf("}");
  printf("__syncthreads();");
  for (int slab = 0; slab < STENCIL_DEPTH * NUM_FIELDS; ++slab) {
    const int depth = slab % STENCIL_DEPTH;
    const int field = slab / STENCIL_DEPTH;
    const int next_slab  = slab + 1;
    const int next_depth = next_slab % STENCIL_DEPTH;
    const int next_field = next_slab / STENCIL_DEPTH;
    if (next_slab < STENCIL_DEPTH * NUM_FIELDS) {
      printf("for (int curr=sid; curr < sx*sy; "
             "curr += blockDim.x * blockDim.y) {");
      printf("const int i = curr %% sx;");
      printf("const int j = curr / sx;");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
      printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
      printf("smem[i + j * sx + (%d) * sx*sy] = ", next_slab % 2);
      printf("vba.in[(%d)]"
             "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
             next_field, next_depth);
      printf("}");
    }
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
          // Skip if the stencil is not used
          if (!stencils_accessed[curr_kernel][field][stencil])
            continue;
          if (stencils[stencil][depth][height][width]) {
            if (!stencil_initialized[field][stencil]) {
              printf("auto f%d_s%d = ", field, stencil);
              printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                     width);
              printf("%s(smem[(threadIdx.x + %d) + "
                     "(threadIdx.y + %d) * sx + "
                     "(%d) * sx * sy]);",
                     stencil_unary_ops[stencil], width, height, slab % 2);
              stencil_initialized[field][stencil] = 1;
            }
            else {
              printf("f%d_s%d = ", field, stencil);
              printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                     stencil);
              printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                     width);
              printf("%s(smem[(threadIdx.x + %d) + "
                     "(threadIdx.y + %d) * sx + "
                     "(%d) * sx * sy])",
                     stencil_unary_ops[stencil], width, height, slab % 2);
              printf(");");
            }
          }
        }
      }
    }
    if (next_slab < STENCIL_DEPTH * NUM_FIELDS - 1)
      printf("__syncthreads();");
  }
}
#endif
// /** Ping-pong 2D txy blocked (TODO need to rewrite)*/
// static void
// prefetch_stencil_elems_to_smem_pingpong_txyblocked_and_compute_stencil_ops(
//     const int curr_kernel)
// {
//   const size_t num_blocks = 7;
//   printf("extern __shared__ AcReal smem[];");
//   printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
//   printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
//   printf("const int sz = blockDim.y + STENCIL_HEIGHT - 1;");
//   printf("const int sb = 2;");
//   printf("const int sid = threadIdx.x + threadIdx.y * blockDim.x;");
//   printf("const int3 baseIdx = (int3){"
//          "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
//          "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
//          "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
//          "(STENCIL_DEPTH-1)/2};");
//   int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
//   printf(
//       "for (int curr=sid; curr < sx*sy*sz; curr += blockDim.x * blockDim.y)
//       {");
//   printf("const int i = curr %% sx;");
//   printf("const int j = (curr %% (sx*sy)) / sx;");
//   printf("const int k = curr / (sx*sy);");
//   printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
//   printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
//   printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2) {break;}");
//   printf("smem[i + j * sx + k * sx*sy + (%d)*sx*sy*sz] = ", 0);
//   printf("vba.in[(%d)]", 0);
//   printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)];");
//   printf("}");
//   printf("__syncthreads();");
//   for (int block = 0; block < STENCIL_DEPTH / num_blocks; ++slab) {
//     const int depth = slab % STENCIL_DEPTH;
//     const int field = slab / STENCIL_DEPTH;
//     const int next_slab  = slab + 1;
//     const int next_depth = next_slab % STENCIL_DEPTH;
//     const int next_field = next_slab / STENCIL_DEPTH;
//     if (next_slab < STENCIL_DEPTH * NUM_FIELDS) {
//       printf("for (int curr=sid; curr < sx*sy; "
//              "curr += blockDim.x * blockDim.y) {");
//       printf("const int i = curr %% sx;");
//       printf("const int j = curr / sx;");
//       printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
//       printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
//       printf("smem[i + j * sx + (%d) * sx*sy] = ", next_slab % 2);
//       printf("vba.in[(%d)]"
//              "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
//              next_field, next_depth);
//       printf("}");
//     }
//     for (int height = 0; height < STENCIL_HEIGHT; ++height) {
//       for (int width = 0; width < STENCIL_WIDTH; ++width) {
//         for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
//           // Skip if the stencil is not used
//           if (!stencils_accessed[curr_kernel][field][stencil])
//             continue;
//           if (stencils[stencil][depth][height][width]) {
//             if (!stencil_initialized[field][stencil]) {
//               printf("auto f%d_s%d = ", field, stencil);
//               printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
//                      width);
//               printf("%s(smem[(threadIdx.x + %d) + "
//                      "(threadIdx.y + %d) * sx + "
//                      "(%d) * sx * sy]);",
//                      stencil_unary_ops[stencil], width, height, slab % 2);
//               stencil_initialized[field][stencil] = 1;
//             }
//             else {
//               printf("f%d_s%d = ", field, stencil);
//               printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
//                      stencil);
//               printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
//                      width);
//               printf("%s(smem[(threadIdx.x + %d) + "
//                      "(threadIdx.y + %d) * sx + "
//                      "(%d) * sx * sy])",
//                      stencil_unary_ops[stencil], width, height, slab % 2);
//               printf(");");
//             }
//           }
//         }
//       }
//     }
//     if (next_slab < STENCIL_DEPTH * NUM_FIELDS - 1)
//       printf("__syncthreads();");
//   }
// }
// /** Ping-pong 3D txyz (TODO)*/
// static void
// prefetch_stencil_elems_to_smem_pingpong_txyz_and_compute_stencil_ops(
//     const int curr_kernel)
// {
//   printf("extern __shared__ AcReal smem[];");
//   printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
//   printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
//   printf("const int sz = blockDim.z + STENCIL_HEIGHT - 1;");
//   printf("const int sb = 2;");
//   printf("const int sid = threadIdx.x + threadIdx.y * blockDim.x +
//   threadIdx.z "
//          "* blockDim.x * blockDim.y;");
//   printf("const int3 baseIdx = (int3){"
//          "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
//          "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
//          "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
//   int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
//   printf("for (int curr=sid; curr < sx*sy*sz; curr += blockDim.x * blockDim.y
//   "
//          "* blockDim.z) {");
//   printf("const int i = curr %% sx;");
//   printf("const int j = (curr %% ) / sx;");
//   printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
//   printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
//   printf("smem[i + j * sx + 0 * sx*sy] = ");
//   printf("vba.in[0]"
//          "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + 0)];");
//   printf("}");
//   printf("__syncthreads();");
//   for (int slab = 0; slab < STENCIL_DEPTH * NUM_FIELDS; ++slab) {
//     const int depth = slab % STENCIL_DEPTH;
//     const int field = slab / STENCIL_DEPTH;
//     const int next_slab  = slab + 1;
//     const int next_depth = next_slab % STENCIL_DEPTH;
//     const int next_field = next_slab / STENCIL_DEPTH;
//     if (next_slab < STENCIL_DEPTH * NUM_FIELDS) {
//       printf("for (int curr=sid; curr < sx*sy; "
//              "curr += blockDim.x * blockDim.y) {");
//       printf("const int i = curr %% sx;");
//       printf("const int j = curr / sx;");
//       printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2) {break;}");
//       printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2) {break;}");
//       printf("smem[i + j * sx + (%d) * sx*sy] = ", next_slab % 2);
//       printf("vba.in[(%d)]"
//              "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
//              next_field, next_depth);
//       printf("}");
//     }
//     for (int height = 0; height < STENCIL_HEIGHT; ++height) {
//       for (int width = 0; width < STENCIL_WIDTH; ++width) {
//         for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
//           // Skip if the stencil is not used
//           if (!stencils_accessed[curr_kernel][field][stencil])
//             continue;
//           if (stencils[stencil][depth][height][width]) {
//             if (!stencil_initialized[field][stencil]) {
//               printf("auto f%d_s%d = ", field, stencil);
//               printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
//                      width);
//               printf("%s(smem[(threadIdx.x + %d) + "
//                      "(threadIdx.y + %d) * sx + "
//                      "(%d) * sx * sy]);",
//                      stencil_unary_ops[stencil], width, height, slab % 2);
//               stencil_initialized[field][stencil] = 1;
//             }
//             else {
//               printf("f%d_s%d = ", field, stencil);
//               printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
//                      stencil);
//               printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
//                      width);
//               printf("%s(smem[(threadIdx.x + %d) + "
//                      "(threadIdx.y + %d) * sx + "
//                      "(%d) * sx * sy])",
//                      stencil_unary_ops[stencil], width, height, slab % 2);
//               printf(");");
//             }
//           }
//         }
//       }
//     }
//     if (next_slab < STENCIL_DEPTH * NUM_FIELDS - 1)
//       printf("__syncthreads();");
//   }
// }
/** Rolling ping-pong, original, working */
#if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
static void
prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops_original(
    const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("__syncthreads();");
    // Load the main block
    printf("for (int curr = sid; curr < sx * sy * blockDim.z; curr += tpb) {");
    printf("const int i = curr %% sx;");
    printf("const int j = (curr %% (sx * sy)) / sx;");
    printf("const int k = curr / (sx * sy);");
    printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
    printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
    printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
    printf("smem[i + j * sx + k * sx * sy] = ");
    printf("__ldg(&");
    printf("vba.in[%d]", field);
    printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
    printf(")");
    printf(";");
    printf("}");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("__syncthreads();");
      if (depth + 1 < STENCIL_DEPTH) {
        // Load the rolling block
        printf("for (int curr = sid; curr < sx * sy; curr += tpb) {");
        printf("const int i = curr %% sx;");
        printf("const int j = (curr %% (sx * sy)) / sx;");
        printf("const int k = blockDim.z + %d;", depth);
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
        printf("smem[i + j * sx + (k%%sz) * sx * sy] = ");
        printf("__ldg(&");
        printf("vba.in[%d]", field);
        printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
        printf(")");
        printf(";");
        printf("}");
      }
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;
            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z + %d)%%sz) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, depth);
                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z + %d)%%sz) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, depth);
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
/** Rolling ping-pong, optimized: multiple fields */
#if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
static void
prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops_v2(
    const int curr_kernel)
{
  const int BLOCK_SIZE = EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE;
  const int NUM_BLOCKS = (NUM_FIELDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (BLOCK_SIZE * NUM_BLOCKS < NUM_FIELDS)
    raise_error(
        "Invalid NUM_BLOCKS computed in stencilgen.c (rolling pingpong)");
  if (BLOCK_SIZE > NUM_FIELDS)
    raise_error(
        "Invalid EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE. Must be <= NUM_FIELDS");
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + 1;");
  // printf("const int sw = %d;", BLOCK_SIZE);
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int block = 0; block < NUM_BLOCKS; ++block) {
    printf("__syncthreads();");
    for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
      const int field = block_offset + block * BLOCK_SIZE;
      if (field >= NUM_FIELDS)
        break;
      // Load the main block
      printf(
          "for (int curr = sid; curr < sx * sy * blockDim.z; curr += tpb) {");
      printf("const int i = curr %% sx;");
      printf("const int j = (curr %% (sx * sy)) / sx;");
      printf("const int k = curr / (sx * sy);");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ continue;}");
      printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){continue; }");
      printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ continue; }");
      printf("smem[i + j * sx + k * sx * sy + (%d) * sx * sy * sz] = ",
             field % BLOCK_SIZE);
      printf("__ldg(&");
      printf("vba.in[%d]", field);
      printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
      printf(")");
      printf(";");
      printf("}");
    }
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("__syncthreads();");
      for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
        const int field = block_offset + block * BLOCK_SIZE;
        if (field >= NUM_FIELDS)
          break;
        if (depth + 1 < STENCIL_DEPTH) {
          // Load the rolling block
          printf("for (int curr = sid; curr < sx * sy; curr += tpb) {");
          printf("const int i = curr %% sx;");
          printf("const int j = (curr %% (sx * sy)) / sx;");
          printf("const int k = blockDim.z + %d;", depth);
          printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ "
                 "continue; }");
          printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ "
                 "continue; }");
          printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ "
                 "continue; }");
          printf("smem[i + j * sx + (k%%sz) * sx * sy + "
                 "(%d) * sx * sy * sz] =",
                 field % BLOCK_SIZE);
          printf("__ldg(&");
          printf("vba.in[%d]", field);
          printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
          printf(")");
          printf(";");
          printf("}");
        }
      }
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            for (int block_offset = 0; block_offset < BLOCK_SIZE;
                 ++block_offset) {
              const int field = block_offset + block * BLOCK_SIZE;
              if (field >= NUM_FIELDS)
                break;
              // Skip if the stencil is not used
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("auto f%d_s%d = ", field, stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((threadIdx.z + %d)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz]);",
                         stencil_unary_ops[stencil], width, height, depth,
                         field % BLOCK_SIZE);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                         stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((threadIdx.z + %d)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz])",
                         stencil_unary_ops[stencil], width, height, depth,
                         field % BLOCK_SIZE);
                  printf(");");
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif
/** Rolling ping-pong, original, working, rolling base slab, test, remove,
 * WORKS
 */
#if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
static void
prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops_v3(
    const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  printf("int curr_slab = 0;");
  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("__syncthreads();");
    // Load the main block
    printf("for (int curr = sid; curr < sx * sy * blockDim.z; curr += tpb) {");
    printf("const int i = curr %% sx;");
    printf("const int j = (curr %% (sx * sy)) / sx;");
    printf("const int k = curr / (sx * sy);");
    printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
    printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
    printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
    printf("smem[i + j * sx + ((curr_slab + k)%%sz) * sx * sy] = ");
    printf("__ldg(&");
    printf("vba.in[%d]", field);
    printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
    printf(")");
    printf(";");
    printf("}");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("__syncthreads();");
      if (depth + 1 < STENCIL_DEPTH) {
        // Load the rolling block
        printf("for (int curr = sid; curr < sx * sy; curr += tpb) {");
        printf("const int i = curr %% sx;");
        printf("const int j = (curr %% (sx * sy)) / sx;");
        printf("const int k = blockDim.z + %d;", depth);
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
        printf("smem[i + j * sx + ((curr_slab+blockDim.z)%%sz) * sx * sy] = ");
        printf("__ldg(&");
        printf("vba.in[%d]", field);
        printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
        printf(")");
        printf(";");
        printf("}");
      }
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;
            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((curr_slab+threadIdx.z)%%sz) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height);
                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((curr_slab + threadIdx.z)%%sz) * sx * sy])",
                       stencil_unary_ops[stencil], width, height);
                printf(");");
              }
            }
          }
        }
      }
      printf("++curr_slab;");
    }
  }
}
#endif
/** Rolling ping-pong, optimized: multiple fields, rolling base slab, basic
 * WORKING (only on V100 and MI250X likely due to warp lockstepping. Breaks
 on A100 likely because of the removal of lockstep) */
#if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
static void
prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops_v4(
    const int curr_kernel)
{
  const int BLOCK_SIZE = EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE;
  const int NUM_BLOCKS = (NUM_FIELDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (BLOCK_SIZE * NUM_BLOCKS < NUM_FIELDS)
    raise_error(
        "Invalid NUM_BLOCKS computed in stencilgen.c (rolling pingpong)");
  if (BLOCK_SIZE > NUM_FIELDS)
    raise_error(
        "Invalid EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE. Must be <= NUM_FIELDS");
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + 1;");
  // printf("const int sw = %d;", BLOCK_SIZE);
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  printf("int curr_slab = 0;");
  for (int block = 0; block < NUM_BLOCKS; ++block) {
    printf("__syncthreads();");
    for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
      const int field = block_offset + block * BLOCK_SIZE;
      if (field >= NUM_FIELDS)
        break;
      // Load the main block
      printf(
          "for (int curr = sid; curr < sx * sy * blockDim.z; curr += tpb) {");
      printf("const int i = curr %% sx;");
      printf("const int j = (curr %% (sx * sy)) / sx;");
      printf("const int k = curr / (sx * sy);");
      printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
      printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
      printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
      printf("smem[i + j * sx + ((curr_slab+k)%%sz) * sx * sy + (%d) * sx * sy "
             "* sz] = ",
             field % BLOCK_SIZE);
      printf("__ldg(&");
      printf("vba.in[%d]", field);
      printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
      printf(")");
      printf(";");
      printf("}");
    }
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("__syncthreads();");
      for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
        const int field = block_offset + block * BLOCK_SIZE;
        if (field >= NUM_FIELDS)
          break;
        if (depth + 1 < STENCIL_DEPTH) {
          // Load the rolling block
          printf("for (int curr = sid; curr < sx * sy; curr += tpb) {");
          printf("const int i = curr %% sx;");
          printf("const int j = (curr %% (sx * sy)) / sx;");
          printf("const int k = blockDim.z + %d;", depth);
          printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
          printf(
              "if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
          printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
          printf("smem[i + j * sx + ((curr_slab+blockDim.z)%%sz) * sx * sy + "
                 "(%d) * sx "
                 "* sy * sz] = ",
                 field % BLOCK_SIZE);
          printf("__ldg(&");
          printf("vba.in[%d]", field);
          printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
          printf(")");
          printf(";");
          printf("}");
        }
      }
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            for (int block_offset = 0; block_offset < BLOCK_SIZE;
                 ++block_offset) {
              const int field = block_offset + block * BLOCK_SIZE;
              if (field >= NUM_FIELDS)
                break;
              // Skip if the stencil is not used
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("auto f%d_s%d = ", field, stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((curr_slab + threadIdx.z)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz]);",
                         stencil_unary_ops[stencil], width, height,
                         field % BLOCK_SIZE);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                         stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((curr_slab + threadIdx.z)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz])",
                         stencil_unary_ops[stencil], width, height,
                         field % BLOCK_SIZE);
                  printf(");");
                }
              }
            }
          }
        }
      }
      printf("++curr_slab;");
    }
  }
}
#endif
/** Rolling ping-pong, optimized: multiple fields, unrolled
Note: requires sufficiently large tbdims s.t. can be unrolled with 2 by 2 blocks
*/
#if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
static void
prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops(
    const int curr_kernel)
{
  const int BLOCK_SIZE = EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE;
  const int NUM_BLOCKS = (NUM_FIELDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("// BLOCK_SIZE = %d\n", BLOCK_SIZE);
  printf("// NUM_BLOCKS = %d\n", NUM_BLOCKS);
  if (BLOCK_SIZE * NUM_BLOCKS < NUM_FIELDS)
    raise_error(
        "Invalid NUM_BLOCKS computed in stencilgen.c (rolling pingpong)");
  if (BLOCK_SIZE > NUM_FIELDS)
    raise_error(
        "Invalid EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE. Must be <= NUM_FIELDS");
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + 1;");
  // printf("const int sw = %d;", BLOCK_SIZE);
  // printf("const int sid = threadIdx.x + "
  //        "threadIdx.y * blockDim.x + threadIdx.z * blockDim.x *
  //        blockDim.y;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  // printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int block = 0; block < NUM_BLOCKS; ++block) {
    printf("__syncthreads();");
    // Unrolled (note: need to do a minimum of STENCIL_WIDTH or _HEIGHT
    // iterations to be sure all stencil points are covered with if the tbdim
    // is 1
    printf("{");
    printf("const int k = threadIdx.z;");
    printf("if (baseIdx.z + k < end.z + (STENCIL_DEPTH-1)/2){ ");
    // printf("if (k < blockDim.z){ ");
    for (int by = 0; by < 2; ++by) {
      printf("{");
      printf("const int j = threadIdx.y + (%d) * blockDim.y;", by);
      printf("if (baseIdx.y + j < end.y + (STENCIL_HEIGHT-1)/2){ ");
      printf("if (j < sy){ ");
      for (int bx = 0; bx < 2; ++bx) {
        printf("{");
        printf("const int i = threadIdx.x + (%d) * blockDim.x;", bx);
        printf("if (baseIdx.x + i < end.x + (STENCIL_WIDTH-1)/2){ ");
        printf("if (i < sx){ ");
        for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
          const int field = block_offset + block * BLOCK_SIZE;
          if (field >= NUM_FIELDS)
            break;
          printf("smem[i + j * sx + k * sx * sy + (%d) * sx * sy * sz] = ",
                 field % BLOCK_SIZE);
          printf("__ldg(&");
          printf("vba.in[%d]", field);
          printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
          printf(")");
          printf(";");
        }
        printf("}");
        printf("}");
        printf("}");
      }
      printf("}");
      printf("}");
      printf("}");
    }
    // printf("}");
    printf("}");
    printf("}");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("__syncthreads();");
      if (depth + 1 < STENCIL_DEPTH) {
        printf("{");
        printf("if (threadIdx.z == 0){ ");
        printf("const int k = blockDim.z + %d;", depth);
        printf("if (baseIdx.z + k < end.z + (STENCIL_DEPTH-1)/2){ ");
        for (int by = 0; by < 2; ++by) {
          printf("{");
          printf("const int j = threadIdx.y + (%d) * blockDim.y;", by);
          printf("if (baseIdx.y + j < end.y + (STENCIL_HEIGHT-1)/2){ ");
          printf("if (j < sy){ ");
          for (int bx = 0; bx < 2; ++bx) {
            printf("{");
            printf("const int i = threadIdx.x + (%d) * blockDim.x;", bx);
            printf("if (baseIdx.x + i < end.x + (STENCIL_WIDTH-1)/2){ ");
            printf("if (i < sx){ ");
            for (int block_offset = 0; block_offset < BLOCK_SIZE;
                 ++block_offset) {
              const int field = block_offset + block * BLOCK_SIZE;
              if (field >= NUM_FIELDS)
                break;
              // Load the rolling block
              printf("smem[i + j * sx + (k%%sz) * sx * sy + "
                     "(%d) * sx * sy * sz] =",
                     field % BLOCK_SIZE);
              printf("__ldg(&");
              printf("vba.in[%d]", field);
              printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
              printf(")");
              printf(";");
            }
            printf("}");
            printf("}");
            printf("}");
          }
          printf("}");
          printf("}");
          printf("}");
        }
        printf("}");
        printf("}");
        printf("}");
      }
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            for (int block_offset = 0; block_offset < BLOCK_SIZE;
                 ++block_offset) {
              const int field = block_offset + block * BLOCK_SIZE;
              if (field >= NUM_FIELDS)
                break;
              // Skip if the stencil is not used
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("auto f%d_s%d = ", field, stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((threadIdx.z + %d)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz]);",
                         stencil_unary_ops[stencil], width, height, depth,
                         field % BLOCK_SIZE);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                         stencil);
                  printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                         width);
                  printf("%s(smem[(threadIdx.x + %d) + "
                         "(threadIdx.y + %d) * sx + "
                         "((threadIdx.z + %d)%%sz) * sx * sy + "
                         "(%d) * sx * sy * sz])",
                         stencil_unary_ops[stencil], width, height, depth,
                         field % BLOCK_SIZE);
                  printf(");");
                }
              }
            }
          }
        }
      }
    }
  }
}
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
gen_kernel_body(const int curr_kernel)
{
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING: {
    gen_kernel_prefix();
    gen_return_if_oob();
    prefetch_output_elements_and_gen_prev_function(false);

    int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};

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
    // BLOCK_SIZE=NUM_FIELDS by default (the original implementation)
    // tradeoff:
    //  A) larger BLOCK_SIZE
    //    + deeper instruction pipeline (instruction-level parallelism)
    //    - larger working set (can cause cache thrashing)
    //  B) smaller BLOCK_SIZE =
    //    + smaller working set (better cache locality)
    //    - shallower instruction pipeline (more stalling due to data
    //    dependencies)
    const int BLOCK_SIZE = NUM_FIELDS;
    const int NUM_BLOCKS = (NUM_FIELDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (BLOCK_SIZE * NUM_BLOCKS < NUM_FIELDS)
      raise_error("Invalid BLOCK_SIZE * NUM_BLOCKS, was smaller than "
                  "NUM_FIELDS in stencilgen.c\n");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int field_block = 0; field_block < NUM_BLOCKS; ++field_block) {
            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
              for (int foffset = 0; foffset < BLOCK_SIZE; ++foffset) {
                const int field = foffset + field_block * BLOCK_SIZE;
                if (field >= NUM_FIELDS)
                  break;

                // Skip if the stencil is not used
                if (!stencils_accessed[curr_kernel][field][stencil])
                  continue;

                if (stencils[stencil][depth][height][width]) {
                  if (!stencil_initialized[field][stencil]) {
                    printf("auto f%d_s%d = ", field, stencil);
                    printf("stencils[%d][%d][%d][%d] *", //
                           stencil, depth, height, width);
                    printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                    printf("__ldg(&");
#endif
                    printf("vba.in[%d]"
                           "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                           "vertexIdx.z+(%d))])",
                           field, -STENCIL_ORDER / 2 + width,
                           -STENCIL_ORDER / 2 + height,
                           -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
                    printf(")");
#endif
                    printf(";");

                    stencil_initialized[field][stencil] = 1;
                  }
                  else {
                    printf("f%d_s%d = ", field, stencil);
                    printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                           stencil);
                    printf("stencils[%d][%d][%d][%d] *", //
                           stencil, depth, height, width);
                    printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                    printf("__ldg(&");
#endif
                    printf("vba.in[%d]"
                           "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                           "vertexIdx.z+(%d))])",
                           field, -STENCIL_ORDER / 2 + width,
                           -STENCIL_ORDER / 2 + height,
                           -STENCIL_ORDER / 2 + depth);
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

    gen_stencil_functions(curr_kernel);
    /*
    gen_kernel_prefix();
    gen_return_if_oob();

    prefetch_output_elements_and_gen_prev_function(false);
    prefetch_stencil_elements(curr_kernel);
    prefetch_stencil_coeffs(curr_kernel, false);

    compute_stencil_ops(curr_kernel);
    gen_stencil_functions(curr_kernel);
    */

    return;
  }
  case EXPLICIT_CACHING: {
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    return;
  }
  case EXPLICIT_CACHING_3D_BLOCKING: {
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    return;
  }
  case EXPLICIT_CACHING_4D_BLOCKING: {
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_4d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    return;
  }
  case EXPLICIT_PINGPONG_txw: {
    #if IMPLEMENTATION == EXPLICIT_PINGPONG_txw
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_pingpong_txw_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    #endif
    return;
  }
  case EXPLICIT_PINGPONG_txy: {
    #if IMPLEMENTATION == EXPLICIT_PINGPONG_txy
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_pingpong_txy_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    #endif
    return;
  }
  case EXPLICIT_PINGPONG_txyblocked: {
    gen_kernel_prefix(); // Note no bounds check

    // prefetch_stencil_elems_to_smem_pingpong_txyblocked_and_compute_stencil_ops(
    //     curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
    return;
  }
  case EXPLICIT_ROLLING_PINGPONG: {
    #if IMPLEMENTATION == EXPLICIT_ROLLING_PINGPONG
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_rolling_pingpong_and_compute_stencil_ops(
        curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function(false);
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
  else if (argc == 2 && !strcmp(argv[1], "-mem-accesses")) {
    gen_stencil_accesses();
  }
  // Generate memory accesses for the DSL kernels
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
