#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "user_defines.h"

#include "stencil_accesses.h"
#include "stencilgen.h"

#define ORIGINAL (0)
#define ORIGINAL_WITH_ILP (1)
#define EXPL_REG_VARS (2)
#define FULLY_EXPL_REG_VARS (3)
#define EXPL_REG_VARS_AND_CT_CONST_STENCILS (4)
#define IMPLEMENTATION (1)

void
gen_stencil_definitions(void)
{
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
  printf("const int3 globalGridN = d_mesh_info.int3_params[AC_global_grid_n];");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");
  printf("const auto previous __attribute__((unused)) =[&](const Field field)"
         "{ return vba.out[field][idx]; };");
  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");
  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");
}

void
gen_stencil_accesses(void)
{
  gen_kernel_prefix();
  printf("AcReal __restrict__ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s=[&](const auto field)"
           "{stencils_accessed[field][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
}

#if IMPLEMENTATION == ORIGINAL
// Original
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();
  printf("AcReal __restrict__ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              printf("processed_stencils[%d][%d] = ", field, stencil);
              if (!stencil_initialized[field][stencil]) {
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf( //
                    "%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s __attribute__((unused)) =[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
}
#elif IMPLEMENTATION == ORIGINAL_WITH_ILP
// Original + improved ILP (field-stencil to inner loop)
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();
  printf("AcReal __restrict__ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

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
              printf("processed_stencils[%d][%d] = ", field, stencil);
              if (!stencil_initialized[field][stencil]) {
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf( //
                    "%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s __attribute__((unused)) =[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
}
#elif IMPLEMENTATION == EXPL_REG_VARS
// Explicit register variables instead of a processed_stencils array
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

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
                printf("auto register f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS
// Everything prefetched
void
gen_kernel_body(const int curr_kernel)
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
  printf("const int3 globalGridN = d_mesh_info.int3_params[AC_global_grid_n];");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");
  printf("const auto previous __attribute__((unused)) =[&](const Field field)"
         "{ return vba.out[field][idx]; };");
  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");
  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("const auto __restrict__ in%d = vba.in[%d];", field, field);

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
              printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))];",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }

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

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

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
                printf("auto register f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == EXPL_REG_VARS_AND_CT_CONST_STENCILS
// Explicit register variables & compile-time constant stencils
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

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
                printf("auto register f%d_s%d = ", field, stencil);
                printf("%s((%s)*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil],
                       stencils[stencil][depth][height][width], field,
                       -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                       -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s((%s)*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil],
                    stencils[stencil][depth][height][width], field,
                    -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                    -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#endif

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