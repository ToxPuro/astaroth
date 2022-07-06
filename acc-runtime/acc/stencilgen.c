#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "user_defines.h"

#include "stencil_accesses.h"
#include "stencilgen.h"

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
  printf("AcReal processed_stencils[NUM_FIELDS][NUM_STENCILS];");
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
  printf("const auto previous=[&](const Field field)"
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
  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s=[&](const auto field)"
           "{stencils_accessed[field][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
}

void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            // Increases the compilation time notably
            if (!stencils_accessed[curr_kernel][field][stencil]) {
              // printf("processed_stencils[%d][%d] = (AcReal)NAN;", field,
              // stencil);
              continue;
            }

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
    printf("const auto %s=[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
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