#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stencilgen.h"

// clang-format off
int
main(int argc, char** argv)
{
  if (argc == 2 && !strcmp(argv[1], "-definitions")) { // Generate stencil
                                                       // definitions
    printf("static __device__ /*const*/ AcReal /*__restrict__*/ "
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
  else if (argc == 3) { // Generate stencil reductions
    const int curr_kernel      = atoi(argv[2]);
    
    int field_used[NUM_FIELDS] = {0};
    for (size_t field = 0; field < NUM_FIELDS; ++field)
      for (size_t stencil = 0; stencil < NUM_STENCILS; ++stencil)
        field_used[field] |= stencils_accessed[curr_kernel][field][stencil];

    int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (!field_used[field])
        continue;
      printf("{const AcReal* __restrict__ in=vba.in[%d];", field);
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*in[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*in[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                         field, stencil, stencil_binary_ops[stencil], field,
                         stencil, stencil_unary_ops[stencil], stencil, depth,
                         height, width, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                }
              }
            }
          }
        }
      }
      printf("}");
    }
  }
  else {
    fprintf(stderr, "Fatal error: invalid arguments passed to stencilgen.c");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
};
// clang-format on