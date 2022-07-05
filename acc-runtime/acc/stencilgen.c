#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "user_defines.h"

#include "stencilgen.h"

// Remember to modify USE_SMEM also in acc_runtime.cu
// Check also if need to comment out "if (vertexIdx.x >= end...." in ac.y
// to avoid having threads exit before loading data to smem
#define USE_SMEM (0)

// clang-format off
int
main(int argc, char** argv)
{
  (void)vtxbuf_names; // Unused
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
      
      #if USE_SMEM
      printf("extern __shared__ AcReal smem[];");
      printf("const int pad = 0;");
      printf("const int sx = blockDim.x + STENCIL_WIDTH - 1 + pad;");
      printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
      printf("int3 baseIdx = (int3){blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
                                         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
                                         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        printf("for (int j = 0; j < sy; ++j){");
          printf("for (int i = threadIdx.x; i < sx - pad; i += blockDim.x){");
            printf("smem[i + j * sx] = in[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + %d)];", depth);
          printf("}");
        printf("}");
        printf("__syncthreads();");
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*smem[(threadIdx.x + %d) + (threadIdx.y + %d) * sx]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, width, height);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*smem[(threadIdx.x + %d) + (threadIdx.y + %d) * sx]));",
                         field, stencil, stencil_binary_ops[stencil], field,
                         stencil, stencil_unary_ops[stencil], stencil, depth,
                         height, width, width, height);
                }
              }
            }
          }
        }
      }
      #else
      /*
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            //printf("{const AcReal tmp = in[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))];", -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
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
            //printf("};");
          }
        }
      }*/
      /*
      // Original
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
      */
     // Alt, seems to be slightly faster
     // Findings: nvcc does some very nice reordering, does not seem to be much room for ILP improvements
      for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            
            
              if (!stencils_accessed[curr_kernel][field][stencil]) {
                printf("processed_stencils[%d][%d] = (AcReal)NAN;", field, stencil);
                continue;
              }
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                         field, stencil, stencil_binary_ops[stencil], field,
                         stencil, stencil_unary_ops[stencil], stencil, depth,
                         height, width, field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                }
              }
            }
          }
        }
      }
     /*
      // Prefetching to local memory for better ILP using vectorized loads (1D)
      // Works but is slow, likely caused by excessive integer arithmetic to compute indices
      // Also incurs dataa dependency which limits ILP (cannot compute stencil and update 'vec' before data have arrived)
      //
      // Need to define these in the beginning of the kernel for this to work 
      //const size_t veclen = STENCIL_WIDTH/4 + 2;
      //printf("\n#define VECLEN (%lu)\n", veclen);
      //printf("AcReal vec[%lu];", 4 * veclen);
      //printf("int base_idx, offset;");
      //printf("AcReal tmp[STENCIL_HEIGHT][8];");
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {

          printf("base_idx = 4 * (IDX(vertexIdx.x + (%d), vertexIdx.y + (%d), vertexIdx.z + (%d)) / 4);", -STENCIL_ORDER/2, -STENCIL_ORDER/2 + height, -STENCIL_ORDER/2 + depth);
          printf("offset = IDX(vertexIdx.x + (%d), vertexIdx.y + (%d), vertexIdx.z + (%d)) - base_idx;", -STENCIL_ORDER/2, -STENCIL_ORDER/2 + height, -STENCIL_ORDER/2 + depth);
          for (size_t i = 0; i < veclen; ++i) {
            printf("reinterpret_cast<double4*>(vec)[%lu] = reinterpret_cast<double4*>(&vba.in[%d][base_idx])[%lu];", i, field, i);
          }

          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*vec[%d + offset]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, width);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*vec[%d + offset]));",
                         field, stencil, stencil_binary_ops[stencil], field,
                         stencil, stencil_unary_ops[stencil], stencil, depth,
                         height, width, width);
                }
              }
            }
          }
        }
      }
      */
     /*
     // Prefetching to local memory for better ILP using vectorized loads (2D)
     // Proof of concept, does not actually access the proper indices
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
          //printf("{");
          for (int h = 0; h < STENCIL_HEIGHT; ++h) {
            printf("reinterpret_cast<double4*>(tmp[%d])[0] = reinterpret_cast<double4*>(&vba.in[%d][0])[2*%d + 0];", h, field, h);
            printf("reinterpret_cast<double4*>(tmp[%d])[1] = reinterpret_cast<double4*>(&vba.in[%d][0])[2*%d + 1];", h, field, h);
          }
          //printf("}");

        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          
          //for (int i = 0; i < STENCIL_WIDTH; ++i)
          //      printf("tmp[%d] = vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))];", i, field, -STENCIL_ORDER / 2 + i, -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*tmp[%d][%d]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, height, width);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*tmp[%d][%d]));",
                         field, stencil, stencil_binary_ops[stencil], field,
                         stencil, stencil_unary_ops[stencil], stencil, depth,
                         height, width, height, width);
                }
              }
            }
          }
        }
      }
      */
     /*
     // Bad performance and wrong
     for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
     printf("auto %s = [&](const int field){", stencil_names[stencil]);
     //printf("auto %s = [&](const int field){const AcReal* __restrict__ in=vba.in[%d];", stencil_names[stencil], field);
     printf("switch (field) {");
     for (int field = 0; field < NUM_FIELDS; ++field) {
       printf("case %d: {", field);
      printf("if (processed_stencils[%d][%d]) return processed_stencils[%d][%d];", field, stencil, field, stencil);
      printf("const AcReal* __restrict__ in=vba.in[%d];", field);
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
          for (int height = 0; height < STENCIL_HEIGHT; ++height) {
            for (int width = 0; width < STENCIL_WIDTH; ++width) {
              
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
        printf("return processed_stencils[%d][%d];", field, stencil);
        printf("}");
      }
        printf("default: return (AcReal)NAN;");
        printf("}"); // Switch end
        printf("};"); // Lambda end
      }
      break;
      */
     /*
      for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=stencils[%d][%d][%d][%d]*vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))]",
                         field, stencil, stencil,
                         depth, height, width, field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("+stencils[%d][%d][%d][%d]*vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]",
                         stencil, depth,
                         height, width, field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                }
              }
            }
          }
        }
        printf(";");
      }
      */
     /*
     printf("auto idxfn = [](const int i, const int j, const int k) { return i + j * (64 + STENCIL_WIDTH-1) + k * (64 + STENCIL_WIDTH-1) * (64 + STENCIL_HEIGHT-1); };");
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {
            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
              if (!stencils_accessed[curr_kernel][field][stencil])
                continue;
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field][stencil]) {
                  printf("processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*in[idxfn(vertexIdx.x+(%d),vertexIdx.y+(%d), vertexIdx.z+(%d))]);",
                         field, stencil, stencil_unary_ops[stencil], stencil,
                         depth, height, width, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*in[idxfn(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
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
      }*/
      #endif
    }
  }
  else {
    fprintf(stderr, "Fatal error: invalid arguments passed to stencilgen.c");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
};
// clang-format on