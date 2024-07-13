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
    printf("__ldg(&");
    printf("vba.in[%d]", field);
    printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
    printf(")");
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
