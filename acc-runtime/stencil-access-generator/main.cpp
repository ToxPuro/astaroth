#include <assert.h>
#include <cuda_runtime_api.h>
#include <string.h>

//#define __device__
//#define __global__

#define threadIdx ((int3){0, 0, 0})
#define blockIdx ((int3){0, 0, 0})
#define blockDim ((int3){0, 0, 0})
#define make_int3(x, y, z) ((int3){x, y, z})
#define Field3(x, y, z) make_int3((x), (y), (z))
#define make_float3(x, y, z) ((float3){x, y, z})
#define make_double3(x, y, z) ((double3){x, y, z})

// Just nasty: Must evaluate all code branches given arbitrary input
// if we want automated stencil generation to work in every case
#define DCONST(x) (2)
#define d_multigpu_offset ((int3){0, 0, 0})

constexpr int
IDX(const int i)
{
  return 0;
}

int
IDX(const int i, const int j, const int k)
{
  return 0;
}

int
IDX(const int3 idx)
{
  return 0;
}

#include "math_utils.h"

#include "acc_runtime.h"

static bool stencils_accessed[NUM_FIELDS][NUM_STENCILS] = {{0}};
static AcMeshInfo d_mesh_info;
#include "build/api/user_kernels.h"

VertexBufferArray
vbaCreate(const size_t count)
{
  VertexBufferArray vba;

  const size_t bytes = sizeof(vba.in[0][0]) * count;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.in[i]  = (AcReal*)malloc(bytes);
    vba.out[i] = (AcReal*)malloc(bytes);
  }

  return vba;
}

void
vbaDestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    free(vba->in[i]);
    free(vba->out[i]);
    vba->in[i]  = NULL;
    vba->out[i] = NULL;
  }
}

int
main(void)
{
  FILE* fp = fopen("stencil_accesses.h", "w+");
  assert(fp);

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    memset(stencils_accessed, 0, sizeof(bool) * NUM_FIELDS * NUM_STENCILS);
    VertexBufferArray vba = vbaCreate(1);
    kernel_lookup[k]((int3){0, 0, 0}, (int3){1, 1, 1}, vba);
    vbaDestroy(&vba);

    for (size_t j = 0; j < NUM_FIELDS; ++j)
      for (size_t i = 0; i < NUM_STENCILS; ++i)
        if (stencils_accessed[j][i])
          fprintf(fp, "[%lu][%lu][%lu] = 1,", k, j, i);
  }
  fprintf(fp, "};");

  fclose(fp);
}