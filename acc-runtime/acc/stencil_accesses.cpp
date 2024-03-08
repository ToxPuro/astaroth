#include <assert.h>
/*
#if AC_USE_HIP || __HIP_PLATFORM_HCC__ // Hack to ensure hip is used even if
                                       // USE_HIP is not propagated properly
                                       // TODO figure out better way
#include <hip/hip_runtime.h>           // Needed in files that include kernels
#else
#include <cuda_runtime_api.h>
#endif
*/
#include <string.h>

#include "errchk.h"
#include "datatypes.h"
#include "user_defines.h"

#undef __device__
#define __device__
#undef __global__
#define __global__
#undef __launch_bounds__
#define __launch_bounds__(x, y)
#undef __syncthreads
#define __syncthreads()
#undef __shared__
#define __shared__

#define threadIdx ((int3){0, 0, 0})
#define blockIdx ((int3){0, 0, 0})
#define blockDim ((int3){0, 0, 0})
#define make_int3(x, y, z) ((int3){x, y, z})
#define Field3(x, y, z) make_int3((x), (y), (z))
#define make_float3(x, y, z) ((float3){x, y, z})
#define make_double3(x, y, z) ((double3){x, y, z})
#define print printf
#define len(arr) sizeof(arr) / sizeof(arr[0])
#define rand_uniform() (0.5065983774206012) // Chosen by a fair dice roll.
// Guaranteed to be random.
// ref. xkcd :)

// Just nasty: Must evaluate all code branches given arbitrary input
// if we want automated stencil generation to work in every case
#define d_multigpu_offset ((int3){0, 0, 0})

int
DCONST(const AcIntParam param)
{
  return 0;
}
int3
DCONST(const AcInt3Param param)
{
  return (int3){0,0,0};
}
AcReal
DCONST(const AcRealParam param)
{
  return 0.0;
}
AcReal3
DCONST(const AcReal3Param param)
{
  return (AcReal3){0,0,0};
}

constexpr int
IDX(const int i)
{
  (void)i; // Unused
  return 0;
}

int
IDX(const int i, const int j, const int k)
{
  (void)i; // Unused
  (void)j; // Unused
  (void)k; // Unused
  return 0;
}

int
IDX(const int3 idx)
{
  (void)idx; // Unused
  return 0;
}

int
LOCAL_COMPDOMAIN_IDX(const int3 coord)
{
  (void)coord; // Unused
  return 0;
}

#include "math_utils.h"

#include "acc_runtime.h"

AcReal smem[8 * 1024 * 1024]; // NOTE: arbitrary limit: need to allocate at
                              // least the max smem size of the device
AcReal big_array[8*1024*1024]{0.0};

static int stencils_accessed[NUM_FIELDS][NUM_STENCILS] = {{0}};
static AcMeshInfo d_mesh_info;
#include "user_kernels.h"
#include "user_cpu_kernels.h"

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
main(int argc, char* argv[])
{
  if (argc != 2) {
    fprintf(stderr, "Usage: ./main <output_file>\n");
    return EXIT_FAILURE;
  }
  const char* output = argv[1];
  FILE* fp           = fopen(output, "w+");
  assert(fp);

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    memset(stencils_accessed, 0,
           sizeof(stencils_accessed[0][0]) * NUM_FIELDS * NUM_STENCILS);
    VertexBufferArray vba = vbaCreate(1);
    cpu_kernels[k]((int3){0, 0, 0}, (int3){1, 1, 1}, vba);
    vbaDestroy(&vba);

    for (size_t j = 0; j < NUM_FIELDS; ++j)
      for (size_t i = 0; i < NUM_STENCILS; ++i)
        if (stencils_accessed[j][i])
          fprintf(fp, "[%lu][%lu][%lu] = 1,", k, j, i);
  }
  fprintf(fp, "};");

  fclose(fp);
  return EXIT_SUCCESS;
}
