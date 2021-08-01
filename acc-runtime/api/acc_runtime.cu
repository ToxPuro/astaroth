#include "acc_runtime.h"

#include "math_utils.h"

__device__ AcMeshInfo d_mesh_info;

// Astaroth 2.0 backwards compatibility START
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])

int __device__ __forceinline__
DCONST(const AcIntParam param)
{
  return d_mesh_info.int_params[param];
}
int3 __device__ __forceinline__
DCONST(const AcInt3Param param)
{
  return d_mesh_info.int3_params[param];
}
AcReal __device__ __forceinline__
DCONST(const AcRealParam param)
{
  return d_mesh_info.real_params[param];
}
AcReal3 __device__ __forceinline__
DCONST(const AcReal3Param param)
{
  return d_mesh_info.real3_params[param];
}

#define DEVICE_VTXBUF_IDX(i, j, k)                                             \
  ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))

__device__ constexpr int
IDX(const int i)
{
  return i;
}

__device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
  return DEVICE_VTXBUF_IDX(i, j, k);
}

__device__ __forceinline__ int
IDX(const int3 idx)
{
  return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z);
}
/*
__device__ constexpr inline VertexBufferHandle
DCONST(const VertexBufferHandle handle)
{
  return handle;
}*/
//#define IDX(i, j, k) ((i) + (j)*mm.x + (k)*mm.x * mm.y)
// #define IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mx) *
// DCONST(AC_my)) Astaroth 2.0 backwards compatibility END

#define Field3(x, y, z) make_int3((x), (y), (z))
#define real3(i, j, k) ((AcReal3){(i), (j), (k)})
#define print printf // TODO is this a good idea?

#include "user_kernels.h"

AcResult
acLaunchKernel(Kernel func, const cudaStream_t stream, const int3 start,
               const int3 end, VertexBufferArray vba)
{
  const int3 n = end - start;

  // const dim3 tpb = getOptimalTBConfig(n, vba).tpb; // TODO
  const dim3 tpb = dim3(32, 4, 1);
  const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                 (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                 (unsigned int)ceil(n.z / AcReal(tpb.z)));
  const size_t smem = 0;

  func<<<bpg, tpb, smem, stream>>>(start, end, vba);
  ERRCHK_CUDA_KERNEL();

  return AC_SUCCESS;
}

#define GEN_LOAD_UNIFORM(TYPE)                                                 \
  GEN_LOAD_UNIFORM_DECLARATION(TYPE)                                           \
  {                                                                            \
    cudaError_t retval = cudaMemcpyToSymbolAsync(                              \
        symbol, &value, sizeof(value), 0, cudaMemcpyHostToDevice, stream);     \
    return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;                    \
  }

#define GEN_STORE_UNIFORM(TYPE)                                                \
  GEN_STORE_UNIFORM_DECLARATION(TYPE)                                          \
  {                                                                            \
    cudaError_t retval = cudaMemcpyFromSymbolAsync(                            \
        dst, symbol, sizeof(*dst), 0, cudaMemcpyDeviceToHost, stream);         \
    return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;                    \
  }

GEN_LOAD_UNIFORM(AcReal)
GEN_LOAD_UNIFORM(AcReal3)
GEN_LOAD_UNIFORM(int)
GEN_LOAD_UNIFORM(int3)

GEN_STORE_UNIFORM(AcReal)
GEN_STORE_UNIFORM(AcReal3)
GEN_STORE_UNIFORM(int)
GEN_STORE_UNIFORM(int3)

/*
int
main(void)
{
  printf("Launching kernel... \n");
  acLaunchKernel(solve, 0, (int3){0, 0, 0}, (int3){1, 0, 0},
                 (VertexBufferArray){0});
  printf("done\n");
  return EXIT_SUCCESS;
}
*/