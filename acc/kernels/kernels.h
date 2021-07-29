#pragma once
#include <stdio.h>

#include "datatypes.h"

typedef AcReal AcRealPacked;

#if defined(__CUDA_RUNTIME_API_H__)
static inline void
cuda_assert(cudaError_t code, const char* file, int line, bool abort)
{
  if (code != cudaSuccess) {
    time_t terr;
    time(&terr);
    fprintf(stderr, "%s", ctime(&terr));
    fprintf(stderr, "\tCUDA error in file %s line %d: %s\n", file, line,
            cudaGetErrorString(code));
    fflush(stderr);

    if (abort)
      exit(code);
  }
}

#ifdef NDEBUG
#undef ERRCHK
#undef WARNCHK
#define ERRCHK(params)
#define WARNCHK(params)
#define ERRCHK_CUDA(params) params
#define WARNCHK_CUDA(params) params
#define ERRCHK_CUDA_KERNEL()
#else
#define ERRCHK_CUDA(params)                                                    \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }
#define WARNCHK_CUDA(params)                                                   \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, false);                          \
  }

#define ERRCHK_CUDA_KERNEL()                                                   \
  {                                                                            \
    ERRCHK_CUDA(cudaPeekAtLastError());                                        \
    ERRCHK_CUDA(cudaDeviceSynchronize());                                      \
  }
#endif

#define ERRCHK_CUDA_ALWAYS(params)                                             \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }

#define ERRCHK_CUDA_KERNEL_ALWAYS()                                            \
  {                                                                            \
    ERRCHK_CUDA_ALWAYS(cudaPeekAtLastError());                                 \
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());                               \
  }

#define WARNCHK_CUDA_ALWAYS(params)                                            \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, false);                          \
  }
#endif // __CUDA_RUNTIME_API_H__

#include "user_defines.h"

typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

extern __device__ AcMeshInfo d_mesh_info;
extern __device__ dim3 mm;
extern __device__ dim3 multigpu_offset;
#define IDX(i, j, k) ((i) + (j)*mm.x + (k)*mm.x * mm.y)

// Astaroth 2.0 backwards compatibility START
static int __device__ __forceinline__
DCONST(const AcIntParam param)
{
  return d_mesh_info.int_params[param];
}
static int3 __device__ __forceinline__
DCONST(const AcInt3Param param)
{
  return d_mesh_info.int3_params[param];
}
static AcReal __device__ __forceinline__
DCONST(const AcRealParam param)
{
  return d_mesh_info.real_params[param];
}
static AcReal3 __device__ __forceinline__
DCONST(const AcReal3Param param)
{
  return d_mesh_info.real3_params[param];
}
static __device__ constexpr VertexBufferHandle
DCONST(const VertexBufferHandle handle)
{
  return handle;
}
// Astaroth 2.0 backwards compatibility END

typedef struct {
  AcReal* in[NUM_FIELDS];
  AcReal* out[NUM_FIELDS];
} VertexBufferArray;

#ifdef __cplusplus
extern "C" {
#endif

#include "user_declarations.h"

typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);

AcResult acLaunchKernel(Kernel func, const cudaStream_t stream,
                        const int3 start, const int3 end,
                        VertexBufferArray vba);

#define GEN_LOAD_UNIFORM_DECLARATION(TYPE)                                     \
  AcResult acLoad##TYPE##Uniform(const cudaStream_t stream, TYPE symbol,       \
                                 const TYPE value)

#define GEN_STORE_UNIFORM_DECLARATION(TYPE)                                    \
  AcResult acStore##TYPE##Uniform(const cudaStream_t stream, TYPE* dst,        \
                                  const TYPE symbol)

GEN_LOAD_UNIFORM_DECLARATION(AcReal);
GEN_LOAD_UNIFORM_DECLARATION(AcReal3);
GEN_LOAD_UNIFORM_DECLARATION(int);
GEN_LOAD_UNIFORM_DECLARATION(int3);

GEN_STORE_UNIFORM_DECLARATION(AcReal);
GEN_STORE_UNIFORM_DECLARATION(AcReal3);
GEN_STORE_UNIFORM_DECLARATION(int);
GEN_STORE_UNIFORM_DECLARATION(int3);

#ifdef __cplusplus
} // extern "C"
#endif
