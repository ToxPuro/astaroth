#pragma once
#include <stdio.h>

#if AC_USE_HIP
#include "hip.h"
#endif

#include "datatypes.h"
#include "errchk.h"

#include <cuda_runtime_api.h> // cudaStream_t

#include "user_defines.h"

typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

typedef struct {
  AcReal* in[NUM_FIELDS];
  AcReal* out[NUM_FIELDS];
} VertexBufferArray;

typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);

#ifdef __cplusplus
extern "C" {
#endif

#include "user_declarations.h"

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
