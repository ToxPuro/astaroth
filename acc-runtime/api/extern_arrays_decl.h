#pragma once

// TP: We do this ugly macro because I want to keep the generated headers the
// same if we are compiling CPU analysis and the actual GPU code.
// OM: After trying to make it possible to compile kernels across multiple CUs
// the variables need to be referenced using 'extern' and then linked using
// device linking.
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME)                    \
  extern __device__ __constant__ DATATYPE*                                     \
      AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME
#define DECLARE_CONST_DIMS_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, LEN)    \
  extern __device__ DATATYPE                                                   \
      AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME[LEN]
#define DECLARE_DCONST_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, LEN)             \
  extern UNUSED __device__ __constant__ DATATYPE                               \
      AC_INTERNAL_d_##DEFINE_NAME##_arrays_##ARR_NAME[LEN];
#include "dconst_arrays_decl.h"
#include "gmem_arrays_decl.h"
#undef DECLARE_GMEM_ARRAY
#undef DECLARE_CONST_DIMS_GMEM_ARRAY
#undef DECLARE_DCONST_ARRAY
