  /*
      Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

      This file is part of Astaroth.

      Astaroth is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      Astaroth is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
  */
  #pragma once
  #include <stdio.h>

  #if AC_USE_HIP
  #include "hip.h"

  #include <hip/hip_runtime_api.h> // Streams
  #include <roctracer_ext.h>       // Profiling
  #else
  #include <cuda_profiler_api.h> // Profiling
  #include <cuda_runtime_api.h>  // Streams
  #endif

  #include "datatypes.h"
  #include "errchk.h"

  #include "user_defines.h"

  #define NUM_REDUCE_SCRATCHPADS (2)

  typedef struct {
    int int_params[NUM_INT_PARAMS];
    int3 int3_params[NUM_INT3_PARAMS];
    AcReal real_params[NUM_REAL_PARAMS];
    AcReal3 real3_params[NUM_REAL3_PARAMS];
    AcReal* profiles[NUM_PROFILES];
    AcReal* arrays[NUM_REAL_ARRAYS];
  } AcMeshInfo;

  typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
    AcReal* profiles[NUM_PROFILES];
    AcReal* w[NUM_WORK_BUFFERS];
    AcReal* arrays[NUM_REAL_ARRAYS];
    size_t bytes;
  } VertexBufferArray;

  typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);
  #ifdef __cplusplus
  #include <functional>
  typedef std::function<void(const dim3 bpg, const dim3 tpb, const size_t smem, const cudaStream_t stream, const int3 start, const int3 end, VertexBufferArray vba)> kernel_lambda; 
  typedef struct KernelLambda {
    kernel_lambda lambda;
    //used in lookup into tbconfig, otherwise not needed
    void* kernel;
  } KernelLambda;

  KernelLambda
  kernel_to_kernel_lambda(const Kernel kernel);


#define GEN_BIND_SINGLE_HEADER(TYPE)                                                  \
  KernelLambda bind_single_param(void (*kernel)(const int3 start, const int3 end, VertexBufferArray vba, TYPE input_param), TYPE input_param);
  GEN_BIND_SINGLE_HEADER(int)
  GEN_BIND_SINGLE_HEADER(AcReal)
  GEN_BIND_SINGLE_HEADER(AcReal*)
  GEN_BIND_SINGLE_HEADER(int*)
  GEN_BIND_SINGLE_HEADER(bool)
  GEN_BIND_SINGLE_HEADER(bool*)

  template <typename T, typename F>
  KernelLambda
  bind_two_params(void (*kernel)(const int3 start, const int3 end, VertexBufferArray vba, T input_param, F second_input_param), T input_param, F second_input_param);
  template <typename T, typename F, typename H>
  KernelLambda
  bind_three_params(void (*kernel)(const int3 start, const int3 end, VertexBufferArray vba, T input_param, F second_input_param, H third_input_param), T input_param, F second_input_param, H third_input_param);
  #else
  //if not C++ then opaque struct
  typedef struct KernelLambda KernelLambda;
  #endif

  #ifdef __cplusplus
  extern "C" {
  #endif

  #include "user_declarations.h"

  AcResult acKernelFlush(const cudaStream_t stream, AcReal* arr, const size_t n,
                        const AcReal value);

  AcResult acVBAReset(const cudaStream_t stream, VertexBufferArray* vba);

  VertexBufferArray acVBACreate(const AcMeshInfo config);

  void acVBADestroy(VertexBufferArray* vba, const AcMeshInfo config);

  AcResult acRandInit(const uint64_t seed, const Volume m_local,
                      const Volume m_global, const Volume global_offset);

  AcResult acRandInitAlt(const uint64_t seed, const size_t count,
                        const size_t rank);

  void acRandQuit(void);

  AcResult acLaunchKernel(Kernel func, const cudaStream_t stream,
                          const int3 start, const int3 end, VertexBufferArray);

  #ifdef __cplusplus
  //start and end C linkage
  }
  AcResult acLaunchKernel(KernelLambda lambda, const cudaStream_t stream,
                          const int3 start, const int3 end,
                          VertexBufferArray vba);
  extern "C" {
  #endif 

  AcResult acBenchmarkKernel(Kernel kernel, const int3 start, const int3 end,
                            VertexBufferArray vba);

  #ifdef __cplusplus
  //start and end C linkage
  }
  AcResult acBenchmarkKernel(KernelLambda kernel, const int3 start, const int3 end,
                            VertexBufferArray vba);
  extern "C" {
  #endif

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult
  acLoadStencil(const Stencil stencil, const cudaStream_t stream,
                const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult
  acStoreStencil(const Stencil stencil, const cudaStream_t stream,
                AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acLoadRealUniform(const cudaStream_t stream, const AcRealParam param,
                            const AcReal value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acLoadReal3Uniform(const cudaStream_t stream, const AcReal3Param param,
                              const AcReal3 value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acLoadIntUniform(const cudaStream_t stream, const AcIntParam param,
                            const int value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acLoadInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                            const int3 value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acStoreRealUniform(const cudaStream_t stream, const AcRealParam param,
                              AcReal* value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acStoreReal3Uniform(const cudaStream_t stream,
                              const AcReal3Param param, AcReal3* value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acStoreIntUniform(const cudaStream_t stream, const AcIntParam param,
                            int* value);

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  AcResult acStoreInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                              int3* value);

  // Diagnostics
  Volume acKernelLaunchGetLastTPB(void);

  #ifdef __cplusplus
  } // extern "C"
  #endif
