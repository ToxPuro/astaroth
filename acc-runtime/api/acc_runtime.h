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
  #include <stdbool.h>
  #if AC_MPI_ENABLED
  #include <mpi.h>
  #endif

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

  //copied from the sample setup
  #include "user_defines.h"
  #include "kernel_reduce_outputs.h"
  #include "user_input_typedefs.h"

#if AC_RUNTIME_COMPILATION
  #include <dlfcn.h>
#endif

  #define NUM_REDUCE_SCRATCHPADS (2)

  typedef struct {
#include "device_mesh_info_decl.h"
  } AcDeviceMeshInfo;

  typedef struct {
    int3 nn;
    AcReal3 lengths;
  } AcGridInfo;


  typedef struct {
#include "output_decl.h"
  } AcDeviceKernelOutput;

  //could combine these into base struct
  //with struct inheritance, but not sure would that break C ABI
  typedef struct {
#include "array_decl.h"
#include "device_mesh_info_decl.h"
#if AC_MPI_ENABLED
    MPI_Comm comm;
#endif
  } AcMeshInfo;

  typedef struct {
#include "comp_loaded_decl.h"
  } AcCompInfoLoaded;

  typedef struct {
#include "comp_decl.h"
  } AcCompInfoConfig;

  typedef struct {
	  AcCompInfoConfig config;
	  AcCompInfoLoaded is_loaded;
  } AcCompInfo;

  typedef struct {
#include "input_decl.h"
  } AcInputs;

  typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
    AcReal* w[NUM_WORK_BUFFERS];

    size_t bytes;
    acKernelInputParams kernel_input_params;
    AcReal* reduce_scratchpads[NUM_REAL_OUTPUTS+1][NUM_REDUCE_SCRATCHPADS];
    int reduce_offset;
    size_t scratchpad_size;
  } VertexBufferArray;




#if AC_RUNTIME_COMPILATION

#ifndef BASE_FUNC_NAME

#ifdef __cplusplus
#define BASE_FUNC_NAME(func_name) func_name##_BASE
#else
#define BASE_FUNC_NAME(func_name) func_name
#endif

#endif

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) static return_type (*func_name) __VA_ARGS__
#endif

#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) return_type func_name __VA_ARGS__
#endif

#ifndef BASE_FUNC_NAME 
#define BASE_FUNC_NAME(func_name) func_name
#endif

#endif
  #ifdef __cplusplus
  extern "C" {
  #endif

  #include "user_declarations.h"

  FUNC_DEFINE(const AcKernel*, acGetKernels,());
  FUNC_DEFINE(AcResult, acKernelFlush,(const cudaStream_t stream, AcReal* arr, const size_t n, const AcReal value));

  FUNC_DEFINE(AcResult, acVBAReset,(const cudaStream_t stream, VertexBufferArray* vba));

  FUNC_DEFINE(VertexBufferArray, acVBACreate,(const AcMeshInfo config));

  FUNC_DEFINE(void, acVBAUpdate,(VertexBufferArray* vba, const AcMeshInfo config));

  FUNC_DEFINE(void, acVBADestroy,(VertexBufferArray* vba, const AcMeshInfo config));

  FUNC_DEFINE(AcResult, acRandInitAlt,(const uint64_t seed, const size_t count, const size_t rank));

  FUNC_DEFINE(void, acRandQuit,(void));

  FUNC_DEFINE(AcResult, acLaunchKernel,(AcKernel func, const cudaStream_t stream, const int3 start, const int3 end, VertexBufferArray));

  FUNC_DEFINE(AcResult, acBenchmarkKernel,(AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
#if TWO_D == 0
  FUNC_DEFINE(AcResult, acLoadStencil,(const Stencil stencil, const cudaStream_t stream, const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreStencil,(const Stencil stencil, const cudaStream_t stream, AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));
#else
  FUNC_DEFINE(AcResult,acLoadStencil,(const Stencil stencil, const cudaStream_t stream, const AcReal data[STENCIL_HEIGHT][STENCIL_WIDTH]));

  FUNC_DEFINE(AcResult, acStoreStencil(const Stencil stencil, const cudaStream_t stream, AcReal data[STENCIL_HEIGHT][STENCIL_WIDTH]));
#endif

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
#include "load_and_store_uniform_header.h"

  // Diagnostics
  FUNC_DEFINE(Volume, acKernelLaunchGetLastTPB,(void));

  FUNC_DEFINE(AcKernel, GetOptimizedKernel,(const AcKernel, const VertexBufferArray vba));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadSize,(const AcKernel kernel));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadMinSize,());

#if AC_RUNTIME_COMPILATION
  static AcResult __attribute__((unused)) acLoadRunTime()
  {
 	void* handle = dlopen(runtime_astaroth_runtime_path,RTLD_NOW | RTLD_GLOBAL);
	if(!handle)
	{
    		fprintf(stderr,"%s","Fatal error was not able to load Astaroth runtime\n"); 
		fprintf(stderr,"Error message: %s\n",dlerror());
		exit(EXIT_FAILURE);
	}
	*(void**)(&acKernelFlush) = dlsym(handle,"acKernelFlush");
	if(!acKernelFlush) fprintf(stderr,"Astaroth error: was not able to load %s\n","acKernelFlush");
	*(void**)(&acVBAReset) = dlsym(handle,"acVBAReset");
	if(!acVBAReset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVBAReset");
	*(void**)(&acVBACreate) = dlsym(handle,"acVBACreate");
	if(!acVBACreate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVBACreate");
	*(void**)(&acVBAUpdate) = dlsym(handle,"acVBAUpdate");
	if(!acVBAUpdate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVBAUpdate");
	*(void**)(&acVBADestroy) = dlsym(handle,"acVBADestroy");
	if(!acVBADestroy) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVBADestroy");
	*(void**)(&acRandInitAlt) = dlsym(handle,"acRandInitAlt");
	if(!acRandInitAlt) fprintf(stderr,"Astaroth error: was not able to load %s\n","acRandInitAlt");
	*(void**)(&acRandQuit) = dlsym(handle,"acRandQuit");
	if(!acRandQuit) fprintf(stderr,"Astaroth error: was not able to load %s\n","acRandQuit");
	*(void**)(&acLaunchKernel) = dlsym(handle,"acLaunchKernel");
	if(!acLaunchKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLaunchKernel");
	*(void**)(&acBenchmarkKernel) = dlsym(handle,"acBenchmarkKernel");
	if(!acBenchmarkKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","acBenchmarkKernel");
	*(void**)(&acLoadStencil) = dlsym(handle,"acLoadStencil");
	if(!acLoadStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadStencil");
	*(void**)(&acStoreStencil) = dlsym(handle,"acStoreStencil");
	if(!acStoreStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreStencil");
	*(void**)(&acLoadRealUniform) = dlsym(handle,"acLoadRealUniform");
	if(!acLoadRealUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadRealUniform");
	*(void**)(&acLoadRealArrayUniform) = dlsym(handle,"acLoadRealArrayUniform");
	if(!acLoadRealArrayUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadRealArrayUniform");
	*(void**)(&acLoadReal3Uniform) = dlsym(handle,"acLoadReal3Uniform");
	if(!acLoadReal3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadReal3Uniform");
	*(void**)(&acLoadIntUniform) = dlsym(handle,"acLoadIntUniform");
	if(!acLoadIntUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadIntUniform");
	*(void**)(&acLoadBoolUniform) = dlsym(handle,"acLoadBoolUniform");
	if(!acLoadBoolUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadBoolUniform");
	*(void**)(&acLoadIntArrayUniform) = dlsym(handle,"acLoadIntArrayUniform");
	if(!acLoadIntArrayUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadIntArrayUniform");
	*(void**)(&acLoadInt3Uniform) = dlsym(handle,"acLoadInt3Uniform");
	if(!acLoadInt3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadInt3Uniform");
	*(void**)(&acStoreRealUniform) = dlsym(handle,"acStoreRealUniform");
	if(!acStoreRealUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreRealUniform");
	*(void**)(&acStoreReal3Uniform) = dlsym(handle,"acStoreReal3Uniform");
	if(!acStoreReal3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreReal3Uniform");
	*(void**)(&acStoreIntUniform) = dlsym(handle,"acStoreIntUniform");
	if(!acStoreIntUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreIntUniform");
	*(void**)(&acStoreBoolUniform) = dlsym(handle,"acStoreBoolUniform");
	if(!acStoreBoolUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreBoolUniform");
	*(void**)(&acStoreInt3Uniform) = dlsym(handle,"acStoreInt3Uniform");
	if(!acStoreInt3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acStoreInt3Uniform");
	*(void**)(&acKernelLaunchGetLastTPB) = dlsym(handle,"acKernelLaunchGetLastTPB");
	if(!acKernelLaunchGetLastTPB) fprintf(stderr,"Astaroth error: was not able to load %s\n","acKernelLaunchGetLastTPB");
	*(void**)(&GetOptimizedKernel) = dlsym(handle,"GetOptimizedKernel");
	if(!GetOptimizedKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","GetOptimizedKernel");
	*(void**)(&acGetKernelReduceScratchPadSize) = dlsym(handle,"acGetKernelReduceScratchPadSize");
	if(!acGetKernelReduceScratchPadSize) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetKernelReduceScratchPadSize");
	*(void**)(&acGetKernelReduceScratchPadMinSize) = dlsym(handle,"acGetKernelReduceScratchPadMinSize");
	if(!acGetKernelReduceScratchPadMinSize) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetKernelReduceScratchPadMinSize");
	*(void**)(&acGetKernels) = dlsym(handle,"acGetKernels");
	if(!acGetKernels) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetKernels");
	return AC_SUCCESS;
  }
#endif

  #ifdef __cplusplus
  } // extern "C"
    //
    //
#ifndef AC_RUNTIME_SOURCE
#include <type_traits>
#include <string.h>

  #ifdef __cplusplus
#include  "push_to_config.h"

  template <typename P, typename V>
  void
  acPushToConfig(AcMeshInfo& config, AcCompInfo& comp_info, P param, V val)
  {
          if constexpr(IsCompParam(param))
                  acLoadCompInfo(param, val, &comp_info);
          else
                  acPushToConfig(config, param, val);
  }

  #endif
  static AcCompInfo __attribute__((unused)) acInitCompInfo()
  {
	  AcCompInfo res;
	  memset(&res.is_loaded,0,sizeof(res.is_loaded));
	  return res;
  }
#include "load_comp_info.h"

#ifdef __cplusplus
#include "is_comptime_param.h"
#endif

#ifdef __cplusplus

#define GEN_LOAD_COMP_INFO(PARAM_TYPE,VAL_TYPE,TYPE) \
  static AcResult __attribute__((unused)) acLoadCompInfo(const PARAM_TYPE param, const VAL_TYPE val, AcCompInfo* info) {return acLoad##TYPE##CompInfo(param,val,info);};
#include "load_comp_info_overloads.h"

#endif

  template <typename P>
  constexpr static array_info
  get_array_info(const P array)
  {
#include "get_array_info.h"
  }

  template <typename P>
  constexpr static bool
  is_dconst(const P array)
  {
	  return get_array_info(array).is_dconst;
  }

  template <typename P>
  constexpr static auto
  get_array_dims(const P array)
  {
	  return get_array_info(array).dims;
  }

  template <typename P>
  constexpr static int
  get_array_n_dims(const P array)
  {
	  return get_array_info(array).num_dims;
  }

  template <typename P>
  constexpr const char*
  get_param_name(const P param)
  {
#include "get_param_name.h"
  }
  template <typename P>
  constexpr const char*
  get_array_name(const P array)
  {
	  return get_array_info(array).name;
  }
  template <typename P>
  constexpr bool
  get_is_loaded(const P param, const AcCompInfoLoaded config)
  {
#include "get_from_comp_config.h"
  };

  template <typename P>
  auto
  get_loaded_val(const P param, const AcCompInfoConfig config)
  {
#include "get_from_comp_config.h"
  };

  template <typename P>
  constexpr static int
  get_array_length(const P array, const AcMeshInfo host_info)
  {
	  AcArrayDims dims     = get_array_info(array).dims;
	  int num_dims         = get_array_info(array).num_dims;
	  int res = 1;
	  for(int i = 0; i < num_dims; ++i)
		  res *= dims.from_config[i] ? host_info.int_params[dims.len[i]] : dims.len[i];
	  return res;
  }

  template <typename P>
  constexpr static int
  get_dconst_array_length(const P array)
  {
	  return get_array_info(array).length;
  }

  template <typename P>
  static int
  get_dconst_array_offset(const P array)
  {
	  return get_array_info(array).d_offset;
  }

  template <typename P>
  constexpr static size_t
  get_num_params()
  {
	  const int res=
#include "get_num_params.h"
		  -1;
	  static_assert(res >= 0);
	  return res;
  }

  // Helper to generate a sequence of integers at compile-time
  template<std::size_t... Is>
  constexpr std::array<int, sizeof...(Is)> make_array(std::index_sequence<Is...>) {
      return {{ Is... }};
  }
  
  // Function to create an array of integers from 0 to N-1
  template<std::size_t N>
  constexpr std::array<int, N> createIntArray() {
      return make_array(std::make_index_sequence<N>{});
  }
  
  // Function to create an array of enums from an array of integers
  template<typename P, std::size_t N, std::size_t... Is>
  constexpr std::array<P, N> createEnumArrayImpl(const std::array<int, N>& intArray, std::index_sequence<Is...>) {
      return {{ static_cast<P>(intArray[Is])... }};
  }
  
  template<typename P, std::size_t N>
  constexpr std::array<P, N> createEnumArray(const std::array<int, N>& intArray) {
      return createEnumArrayImpl<P>(intArray, std::make_index_sequence<N>{});
  }

  template <typename P>
  constexpr auto
  get_params()
  {
	  return createEnumArray<P>(
			  createIntArray<get_num_params<P>()>()
	  );
  }
  constexpr auto
  get_vtxbuf_handles()
  {
	  return createEnumArray<Field>(createIntArray<NUM_VTXBUF_HANDLES>());
  }

	


  template <typename P>
  constexpr auto
  get_config_param(const P param, const AcMeshInfo& config)
  {
#include "get_config_param.h"
  }	  

  

#include "load_and_store_uniform_overloads.h"
  
  template<typename T, typename... Ts>
  struct ForEach
  {
      template<template<typename> typename F, typename... Args>
      static constexpr void run(Args&&... args)
      {
          ForEach<T>::template run<F>(std::forward<Args>(args)...);
          ForEach<Ts...>::template run<F>(std::forward<Args>(args)...);
      }
  };
  
  template<typename T>
  struct ForEach<T>
  {
      template<template<typename> typename F, typename... Args>
      static constexpr void run(Args&&... args)
      {
          F<T>{}(std::forward<Args>(args)...);
      }
  };


  using AcScalarTypes = ForEach<
#include "scalar_types.h"
  AcIntParam
  >;

  using AcScalarCompTypes = ForEach<
#include "scalar_comp_types.h"
  AcIntCompParam
  >;
  

  using AcArrayTypes = ForEach<
#include "array_types.h"
  AcIntArrayParam
  >;

  using AcArrayCompTypes = ForEach<
#include "array_comp_types.h"
  AcIntCompArrayParam
  >;

#endif
  #endif
