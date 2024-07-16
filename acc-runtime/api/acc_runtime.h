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
    AcReal real_outputs[NUM_REAL_OUTPUTS];
    AcReal3 real3_outputs[NUM_REAL3_OUTPUTS];
    int int_outputs[NUM_INT_OUTPUTS];
    int3 int3_outputs[NUM_INT3_OUTPUTS];
    bool bool_outputs[NUM_BOOL_OUTPUTS];
  } AcDeviceKernelOutput;

  //could combine these into base struct
  //with struct inheritance, but not sure would that break C ABI
  typedef struct {
    AcReal* real_arrays[NUM_REAL_ARRAYS];
    int* int_arrays[NUM_INT_ARRAYS];
    bool* bool_arrays[NUM_BOOL_ARRAYS];
    AcReal3* real3_arrays[NUM_REAL3_ARRAYS];
    int3* int3_arrays[NUM_INT3_ARRAYS];
#include "device_mesh_info_decl.h"
#if AC_MPI_ENABLED
    MPI_Comm comm;
#endif
  } AcMeshInfo;

  typedef struct {
    bool int_params[NUM_INT_COMP_PARAMS];
    bool int3_params[NUM_INT3_COMP_PARAMS];
    bool real_params[NUM_REAL_COMP_PARAMS];
    bool real3_params[NUM_REAL3_COMP_PARAMS];
    bool real_arrays[NUM_REAL_COMP_ARRAYS];
    bool bool_params[NUM_BOOL_COMP_PARAMS];
    bool int_arrays[NUM_INT_COMP_ARRAYS];
    bool bool_arrays[NUM_BOOL_COMP_ARRAYS];
    bool int3_arrays[NUM_INT3_COMP_ARRAYS];
    bool real3_arrays[NUM_REAL3_COMP_ARRAYS];
  } AcCompInfoLoaded;

  typedef struct {
    int int_params[NUM_INT_COMP_PARAMS];
    int3 int3_params[NUM_INT3_COMP_PARAMS];
    AcReal real_params[NUM_REAL_COMP_PARAMS];
    AcReal3 real3_params[NUM_REAL3_COMP_PARAMS];
    bool bool_params[NUM_BOOL_COMP_PARAMS];
    const AcReal*    real_arrays[NUM_REAL_COMP_ARRAYS];
    const int*       int_arrays[NUM_INT_COMP_ARRAYS];
    const bool*      bool_arrays[NUM_BOOL_COMP_ARRAYS];
    const AcReal3*   real3_arrays[NUM_REAL3_COMP_ARRAYS]; 
    const int3*      int3_arrays[NUM_INT3_COMP_ARRAYS]; 
  } AcCompInfoConfig;

  typedef struct {
	  AcCompInfoConfig config;
	  AcCompInfoLoaded is_loaded;
  } AcCompInfo;

  //pad by one to avoid 0 length arrays that might not work
  typedef struct {
    int int_params[NUM_INT_INPUT_PARAMS+1];
    AcReal real_params[NUM_REAL_INPUT_PARAMS+1];
    bool bool_params[NUM_BOOL_INPUT_PARAMS+1];
  } AcInputs;

  typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
    AcReal* w[NUM_WORK_BUFFERS];

    AcReal*   real_arrays[NUM_REAL_ARRAYS];
    int*      int_arrays[NUM_INT_ARRAYS];
    bool*     bool_arrays[NUM_BOOL_ARRAYS];
    AcReal3*  real3_arrays[NUM_REAL3_ARRAYS];
    int3*     int3_arrays[NUM_INT3_ARRAYS];

    size_t bytes;
    acKernelInputParams kernel_input_params;
    AcReal* reduce_scratchpads[NUM_REAL_OUTPUTS+1][NUM_REDUCE_SCRATCHPADS];
    int reduce_offset;
    size_t scratchpad_size;
  } VertexBufferArray;


  typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);

#if AC_RUNTIME_COMPILATION

#ifndef BASE_FUNC_NAME

#if __cplusplus
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
  FUNC_DEFINE(AcResult, acKernelFlush,(const cudaStream_t stream, AcReal* arr, const size_t n, const AcReal value));

  FUNC_DEFINE(AcResult, acVBAReset,(const cudaStream_t stream, VertexBufferArray* vba));

  FUNC_DEFINE(VertexBufferArray, acVBACreate,(const AcMeshInfo config));

  FUNC_DEFINE(void, acVBAUpdate,(VertexBufferArray* vba, const AcMeshInfo config));

  FUNC_DEFINE(void, acVBADestroy,(VertexBufferArray* vba, const AcMeshInfo config));

  FUNC_DEFINE(AcResult, acRandInitAlt,(const uint64_t seed, const size_t count, const size_t rank));

  FUNC_DEFINE(void, acRandQuit,(void));

  FUNC_DEFINE(AcResult, acLaunchKernel,(Kernel func, const cudaStream_t stream, const int3 start, const int3 end, VertexBufferArray));

  FUNC_DEFINE(AcResult, acBenchmarkKernel,(Kernel kernel, const int3 start, const int3 end, VertexBufferArray vba));

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
  FUNC_DEFINE(AcResult, acLoadRealUniform,(const cudaStream_t stream, const AcRealParam param, const AcReal value));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadRealArrayUniform,(const cudaStream_t stream, const AcRealArrayParam param, const AcReal* values));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadReal3Uniform,(const cudaStream_t stream, const AcReal3Param param, const AcReal3 value));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadReal3ArrayUniform,(const cudaStream_t stream, const AcReal3ArrayParam param, const AcReal3* values));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadIntUniform,(const cudaStream_t stream, const AcIntParam param, const int value));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadBoolUniform,(const cudaStream_t stream, const AcBoolParam param, const bool value));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadIntArrayUniform,(const cudaStream_t stream, const AcIntArrayParam param, const int* values));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadBoolArrayUniform,(const cudaStream_t stream, const AcBoolArrayParam param, const bool* values));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadInt3Uniform,(const cudaStream_t stream, const AcInt3Param param, const int3 value));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadInt3ArrayUniform,(const cudaStream_t stream, const AcInt3ArrayParam param, const int3* values));


  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreRealUniform,(const cudaStream_t stream, const AcRealParam param, AcReal* value));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreReal3Uniform,(const cudaStream_t stream, const AcReal3Param param, AcReal3* value));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreIntUniform,(const cudaStream_t stream, const AcIntParam param, int* value));
  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreBoolUniform,(const cudaStream_t stream, const AcBoolParam param, bool* value));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreInt3Uniform,(const cudaStream_t stream, const AcInt3Param param, int3* value));

  // Diagnostics
  FUNC_DEFINE(Volume, acKernelLaunchGetLastTPB,(void));

  FUNC_DEFINE(Kernel, GetOptimizedKernel,(const AcKernel, const VertexBufferArray vba));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadSize,(const AcKernel kernel));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadMinSize,());

#if AC_RUNTIME_COMPILATION
  static AcResult __attribute__((unused)) acLoadRunTime(void* handle)
  {
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
	return AC_SUCCESS;
  }
  static AcCompInfo __attribute__((unused)) acInitCompInfo()
  {
	  AcCompInfo res;
	  memset(&res.is_loaded,0,sizeof(res.is_loaded));
	  return res;
  }
  static AcResult __attribute__((unused)) acLoadRealCompInfo(const AcRealCompParam param, const AcReal val, AcCompInfo* info)
  {
	  info->is_loaded.real_params[(int)param] = true;
	  info->config.real_params[(int)param] = val;
	  return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acLoadIntCompInfo(const AcIntCompParam param, const int val, AcCompInfo* info)
  {
	  info->is_loaded.int_params[(int)param] = true;
	  info->config.int_params[(int)param] = val;
	  return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acLoadReal3CompInfo(const AcReal3CompParam param, const AcReal3 val, AcCompInfo* info)
  {
	  info->is_loaded.real3_params[(int)param] = true;
	  info->config.real3_params[(int)param] = val;
	  return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acLoadInt3CompInfo(const AcInt3CompParam param, const int3 val, AcCompInfo* info)
  {
	  info->is_loaded.int3_params[(int)param] = true;
	  info->config.int3_params[(int)param] = val;
	  return AC_SUCCESS;
  }

  static AcResult __attribute__((unused)) acLoadRealArrayCompInfo(const AcRealCompArrayParam param, const AcReal* val, AcCompInfo* info)
  {
	  info->is_loaded.real_arrays[(int)param] = true;
	  info->config.real_arrays[(int)param] = val;
	  return AC_SUCCESS;
  }

  static AcResult __attribute__((unused)) acLoadIntArrayCompInfo(const AcIntCompArrayParam param, const int* val, AcCompInfo* info)
  {
	  info->is_loaded.int_arrays[(int)param] = true;
	  info->config.int_arrays[(int)param] = val;
	  return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acLoadBoolCompInfo(const AcBoolCompParam param, const bool val, AcCompInfo* info)
  {
	  info->is_loaded.bool_params[(int)param] = true;
	  info->config.bool_params[(int)param] = val;
	  return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acLoadBoolArrayCompInfo(const AcBoolCompArrayParam param, const bool* val, AcCompInfo* info)
  {
	  info->is_loaded.bool_arrays[(int)param] = true;
	  info->config.bool_arrays[(int)param] = val;
	  return AC_SUCCESS;
  }

#if __cplusplus
#define GEN_LOAD_COMP_INFO(PARAM_TYPE,VAL_TYPE,TYPE) \
  static AcResult __attribute__((unused)) acLoadCompInfo(const PARAM_TYPE param, const VAL_TYPE val, AcCompInfo* info) {return acLoad##TYPE##CompInfo(param,val,info);};

  GEN_LOAD_COMP_INFO(AcBoolCompParam, bool, Bool)
  GEN_LOAD_COMP_INFO(AcIntCompParam,  int, Int)
  GEN_LOAD_COMP_INFO(AcInt3CompParam, int3, Int3)
  GEN_LOAD_COMP_INFO(AcRealCompParam, AcReal,Real)
  GEN_LOAD_COMP_INFO(AcReal3CompParam,AcReal3,Real3)
  GEN_LOAD_COMP_INFO(AcRealCompArrayParam,AcReal*,RealArray)
  GEN_LOAD_COMP_INFO(AcIntCompArrayParam,int*,IntArray)
  GEN_LOAD_COMP_INFO(AcBoolCompArrayParam,bool*,BoolArray)
#endif
#endif

  #ifdef __cplusplus
  } // extern "C"
    //
    //
#include <type_traits>
  template <typename P>
  static bool
  is_dconst(const P array)
  {
	  constexpr const bool* dconst_arr =
		  std::is_same<P,AcRealArrayParam>::value  ? real_array_is_dconst :
		  std::is_same<P,AcIntArrayParam>::value   ? int_array_is_dconst  :
		  std::is_same<P,AcBoolArrayParam>::value  ? bool_array_is_dconst :
		  std::is_same<P,AcReal3ArrayParam>::value ? real3_array_is_dconst :
		  std::is_same<P,AcInt3ArrayParam>::value  ? real3_array_is_dconst : 
		  NULL;
	  return dconst_arr[(int)array];
  }

  template <typename P>
  static int
  get_array_length(const P array, const AcMeshInfo host_info)
  {
	  constexpr const int* length_arr=
		  std::is_same<P,AcRealArrayParam>::value  ? real_array_lengths:
		  std::is_same<P,AcIntArrayParam>::value   ? int_array_lengths:
		  std::is_same<P,AcBoolArrayParam>::value  ? bool_array_lengths:
		  std::is_same<P,AcReal3ArrayParam>::value ? real3_array_lengths:
		  std::is_same<P,AcInt3ArrayParam>::value  ? real3_array_lengths: 
		  NULL;
	  static_assert(length_arr != NULL);
	  if(is_dconst(array))
		  return length_arr[(int)array];
	  return host_info.int_params[length_arr[(int)array]];
  }

  template <typename P>
  static int
  get_dconst_array_length(const P array)
  {
	  constexpr const int* length_arr=
		  std::is_same<P,AcRealArrayParam>::value  ? real_array_lengths:
		  std::is_same<P,AcIntArrayParam>::value   ? int_array_lengths:
		  std::is_same<P,AcBoolArrayParam>::value  ? bool_array_lengths:
		  std::is_same<P,AcReal3ArrayParam>::value ? real3_array_lengths:
		  std::is_same<P,AcInt3ArrayParam>::value  ? real3_array_lengths: 
		  NULL;
	  static_assert(length_arr != NULL);
	  return length_arr[(int)array];
  }

  template <typename P>
  static int
  get_dconst_array_offset(const P array)
  {
	  constexpr const int* offset_arr =
		  std::is_same<P,AcRealArrayParam>::value  ? d_real_array_offsets:
		  std::is_same<P,AcIntArrayParam>::value   ? d_int_array_offsets:
		  std::is_same<P,AcBoolArrayParam>::value  ? d_bool_array_offsets:
		  std::is_same<P,AcReal3ArrayParam>::value ? d_real3_array_offsets:
		  std::is_same<P,AcInt3ArrayParam>::value  ? d_real3_array_offsets: 
		  NULL;
	  if(offset_arr == NULL)
	  {
		  fprintf(stderr,"%s\n","FATAL ERROR: missing array type\n");
		  exit(EXIT_FAILURE);
	  }
	  return offset_arr[(int)array];
  }

  template <typename P>
  static int
  get_num_params()
  {
	  constexpr const int res=
		  std::is_same<P,AcRealParam>::value    ? NUM_REAL_PARAMS:
		  std::is_same<P,AcIntParam>::value     ? NUM_INT_PARAMS:
		  std::is_same<P,AcBoolParam>::value    ? NUM_BOOL_PARAMS:
		  std::is_same<P,AcReal3Param>::value   ? NUM_REAL3_PARAMS:
		  std::is_same<P,AcInt3Param>::value    ? NUM_INT3_PARAMS:

		  std::is_same<P,AcRealArrayParam>::value  ? NUM_REAL_ARRAYS:
		  std::is_same<P,AcIntArrayParam>::value   ? NUM_INT_ARRAYS:
		  std::is_same<P,AcBoolArrayParam>::value  ? NUM_BOOL_ARRAYS:
		  std::is_same<P,AcReal3ArrayParam>::value ? NUM_REAL_ARRAYS:
		  std::is_same<P,AcInt3ArrayParam>::value  ? NUM_INT_ARRAYS: 
		  -1;

	  static_assert(res >= 0);
	  return res;
  }

  template <typename T>
  auto
  get_config_params(const AcMeshInfo& config)
  {
	  if constexpr(std::is_same<T,AcRealParam>::value)
	  	return config.real_params;
	  else if constexpr(std::is_same<T,AcIntParam>::value)
	  	return config.int_params;
	  else if constexpr(std::is_same<T,AcBoolParam>::value)
	  	return config.bool_params;
	  else if constexpr(std::is_same<T,AcReal3Param>::value)
	  	return config.real3_params;
	  else if constexpr(std::is_same<T,AcInt3Param>::value)
	  	return config.int3_params;
  }	  
  
  template <typename T>
  auto
  get_config_arrays(const AcMeshInfo& config)
  {
	  if constexpr(std::is_same<T,AcRealArrayParam>::value)
	  	return config.real_arrays;
	  else if constexpr(std::is_same<T,AcIntArrayParam>::value)
	  	return config.int_arrays;
	  else if constexpr(std::is_same<T,AcBoolArrayParam>::value)
	  	return config.bool_arrays;
	  else if constexpr(std::is_same<T,AcReal3ArrayParam>::value)
	  	return config.real3_arrays;
	  else if constexpr(std::is_same<T,AcInt3ArrayParam>::value)
	  	return config.int3_arrays;
  }	  

  template <typename T>
  auto
  get_vba_arrays(VertexBufferArray& vba)
  {
	  if constexpr(std::is_same<T,AcRealArrayParam>::value)
	  	return vba.real_arrays;
	  else if constexpr(std::is_same<T,AcIntArrayParam>::value)
	  	return vba.int_arrays;
	  else if constexpr(std::is_same<T,AcBoolArrayParam>::value)
	  	return vba.bool_arrays;
	  else if constexpr(std::is_same<T,AcReal3ArrayParam>::value)
	  	return vba.real3_arrays;
	  else if constexpr(std::is_same<T,AcInt3ArrayParam>::value)
	  	return vba.int3_arrays;
  }	  
  

#ifndef AC_RUNTIME_SOURCE
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcRealParam param, const AcReal value)          { return BASE_FUNC_NAME(acLoadRealUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcIntParam param, const int value)              { return BASE_FUNC_NAME(acLoadIntUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcBoolParam param, const int value)             { return BASE_FUNC_NAME(acLoadBoolUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcReal3Param param, const AcReal3 value)        { return BASE_FUNC_NAME(acLoadReal3Uniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcInt3Param param, const int3 value)            { return BASE_FUNC_NAME(acLoadInt3Uniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcRealArrayParam param, const AcReal* values)   { return BASE_FUNC_NAME(acLoadRealArrayUniform)(stream,param,values);}

  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcIntArrayParam param, const int* values)       { return BASE_FUNC_NAME(acLoadIntArrayUniform)(stream,param,values);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcBoolArrayParam param, const bool* values)     { return BASE_FUNC_NAME(acLoadBoolArrayUniform)(stream,param,values);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcReal3ArrayParam param, const AcReal3* values) { return BASE_FUNC_NAME(acLoadReal3ArrayUniform)(stream,param,values);}
  
  static AcResult __attribute__((unused))
  acLoadUniform(const cudaStream_t stream, const AcInt3ArrayParam param, const int3* values)     { return BASE_FUNC_NAME(acLoadInt3ArrayUniform)(stream,param,values);}

  static AcResult __attribute__((unused))
  acStoreUniform(const cudaStream_t stream, const AcRealParam param, AcReal* value)         { return BASE_FUNC_NAME(acStoreRealUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acStoreUniform(const cudaStream_t stream, const AcIntParam param,  int* value)                 { return BASE_FUNC_NAME(acStoreIntUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acStoreUniform(const cudaStream_t stream, const AcBoolParam param, bool* value)                 { return BASE_FUNC_NAME(acStoreBoolUniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acStoreUniform(const cudaStream_t stream, const AcReal3Param param,AcReal3* value)             { return BASE_FUNC_NAME(acStoreReal3Uniform)(stream,param,value);}
  
  static AcResult __attribute__((unused))
  acStoreUniform(const cudaStream_t stream, const AcInt3Param param, int3* value)                { return BASE_FUNC_NAME(acStoreInt3Uniform)(stream,param,value);}

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
  	AcRealParam,
	AcIntParam,
	AcBoolParam,
	AcInt3Param,
	AcReal3Param
  >;

  using AcArrayTypes = ForEach<
	AcRealArrayParam,
	AcIntArrayParam,
	AcBoolArrayParam,
	AcInt3ArrayParam,
	AcReal3ArrayParam
  >;

#endif
  #endif
