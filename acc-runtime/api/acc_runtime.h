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

  #include "builtin_enums.h"
  #include "datatypes.h"
  #include "errchk.h"

#define ONE_DIMENSIONAL_PROFILE (1 << 20)
#define TWO_DIMENSIONAL_PROFILE (1 << 21)
typedef enum {
	PROFILE_X  = (1 << 0) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Y  = (1 << 1) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Z  = (1 << 2) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_XY = (1 << 3) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_XZ = (1 << 4) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YX = (1 << 5) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YZ = (1 << 6) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZX = (1 << 7) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZY = (1 << 8) | TWO_DIMENSIONAL_PROFILE,
} AcProfileType;

  //copied from the sample setup
#ifdef __cplusplus
#define CONSTEXPR constexpr
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define CONSTEXPR
#define MAYBE_UNUSED
#endif
  #include "user_defines.h"
  #include "profiles_info.h"
  #include "user_built-in_constants.h"
  #include "user_builtin_non_scalar_constants.h"
  #include "func_attributes.h"

#ifdef __cplusplus


HOST_DEVICE_INLINE Field3 
MakeField3(const Field& x, const Field& y, const Field& z)
{
	return (Field3){x,y,z};
}
template <size_t N>
HOST_DEVICE_INLINE
AcArray<Field3,N>
MakeField3(const AcArray<Field,N>& x, const AcArray<Field,N>& y, const AcArray<Field,N>& z)
{
	AcArray<Field3,N> res{};
	for(size_t i = 0; i < N; ++i)
		res[i] = (Field3){x[i],y[i],z[i]};
	return res;
}
#endif
#define N_DIMS (3)
#define X_ORDER_INT (0)
#define Y_ORDER_INT (1)
#define Z_ORDER_INT (2)

typedef enum {
	XYZ = X_ORDER_INT + N_DIMS*Y_ORDER_INT + N_DIMS*N_DIMS*Z_ORDER_INT,
	XZY = X_ORDER_INT + N_DIMS*Z_ORDER_INT + N_DIMS*N_DIMS*Y_ORDER_INT,
	YXZ = Y_ORDER_INT + N_DIMS*X_ORDER_INT + N_DIMS*N_DIMS*Z_ORDER_INT,
	YZX = Y_ORDER_INT + N_DIMS*Z_ORDER_INT + N_DIMS*N_DIMS*X_ORDER_INT,
	ZXY = Z_ORDER_INT + N_DIMS*X_ORDER_INT + N_DIMS*N_DIMS*Y_ORDER_INT,
	ZYX = Z_ORDER_INT + N_DIMS*Y_ORDER_INT + N_DIMS*N_DIMS*X_ORDER_INT,
} AcMeshOrder;


typedef enum KernelReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} KernelReduceOp;
typedef struct {
  int variable;
  AcType type;
  bool called;
} KernelReduceOutput;
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


  typedef struct AcCompInfoLoaded {
#include "comp_loaded_decl.h"
#ifdef __cplusplus
#include "loaded_info_access_operators.h"
#endif
  } AcCompInfoLoaded;

  typedef struct AcCompInfoConfig{
#include "comp_decl.h"
#ifdef __cplusplus
#include "comp_info_access_operators.h"
#endif
  } AcCompInfoConfig;

  typedef struct {
	  AcCompInfoConfig config;
	  AcCompInfoLoaded is_loaded;
  } AcCompInfo;

  #ifdef __cplusplus
#include "is_comptime_param.h"
#endif

  typedef struct AcMeshInfo{
#include "array_decl.h"
#include "device_mesh_info_decl.h"
#if AC_MPI_ENABLED
    MPI_Comm comm;
#endif
#ifdef __cplusplus
#include "info_access_operators.h"
#endif
    AcCompInfo run_consts;
  } AcMeshInfo;

  typedef struct {
#include "input_decl.h"
  } AcInputs;

typedef struct {
  AcReal* in[NUM_PROFILES];
  AcReal* out[NUM_PROFILES];
  size_t count;
} ProfileBufferArray;


  typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
    AcReal* w[NUM_WORK_BUFFERS];
    size_t bytes;
    size_t mx, my, mz;
    acKernelInputParams kernel_input_params;
    AcReal* reduce_scratchpads_real[NUM_REAL_SCRATCHPADS][NUM_REDUCE_SCRATCHPADS];
    int* reduce_scratchpads_int[NUM_INT_OUTPUTS+1][NUM_REDUCE_SCRATCHPADS];
    int reduce_offset;
    size_t scratchpad_size;
    ProfileBufferArray profiles;
  } VertexBufferArray;


  typedef struct
  {
  	int read_fields[NUM_KERNELS][NUM_ALL_FIELDS];
  	int field_has_stencil_op[NUM_KERNELS][NUM_ALL_FIELDS];
  	int written_fields[NUM_KERNELS][NUM_ALL_FIELDS];
	int read_profiles[NUM_KERNELS][NUM_PROFILES];
	int reduced_profiles[NUM_KERNELS][NUM_PROFILES];
	int written_profiles[NUM_KERNELS][NUM_PROFILES];

  } KernelAnalysisInfo;

  typedef struct {
    size_t x, y, z, w;
  } AcShape;
  typedef AcShape AcIndex;
  
  typedef struct AcBuffer{
      AcReal* data;
      size_t count;
      bool on_device;
      AcShape shape;
#ifdef __cplusplus
      const AcReal& operator[](const int index) {return data[index];}
#endif
  } AcBuffer;

  typedef struct
  {
  	bool larger_input;
  	bool larger_output;
  } acAnalysisBCInfo;




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

#if AC_MPI_ENABLED
   FUNC_DEFINE(AcResult, acInitializeRuntimeMPI, (const MPI_Comm comm));
#endif

  FUNC_DEFINE(AcResult, acTranspose,(const AcMeshOrder order, const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream));
  FUNC_DEFINE(const AcKernel*, acGetKernels,());
  FUNC_DEFINE(AcResult, acKernelFlush,(const cudaStream_t stream, AcReal* arr, const size_t n, const AcReal value));
  FUNC_DEFINE(AcResult, acKernelFlushInt,(const cudaStream_t stream, int* arr, const size_t n, const int value));

  FUNC_DEFINE(AcResult, acVBAReset,(const cudaStream_t stream, VertexBufferArray* vba));

  FUNC_DEFINE(VertexBufferArray, acVBACreate,(const AcMeshInfo config));

  FUNC_DEFINE(void, acUpdateArrays ,(const AcMeshInfo config));

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
	*(void**)(&acUpdateArrays) = dlsym(handle,"acUpdateArrays");
	if(!acUpdateArrays) fprintf(stderr,"Astaroth error: was not able to load %s\n","acUpdateArrays");
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

  template <typename P, typename V>
  void
  acPushToConfig(AcMeshInfo& config, P param, V val)
  {
	  static_assert(!std::is_same<P,int>::value);
          if constexpr(IsCompParam(param))
	  {
	  	  config.run_consts.config[param] = val;
	  	  config.run_consts.is_loaded[param] = true;
	  }
          else
		  config[param] = val;
  }

  #endif
  static AcCompInfo __attribute__((unused)) acInitCompInfo()
  {
	  AcCompInfo res;
	  memset(&res.is_loaded,0,sizeof(res.is_loaded));
	  return res;
  }
  static AcMeshInfo __attribute__((unused)) acInitInfo()
  {
	  AcMeshInfo res;
    	  // memset reads the second parameter as a byte even though it says int in
          // the function declaration
    	  memset(&res, (uint8_t)0xFF, sizeof(res));
    	  //these are set to nullpointers for the users convenience that the user doesn't have to set them to null elsewhere
    	  //if they are present in the config then they are initialized correctly
    	  memset(res.real_arrays, 0,NUM_REAL_ARRAYS *sizeof(AcReal*));
    	  memset(res.int_arrays,  0,NUM_INT_ARRAYS  *sizeof(int*));
    	  memset(res.bool_arrays, 0,NUM_BOOL_ARRAYS *sizeof(bool*));
    	  memset(res.int3_arrays, 0,NUM_INT3_ARRAYS *sizeof(int*));
    	  memset(res.real3_arrays,0,NUM_REAL3_ARRAYS*sizeof(int*));

#if AC_MPI_ENABLED
	  res.comm = MPI_COMM_NULL;
#endif
	  res.run_consts = acInitCompInfo();
	  return res;
  }
#include "load_comp_info.h"

void acVBASwapBuffer(const Field field, VertexBufferArray* vba);

void acVBASwapBuffers(VertexBufferArray* vba);

void acPBASwapBuffer(const Profile profile, VertexBufferArray* vba);

void acPBASwapBuffers(VertexBufferArray* vba);

void acLoadMeshInfo(const AcMeshInfo info, const cudaStream_t stream);


// Returns the number of elements contained within shape
size_t acShapeSize(const AcShape shape);

AcResult acReindex(const cudaStream_t stream, //
                   const AcReal* in, const AcIndex in_offset,
                   const AcIndex in_shape, //
                   AcReal* out, const AcIndex out_offset,
                   const AcIndex out_shape, const AcShape block_shape);

AcResult acReindexCross(const cudaStream_t stream, //
                        const VertexBufferArray vba, const AcIndex in_offset,
                        const AcShape in_shape, //
                        AcReal* out, const AcIndex out_offset,
                        const AcShape out_shape, const AcShape block_shape);

AcResult acSegmentedReduce(const cudaStream_t stream, const AcReal* d_in,
                           const size_t count, const size_t num_segments,
                           AcReal* d_out);

AcResult
acReduce(const cudaStream_t stream, const AcReal* d_in, const size_t count, AcReal* d_out);

AcResult acMultiplyInplace(const AcReal value, const size_t count,
                           AcReal* array);

  FUNC_DEFINE(AcResult, acPBAReset,(const cudaStream_t stream, ProfileBufferArray* pba, const size3_t counts));

  FUNC_DEFINE(ProfileBufferArray, acPBACreate,(const size3_t count));

  FUNC_DEFINE(void, acPBADestroy,(ProfileBufferArray* pba));

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

  static UNUSED const char* 
  get_name(const AcRealOutputParam& param)
  {
          return real_output_names[param];
  }
  static UNUSED const char* 
  get_name(const AcIntOutputParam& param)
  {
          return int_output_names[param];
  }
  
  static UNUSED const char* 
  get_name(const Profile& param)
  {
          return profile_names[param];
  }

  static UNUSED const char* 
  get_name(const Field& param)
  {
          return field_names[param];
  }

  template <typename P>
  constexpr static const char*
  get_name(const P array)
  {
	  return get_array_info(array).name;
  }

  template <typename P>
  constexpr static bool
  is_dconst(const P array)
  {
	  return get_array_info(array).is_dconst;
  }

  template <typename P>
  constexpr static bool
  is_alive(const P array)
  {
	  return get_array_info(array).is_alive;
  }


  template <typename P>
  constexpr static auto
  get_array_dims(const P array)
  {
	  return get_array_info(array).dims;
  }

  template <typename P>
  constexpr static bool
  has_const_dims(const P array)
  {
	  auto dims = get_array_dims(array);
	  int num_dims         = get_array_info(array).num_dims;
	  bool res = true;
	  for(int i = 0; i < num_dims; ++i)
		  res &= !dims.from_config[i];
	  return res;
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
  constexpr static int
  get_const_dims_array_length(const P array)
  {
	  AcArrayDims dims     = get_array_info(array).dims;
	  int num_dims         = get_array_info(array).num_dims;
	  int res = 1;
	  for(int i = 0; i < num_dims; ++i)
		  res *= dims.len[i];
	  return res;
  }

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
  get_empty_pointer(const P param)
  {
   (void)param;
#include "get_empty_pointer.h"
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

static UNUSED size_t
prof_count(const Profile prof, const size3_t counts)
{
    if(NUM_PROFILES == 0) return 0;
    return 
	    	prof_types[prof] == PROFILE_X  ? counts.x  :
	    	prof_types[prof] == PROFILE_Y  ? counts.y  :
	    	prof_types[prof] == PROFILE_Z  ? counts.z  :
	    	prof_types[prof] == PROFILE_XY ? counts.x*counts.y  :
	    	prof_types[prof] == PROFILE_XZ ? counts.x*counts.z  :
	    	prof_types[prof] == PROFILE_YX ? counts.y*counts.x  :
	    	prof_types[prof] == PROFILE_YZ ? counts.y*counts.z  :
	    	prof_types[prof] == PROFILE_ZX ? counts.z*counts.x  :
	    	prof_types[prof] == PROFILE_ZY ? counts.z*counts.y  :
		0;
}
static UNUSED size_t
prof_size(const Profile prof, const size3_t counts)
{
    return prof_count(prof,counts)*sizeof(AcReal);
}
