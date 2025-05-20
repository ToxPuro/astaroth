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

  #define AC_INDEX_ORDER(i,j,k,x,y,z) \
  	i + x*j + x*y*k

  #include <stdio.h>
  #include <stdbool.h>


  #if AC_USE_HIP
  #include "hip.h"

  #include <hip/hip_runtime_api.h> // Streams
  #include <roctracer/roctracer_ext.h>       // Profiling
  #else
  #include <cuda_profiler_api.h> // Profiling
  #include <cuda_runtime_api.h>  // Streams
  #endif

  #include "builtin_enums.h"
  #include "datatypes.h"
  #include "errchk.h"

#define AC_SIZE(arr) sizeof(arr)/sizeof(arr[0])
#define ONE_DIMENSIONAL_PROFILE (1 << 20)
#define TWO_DIMENSIONAL_PROFILE (1 << 21)
typedef enum {
	PROFILE_NONE = 0,
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
  //#include "user_builtin_non_scalar_constants.h"
  #include "func_attributes.h"
#include "device_headers.h"

#ifdef __cplusplus


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


typedef enum AcReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} AcReduceOp;

typedef struct {
    size3_t n0, n1;
    size3_t m0, m1;
    size3_t nn;
    size3_t reduction_tile;
} AcMeshDims;


typedef struct KernelReduceOutput {
  int variable;
  AcType type;
  AcReduceOp op;
  AcKernel kernel;
#ifdef __cplusplus
  bool operator==(const KernelReduceOutput& other) const
  {
	  return (variable == other.variable) &&
		 (type == other.type) &&
		 (op == other.op) &&
		 (kernel == other.kernel);
  }
#endif
} KernelReduceOutput;

typedef enum {
	AC_NO_REDUCE_POST_PROCESSING,
	AC_RMS,
	AC_RADIAL_WINDOW_RMS
} AcReductionPostProcessingOp;

typedef struct
{
	const AcReduceOp reduce_op;
	const AcReductionPostProcessingOp post_processing_op;
	const AcKernel map_vtxbuf_single;
	const AcKernel map_vtxbuf_vec;
	const AcKernel map_vtxbuf_vec_scal;
	const char* name;
} AcReduction;

  #include "user_input_typedefs.h"

#if AC_RUNTIME_COMPILATION
  #include <dlfcn.h>
#endif

  #define NUM_REDUCE_SCRATCHPADS (2)


  typedef struct {
    Volume nn;
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

  typedef struct AcCompInfoHasDefaultValue {
#include "comp_loaded_decl.h"
#ifdef __cplusplus
#include "loaded_info_access_operators.h"
#endif
  } AcCompInfoHasDefaultValue;

  typedef struct AcCompInfoConfig{
#include "comp_decl.h"
#ifdef __cplusplus
#include "comp_info_access_operators.h"
#endif
  } AcCompInfoConfig;

  typedef struct {
	  AcCompInfoConfig config;
	  AcCompInfoLoaded is_loaded;
	  AcCompInfoHasDefaultValue has_default_value;
  } AcCompInfo;

  #ifdef __cplusplus
#include "is_comptime_param.h"
#include "is_array_param.h"
#include "is_output_param.h"
#endif

  typedef struct AcMeshInfoLoaded {
#include "info_loaded_decl.h"

#ifdef __cplusplus
#include "info_loaded_operator_decl.h"
#endif
  } AcMeshInfoLoadedInfo;


  typedef struct AcMeshInfoScalars
  {
#include "device_mesh_info_decl.h"
  } AcMeshInfoScalars;

//TP: opaque pointer for the MPI comm to enable having the opaque type in modules which do not about MPI_Comm
typedef struct AcCommunicator AcCommunicator;

  typedef struct AcMeshInfo{

#include "device_mesh_info_decl.h"
#include "array_decl.h"

  AcMeshInfoLoadedInfo is_loaded;
  const char* runtime_compilation_log_dst;
  const char* runtime_compilation_build_path;
  const char* runtime_compilation_base_path;
  AcCommunicator* comm;
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
} ProfileBufferArray;



  typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
    AcReal* w[NUM_WORK_BUFFERS];
    acKernelInputParams kernel_input_params;
    int reduce_offset;
    ProfileBufferArray profiles;
    int3 block_factor;
  } DeviceVertexBufferArray;

  typedef struct
  {
  	AcReduceOp reals[NUM_REAL_SCRATCHPADS+1];	
  	AcReduceOp ints[NUM_INT_OUTPUTS+1];	
#if AC_DOUBLE_PRECISION
  	AcReduceOp floats[NUM_FLOAT_OUTPUTS+1];	
#endif
  } AcScratchpadStates;

  typedef struct {
    size_t x, y, z, w;
  } AcShape;

  typedef struct AcBuffer{
      AcReal* data;
      size_t count;
      bool on_device;
      AcShape shape;
#ifdef __cplusplus
      const AcReal& operator[](const int index) {return data[index];}
#endif
  } AcBuffer;

#include "scalar_reduce_buffer_defs.h"

  typedef struct 
  {
	  AcBuffer src;
	  AcBuffer transposed;
	  AcMeshOrder mem_order;
	  AcReal** cub_tmp;
	  size_t* cub_tmp_size;
  } AcReduceBuffer;

  typedef struct {
    //Auxiliary metadata
    size_t counts[NUM_ALL_FIELDS];
    size_t bytes[NUM_ALL_FIELDS];
    AcMeshDims dims[NUM_ALL_FIELDS];
    AcMeshDims profile_dims[NUM_PROFILES+1];
    size_t profile_counts[NUM_PROFILES+1];
    AcMeshDims computational_dims;
    //All kernel parameters and memory allocated on the device
    DeviceVertexBufferArray on_device;
    size_t profile_count;

#include "scalar_reduce_buffers_in_vba.h"

    AcScratchpadStates* scratchpad_states;
    AcReduceBuffer profile_reduce_buffers[NUM_PROFILES];

  } VertexBufferArray;


  typedef struct
  {
	int read_fields[NUM_ALL_FIELDS];
	int field_has_stencil_op[NUM_ALL_FIELDS];
  	int stencils_accessed[NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS];
  	int written_fields[NUM_ALL_FIELDS];
	int read_profiles[NUM_PROFILES+1];
	int reduced_profiles[NUM_PROFILES+1];
	int written_profiles[NUM_PROFILES+1];
  	int profile_has_stencil_op[NUM_PROFILES+1];
	int incoming_ray_accessed[NUM_ALL_FIELDS][NUM_RAYS+1];
	KernelReduceOutput reduce_inputs[NUM_OUTPUTS+1];
	KernelReduceOutput reduce_outputs[NUM_OUTPUTS+1];
	size_t n_reduce_inputs;
	size_t n_reduce_outputs;
  } KernelAnalysisInfo;


  typedef AcShape AcIndex;
  

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
#define FUNC_DEFINE(return_type, func_name, ...) static UNUSED return_type (*func_name) __VA_ARGS__
#endif
#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) return_type func_name __VA_ARGS__
#endif

#ifndef BASE_FUNC_NAME 
#define BASE_FUNC_NAME(func_name) func_name
#endif

#endif

typedef struct
{
        float time;
        dim3 tpb;
} AcAutotuneMeasurement;

typedef AcAutotuneMeasurement (*AcMeasurementGatherFunc)(const AcAutotuneMeasurement);
  #ifdef __cplusplus
  extern "C" {
  #endif

  #include "user_declarations.h"

#if AC_MPI_ENABLED
   FUNC_DEFINE(AcResult, acInitializeRuntimeMPI,(const int grid_pid, const int nprocs, AcMeasurementGatherFunc));
#endif

  FUNC_DEFINE(AcResult, acTransposeWithBounds,(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream));
  FUNC_DEFINE(AcResult, acTranspose,(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const cudaStream_t stream));
  FUNC_DEFINE(const AcKernel*, acGetKernels,());
  FUNC_DEFINE(AcResult, acKernelFlush,(const cudaStream_t stream, AcReal* arr, const size_t n, const AcReal value));
  FUNC_DEFINE(AcResult, acKernelFlushInt,(const cudaStream_t stream, int* arr, const size_t n, const int value));

  FUNC_DEFINE(AcResult, acVBAReset,(const cudaStream_t stream, VertexBufferArray* vba));
  FUNC_DEFINE(size_t,acGetRealScratchpadSize,(const size_t i));

  FUNC_DEFINE(AcMeshOrder, acGetMeshOrderForProfile,(const AcProfileType type));
  FUNC_DEFINE(size3_t, acGetProfileReduceScratchPadDims,(const int profile, const AcMeshDims dims));

  FUNC_DEFINE(AcResult,acPreprocessScratchPad,(VertexBufferArray, const int variable, const AcType type,const AcReduceOp op));

  FUNC_DEFINE(VertexBufferArray, acVBACreate,(const AcMeshInfo config));

  FUNC_DEFINE(void, acUpdateArrays ,(const AcMeshInfo config));

  FUNC_DEFINE(void, acVBADestroy,(VertexBufferArray* vba, const AcMeshInfo config));

  FUNC_DEFINE(AcResult, acRandInitAlt,(const uint64_t seed, const size_t count, const size_t rank));

  FUNC_DEFINE(void, acRandQuit,(void));

  FUNC_DEFINE(AcResult, acLaunchKernel,(AcKernel func, const cudaStream_t stream, const Volume start, const Volume end, VertexBufferArray));

  FUNC_DEFINE(AcResult, acBenchmarkKernel,(AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba));

  FUNC_DEFINE(int3, acReadOptimTBConfig,(const AcKernel, const int3 dims, const int3 block_factors));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acLoadStencil,(const Stencil stencil, const cudaStream_t stream, const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
  FUNC_DEFINE(AcResult, acStoreStencil,(const Stencil stencil, const cudaStream_t stream, AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

  /** NOTE: stream unused. acUniform functions are completely synchronous. */
#include "load_and_store_uniform_header.h"

  // Diagnostics
  FUNC_DEFINE(Volume, acKernelLaunchGetLastTPB,(void));

  FUNC_DEFINE(AcKernel, acGetOptimizedKernel,(const AcKernel, const VertexBufferArray vba));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadSize,(const AcKernel kernel));

  FUNC_DEFINE(int, acGetKernelReduceScratchPadMinSize,());
  FUNC_DEFINE(size_t,  acGetSmallestRealReduceScratchPadSizeBytes,());

#if AC_RUNTIME_COMPILATION
#define LOAD_DSYM(FUNC_NAME,STREAM) *(void**)(&FUNC_NAME) = dlsym(handle,#FUNC_NAME); \
			     if(!FUNC_NAME && STREAM) fprintf(STREAM,"Astaroth error: was not able to load %s\n",#FUNC_NAME);
  static UNUSED void* acLoadRunTime(FILE* stream, const AcMeshInfo info)
  {
	char runtime_astaroth_runtime_path[40000];
	sprintf(runtime_astaroth_runtime_path,"%s/runtime_build/src/core/kernels/libkernels.so",info.runtime_compilation_build_path ? info.runtime_compilation_build_path : astaroth_binary_path);
 	void* handle = dlopen(runtime_astaroth_runtime_path,RTLD_LAZY | RTLD_GLOBAL);
	if(!handle)
	{
    		fprintf(stderr,"%s","Fatal error was not able to load Astaroth runtime\n"); 
		fprintf(stderr,"Error message: %s\n",dlerror());
		exit(EXIT_FAILURE);
	}
	LOAD_DSYM(acKernelFlush,stream);
	LOAD_DSYM(acVBAReset,stream);
	LOAD_DSYM(acVBACreate,stream);
	LOAD_DSYM(acUpdateArrays,stream);
	LOAD_DSYM(acVBADestroy,stream);
	LOAD_DSYM(acRandInitAlt,stream);
	LOAD_DSYM(acRandQuit,stream);
	LOAD_DSYM(acLaunchKernel,stream);
	LOAD_DSYM(acBenchmarkKernel,stream);
	LOAD_DSYM(acLoadStencil,stream);
	LOAD_DSYM(acStoreStencil,stream);
	LOAD_DSYM(acLoadRealUniform,stream);
	LOAD_DSYM(acLoadRealArrayUniform,stream);
	LOAD_DSYM(acLoadReal3Uniform,stream);
	LOAD_DSYM(acLoadIntUniform,stream)
	LOAD_DSYM(acLoadIntUniform,stream)
	LOAD_DSYM(acLoadIntArrayUniform,stream)
	LOAD_DSYM(acLoadBoolUniform,stream)
	LOAD_DSYM(acLoadIntArrayUniform,stream)
	LOAD_DSYM(acLoadInt3Uniform,stream)
	LOAD_DSYM(acStoreRealUniform,stream)
	LOAD_DSYM(acStoreReal3Uniform,stream)
	LOAD_DSYM(acStoreIntUniform,stream)
	LOAD_DSYM(acStoreBoolUniform,stream)
	LOAD_DSYM(acStoreInt3Uniform,stream)
	LOAD_DSYM(acKernelLaunchGetLastTPB,stream)
	LOAD_DSYM(acGetOptimizedKernel,stream)
	LOAD_DSYM(acGetKernelReduceScratchPadSize,stream)
	LOAD_DSYM(acGetKernelReduceScratchPadMinSize,stream)
	LOAD_DSYM(acGetKernels,stream)
	LOAD_DSYM(acReadOptimTBConfig,stream);

	return handle;
  }
#endif

  #ifdef __cplusplus
  } // extern "C"
    //
    //
#ifndef AC_RUNTIME_SOURCE
#include <type_traits>
#include <string.h>

#include "load_comp_info.h"

void acVBASwapBuffer(const Field field, VertexBufferArray* vba);

void acVBASwapBuffers(VertexBufferArray* vba);

void acPBASwapBuffer(const Profile profile, VertexBufferArray* vba);

void acPBASwapBuffers(VertexBufferArray* vba);

AcResult acLoadMeshInfo(const AcMeshInfo info, const cudaStream_t stream);


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
                           AcReal* d_out, AcReal** tmp, size_t* tmp_size);

FUNC_DEFINE(AcResult, acReduceReal,(const cudaStream_t stream, const AcReduceOp, AcRealScalarReduceBuffer, const size_t count));

FUNC_DEFINE(AcResult, acReduceInt,(const cudaStream_t stream, const AcReduceOp, AcIntScalarReduceBuffer, const size_t count));

#if AC_DOUBLE_PRECISION
FUNC_DEFINE(AcResult, acReduceFloat,(const cudaStream_t stream, const AcReduceOp, AcFloatScalarReduceBuffer, const size_t count));
#endif

FUNC_DEFINE(AcResult, acLoadRealReduceRes,(cudaStream_t stream, const AcRealOutputParam param, const AcReal* value));
FUNC_DEFINE(AcResult, acLoadIntReduceRes,(cudaStream_t stream, const AcIntOutputParam param, const int* value));

#if AC_DOUBLE_PRECISION
FUNC_DEFINE(AcResult, acLoadFloatReduceRes,(cudaStream_t stream, const AcFloatOutputParam param, const float* value));
#endif


static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcRealScalarReduceBuffer buffer, const size_t count)
{
	return acReduceReal(stream,op,buffer,count);
}

static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcIntScalarReduceBuffer buffer, const size_t count)
{
	return acReduceInt(stream,op,buffer,count);
}

#if AC_DOUBLE_PRECISION
static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcFloatScalarReduceBuffer buffer, const size_t count)
{
	return acReduceFloat(stream,op,buffer,count);
}
#endif


AcResult acMultiplyInplace(const AcReal value, const size_t count,
                           AcReal* array);

  FUNC_DEFINE(AcResult, acPBAReset,(const cudaStream_t stream, ProfileBufferArray* pba, const size3_t counts));

  FUNC_DEFINE(ProfileBufferArray, acPBACreate,(const size3_t count));

  FUNC_DEFINE(void, acPBADestroy,(ProfileBufferArray* pba));
/**
 * Checks the mesh info for uninitialized values.
 * Returns 0 on succes and -1 on failure.
 */
FUNC_DEFINE(int, acVerifyMeshInfo,(const AcMeshInfo info));

#ifdef __cplusplus

#define GEN_LOAD_COMP_INFO(PARAM_TYPE,VAL_TYPE,TYPE) \
  static AcResult __attribute__((unused)) acLoadCompInfo(const PARAM_TYPE param, const VAL_TYPE val, AcCompInfo* info) {return acLoad##TYPE##CompInfo(param,val,info);};
#include "load_comp_info_overloads.h"
#include "load_ac_kernel_params_def.h"

#endif

  template <typename P>
  constexpr static array_info
  get_array_info(const P array)
  {
#include "get_array_info.h"
	  ERRCHK_ALWAYS(false); //did not find array info
	  return (array_info){};
  }

  static UNUSED const char* 
  get_name(const AcRealOutputParam& param)
  {
	  if constexpr(NUM_REAL_OUTPUTS == 0) return "NO REAL OUTPUTS";
          return real_output_names[param];
  }
  static UNUSED const char* 
  get_name(const AcIntOutputParam& param)
  {
	  if constexpr(NUM_INT_OUTPUTS == 0) return "NO INT OUTPUTS";
	  else return int_output_names[param];
  }
#if AC_DOUBLE_PRECISION
  static UNUSED const char* 
  get_name(const AcFloatOutputParam& param)
  {
	  if constexpr(NUM_FLOAT_OUTPUTS == 0) return "NO FLOAT OUTPUTS";
	  else return float_output_names[param];
  }
#endif
  
  static UNUSED const char* 
  get_name(const Profile& param)
  {
	  if constexpr(NUM_PROFILES==0) return "NO_PROFILES";
	  else return profile_names[param];
  }

  static UNUSED const char* 
  get_name(const Field& param)
  {
          return field_names[param];
  }

  template <typename P>
  constexpr const char*
  get_array_name(const P array)
  {
	  return get_array_info(array).name;
  }

  template <typename P>
  constexpr const char*
  get_param_name(const P param)
  {
#include "get_param_name.h"
	  //ERRCHK_ALWAYS(false); //did not find name
	  return "NOT FOUND!";
  }


  template <typename P>
  constexpr static const char*
  get_name(const P param)
  {
	  if constexpr (IsArrayParam(param)) return get_array_name(param);
	  else return get_param_name(param);
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
		  res &= !dims[i].from_config;
	  return res;
  }

  template <typename P>
  constexpr static int
  get_array_n_dims(const P array)
  {
	  return get_array_info(array).num_dims;
  }

  template <typename P>
  constexpr static int
  get_const_dims_array_length(const P array)
  {
	  auto dims     = get_array_info(array).dims;
	  int num_dims         = get_array_info(array).num_dims;
	  int res = 1;
	  for(int i = 0; i < num_dims; ++i)
		  res *= dims[i].base;
	  return res;
  }

  template <typename P>
  constexpr static auto
  get_array_dim_sizes(const P array, const AcMeshInfo info)
  {
	  auto dims            = get_array_info(array).dims;
	  int num_dims         = get_array_info(array).num_dims;
	  std::array<size_t,20> res{};
	  for(int i = 0; i < num_dims; ++i)
	  {
		  if(!dims[i].from_config)
		  {
			  res[i] = dims[i].base;
			  continue;
		  }
		  if(dims[i].member == NULL)
		  {
			  res[i] = info.int_params[dims[i].base];
			  continue;
		  }
		  if(!strcmp(dims[i].member,"x"))
		  {
			  res[i] = info.int3_params[dims[i].base].x;
			  continue;
		  }
		  if(!strcmp(dims[i].member,"y"))
		  {
			  res[i] = info.int3_params[dims[i].base].y;
			  continue;
		  }
		  if(!strcmp(dims[i].member,"z"))
		  {
			  res[i] = info.int3_params[dims[i].base].z;
			  continue;
		  }
	  }
	  return res;
  }

  template <typename P>
  constexpr static size_t
  get_array_length(const P array, const AcMeshInfo info)
  {
	  auto sizes = get_array_dim_sizes(array,info);
	  size_t res = 1;
	  int num_dims         = get_array_info(array).num_dims;
	  for(int i = 0; i < num_dims; ++i)
		  res *= sizes[i];
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
