#pragma once
#pragma GCC system_header // NOTE: Silences errors originating from CUDA/HIP
                          // headers

#if AC_CPU_BUILD

#include <limits.h>
#include <cstdint>
typedef int cudaStream_t;
typedef int cudaError_t;
typedef int cudaSharedMemConfig;
typedef int cudaStream_t;
typedef int cudaMemcpyKind;
typedef int cudaEvent_t;
typedef struct
{
	int warpSize;
	int multiProcessorCount;
	int maxThreadsPerBlock;
	int sharedMemPerBlock;
	const char* name;
	int major,minor;
	double clockRate;
	int singleToDoublePrecisionPerfRatio;
	int computeMode;
	int memoryClockRate;
	int memoryBusWidth;
	int   ECCEnabled;
	double totalGlobalMem;
	int localL1CacheSupported;
	int globalL1CacheSupported;
	int l2CacheSize;
	int regsPerBlock;
	int streamPrioritiesSupported;
} cudaDeviceProp; 
typedef int cudaDeviceAttribute_t;
#define cudaStreamNonBlocking (0)

#define cudaSharedMemBankSizeEightByte (0)

#define cudaSuccess (0)
#define cudaFailure (1)
#define cudaErrorInvalidConfiguration (2)
#define cudaErrorLaunchOutOfResources (3)
#define cudaErrorNotReady (4)

#define cudaMemcpyDeviceToHost   (0)
#define cudaMemcpyHostToDevice   (1)
#define cudaMemcpyDeviceToDevice (2)
#define cudaMemcpyHostToHost     (3)
#define cudaMemcpyDefault        (4)
#define cudaDevAttrCooperativeLaunch (0)

#undef __host__
#define __host__
#undef __forceinline__
#define __forceinline__
#undef __device__
#define __device__
#undef __global__
#define __global__
#undef __launch_bounds__
#define __launch_bounds__(x)
#undef __syncthreads
#define __syncthreads()
#undef __shared__
#define __shared__
#undef __constant__
#define __constant__

#define blockIdx ((dim3){0, 0, 0})
#define blockDim ((dim3){1, 1, 1})
#define gridDim ((dim3){1, 1, 1})
#define make_int3(x, y, z) ((int3){x, y, z})
#define make_float3(x, y, z) ((float3){x, y, z})
#define make_double3(x, y, z) ((double3){x, y, z})
#define KERNEL_LAUNCH(func,bgp,tpb,...) \
	func

#define KERNEL_VBA_LAUNCH(func,bgp,tpb,...) \
	func

#else 

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime_api.h>     // Streams
#if PROFILING_ENABLED
#include <roctracer/roctracer_ext.h> // Profiling
#endif
#else
#if PROFILING_ENABLED
#include <cuda_profiler_api.h> // Profiling
#endif
#include <cuda_runtime_api.h>  // Streams
#endif

#define KERNEL_LAUNCH(func,bgp,tpb,...) \
	func<<<bpg,tpb,__VA_ARGS__>>>

#define KERNEL_VBA_LAUNCH(func,bgp,tpb,...) \
	func<<<bpg,tpb,__VA_ARGS__>>>

#endif
