#pragma once

// Types
using cudaError_t  = hipError_t;
using cudaStream_t = hipStream_t;

// Constants
constexpr auto cudaStreamDefault{hipStreamDefault};
constexpr auto cudaSuccess{hipSuccess};

// Functions
constexpr auto cudaDeviceSynchronize{hipDeviceSynchronize};
constexpr auto cudaFree{hipFree};
constexpr auto cudaFreeHost{hipHostFree};
constexpr auto cudaGetDeviceCount{hipGetDeviceCount};
constexpr auto cudaGetErrorString{hipGetErrorString};
constexpr auto cudaMemcpyDeviceToDevice{hipMemcpyDeviceToDevice};
constexpr auto cudaMemcpyDeviceToHost{hipMemcpyDeviceToHost};
constexpr auto cudaMemcpyHostToDevice{hipMemcpyHostToDevice};
constexpr auto cudaMemcpyHostToHost{hipMemcpyHostToHost};
constexpr auto cudaPeekAtLastError{hipPeekAtLastError};
constexpr auto cudaSetDevice{hipSetDevice};
constexpr auto cudaStreamCreate{hipStreamCreate};
constexpr auto cudaStreamCreateWithFlags{hipStreamCreateWithFlags};
constexpr auto cudaStreamDestroy{hipStreamDestroy};
constexpr auto cudaStreamSynchronize{hipStreamSynchronize};

// Overloaded functions cannot be aliased
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyAsync hipMemcpyAsync
