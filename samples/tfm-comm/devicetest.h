#pragma once

#if defined(CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "hip.h"
#include <hip/hip_runtime.h>
#endif

template <typename T>
void call_device(const cudaStream_t stream, const size_t count, const T* in, T* out);

extern template void call_device<double>(const cudaStream_t, const size_t, const double*, double*);
