#pragma once
#include "datatypes.h"

#if defined(CUDA_ENABLED)
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
#endif

#include "math_utils.h"

constexpr size_t PACK_MAX_INPUTS         = 27;
template <typename T> using PackPtrArray = StaticArray<T, PACK_MAX_INPUTS>;

template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const PackPtrArray<T*>& inputs, T* output);

template <typename T>
void unpack(const T* input, const Shape& mm, const Shape& block_shape, const Index& block_offset,
            PackPtrArray<T*>& outputs);

extern template void pack<AcReal>(const Shape&, const Shape&, const Index&,
                                  const PackPtrArray<AcReal*>&, AcReal*);

extern template void unpack<AcReal>(const AcReal*, const Shape&, const Shape&, const Index&,
                                    PackPtrArray<AcReal*>&);
