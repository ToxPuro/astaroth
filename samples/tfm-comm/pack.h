#pragma once
#include "buffer.h"
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

template <typename T, typename MemoryResource>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const PackPtrArray<T*>& inputs, Buffer<T, MemoryResource>& output);

template <typename T, typename MemoryResource>
void unpack(const Buffer<T, MemoryResource>& input, const Shape& mm, const Shape& block_shape,
            const Index& block_offset, PackPtrArray<T*>& outputs);

extern template void pack<AcReal, HostMemoryResource>(const Shape&, const Shape&, const Index&,
                                                      const PackPtrArray<AcReal*>&,
                                                      Buffer<AcReal, HostMemoryResource>&);

extern template void unpack<AcReal, HostMemoryResource>(const Buffer<AcReal, HostMemoryResource>&,
                                                        const Shape&, const Shape&, const Index&,
                                                        PackPtrArray<AcReal*>&);

extern template void pack<AcReal, DeviceMemoryResource>(const Shape&, const Shape&, const Index&,
                                                        const PackPtrArray<AcReal*>&,
                                                        Buffer<AcReal, DeviceMemoryResource>&);

extern template void
unpack<AcReal, DeviceMemoryResource>(const Buffer<AcReal, DeviceMemoryResource>&, const Shape&,
                                     const Shape&, const Index&, PackPtrArray<AcReal*>&);
