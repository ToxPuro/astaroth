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

// User types
constexpr size_t UserNdims = 2;
using UserShape            = ac::shape<UserNdims>;
using UserIndex            = ac::shape<UserNdims>;
using UserType             = double;

// Forward declarations
template <typename T, size_t ndims>
void pack(const ac::shape<ndims>& mm, const ac::shape<ndims>& block_shape,
          const ac::shape<ndims>& block_offset, const std::vector<T*>& inputs, T* output);

template <typename T, size_t ndims>
void unpack(const T* input, const ac::shape<ndims>& mm, const ac::shape<ndims>& block_shape,
            const ac::shape<ndims>& block_offset, std::vector<T*>& outputs);

// The actual kernel
extern template void pack<UserType, UserNdims>(const UserShape& mm, const UserShape& block_shape,
                                               const UserIndex& block_offset,
                                               const std::vector<UserType*>& inputs,
                                               UserType* output);

extern template void unpack(const UserType* input, const UserShape& mm,
                            const UserShape& block_shape, const UserShape& block_offset,
                            std::vector<UserType*>& outputs);

#if defined(DEVICE_ENABLED)

template <typename T, size_t N, size_t M>
void pack(const ac::shape<N>& mm, const ac::shape<N>& block_shape,
          const ac::shape<ndims>& block_offset, const ac::array<T*, M>& inputs,
          Buffer<T, DeviceMemoryResource>& output);

template <typename T, size_t N, size_t M>
void unpack(const Buffer<T, DeviceMemoryResource>& input, const ac::shape<N>& mm,
            const ac::shape<N>& block_shape, const ac::shape<ndims>& block_offset,
            ac::array<T*, M>& outputs);

extern template void
pack<AcReal, PACK_NDIMS, PACK_MAX_NAGGR_BUFS>(const ac::shape<PACK_NDIMS>& mm,
                                              const ac::shape<PACK_NDIMS>& block_shape,
                                              const ac::shape<PACK_NDIMS>& block_offset,
                                              const ac::array<AcReal*, PACK_MAX_NAGGR_BUFS>& inputs,
                                              Buffer<AcReal, DeviceMemoryResource>& output);

extern template void unpack<AcReal, PACK_NDIMS, PACK_MAX_NAGGR_BUFS>(
    const Buffer<AcReal, DeviceMemoryResource>& input, const ac::shape<PACK_NDIMS>& mm,
    const ac::shape<PACK_NDIMS>& block_shape, const ac::shape<PACK_NDIMS>& block_offset,
    ac::array<AcReal*, PACK_MAX_NAGGR_BUFS>& outputs);

#endif

void test_pack(void);
