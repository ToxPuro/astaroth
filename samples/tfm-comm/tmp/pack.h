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
template <typename T, size_t N, typename MemoryResource>
void pack(const ac::shape<N>& mm, const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
          const std::vector<T*>& inputs, T* output);

template <typename T, size_t N, typename MemoryResource>
void unpack(const T* input, const ac::shape<N>& mm, const ac::shape<N>& block_shape,
            const ac::index<N>& block_offset, std::vector<T*>& outputs);

// The actual kernel
extern template void pack<UserType, UserNdims, ac::mr::host_memory_resource>(
    const UserShape& mm, const UserShape& block_shape, const UserIndex& block_offset,
    const std::vector<UserType*>& inputs, UserType* output);

extern template void unpack<UserType, UserNdims, ac::mr::host_memory_resource>(
    const UserType* input, const UserShape& mm, const UserShape& block_shape,
    const UserShape& block_offset, std::vector<UserType*>& outputs);

#if defined(DEVICE_ENABLED)

extern template void pack<UserType, UserNdims, ac::mr::device_memory_resource>(
    const UserShape& mm, const UserShape& block_shape, const UserIndex& block_offset,
    const std::vector<UserType*>& inputs, UserType* output);

extern template void unpack<UserType, UserNdims, ac::mr::device_memory_resource>(
    const UserType* input, const UserShape& mm, const UserShape& block_shape,
    const UserShape& block_offset, std::vector<UserType*>& outputs);

#endif

void test_pack(void);
