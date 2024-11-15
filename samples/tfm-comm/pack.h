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

constexpr size_t PACK_MAX_INPUTS         = 1;
template <typename T> using PackPtrArray = ac::array<T, PACK_MAX_INPUTS>;

template <typename T, size_t N>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const ac::array<T*, N>& inputs, ac::device_vector<T>& output);

template <typename T, size_t N>
void unpack(const ac::device_vector<T>& input, const Shape& mm, const Shape& block_shape,
            const Index& block_offset, ac::array<T*, N>& outputs);

extern template void pack<AcReal, PACK_MAX_INPUTS>(const Shape&, const Shape&, const Index&,
                                                   const ac::array<AcReal*, PACK_MAX_INPUTS>&,
                                                   ac::host_vector<AcReal>&);

extern template void unpack<AcReal, PACK_MAX_INPUTS>(const ac::host_vector<AcReal>&, const Shape&,
                                                     const Shape&, const Index&,
                                                     ac::array<AcReal*, PACK_MAX_INPUTS>&);

#if defined(DEVICE_ENABLED)
extern template void pack<AcReal, PACK_MAX_INPUTS>(const Shape&, const Shape&, const Index&,
                                                   const ac::array<AcReal*>&,
                                                   ac::device_vector<AcReal>&);

extern template void unpack<AcReal, PACK_MAX_INPUTS>(const ac::device_vector<AcReal>&, const Shape&,
                                                     const Shape&, const Index&,
                                                     ac::array<AcReal*>&);
#endif

void test_pack(void);
