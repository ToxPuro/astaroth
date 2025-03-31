#pragma once

#include "acm/detail/math_utils.h"
#include "ntuple.h"
#include "pointer.h"
#include "segment.h"

template <typename T>
void
pack(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
     const std::vector<ac::mr::host_pointer<T>>& inputs, ac::mr::host_pointer<T> output)
{
    ERRCHK(inputs.size() * prod(block_shape) <= output.size());
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::shape block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
void
unpack(const ac::mr::host_pointer<T>& input, const ac::shape& mm, const ac::shape& block_shape,
       const ac::index& block_offset, std::vector<ac::mr::host_pointer<T>> outputs)
{
    ERRCHK(outputs.size() * prod(block_shape) <= input.size());
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::shape block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

template <typename T>
void
pack_batched(const ac::shape& mm, const std::vector<ac::mr::host_pointer<T>>& inputs,
             const std::vector<ac::segment>& segments, std::vector<ac::mr::host_pointer<T>> outputs)
{
    ERRCHK(same_size(segments, outputs));

    for (size_t i{0}; i < segments.size(); ++i)
        pack(mm, segments[i].dims, segments[i].offset, inputs, outputs[i]);
}

template <typename T>
void
unpack_batched(const std::vector<ac::segment>&             segments,
               const std::vector<ac::mr::host_pointer<T>>& inputs, const ac::shape& mm,
               std::vector<ac::mr::host_pointer<T>> outputs)
{
    for (size_t i{0}; i < segments.size(); ++i)
        unpack(inputs[i], mm, segments[i].dims, segments[i].offset, outputs);
}

#if defined(ACM_DEVICE_ENABLED)

template <typename T>
void pack(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
          const std::vector<ac::mr::device_pointer<T>>& inputs, ac::mr::device_pointer<T> output);

template <typename T>
void unpack(const ac::mr::device_pointer<T>& input, const ac::shape& mm,
            const ac::shape& block_shape, const ac::index& block_offset,
            std::vector<ac::mr::device_pointer<T>> outputs);

template <typename T>
void pack_batched(const ac::shape& mm, const std::vector<ac::mr::device_pointer<T>>& inputs,
                  const std::vector<ac::segment>&        segments,
                  std::vector<ac::mr::device_pointer<T>> outputs);

template <typename T>
void unpack_batched(const std::vector<ac::segment>&               segments,
                    const std::vector<ac::mr::device_pointer<T>>& inputs, const ac::shape& mm,
                    std::vector<ac::mr::device_pointer<T>> outputs);

#define PACK_DTYPE double
extern template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                                      const ac::index& block_offset,
                                      const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                                      ac::mr::device_pointer<PACK_DTYPE> output);

extern template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                        const ac::shape& mm, const ac::shape& block_shape,
                                        const ac::index& block_offset,
                                        std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);

extern template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

extern template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
extern template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                                      const ac::index& block_offset,
                                      const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                                      ac::mr::device_pointer<PACK_DTYPE> output);

extern template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                        const ac::shape& mm, const ac::shape& block_shape,
                                        const ac::index& block_offset,
                                        std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);

extern template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

extern template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE

#define PACK_DTYPE int
extern template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                                      const ac::index& block_offset,
                                      const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                                      ac::mr::device_pointer<PACK_DTYPE> output);

extern template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                        const ac::shape& mm, const ac::shape& block_shape,
                                        const ac::index& block_offset,
                                        std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);

extern template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

extern template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE

#endif
