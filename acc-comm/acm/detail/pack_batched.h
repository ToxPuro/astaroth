#pragma once

#include "buffer.h"
#include "segment.h"
#include "vector.h"

#include "math_utils.h"

template <typename T>
void
pack_batched_host(const Shape& local_mm, //
                  const std::vector<T*>& inputs, const std::vector<ac::segment>& segments,
                  std::vector<T*>&& outputs)
{
    ERRCHK(segments.size() == outputs.size());

    for (uint64_t segid{0}; segid < segments.size(); ++segid) {

        const auto block_shape{segments[segid].dims};
        const auto block_offset{segments[segid].offset};

        const uint64_t block_nelems{prod(block_shape)};
        for (uint64_t i{0}; i < block_nelems; ++i) {
            for (size_t j{0}; j < inputs.size(); ++j) {

                // Block coords
                const Shape block_coords{to_spatial(i, block_shape)};

                // Input coords
                const Shape in_coords{block_offset + block_coords};

                const uint64_t in_idx{to_linear(in_coords, local_mm)};
                ERRCHK(in_idx < prod(local_mm));

                outputs[segid][i + j * block_nelems] = inputs[j][in_idx];
            }
        }
    }
}

template <typename T>
void
unpack_batched_host(const std::vector<ac::segment>& segments, const std::vector<T*>& inputs,
                    const Shape& local_mm, //
                    std::vector<T*>&& outputs)
{
    ERRCHK(segments.size() == inputs.size());
    for (uint64_t segid{0}; segid < segments.size(); ++segid) {

        const auto block_shape{segments[segid].dims};
        const auto block_offset{segments[segid].offset};

        const uint64_t block_nelems{prod(block_shape)};
        for (uint64_t i{0}; i < block_nelems; ++i) {
            for (size_t j{0}; j < outputs.size(); ++j) {

                // Block coords
                const Shape block_coords{to_spatial(i, block_shape)};

                // Input coords
                const Shape in_coords{block_offset + block_coords};

                const uint64_t in_idx{to_linear(in_coords, local_mm)};
                ERRCHK(in_idx < prod(local_mm));

                outputs[j][in_idx] = inputs[segid][i + j * block_nelems];
            }
        }
    }
}

template <typename T>
void
pack_batched(const Shape& local_mm, //
             const std::vector<ac::mr::host_ptr<T>>& inputs,
             const std::vector<ac::segment>& segments, std::vector<ac::mr::host_ptr<T>> outputs)
{
    ERRCHK(segments.size() == outputs.size());

    for (uint64_t segid{0}; segid < segments.size(); ++segid) {

        const auto block_shape{segments[segid].dims};
        const auto block_offset{segments[segid].offset};

        const uint64_t block_nelems{prod(block_shape)};
        for (uint64_t i{0}; i < block_nelems; ++i) {
            for (size_t j{0}; j < inputs.size(); ++j) {

                // Block coords
                const Shape block_coords{to_spatial(i, block_shape)};

                // Input coords
                const Shape in_coords{block_offset + block_coords};

                const uint64_t in_idx{to_linear(in_coords, local_mm)};
                ERRCHK(in_idx < prod(local_mm));

                outputs[segid][i + j * block_nelems] = inputs[j][in_idx];
            }
        }
    }
}

template <typename T>
void
unpack_batched(const std::vector<ac::segment>& segments,
               const std::vector<ac::mr::host_ptr<T>>& inputs,
               const Shape& local_mm, //
               std::vector<ac::mr::host_ptr<T>>& outputs)
{
    ERRCHK(segments.size() == inputs.size());
    for (uint64_t segid{0}; segid < segments.size(); ++segid) {

        const auto block_shape{segments[segid].dims};
        const auto block_offset{segments[segid].offset};

        const uint64_t block_nelems{prod(block_shape)};
        for (uint64_t i{0}; i < block_nelems; ++i) {
            for (size_t j{0}; j < outputs.size(); ++j) {

                // Block coords
                const Shape block_coords{to_spatial(i, block_shape)};

                // Input coords
                const Shape in_coords{block_offset + block_coords};

                const uint64_t in_idx{to_linear(in_coords, local_mm)};
                ERRCHK(in_idx < prod(local_mm));

                outputs[j][in_idx] = inputs[segid][i + j * block_nelems];
            }
        }
    }
}

#if defined(ACM_DEVICE_ENABLED)

// template <typename T>
// void pack_batched(const Shape& local_mm, //
//                   const std::vector<ac::mr::host_ptr<T>>& inputs,
//                   const std::vector<ac::segment>& segments,
//                   std::vector<ac::mr::host_ptr<T>>& outputs);

// template <typename T>
// void unpack_batched(const std::vector<ac::segment>& segments,
//                     const std::vector<ac::mr::host_ptr<T>>& inputs,
//                     const Shape& local_mm, //
//                     std::vector<ac::mr::host_ptr<T>>& outputs);

template <typename T>
void pack_batched(const Shape& local_mm, //
                  const std::vector<ac::mr::device_ptr<T>>& inputs,
                  const std::vector<ac::segment>& segments,
                  std::vector<ac::mr::device_ptr<T>> outputs);

template <typename T>
void unpack_batched(const std::vector<ac::segment>& segments,
                    const std::vector<ac::mr::device_ptr<T>>& inputs,
                    const Shape& local_mm, //
                    std::vector<ac::mr::device_ptr<T>>& outputs);

#define PACK_DTYPE double
extern template void
pack_batched<PACK_DTYPE>(const Shape& local_mm, //
                         const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>& segments,
                         std::vector<ac::mr::device_ptr<PACK_DTYPE>> outputs);

extern template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>& segments,
                           const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                           const Shape& local_mm, //
                           std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
extern template void
pack_batched<PACK_DTYPE>(const Shape& local_mm, //
                         const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>& segments,
                         std::vector<ac::mr::device_ptr<PACK_DTYPE>> outputs);

extern template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>& segments,
                           const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                           const Shape& local_mm, //
                           std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#endif
