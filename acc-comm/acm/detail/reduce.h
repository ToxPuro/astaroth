#pragma once
#include <algorithm>

#include "ntuple.h"
#include "pointer.h"

namespace ac {

template <typename T, typename Allocator>
void segmented_reduce(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset,
                      const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                      ac::mr::pointer<T, Allocator>                     output);

template <typename T>
void
segmented_reduce(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset,
                 const std::vector<ac::mr::host_pointer<T>>& inputs, ac::mr::host_pointer<T> output)
{
    // Check that the output memory resource can hold all segments
    ERRCHK(inputs.size() == output.size());

    std::fill(output.begin(), output.end(), 0);
    for (size_t j{0}; j < inputs.size(); ++j) {
        const uint64_t count{prod(subdims)};
        for (uint64_t i{0}; i < count; ++i) {

            // Block coords
            const ac::index block_coords{to_spatial(i, subdims)};

            // Input coords
            const ac::index in_coords{offset + block_coords};
            const uint64_t  in_idx{ac::to_linear(in_coords, dims)};
            output[j] += inputs[j][in_idx];
        }
    }
}

} // namespace ac

#include "datatypes.h"
namespace ac {
template <>
void segmented_reduce(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset,
                      const std::vector<DevicePointer>& inputs, DevicePointer output);

} // namespace ac

// #include "mpi_utils.h"

// namespace ac::mpi {

// template <typename T>
// void
// segmented_reduce(const MPI_Comm& comm, const size_t& axis, const ac::shape& dims,
//                  const ac::shape& subdims, const ac::index& offset,
//                  const std::vector<ac::mr::host_pointer<T>>& inputs, ac::mr::host_pointer<T>
//                  output)
// {
//     ac::segmented_reduce(dims, subdims, offset, inputs, output);
//     const int num_collaborators{ac::mpi::reduce_axis(comm,
//                                                      ac::mpi::get_dtype<T>(),
//                                                      MPI_SUM,
//                                                      axis,
//                                                      output.size(),
//                                                      output.data())};
//     // If used for averaging
//     std::for_each(output.begin(), output.end(), [&num_collaborators](const auto& elem) {
//         return elem / num_collaborators;
//     });
// }

// } // namespace ac::mpi

// #include "buffer.h"
// namespace ac {

// template <typename T>
// void
// segmented_reduce(const size_t num_segments, const size_t stride, const ac::host_buffer<T>& input,
//                  ac::host_buffer<T> output)
// {
//     for (size_t segment{0}; segment < num_segments; ++segment) {
//         std::reduce(input.begin() + segment * stride,
//                     input.begin() + (segment + 1) * stride,
//                     output.begin(),
//                     [](const auto& a, const auto& b) { return a + b; });
//     }
// }

// } // namespace ac

// namespace ac {

// template <typename T>
// void
// pack_transform_reduce(const MPI_Comm& comm, const ac::shape& dims, const ac::shape& subdims,
//                       const ac::index& offset, const std::vector<ac::mr::host_pointer<T>>&
//                       inputs, ac::mr::host_pointer<T> output)
// {
//     static ac::host_buffer<T> pack_buffer{inputs.size() * prod(subdims)};

//     // Pack
//     pack(dims, subdims, offset, inputs, pack_buffer.get());

//     // Transform
//     std::transform(pack_buffer.begin(), pack_buffer.end(), [](const auto& elem) {
//         return elem * elem;
//     });

//     // Reduce
//     const size_t num_segments{inputs.size()};
//     const size_t stride{prod(subdims)};
//     ac::segmented_reduce(num_segments, stride, pack_buffer.get(), output);
// }

// } // namespace ac

void test_reduce_device();
