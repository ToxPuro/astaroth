#pragma once
#include <iomanip>

#include "acm/detail/errchk_print.h"
#include "pointer.h"

#include "errchk.h"
#include "ntuple.h"
#include "print_debug.h"

#include <numeric>
#include <vector>

#include "datatypes.h"

namespace ac {

template <typename T>
void
transform(const ac::ntuple<uint64_t> dims, const ac::ntuple<uint64_t> subdims,
          const ac::ntuple<uint64_t> offset, const T* in, T* out)
{
    for (uint64_t out_idx{0}; out_idx < prod(subdims); ++out_idx) {
        const ac::ntuple<uint64_t> out_coords{to_spatial(out_idx, subdims)};
        ERRCHK(out_coords < subdims);

        const ac::ntuple<uint64_t> in_coords{offset + out_coords};
        ERRCHK(in_coords < dims);

        const uint64_t in_idx{to_linear(in_coords, dims)};
        out[out_idx] = in[in_idx];
    }
}

template <typename T>
void
transform(const Shape dims, const Shape subdims, const Index offset,
          const ac::mr::host_pointer<T> in, ac::mr::host_pointer<T> out)
{
    for (uint64_t out_idx{0}; out_idx < prod(subdims); ++out_idx) {
        const ac::ntuple<uint64_t> out_coords{to_spatial(out_idx, subdims)};
        ERRCHK(out_coords < subdims);

        const ac::ntuple<uint64_t> in_coords{offset + out_coords};
        ERRCHK(in_coords < dims);

        const uint64_t in_idx{to_linear(in_coords, dims)};

        ERRCHK(in_idx < in.size());
        ERRCHK(out_idx < out.size());
        out[out_idx] = in[in_idx];
    }
}

#if defined(ACM_DEVICE_ENABLED)
void transform(const Shape in_dims, const Shape in_subdims, const Index in_offset,
               const DevicePointer in, DevicePointer out);
#endif

// template <typename T> print_recursive(const size_t depth, const ac::ntuple<uint64_t>& ntuple) {
// ERRCHK() }

template <typename T>
void
print_recursive(const ac::ntuple<uint64_t>& shape, const T* data)
{
    if (shape.size() == 1) {
        for (size_t i{0}; i < shape[0]; ++i)
            std::cout << std::setw(4) << data[i];
        std::cout << std::endl;
    }
    else {
        const auto curr_extent{shape[shape.size() - 1]};
        const auto new_shape{slice(shape, 0, shape.size() - 1)};
        const auto offset{prod(new_shape)};
        for (size_t i{0}; i < curr_extent; ++i) {
            if (shape.size() > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, shape.size() - 1);
            if (shape.size() == 4)
                printf("Cube %zu:\n", i);
            if (shape.size() == 3)
                printf("Layer %zu:\n", i);
            if (shape.size() == 2)
                printf("Row %zu: ", i);
            print_recursive(new_shape, &data[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
void
print(const std::string& label, const ac::ntuple<uint64_t>& shape, const T* data)
{
    std::cout << label << ":" << std::endl;
    print_recursive(shape, data);
}

// template <typename T>
// void
// fill(const T& fill_value, const ac::ntuple<uint64_t>& dims, const ac::ntuple<uint64_t>& subdims,
//      const ac::ntuple<uint64_t>& offset, ac::mr::host_pointer<T>& data)
// {
//     WARNING_DESC("Not implemented");
//     // Should also consider movng this to pointer instead:
//     // need to define separate functions for host and device versions
// }

} // namespace ac

void test_transform();
