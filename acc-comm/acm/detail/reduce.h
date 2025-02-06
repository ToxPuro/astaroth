#pragma once
#include <algorithm>

#include "ntuple.h"
#include "pointer.h"

namespace ac {

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

void test_reduce();
