#pragma once

#include "segment.h"

#include <vector>

namespace detail {
template <size_t N>
static void
partition_recursive(const ac::shape<N>& mm, const ac::shape<N>& nn, const ac::index<N>& nn_offset,
                    const Segment<N>& current_segment, const size_t axis,
                    std::vector<Segment<N>>& segments)
{
    if (prod(current_segment.dims) == 0) { // Empty segment
        return;
    }
    else if (axis >= current_segment.dims.size()) { // All axes partitioned
        segments.push_back(current_segment);
    }
    else { // Partition

        // Split points
        const size_t x0{current_segment.offset[axis]};
        const size_t x1{nn_offset[axis]};
        const size_t x2{nn_offset[axis] + nn[axis]};
        const size_t x3{current_segment.offset[axis] + current_segment.dims[axis]};

        { // Left
            Segment<N> new_segment(current_segment);
            new_segment.offset[axis] = x0;
            new_segment.dims[axis]   = x1 - x0;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Center
            Segment<N> new_segment(current_segment);
            new_segment.offset[axis] = x1;
            new_segment.dims[axis]   = x2 - x1;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Right
            Segment<N> new_segment(current_segment);
            new_segment.offset[axis] = x2;
            new_segment.dims[axis]   = x3 - x2;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }
    }
}
} // namespace detail

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
template <size_t N>
std::vector<Segment<N>>
partition(const ac::shape<N>& mm, const ac::shape<N>& nn, const ac::index<N>& nn_offset)
{
    std::vector<Segment<N>> segments;
    Segment<N> initial_segment(mm);
    detail::partition_recursive(mm, nn, nn_offset, initial_segment, 0, segments);
    return segments;
}

void test_partition(void);
