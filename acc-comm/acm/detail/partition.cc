#include "partition.h"

static void
partition_recursive(const ac::shape& mm, const ac::shape& nn, const ac::index& nn_offset,
                    const ac::segment& current_segment, const size_t axis,
                    std::vector<ac::segment>& segments)
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
            ac::segment new_segment{current_segment};
            new_segment.offset[axis] = x0;
            new_segment.dims[axis]   = x1 - x0;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Center
            ac::segment new_segment{current_segment};
            new_segment.offset[axis] = x1;
            new_segment.dims[axis]   = x2 - x1;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Right
            ac::segment new_segment{current_segment};
            new_segment.offset[axis] = x2;
            new_segment.dims[axis]   = x3 - x2;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }
    }
}

std::vector<ac::segment>
partition(const ac::shape& mm, const ac::shape& nn, const ac::index& nn_offset)
{
    std::vector<ac::segment> segments;
    ac::segment              initial_segment{mm};
    partition_recursive(mm, nn, nn_offset, initial_segment, 0, segments);
    return segments;
}
