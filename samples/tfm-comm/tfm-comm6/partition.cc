#include "partition.h"

#include "print_debug.h"

void
partition_recursive(const Shape& mm, const Shape& nn, const Index& nn_offset,
                    const Segment& current_segment, const size_t axis,
                    std::vector<Segment>& segments)
{
    if (prod(current_segment.dims) == 0) { // Empty segment
        return;
    }
    else if (axis >= current_segment.dims.count) { // All axes partitioned
        segments.push_back(current_segment);
    }
    else { // Partition

        // Split points
        const size_t x0 = current_segment.offset[axis];
        const size_t x1 = nn_offset[axis];
        const size_t x2 = nn_offset[axis] + nn[axis];
        const size_t x3 = current_segment.offset[axis] + current_segment.dims[axis];

        { // Left
            Segment new_segment(current_segment);
            new_segment.offset[axis] = x0;
            new_segment.dims[axis]   = x1 - x0;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Center
            Segment new_segment(current_segment);
            new_segment.offset[axis] = x1;
            new_segment.dims[axis]   = x2 - x1;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }

        { // Right
            Segment new_segment(current_segment);
            new_segment.offset[axis] = x2;
            new_segment.dims[axis]   = x3 - x2;
            partition_recursive(mm, nn, nn_offset, new_segment, axis + 1, segments);
        }
    }
}

std::vector<Segment>
partition(const Shape& mm, const Shape& nn, const Index& nn_offset)
{
    std::vector<Segment> segments;
    Segment initial_segment(mm);
    partition_recursive(mm, nn, nn_offset, initial_segment, 0, segments);
    return segments;
}

void
test_partition(void)
{
    Shape rr = {2, 3};
    Shape nn = {128, 128};
    Shape mm = as<uint64_t>(2) * rr + nn;

    auto segments = partition(mm, nn, rr);
    PRINT_DEBUG(segments);
    PRINT_DEBUG(segments.size());
}
