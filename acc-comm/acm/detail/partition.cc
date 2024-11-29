#include "partition.h"

#include "print_debug.h"

static void
partition_recursive(const Shape& mm, const Shape& nn, const Index& nn_offset,
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
partition(const Shape& mm, const Shape& nn, const Index& nn_offset)
{
    std::vector<ac::segment> segments;
    ac::segment initial_segment{mm};
    partition_recursive(mm, nn, nn_offset, initial_segment, 0, segments);
    return segments;
}

void
test_partition(void)
{
    {
        const ac::vector<uint64_t> mm{8};
        const ac::vector<uint64_t> nn{6};
        const ac::vector<uint64_t> nn_offset{1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3);
    }
    {
        const ac::vector<uint64_t> mm{8, 8};
        const ac::vector<uint64_t> nn{6, 6};
        const ac::vector<uint64_t> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 9);
    }
    {
        const ac::vector<uint64_t> mm{8, 8, 8};
        const ac::vector<uint64_t> nn{6, 6, 6};
        const ac::vector<uint64_t> nn_offset{1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 27);
    }
    {
        const ac::vector<uint64_t> mm{5, 6, 7, 8};
        const ac::vector<uint64_t> nn{3, 4, 5, 6};
        const ac::vector<uint64_t> nn_offset{1, 1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3 * 3 * 3 * 3);
    }
    {
        const ac::vector<uint64_t> mm{4, 4, 4};
        const ac::vector<uint64_t> nn{4, 4, 4};
        const ac::vector<uint64_t> nn_offset{0, 0, 0};
        ERRCHK(partition(mm, nn, nn_offset).size() == 1);
    }
    {
        const ac::vector<uint64_t> mm{4, 4};
        const ac::vector<uint64_t> nn{3, 3};
        const ac::vector<uint64_t> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 4);
    }
}
