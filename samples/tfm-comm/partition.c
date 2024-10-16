#include "partition.h"

#include "math_utils.h"
#include "misc.h"
#include "print.h"

static void
partition_recursive(const Segment mm_seg, const Segment nn_seg, const size_t axis,
                    SegmentArray* segments)
{
    if (prod(mm_seg.ndims, mm_seg.dims) == 0) { // Empty segment
        return;
    }
    else if (axis >= mm_seg.ndims) { // All axes partitioned
        dynarr_append(segment_create(mm_seg.ndims, mm_seg.dims, mm_seg.offset), segments);
    }
    else { // Partition

        // Split points
        const size_t x0 = mm_seg.offset[axis];
        const size_t x1 = nn_seg.offset[axis];
        const size_t x2 = nn_seg.offset[axis] + nn_seg.dims[axis];
        const size_t x3 = mm_seg.dims[axis];

        // Left
        Segment new_segment      = segment_create(mm_seg.ndims, mm_seg.dims, mm_seg.offset);
        new_segment.offset[axis] = x0;
        new_segment.dims[axis]   = x1 - x0;
        partition_recursive(new_segment, nn_seg, axis + 1, segments);

        // Center
        segment_copy(mm_seg, &new_segment);
        new_segment.offset[axis] = x1;
        new_segment.dims[axis]   = x2 - x1;
        partition_recursive(new_segment, nn_seg, axis + 1, segments);

        // Right
        segment_copy(mm_seg, &new_segment);
        new_segment.offset[axis] = x2;
        new_segment.dims[axis]   = x3 - x2;
        partition_recursive(new_segment, nn_seg, axis + 1, segments);

        segment_destroy(&new_segment);
    }
}

void
partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
          SegmentArray* segments)
{
    size_t* zeros;
    ncalloc(ndims, zeros);

    Segment mm_seg = segment_create(ndims, mm, zeros);
    Segment nn_seg = segment_create(ndims, nn, nn_offset);

    partition_recursive(mm_seg, nn_seg, 0, segments);

    segment_destroy(&mm_seg);
    segment_destroy(&nn_seg);

    ndealloc(zeros);
}

void
test_partition(void)
{
    {
        const size_t mm[]        = {8};
        const size_t nn[]        = {6, 6};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 3);

        dynarr_destroy(&segments);
    }
    {
        const size_t mm[]        = {8, 8};
        const size_t nn[]        = {6, 6};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 9);

        dynarr_destroy(&segments);
    }
    {
        const size_t mm[]        = {8, 8, 8};
        const size_t nn[]        = {6, 6, 6};
        const size_t nn_offset[] = {1, 1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 27);

        dynarr_destroy(&segments);
    }
    {
        const size_t mm[]        = {5, 6, 7, 8};
        const size_t nn[]        = {3, 4, 5, 6};
        const size_t nn_offset[] = {1, 1, 1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 3 * 3 * 3 * 3);

        dynarr_destroy(&segments);
    }
    {
        const size_t mm[]        = {4, 4, 4};
        const size_t nn[]        = {4, 4, 4};
        const size_t nn_offset[] = {0, 0, 0};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 1);

        dynarr_destroy(&segments);
    }
    {
        const size_t mm[]        = {4, 4};
        const size_t nn[]        = {3, 3};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 4);

        dynarr_destroy(&segments);
    }
}
