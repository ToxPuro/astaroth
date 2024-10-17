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
        const uint64_t x0 = mm_seg.offset[axis];
        const uint64_t x1 = nn_seg.offset[axis];
        const uint64_t x2 = nn_seg.offset[axis] + nn_seg.dims[axis];
        const uint64_t x3 = mm_seg.dims[axis];

        // Left
        {
            Segment new_segment      = segment_dup(mm_seg);
            new_segment.offset[axis] = x0;
            new_segment.dims[axis]   = x1 - x0;
            partition_recursive(new_segment, nn_seg, axis + 1, segments);
            segment_destroy(&new_segment);
        }

        // Center
        {
            Segment new_segment      = segment_dup(mm_seg);
            new_segment.offset[axis] = x1;
            new_segment.dims[axis]   = x2 - x1;
            partition_recursive(new_segment, nn_seg, axis + 1, segments);
            segment_destroy(&new_segment);
        }

        // Right
        {
            Segment new_segment      = segment_dup(mm_seg);
            new_segment.offset[axis] = x2;
            new_segment.dims[axis]   = x3 - x2;
            partition_recursive(new_segment, nn_seg, axis + 1, segments);
            segment_destroy(&new_segment);
        }
    }
}

void
partition(const size_t ndims, const uint64_t* mm, const uint64_t* nn, const uint64_t* nn_offset,
          SegmentArray* segments)
{
    uint64_t* zeros = ac_calloc(ndims, sizeof(zeros));

    Segment mm_seg = segment_create(ndims, mm, zeros);
    Segment nn_seg = segment_create(ndims, nn, nn_offset);

    partition_recursive(mm_seg, nn_seg, 0, segments);

    segment_destroy(&mm_seg);
    segment_destroy(&nn_seg);

    ac_free((void**)&zeros);
}

void
test_partition(void)
{
    {
        const uint64_t mm[]        = {8};
        const uint64_t nn[]        = {6, 6};
        const uint64_t nn_offset[] = {1, 1};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 3);
        dynarr_destroy(&segments);
    }
    {
        const uint64_t mm[]        = {8, 8};
        const uint64_t nn[]        = {6, 6};
        const uint64_t nn_offset[] = {1, 1};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 9);
        dynarr_destroy(&segments);
    }
    {
        const uint64_t mm[]        = {8, 8, 8};
        const uint64_t nn[]        = {6, 6, 6};
        const uint64_t nn_offset[] = {1, 1, 1};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 27);
        dynarr_destroy(&segments);
    }
    {
        const uint64_t mm[]        = {5, 6, 7, 8};
        const uint64_t nn[]        = {3, 4, 5, 6};
        const uint64_t nn_offset[] = {1, 1, 1, 1};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 3 * 3 * 3 * 3);
        dynarr_destroy(&segments);
    }
    {
        const uint64_t mm[]        = {4, 4, 4};
        const uint64_t nn[]        = {4, 4, 4};
        const uint64_t nn_offset[] = {0, 0, 0};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 1);
        dynarr_destroy(&segments);
    }
    {
        const uint64_t mm[]        = {4, 4};
        const uint64_t nn[]        = {3, 3};
        const uint64_t nn_offset[] = {1, 1};
        const size_t ndims         = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 4);
        dynarr_destroy(&segments);
    }
}

static void
partition_hierarchical_recursive(const size_t npartitions, const size_t ndims,
                                 const Segment segment, SegmentArray* segments)
{
    ERRCHK(npartitions > 0);
    if (npartitions == 1) {
        PRINTD_SEGMENT(segment);
        // dynarr_append(segment_create(segment.ndims, segment.dims, segment.offset), segments);
        return;
    }

    // Choose a partitioning axis
    // Inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
    size_t nfactors;
    factorize(npartitions, &nfactors, NULL);
    uint64_t* factors = ac_calloc(nfactors, sizeof(factors[0]));
    factorize(npartitions, &nfactors, factors);

    size_t best_axis     = SIZE_MAX;
    uint64_t best_split  = 0;
    uint64_t best_factor = 0;
    for (size_t i = nfactors - 1; i < nfactors; --i) {
        const uint64_t factor = factors[i];
        for (size_t j = ndims - 1; j < ndims; --j) {
            // If dimension j is divisible by factor
            if ((segment.dims[j] % factor) == 0) {
                // Select axis if the factoring results in the best split
                const uint64_t split = segment.dims[j] / factor;
                if (split > best_split) {
                    best_axis   = j;
                    best_split  = split;
                    best_factor = factor;
                }
            }
        }
    }
    ac_free((void**)&factors);
    ERRCHK(best_axis < ndims);
    ERRCHK((npartitions % best_factor) == 0);

    // for (size_t i = 0; i < best_factor; ++i) {
    for (size_t i = 0; i < 1; ++i) {
        Segment new_segment         = segment_create(ndims, segment.dims, segment.offset);
        new_segment.dims[best_axis] = best_split;
        new_segment.offset[best_axis] += i * best_split;
        partition_hierarchical_recursive(npartitions / best_factor, ndims, new_segment, segments);
        segment_destroy(&new_segment);
    }
}

void
partition_hierarchical(void)
{
    SegmentArray segments;
    dynarr_create_with_destructor(segment_destroy, &segments);

    const uint64_t dims[]   = {512, 64};
    const uint64_t offset[] = {0, 0, 0};
    const size_t ndims      = ARRAY_SIZE(dims);
    Segment segment         = segment_create(ndims, dims, offset);
    partition_hierarchical_recursive(8, ndims, segment, &segments);
    printf("segments len %zu\n", segments.length);
    segment_destroy(&segment);

    dynarr_destroy(&segments);
}
