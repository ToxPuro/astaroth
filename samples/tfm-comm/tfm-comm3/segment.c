#include "segment.h"

#include "alloc.h"
#include "errchk.h"
#include "math_utils.h"
#include "misc.h"
#include "print.h"

Segment
segment_create(const size_t ndims, const uint64_t* dims, const uint64_t* offset)
{
    Segment segment = (Segment){
        .ndims  = ndims,
        .dims   = ac_dup(ndims, sizeof(dims[0]), dims),
        .offset = ac_dup(ndims, sizeof(offset[0]), offset),
    };
    return segment;
}

void
segment_destroy(Segment* segment)
{
    ac_free(segment->offset);
    ac_free(segment->dims);
    segment->offset = NULL;
    segment->dims   = NULL;
    segment->ndims  = 0;
}

void
print_segment(const char* label, const Segment segment)
{
    printf("Segment %s:\n", label);
    print("\tndims", segment.ndims);
    print_array("\tdims", segment.ndims, segment.dims);
    print_array("\toffset", segment.ndims, segment.offset);
}

Segment
segment_copy(const Segment in)
{
    Segment out = segment_create(in.ndims, in.dims, in.offset);
    return out;
}

void
test_segment(void)
{
    {
        const uint64_t dims[]   = {1, 1, 1};
        const uint64_t offset[] = {0, 0, 0};
        const size_t ndims      = ARRAY_SIZE(dims);
        Segment segment         = segment_create(ndims, dims, offset);
        ERRCHK(segment.dims != NULL);
        ERRCHK(segment.offset != NULL);
        segment_destroy(&segment);
        ERRCHK(segment.dims == NULL);
        ERRCHK(segment.offset == NULL);
    }
    {
        const uint64_t dims[]   = {5, 6, 7, 8};
        const uint64_t offset[] = {1, 2, 3, 4};
        const size_t ndims      = ARRAY_SIZE(dims);
        Segment a               = segment_create(ndims, dims, offset);
        Segment b               = segment_copy(a);

        ERRCHK(equals(a.ndims, a.dims, b.dims));
        ERRCHK(equals(a.ndims, a.offset, b.offset));

        segment_destroy(&a);
        segment_destroy(&b);
    }
}
