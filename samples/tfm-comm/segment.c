#include "segment.h"

#include "math_utils.h"
#include "misc.h"
#include "nalloc.h"
#include "print.h"

Segment
segment_create(const size_t ndims, const size_t* dims, const size_t* offset)
{
    Segment segment = (Segment){
        .ndims  = ndims,
        .dims   = NULL,
        .offset = NULL,
    };
    ndup(ndims, dims, segment.dims);
    ndup(ndims, offset, segment.offset);
    return segment;
}

void
segment_destroy(Segment* segment)
{
    ndealloc(segment->offset);
    ndealloc(segment->dims);
    segment->ndims = 0;
}

void
print_segment(const char* label, const Segment segment)
{
    printf("Segment %s:\n", label);
    print("\tndims", segment.ndims);
    print_array("\tdims", segment.ndims, segment.dims);
    print_array("\toffset", segment.ndims, segment.offset);
}

void
segment_copy(const Segment in, Segment* out)
{
    ERRCHK(in.ndims == out->ndims);
    ncopy(in.ndims, in.dims, out->dims);
    ncopy(in.ndims, in.offset, out->offset);
}

void
test_segment(void)
{
    const size_t dims[]   = {1, 1, 1};
    const size_t offset[] = {0, 0, 0};
    const size_t ndims    = ARRAY_SIZE(dims);
    Segment segment       = segment_create(ndims, dims, offset);
    ERRCHK(segment.dims != NULL);
    ERRCHK(segment.offset != NULL);
    segment_destroy(&segment);
    ERRCHK(segment.dims == NULL);
    ERRCHK(segment.offset == NULL);
}
