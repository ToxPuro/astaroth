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

// BoundingBox
// bounding_box_create(const size_t ndims, const size_t* min, const size_t* max)
// {
//     BoundingBox bounding_box = (BoundingBox){
//         .ndims = ndims,
//         .min   = NULL,
//         .max   = NULL,
//     };
//     ndup(ndims, min, bounding_box.min);
//     ndup(ndims, max, bounding_box.max);
//     ERRCHK(all_less_than(ndims, min, max));
//     return bounding_box;
// }

// void
// bounding_box_destroy(BoundingBox* bounding_box)
// {
//     ndealloc(bounding_box->max);
//     ndealloc(bounding_box->min);
//     bounding_box->ndims = 0;
// }

// void
// print_bounding_box(const char* label, const BoundingBox bounding_box)
// {
//     printf("BoundingBox %s:\n", label);
//     print("\tndims", bounding_box.ndims);
//     print_array("\tmin", bounding_box.ndims, bounding_box.min);
//     print_array("\tmax", bounding_box.ndims, bounding_box.max);
// }

// void
// test_bounding_box(void)
// {
//     const size_t min[]       = {1, 1, 1};
//     const size_t max[]       = {0, 0, 0};
//     const size_t ndims       = ARRAY_SIZE(min);
//     BoundingBox bounding_box = bounding_box_create(ndims, min, max);
//     ERRCHK(bounding_box.min != NULL);
//     ERRCHK(bounding_box.max != NULL);
//     bounding_box_destroy(&bounding_box);
//     ERRCHK(bounding_box.min == NULL);
//     ERRCHK(bounding_box.max == NULL);
// }

// BoundingBox
// bounding_box_create_from_segment(const Segment segment)
// {
//     size_t* max;
//     nalloc(segment.ndims, max);
//     add_arrays(segment.ndims, segment.offset, segment.dims, max);

//     BoundingBox bb = bounding_box_create(segment.ndims, segment.offset, max);

//     ndealloc(max);
//     return bb;
// }

// Segment
// segment_create_from_bounding_box(const BoundingBox bounding_box)
// {
//     size_t* dims;
//     nalloc(bounding_box.ndims, dims);
//     subtract_arrays(bounding_box.ndims, bounding_box.max, bounding_box.min, dims);

//     Segment segment = segment_create(bounding_box.ndims, dims, bounding_box.min);

//     ndealloc(dims);
//     return segment;
// }
