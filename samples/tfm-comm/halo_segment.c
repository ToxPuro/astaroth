#include "halo_segment.h"

#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "print.h"

HaloSegment
acCreateHaloSegment(const size_t ndims, const size_t* dims, const size_t* offset,
                    const size_t nbuffers)
{
    HaloSegment segment = (HaloSegment){
        .ndims    = ndims,
        .dims     = malloc(sizeof(segment.dims[0]) * ndims),
        .offset   = malloc(sizeof(segment.offset[0]) * ndims),
        .nbuffers = nbuffers,
        .buffer   = acBufferCreate(nbuffers * prod(ndims, dims)),
    };
    ERRCHK(segment.dims);
    ERRCHK(segment.offset);
    copy(ndims, dims, segment.dims);
    copy(ndims, offset, segment.offset);
    return segment;
}

void
acHaloSegmentPrint(const char* label, const HaloSegment halo_segment)
{
    printf("HaloSegment %s:\n", label);
    print("\tndims", halo_segment.ndims);
    print_array("\tdims", halo_segment.ndims, halo_segment.dims);
    print_array("\toffset", halo_segment.ndims, halo_segment.offset);
    print("\tfields", halo_segment.nbuffers);
    acBufferPrint("\tbuffer", halo_segment.buffer);
}

void
acDestroyHaloSegment(HaloSegment* segment)
{
    acBufferDestroy(&segment->buffer);
    segment->nbuffers = 0;
    free(segment->offset);
    free(segment->dims);
    segment->ndims = 0;
}
