#pragma once
#include <stddef.h>

#include "buffer.h"

typedef size_t FIELDDATATYPE;

typedef struct {
    size_t ndims;
    size_t* dims;
    size_t* offset;

    size_t nbuffers;
    Buffer buffer;
} HaloSegment;

HaloSegment acHaloSegmentCreate(const size_t ndims, const size_t* dims, const size_t* offset,
                                const size_t nbuffers);

void acHaloSegmentPrint(const char* label, const HaloSegment halo_segment);

void acHaloSegmentDestroy(HaloSegment* data);
