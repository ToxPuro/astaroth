#pragma once
#include "stddef.h"

#include "halo_segment.h"

typedef struct {
    size_t npackets;
    HaloSegment* local_packets;
    HaloSegment* remote_packets;
} HaloSegmentBatch;

HaloSegmentBatch acHaloSegmentBatchCreate(const size_t ndims, const size_t* nn, const size_t* rr,
                                          const size_t nfields);

void acHaloSegmentBatchPrint(const char* label, const HaloSegmentBatch batch);

void acHaloSegmentBatchDestroy(HaloSegmentBatch* batch);
