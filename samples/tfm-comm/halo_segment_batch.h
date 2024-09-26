#pragma once
#include <stddef.h>

#include <mpi.h>

#include "halo_segment.h"

typedef struct {
    size_t npackets;
    HaloSegment* local_packets;
    HaloSegment* remote_packets;

    MPI_Request* requests;
    MPI_Datatype* send_subarrays;
    MPI_Datatype* recv_subarrays;
} HaloSegmentBatch;

HaloSegmentBatch acHaloSegmentBatchCreate(const size_t ndims, const size_t* mm, const size_t* nn,
                                          const size_t* nn_offset, const size_t nfields);

void acHaloSegmentBatchPrint(const char* label, const HaloSegmentBatch batch);

void acHaloSegmentBatchDestroy(HaloSegmentBatch* batch);
