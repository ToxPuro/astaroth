#pragma once
#include "stddef.h"

#include "halo_segment.h"

typedef struct {
    size_t npackets;
    HaloSegment* local_packets;
    HaloSegment* remote_packets;
} CommData;

CommData acCommDataCreate(const size_t ndims, const size_t* nn, const size_t* rr,
                          const size_t nfields);

void acCommDataPrint(const char* label, const CommData comm_data);

void acCommDataDestroy(CommData* comm_data);
