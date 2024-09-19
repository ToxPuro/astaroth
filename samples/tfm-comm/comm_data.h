#pragma once
#include "stddef.h"

#include "packed_data.h"

typedef struct {
    size_t npackets;
    PackedData* local_packets;
    PackedData* remote_packets;
} CommData;

CommData acCommDataCreate(const size_t ndims, const size_t nfields);

void acCommDataDestroy(CommData* comm_data);
