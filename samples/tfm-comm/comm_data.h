#pragma once
#include "stddef.h"

#include "packed_data.h"

typedef struct {
    size_t npackets;
    PackedData* local_packets;
    PackedData* remote_packets;
} CommData;

CommData acCommDataCreate(const size_t ndims, const size_t* rr, const size_t* nn,
                          const size_t nfields);

void acCommDataPrint(const char* label, const CommData comm_data);

void acCommDataDestroy(CommData* comm_data);
