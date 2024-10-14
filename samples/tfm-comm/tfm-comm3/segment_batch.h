#pragma once
#include <stddef.h>
#include <stdint.h>

#include "packet.h"

struct HaloSegmentBatch_s {
    size_t ndims;
    uint64_t* local_mm;
    uint64_t* local_nn;
    uint64_t* local_nn_offset;

    size_t npackets;
    Packet* local_packets;
    Packet* remote_packets;
};
