#pragma once
#include <stddef.h>

#include "packet.h"

struct HaloSegmentBatch_s {
    size_t ndims;
    size_t* local_mm;
    size_t* local_nn;
    size_t* local_nn_offset;

    size_t npackets;
    Packet* local_packets;
    Packet* remote_packets;
};
