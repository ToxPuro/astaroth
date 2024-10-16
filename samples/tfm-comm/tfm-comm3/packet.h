#pragma once
#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

#include "buffer.h"
#include "segment.h"

typedef struct {
    Segment segment; // Shape of the data block the packet represents
    Buffer buffer;   // Buffer holding the data
    MPI_Request req; // MPI request for handling synchronization
} Packet;

Packet packet_create(const size_t ndims, const uint64_t* dims, const uint64_t* offset,
                     const size_t nbuffers);

void packet_wait(Packet* packet);

void packet_destroy(Packet* packet);
