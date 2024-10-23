#pragma once

#include "buffer.h"
#include "segment.h"

#include <mpi.h>

template <typename T> struct Packet {
    Segment segment;  // Shape of the data block the packet represents
    Buffer<T> buffer; // Buffer holding the data
    MPI_Request req;  // MPI request for handling synchronization

    Packet(const Segment& segment, const size_t n_aggregate_buffers)
        : segment(segment), buffer(Buffer<T>(n_aggregate_buffers * prod(segment.dims)))
    {
    }
};

template <typename T> __host__ std::ostream& operator<<(std::ostream& os, const Packet<T>& obj);
