#pragma once

#include "segment.h"
#include "shape.h"

#include <mpi.h>

#include <vector>

struct Buffer {
    size_t count;
    double* data;

    Buffer(const size_t count);
    ~Buffer();
    Buffer(const Buffer& other);
    Buffer& operator=(const Buffer& other);
};

struct Segment {
    Shape dims;    // Dimensions of the parent body
    Shape subdims; // Dimensions of the segment
    Index offset;  // Offset of the segment

    Segment(const Shape& dims, const Shape& subdims, const Index& offset);
};
// __host__ std::ostream& operator<<(std::ostream& os, const Segment& obj);

struct Packet {
    Segment segment; // Shape of the data block the packet represents
    Buffer buffer;   // Buffer holding the data
    MPI_Request req; // MPI request for handling synchronization

    Packet(const Segment& segment, const size_t n_aggregate_buffers);
};
__host__ std::ostream& operator<<(std::ostream& os, const Packet& obj);

// struct HaloSegmentBatch {
//     std::vector<Packet> local_packets;
//     std::vector<Packet> remote_packets;

//     HaloSegmentBatch(const Shape& local_mm, const Shape& local_nn, const Index& local_nn_offset,
//                      const size_t n_aggregate_buffers);
//     ~HaloSegmentBatch();
// };
// __host__ std::ostream& operator<<(std::ostream& os, const HaloSegmentBatch& obj);
