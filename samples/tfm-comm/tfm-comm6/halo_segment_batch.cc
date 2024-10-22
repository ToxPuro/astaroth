#include "halo_segment_batch.h"

Buffer::Buffer(const size_t count) : count(count)
{
    std::cout << "created" << std::endl;
    data = new double[count]{};
    ERRCHK(data);
}
Buffer::~Buffer()
{
    std::cout << "deleted" << std::endl;
    count = 0;
    delete[] data;
}

Buffer::Buffer(const Buffer& other)
{
    std::cout << "copied" << std::endl;
    data = new double[other.count]{};
    std::copy(other.data, other.data + count, data);
}

Buffer&
Buffer::operator=(const Buffer& other)
{
    std::cout << "assigned" << std::endl;
    if (this != &other) {
        delete[] data;
        data = new double[other.count]{};
        std::copy(other.data, other.data + count, data);
    }
    return *this;
}

__host__ std::ostream&
operator<<(std::ostream& os, const Buffer& obj)
{
    const size_t max_count = 10;
    os << "{";
    for (size_t i = 0; i < std::min(max_count, obj.count); ++i)
        os << obj.data[i] << (i + 1 < obj.count ? ", " : "");
    if (obj.count > max_count)
        os << ", ...";
    os << "}";
    return os;
}

Segment::Segment(const Shape& dims, const Shape& subdims, const Index& offset)
    : dims(dims), subdims(subdims), offset(offset)
{
}

__host__ std::ostream&
operator<<(std::ostream& os, const Segment& obj)
{
    os << "{\n";
    os << "    dims: " << obj.dims << "," << std::endl;
    os << "    subdims: " << obj.subdims << "," << std::endl;
    os << "    offset: " << obj.offset << std::endl;
    os << "}";
    return os;
}

Packet::Packet(const Segment& segment, const size_t n_aggregate_buffers)
    : segment(segment), buffer(n_aggregate_buffers * prod(segment.subdims)), req(MPI_REQUEST_NULL)
{
}

__host__ std::ostream&
operator<<(std::ostream& os, const Packet& obj)
{
    os << "{\n";
    os << "    segment: " << obj.segment << "," << std::endl;
    os << "    buffer: " << obj.buffer << "," << std::endl;
    os << "    req: " << obj.req << std::endl;
    os << "}";
    return os;
}

HaloSegmentBatch::HaloSegmentBatch(const Shape& local_mm, const Shape& local_nn,
                                   const Index& local_nn_offset, const size_t n_aggregate_buffers)
{
    // std::vector<Segment> segments = partition(local_mm, local_nn, local_nn_offset);
    local_packets.push_back(Packet(Segment(local_mm, local_nn, local_nn_offset), 1));
    local_packets.push_back(Packet(Segment(local_mm, local_nn, local_nn_offset), 1));
    local_packets.erase(local_packets.begin());
}
HaloSegmentBatch::~HaloSegmentBatch() {}

__host__ std::ostream&
operator<<(std::ostream& os, const HaloSegmentBatch& obj)
{
    os << obj.local_packets << std::endl;
    os << obj.remote_packets << std::endl;
    return os;
}
