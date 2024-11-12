#pragma once

#include <memory>

#include "buffer.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

template <typename T> class Packet {

  private:
    Shape local_mm;
    Shape local_nn;
    Index local_rr;

    Segment segment; // Shape of the data block the packet represents (for packing or unpacking)

    Buffer<T, DeviceMemoryResource> pack_buffer;
    Buffer<T, HostMemoryResource> send_buffer;
    Buffer<T, HostMemoryResource> recv_buffer;
    Buffer<T, DeviceMemoryResource> unpack_buffer;

    MPICommWrapper cart_comm;
    MPIRequestWrapper send_req; // MPI request for handling synchronization
    MPIRequestWrapper recv_req; // MPI request for handling synchronization

    bool in_progress = false;

  public:
    Packet(const Shape& in_local_mm, const Shape& in_local_nn, const Index& in_local_rr,
           const Segment& in_segment, const size_t n_aggregate_buffers)
        : local_mm(in_local_mm),
          local_nn(in_local_nn),
          local_rr(in_local_rr),
          segment(in_segment),
          pack_buffer(n_aggregate_buffers * prod(in_segment.dims)),
          send_buffer(n_aggregate_buffers * prod(in_segment.dims)),
          recv_buffer(n_aggregate_buffers * prod(in_segment.dims)),
          unpack_buffer(n_aggregate_buffers * prod(in_segment.dims))
    {
    }

    void launch(const MPI_Comm& parent_comm, const PackPtrArray<T*> inputs);
    void wait(PackPtrArray<T*> outputs);
    bool ready();
    bool complete();

    friend __host__ std::ostream& operator<<(std::ostream& os, const Packet<T>& obj)
    {
        os << "{";
        os << "segment: " << obj.segment << ", ";
        os << "buffer: " << obj.buffer << ", ";
        os << "req: " << obj.req;
        os << "}";
        return os;
    }
};

template <typename T>
void
Packet<T>::launch(const MPI_Comm& parent_comm, const PackPtrArray<T*> inputs)
{
    ERRCHK(!in_progress);
    in_progress = true;

    const size_t count = inputs.count * prod(segment.dims);
    ERRCHK(count <= send_buffer.size());
    ERRCHK(count <= recv_buffer.size());

    // Duplicate the communicator to ensure the operation does not interfere
    // with other operations on the parent communicator
    cart_comm = MPICommWrapper(parent_comm);

    Index send_offset = ((local_nn + segment.offset - local_rr) % local_nn) + local_rr;

    const Direction recv_direction = get_direction(segment.offset, local_nn, local_rr);
    const int recv_neighbor        = get_neighbor(cart_comm.value(), recv_direction);
    const int send_neighbor        = get_neighbor(cart_comm.value(), -recv_direction);

    // Post recv
    const int tag = 0;
    ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data(), as<int>(count), get_mpi_dtype<T>(), recv_neighbor,
                             tag, cart_comm.value(), recv_req.get()));

    // Pack, post send, and ensure the message has left the pack buffer
    pack(local_mm, segment.dims, send_offset, inputs, pack_buffer);
    migrate(pack_buffer, send_buffer);
    ERRCHK_MPI_API(MPI_Isend(send_buffer.data(), as<int>(count), get_mpi_dtype<T>(), send_neighbor,
                             tag, cart_comm.value(), send_req.get()));
}

template <typename T>
void
Packet<T>::wait(PackPtrArray<T*> outputs)
{
    // Wait recv
    ERRCHK_MPI_EXPR_DESC(in_progress, "wait called but no request in flight");
    recv_req.wait();

    // Unpack
    migrate(recv_buffer, unpack_buffer);
    unpack(unpack_buffer, local_mm, segment.dims, segment.offset, outputs);

    // Wait send
    send_req.wait();

    in_progress = false;
}

template <typename T>
bool
Packet<T>::ready()
{
    return send_req.ready() && recv_req.ready();
}

template <typename T>
bool
Packet<T>::complete()
{
    return !in_progress;
}

void test_packet();
