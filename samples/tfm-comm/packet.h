#pragma once

#include <memory>

#include "buf.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

template <typename T> class Packet {

  private:
    MPI_Comm cart_comm;
    Shape local_mm;
    Shape local_nn;
    Index local_rr;

    Segment segment; // Shape of the data block the packet represents (for packing or unpacking)

    GenericBuffer<T> send_buffer;
    GenericBuffer<T> recv_buffer;
    MPI_Request send_req; // MPI request for handling synchronization
    MPI_Request recv_req; // MPI request for handling synchronization

  public:
    Packet(const Shape& in_local_mm, const Shape& in_local_nn, const Index& in_local_rr,
           const Segment& in_segment, const size_t n_aggregate_buffers);
    Packet(const Packet&)            = delete; // Copy
    Packet& operator=(const Packet&) = delete; // Copy assignment
    Packet(Packet&&) noexcept;                 // Move
    Packet& operator=(Packet&&) = delete;      // Move assignment
    ~Packet();

    void launch(const MPI_Comm& parent_comm, const PackPtrArray<T*> inputs);
    void wait(PackPtrArray<T*> outputs);
};

template <typename T>
Packet<T>::Packet(const Shape& in_local_mm, const Shape& in_local_nn, const Index& in_local_rr,
                  const Segment& in_segment, const size_t n_aggregate_buffers)
    : cart_comm(MPI_COMM_NULL),
      local_mm(in_local_mm),
      local_nn(in_local_nn),
      local_rr(in_local_rr),
      segment(in_segment),
      send_buffer(n_aggregate_buffers * prod(in_segment.dims)),
      recv_buffer(n_aggregate_buffers * prod(in_segment.dims)),
      send_req(MPI_REQUEST_NULL),
      recv_req(MPI_REQUEST_NULL)
{
}

template <typename T>
Packet<T>::Packet(Packet&& other) noexcept
    : cart_comm(other.cart_comm),
      local_mm(other.local_mm),
      local_nn(other.local_nn),
      local_rr(other.local_rr),
      segment(other.segment),
      send_buffer(std::move(other.send_buffer)),
      recv_buffer(std::move(other.recv_buffer)),
      send_req(other.send_req),
      recv_req(other.recv_req)
{
    other.cart_comm = MPI_COMM_NULL;
    // other.local_mm    = nullptr;
    // other.local_nn    = nullptr;
    // other.local_rr    = nullptr;
    // other.segment     = nullptr;
    // other.send_buffer = nullptr;
    // other.recv_buffer = nullptr;
    other.send_req = MPI_REQUEST_NULL;
    other.recv_req = MPI_REQUEST_NULL;
}

template <typename T> Packet<T>::~Packet()
{
    ERRCHK_MPI_EXPR_DESC(send_req == MPI_REQUEST_NULL,
                         "Attempted to destroy Packet when there was still "
                         "a request in flight. This should not happen. Call wait after launch "
                         "to synchronize.");
    ERRCHK_MPI_EXPR_DESC(recv_req == MPI_REQUEST_NULL,
                         "Attempted to destroy Packet when there was still "
                         "a request in flight. This should not happen. Call wait after launch "
                         "to synchronize.");
    ERRCHK_MPI(cart_comm == MPI_COMM_NULL);
}

template <typename T>
void
Packet<T>::launch(const MPI_Comm& parent_comm, const PackPtrArray<T*> inputs)
{
    const size_t count = inputs.count * prod(segment.dims);
    ERRCHK(count <= send_buffer.size());
    ERRCHK(count <= recv_buffer.size());

    // Duplicate the communicator to ensure the operation does not interfere
    // with other operations on the parent communicator
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &cart_comm));

    Index send_offset = ((local_nn + segment.offset - local_rr) % local_nn) + local_rr;

    const Direction recv_direction = get_direction(segment.offset, local_nn, local_rr);
    const int recv_neighbor        = get_neighbor(cart_comm, recv_direction);
    const int send_neighbor        = get_neighbor(cart_comm, -recv_direction);

    ERRCHK_MPI_EXPR_DESC(recv_req == MPI_REQUEST_NULL,
                         "A request was still in flight. This should not happen: call "
                         "wait after launch to synchronize.");

    // Post recv
    const int tag = 0;
    ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data(), as<int>(count), get_mpi_dtype<T>(), recv_neighbor,
                             tag, cart_comm, &recv_req));

    // Block until the previous send has completed before reusing the buffer
    ERRCHK_MPI_EXPR_DESC(send_req == MPI_REQUEST_NULL,
                         "A request was still in flight. This should not happen: call "
                         "wait after launch to synchronize.");

    // Pack, post send, and ensure the message has left the pack buffer
    pack(local_mm, segment.dims, send_offset, inputs, send_buffer.data());
    ERRCHK_MPI_API(MPI_Isend(send_buffer.data(), as<int>(count), get_mpi_dtype<T>(), send_neighbor,
                             tag, cart_comm, &send_req));

    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
}

template <typename T>
void
Packet<T>::wait(PackPtrArray<T*> outputs)
{
    ERRCHK_MPI_EXPR_DESC(recv_req != MPI_REQUEST_NULL, "wait called but no request in flight");
    wait_and_destroy_request(recv_req);
    ERRCHK(recv_req == MPI_REQUEST_NULL);
    unpack(recv_buffer.data(), local_mm, segment.dims, segment.offset, outputs);

    ERRCHK_MPI_EXPR_DESC(send_req != MPI_REQUEST_NULL, "wait called but no request in flight");
    wait_and_destroy_request(send_req);
    ERRCHK(send_req == MPI_REQUEST_NULL);
}

template <typename T>
__host__ std::ostream&
operator<<(std::ostream& os, const Packet<T>& obj)
{
    os << "{";
    os << "segment: " << obj.segment << ", ";
    os << "buffer: " << obj.buffer << ", ";
    os << "req: " << obj.req;
    os << "}";
    return os;
}

void test_packet();
