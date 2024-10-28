#pragma once

#include <memory>

#include "buffer.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

template <typename T> struct Packet {
    MPI_Comm cart_comm;
    Shape local_mm;
    Shape local_nn;
    Index local_rr;
    Segment segment; // Shape of the data block the packet represents (for packing or unpacking)
    std::unique_ptr<Buffer<T>> send_buffer; // Buffer holding the data
    std::unique_ptr<Buffer<T>> recv_buffer; // Buffer holding the data
    MPI_Request send_req;                   // MPI request for handling synchronization
    MPI_Request recv_req;                   // MPI request for handling synchronization

    Packet(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
           const Segment& segment, const size_t n_aggregate_buffers)
        : cart_comm(MPI_COMM_NULL),
          local_mm(local_mm),
          local_nn(local_nn),
          local_rr(local_rr),
          segment(segment),
          send_buffer(std::make_unique<Buffer<T>>(n_aggregate_buffers * prod(segment.dims))),
          recv_buffer(std::make_unique<Buffer<T>>(n_aggregate_buffers * prod(segment.dims))),
          send_req(MPI_REQUEST_NULL),
          recv_req(MPI_REQUEST_NULL)
    {
    }

    void launch(const MPI_Comm& parent_comm, const PackInputs<T*> inputs)
    {
        const size_t count = inputs.count * prod(segment.dims);
        ERRCHK(count <= send_buffer->count);
        ERRCHK(count <= recv_buffer->count);

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
        ERRCHK_MPI_API(MPI_Irecv(recv_buffer->data, as<int>(count), MPI_DOUBLE, recv_neighbor, tag,
                                 cart_comm, &recv_req));

        // Block until the previous send has completed before reusing the buffer
        ERRCHK_MPI_EXPR_DESC(send_req == MPI_REQUEST_NULL,
                             "A request was still in flight. This should not happen: call "
                             "wait after launch to synchronize.");

        // Pack, post send, and ensure the message has left the pack buffer
        pack(local_mm, segment.dims, send_offset, inputs, send_buffer->data);
        ERRCHK_MPI_API(MPI_Isend(send_buffer->data, as<int>(count), MPI_DOUBLE, send_neighbor, tag,
                                 cart_comm, &send_req));

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    }

    void wait(PackInputs<T*> outputs)
    {
        ERRCHK_MPI_EXPR_DESC(recv_req != MPI_REQUEST_NULL, "wait called but no request in flight");
        wait_request(recv_req);
        ERRCHK(recv_req == MPI_REQUEST_NULL);
        unpack(recv_buffer->data, local_mm, segment.dims, segment.offset, outputs);

        ERRCHK_MPI_EXPR_DESC(send_req != MPI_REQUEST_NULL, "wait called but no request in flight");
        wait_request(send_req);
        ERRCHK(send_req == MPI_REQUEST_NULL);
    }

    ~Packet()
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

    // Delete all other types of constructors
    Packet(const Packet&)            = delete; // Copy constructor
    Packet& operator=(const Packet&) = delete; // Copy assignment operator
    Packet(Packet&&)                 = delete; // Move constructor
    Packet& operator=(Packet&&)      = delete; // Move assignment operator
    // Packet(Packet&&)            = default; // Move constructor
    // Packet& operator=(Packet&&) = default; // Move assignment operator
};

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
