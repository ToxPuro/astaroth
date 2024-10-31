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
    MPI_Comm m_cart_comm;
    Shape m_local_mm;
    Shape m_local_nn;
    Index m_local_rr;

    Segment m_segment; // Shape of the data block the packet represents (for packing or unpacking)

    std::unique_ptr<Buffer<T>> m_send_buffer; // Buffer holding the data
    std::unique_ptr<Buffer<T>> m_recv_buffer; // Buffer holding the data
    MPI_Request m_send_req;                   // MPI request for handling synchronization
    MPI_Request m_recv_req;                   // MPI request for handling synchronization

  public:
    Packet(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
           const Segment& segment, const size_t n_aggregate_buffers)
        : m_cart_comm(MPI_COMM_NULL),
          m_local_mm(local_mm),
          m_local_nn(local_nn),
          m_local_rr(local_rr),
          m_segment(segment),
          m_send_buffer(std::make_unique<Buffer<T>>(n_aggregate_buffers * prod(m_segment.dims))),
          m_recv_buffer(std::make_unique<Buffer<T>>(n_aggregate_buffers * prod(m_segment.dims))),
          m_send_req(MPI_REQUEST_NULL),
          m_recv_req(MPI_REQUEST_NULL)
    {
    }

    void launch(const MPI_Comm& parent_comm, const PackPtrArray<T*> inputs)
    {
        const size_t count = inputs.count * prod(m_segment.dims);
        ERRCHK(count <= m_send_buffer->count);
        ERRCHK(count <= m_recv_buffer->count);

        // Duplicate the communicator to ensure the operation does not interfere
        // with other operations on the parent communicator
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &m_cart_comm));

        Index send_offset = ((m_local_nn + m_segment.offset - m_local_rr) % m_local_nn) +
                            m_local_rr;

        const Direction recv_direction = get_direction(m_segment.offset, m_local_nn, m_local_rr);
        const int recv_neighbor        = get_neighbor(m_cart_comm, recv_direction);
        const int send_neighbor        = get_neighbor(m_cart_comm, -recv_direction);

        ERRCHK_MPI_EXPR_DESC(m_recv_req == MPI_REQUEST_NULL,
                             "A request was still in flight. This should not happen: call "
                             "wait after launch to synchronize.");

        // Post recv
        const int tag = 0;
        ERRCHK_MPI_API(MPI_Irecv(m_recv_buffer->data, as<int>(count), get_dtype<T>(), recv_neighbor,
                                 tag, m_cart_comm, &m_recv_req));

        // Block until the previous send has completed before reusing the buffer
        ERRCHK_MPI_EXPR_DESC(m_send_req == MPI_REQUEST_NULL,
                             "A request was still in flight. This should not happen: call "
                             "wait after launch to synchronize.");

        // Pack, post send, and ensure the message has left the pack buffer
        pack(m_local_mm, m_segment.dims, send_offset, inputs, m_send_buffer->data);
        ERRCHK_MPI_API(MPI_Isend(m_send_buffer->data, as<int>(count), get_dtype<T>(), send_neighbor,
                                 tag, m_cart_comm, &m_send_req));

        ERRCHK_MPI_API(MPI_Comm_free(&m_cart_comm));
    }

    void wait(PackPtrArray<T*> outputs)
    {
        ERRCHK_MPI_EXPR_DESC(m_recv_req != MPI_REQUEST_NULL,
                             "wait called but no request in flight");
        wait_request(m_recv_req);
        ERRCHK(m_recv_req == MPI_REQUEST_NULL);
        unpack(m_recv_buffer->data, m_local_mm, m_segment.dims, m_segment.offset, outputs);

        ERRCHK_MPI_EXPR_DESC(m_send_req != MPI_REQUEST_NULL,
                             "wait called but no request in flight");
        wait_request(m_send_req);
        ERRCHK(m_send_req == MPI_REQUEST_NULL);
    }

    ~Packet()
    {
        ERRCHK_MPI_EXPR_DESC(m_send_req == MPI_REQUEST_NULL,
                             "Attempted to destroy Packet when there was still "
                             "a request in flight. This should not happen. Call wait after launch "
                             "to synchronize.");
        ERRCHK_MPI_EXPR_DESC(m_recv_req == MPI_REQUEST_NULL,
                             "Attempted to destroy Packet when there was still "
                             "a request in flight. This should not happen. Call wait after launch "
                             "to synchronize.");
        ERRCHK_MPI(m_cart_comm == MPI_COMM_NULL);
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
    os << "m_segment: " << obj.m_segment << ", ";
    os << "buffer: " << obj.buffer << ", ";
    os << "req: " << obj.req;
    os << "}";
    return os;
}

void test_packet();
