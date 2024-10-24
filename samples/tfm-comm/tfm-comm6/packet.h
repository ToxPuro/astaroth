#pragma once

#include "buffer.h"
#include "segment.h"

#include "errchk_mpi.h"

#include <mpi.h>

template <typename T> struct Packet {
    Segment segment;  // Shape of the data block the packet represents
    Buffer<T> buffer; // Buffer holding the data
    MPI_Request req;  // MPI request for handling synchronization

    Packet(const Segment& segment, const size_t n_aggregate_buffers)
        : segment(segment),
          buffer(Buffer<T>(n_aggregate_buffers * prod(segment.dims))),
          req(MPI_REQUEST_NULL)
    {
    }

    void wait()
    {
        if (req != MPI_REQUEST_NULL) {
            // Note: MPI_Status needs to be initialized.
            // Otherwise leads to uninitialized memory access and causes spurious errors
            // because MPI_Wait does not modify the status on successful MPI_Wait
            MPI_Status status = {.MPI_ERROR = MPI_SUCCESS};
            ERRCHK_MPI_API(MPI_Wait(&req, &status));
            ERRCHK_MPI_API(status.MPI_ERROR);
            // Some MPI implementations free the request with MPI_Wait
            if (req != MPI_REQUEST_NULL)
                ERRCHK_MPI_API(MPI_Request_free(&req));
            ERRCHK(req == MPI_REQUEST_NULL);
        }
        else {
            WARNING_DESC("packet_wait called but no there is packet to wait for");
        }
    }

    ~Packet()
    {
        if (req != MPI_REQUEST_NULL) {
            ERROR_DESC(
                "Attempted to destroy Packet when there was a request in flight. This should "
                "not happen.");
            ERRCHK_MPI(req != MPI_REQUEST_NULL);
        }
    }

    // Delete all other types of constructors
    Packet(const Packet&)            = delete; // Copy constructor
    Packet& operator=(const Packet&) = delete; // Copy assignment operator
    Packet(Packet&&)                 = delete; // Move constructor
    Packet& operator=(Packet&&)      = delete; // Move assignment operator
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

void test_packet(void);
