#pragma once

#include <vector>

#include "pack.h"
#include "shape.h"

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

    void wait_request()
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
        ERRCHK_MPI_EXPR_DESC(req != MPI_REQUEST_NULL,
                             "Attempted to destroy Packet when there was still "
                             "a request in flight. This should not happen.");
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

template <typename T> struct HaloExchangeTask {
    Shape local_mm;

    std::vector<Segment> segments;
    std::vector<Buffer<T>> send_buffers;
    std::vector<Buffer<T>> recv_buffers;

    HaloExchangeTask(const Shape& local_mm, const Shape& local_nn, const Index& rr,
                     const size_t n_aggregate_buffers)
        : local_mm(local_mm)
    {
        // Partition the mesh
        segments = partition(local_mm, local_nn, rr);

        // Prune the segment containing the computational domain
        for (size_t i = 0; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }

        // Create packed send/recv buffers
        for (const auto& segment : segments) {
            send_buffers.emplace_back(Buffer<T>(n_aggregate_buffers * prod(segment.dims)));
            recv_buffers.emplace_back(Buffer<T>(n_aggregate_buffers * prod(segment.dims)));
        }
    }

    // void launch(const MPI_Comm& parent_comm, PackInputs<T*> inputs);

    // void wait()
};

// template <typename T>
// HaloExchangeTask<T>::HaloExchangeTask(const Shape& local_mm, const size_t
// n_aggregate_buffers)
//     : local_mm(local_mm)
// {
//     // // Partition the mesh
//     // auto segments = partition(local_mm, local_nn, rr);

//     // // Prune the segment containing the computational domain
//     // for (size_t i = 0; i < segments.size(); ++i) {
//     //     if (within_box(segments[i].offset, local_nn, rr)) {
//     //         segments.erase(segments.begin() + as<long>(i));
//     //         --i;
//     //     }
//     // }

//     // for (const segment& : segments) {
//     //     Index recv_offset = segment.offset;
//     //     Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;

//     //     const Direction recv_direction = get_direction(segment.offset, local_nn, rr);
//     //     const int recv_neighbor        = get_neighbor(cart_comm, recv_direction);
//     //     const int send_neighbor        = get_neighbor(cart_comm, -recv_direction);
//     // }
// }
