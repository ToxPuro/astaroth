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
    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;

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
            send_buffers.push_back(Buffer<T>(n_aggregate_buffers * prod(segment.dims)));
            recv_buffers.push_back(Buffer<T>(n_aggregate_buffers * prod(segment.dims)));
        }
    }

    void wait(const MPI_Comm& parent_comm, const Shape& local_nn, const Shape& rr,
              PackInputs<T*> outputs)
    {
        // Duplicate the communicator to ensure the operation does not interfere
        // with other operations on the parent communicator
        // NOTE: DANGEROUS: SHOULD USE THE SAME, ANY SYNCHRONIZATION WITH THIS DUP IS INVALID
        MPI_Comm cart_comm;
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &cart_comm));

        for (size_t i = 0; i < segments.size(); ++i) {
            Index recv_offset = segments[i].offset;
            Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;

            const Direction recv_direction = get_direction(segments[i].offset, local_nn, rr);
            const int recv_neighbor        = get_neighbor(cart_comm, recv_direction);
            const int send_neighbor        = get_neighbor(cart_comm, -recv_direction);

            wait_request(recv_reqs[i]);
            unpack(recv_buffers[i].data, local_mm, segments[i].dims, recv_offset, outputs);
        }
        recv_reqs.clear();
        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
        // while (!recv_reqs.empty()) {
        //     // Round-robin: choose packet to receive and unpack
        //     // TODO

        //     // Just wait in order
        //     wait_request(recv_reqs.back());
        //     recv_reqs.pop_back();

        //     unpack(recv_buffers[i].)
        // }
    }

    void launch(const MPI_Comm& parent_comm, const Shape& local_nn, const Index& rr,
                const PackInputs<T*> inputs)
    {
        ERRCHK_MPI(recv_reqs.empty()); // Too strict, TODO make leaner

        // Duplicate the communicator to ensure the operation does not interfere
        // with other operations on the parent communicator
        MPI_Comm cart_comm;
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &cart_comm));

        // Post sends and recvs
        for (size_t i = 0; i < segments.size(); ++i) {
            Index recv_offset = segments[i].offset;
            Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;

            const Direction recv_direction = get_direction(segments[i].offset, local_nn, rr);
            const int recv_neighbor        = get_neighbor(cart_comm, recv_direction);
            const int send_neighbor        = get_neighbor(cart_comm, -recv_direction);

            const int tag = get_tag();

            // Post recv
            MPI_Request recv_req;
            ERRCHK_MPI_API(MPI_Irecv(recv_buffers[i].data, as<int>(recv_buffers[i].count),
                                     MPI_DOUBLE, recv_neighbor, tag, cart_comm, &recv_req));
            recv_reqs.push_back(recv_req);

            // Pack and post send
            pack(local_mm, segments[i].dims, send_offset, inputs, send_buffers[i].data);

            MPI_Request send_req;
            ERRCHK_MPI_API(MPI_Isend(send_buffers[i].data, as<int>(send_buffers[i].count),
                                     MPI_DOUBLE, send_neighbor, tag, cart_comm, &send_req));
            send_reqs.push_back(send_req);
        }
        while (!send_reqs.empty()) {
            wait_request(send_reqs.back());
            send_reqs.pop_back();
        }

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    }
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
