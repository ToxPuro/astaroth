#pragma once

#include <memory>
#include <mpi.h>
#include <vector>

#include "buffer.h"
#include "pack.h"
#include "partition.h"
#include "pointer.h"
#include "type_conversion.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include "convert.h"

namespace acm::rev {

template <typename T, typename Allocator> class packet {
  private:
    MPI_Comm m_comm{MPI_COMM_NULL};
    int16_t  m_tag{0};

    int m_recv_neighbor{MPI_PROC_NULL};
    int m_send_neighbor{MPI_PROC_NULL};

    MPI_Request m_recv_req{MPI_REQUEST_NULL};
    MPI_Request m_send_req{MPI_REQUEST_NULL};

  public:
    packet(const MPI_Comm& parent_comm, const int recv_neighbor, const int send_neighbor)
        : m_recv_neighbor{recv_neighbor}, m_send_neighbor{send_neighbor}
    {
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &m_comm));
    }

    ~packet()
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Comm_free(&m_comm));
    }

    packet(const packet&)            = delete; // Copy constructor
    packet& operator=(const packet&) = delete; // Copy assignment operator
    packet(packet&&)                 = delete; // Move constructor
    packet& operator=(packet&&)      = delete; // Move assignment operator

    void launch(const ac::mr::pointer<T, Allocator>& input, ac::mr::pointer<T, Allocator> output)
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(input.size() == output.size());

        const auto rank{ac::mpi::get_rank(m_comm)};
        if (rank == m_recv_neighbor && rank == m_send_neighbor) {
            ac::mr::copy(input, output);

            // Create a dummy request to simplify asynchronous management
            static int dummy{-1};
            ERRCHK_MPI_API(MPI_Irecv(&dummy, 0, MPI_INT, rank, m_tag, m_comm, &m_recv_req));
            ERRCHK_MPI_API(MPI_Isend(&dummy, 0, MPI_INT, rank, m_tag, m_comm, &m_send_req));
        }
        else { // Use MPI communication
            ERRCHK_MPI_API(MPI_Irecv(output.data(),
                                     as<int>(output.size()),
                                     ac::mpi::get_dtype<T>(),
                                     m_recv_neighbor,
                                     m_tag,
                                     m_comm,
                                     &m_recv_req));

            ERRCHK_MPI_API(MPI_Isend(input.data(),
                                     as<int>(input.size()),
                                     ac::mpi::get_dtype<T>(),
                                     m_send_neighbor,
                                     m_tag,
                                     m_comm,
                                     &m_send_req));
        }

        ac::mpi::increment_tag(m_tag);
    }

    void wait()
    {
        ERRCHK_MPI(m_send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req != MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Wait(&m_recv_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Wait(&m_send_req, MPI_STATUS_IGNORE));

        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
    }
};

template <typename T, typename Allocator> class halo_exchange {
  private:
    ac::shape m_local_mm;

    std::vector<ac::segment> m_recv_segments;
    std::vector<ac::segment> m_send_segments;

    std::vector<std::unique_ptr<ac::buffer<T, Allocator>>> m_recv_buffers;
    std::vector<std::unique_ptr<ac::buffer<T, Allocator>>> m_send_buffers;

    std::vector<std::unique_ptr<packet<T, Allocator>>> m_packets;

  public:
    halo_exchange(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
                  const size_t n_max_aggregate_buffers)
        : m_local_mm{ac::mpi::get_local_mm(parent_comm, global_nn, rr)}
    {
        // Get local mesh dimensions
        const auto local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};

        // Partition the domain
        auto segments{partition(m_local_mm, local_nn, rr)};

        // Prune the segment containing the computational domain
        segments = prune(segments, local_nn, rr);

        // Sort the packets from largest to smallest
        std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
            return ac::prod(a.dims) > ac::prod(b.dims);
        });

        for (const auto& segment : segments) {

            const ac::segment recv_segment{segment};
            const ac::segment send_segment{recv_segment.dims,
                                           ((local_nn + recv_segment.offset - rr) % local_nn) + rr};

            const ac::direction recv_direction{
                ac::mpi::get_direction(recv_segment.offset, local_nn, rr)};
            const auto recv_neighbor{ac::mpi::get_neighbor(parent_comm, recv_direction)};
            const auto send_neighbor{ac::mpi::get_neighbor(parent_comm, -recv_direction)};

            m_recv_segments.push_back(recv_segment);
            m_send_segments.push_back(send_segment);

            m_recv_buffers.push_back(std::make_unique<ac::buffer<T, Allocator>>(
                n_max_aggregate_buffers * prod(recv_segment.dims)));
            m_send_buffers.push_back(std::make_unique<ac::buffer<T, Allocator>>(
                n_max_aggregate_buffers * prod(send_segment.dims)));

            m_packets.push_back(
                std::make_unique<packet<T, Allocator>>(parent_comm, recv_neighbor, send_neighbor));
        }
    }

    void launch(const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        // ERRCHK_MPI(
        //     same_size(m_recv_segments, m_send_segments, m_recv_buffers, m_send_buffers,
        //     m_packets));

        auto unwrap = [](const std::vector<std::unique_ptr<ac::buffer<T, Allocator>>>& vec) {
            std::vector<ac::mr::pointer<T, Allocator>> ptrs;
            for (const auto& elem : vec)
                ptrs.push_back(elem->get());
            return ptrs;
        };

        pack_batched(m_local_mm, inputs, m_send_segments, unwrap(m_send_buffers));

        for (size_t i{0}; i < m_packets.size(); ++i)
            m_packets[i]->launch(m_send_buffers[i]->get(), m_recv_buffers[i]->get());
    }

    void wait(std::vector<ac::mr::pointer<T, Allocator>> outputs)
    {
        for (auto& packet : m_packets)
            packet->wait();

        auto unwrap = [](const std::vector<std::unique_ptr<ac::buffer<T, Allocator>>>& vec) {
            std::vector<ac::mr::pointer<T, Allocator>> ptrs;
            for (const auto& elem : vec)
                ptrs.push_back(elem->get());
            return ptrs;
        };

        unpack_batched(m_recv_segments, unwrap(m_recv_buffers), m_local_mm, outputs);
    }
};

} // namespace acm::rev
