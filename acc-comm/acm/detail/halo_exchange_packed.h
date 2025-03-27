#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "packet.h"
#include "partition.h"

namespace ac::comm {

template <typename T, typename Allocator = ac::mr::device_allocator>
class async_halo_exchange_task {
  private:
    std::vector<std::unique_ptr<ac::comm::packet<T, Allocator>>> m_packets{};

  public:
    async_halo_exchange_task() = default;

    async_halo_exchange_task(const ac::shape& local_mm, const ac::shape& local_nn,
                             const ac::index& local_rr, const size_t n_aggregate_buffers)
    {
        // Must be larger than the boundary area to avoid boundary artifacts
        ERRCHK_MPI(local_nn >= local_rr);

        // Partition the mesh
        auto segments{partition(local_mm, local_nn, local_rr)};

        // Prune the segment containing the computational domain
        auto it{std::remove_if(segments.begin(),
                               segments.end(),
                               [local_nn, local_rr](const ac::segment& segment) {
                                   return within_box(segment.offset, local_nn, local_rr);
                               })};
        segments.erase(it, segments.end());

        // Create packed send/recv buffers
        for (const auto& segment : segments) {
            m_packets.push_back(
                std::make_unique<ac::comm::packet<T, Allocator>>(local_mm,
                                                                 local_nn,
                                                                 local_rr,
                                                                 segment,
                                                                 n_aggregate_buffers));
        }
    }

    void launch(const MPI_Comm&                                   comm,
                const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        static int16_t tag{0};
        for (auto& packet : m_packets){
            
            constexpr int MPI_TAG_UB_MIN_VALUE{32767}; // Note duplicated code: also set in mpi_utils. TODO fix
            ERRCHK_MPI(m_packets.size() < MPI_TAG_UB_MIN_VALUE);
            
            packet->launch(comm, tag, inputs);
            
            ac::mpi::increment_tag(&tag);
        }
    }

    void wait(std::vector<ac::mr::pointer<T, Allocator>> outputs)
    {
        // Round-robin busy-wait to choose packet to unpack
        while (!complete()) {
            for (auto& packet : m_packets)
                if (!packet->complete() && packet->ready())
                    packet->wait(outputs);
        }
        // Simple loop over the packets
        // for (auto& packet : m_packets)
        //     packet->wait(outputs);
    }

    void launchwait(const MPI_Comm& comm, const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                    std::vector<ac::mr::pointer<T, Allocator>> outputs)
    {
        static int16_t tag{0};
        if (ac::mpi::get_rank(comm) == 0)
            std::cerr << "-----------------" << std::endl;
        for (auto& packet : m_packets) {
            constexpr int MPI_TAG_UB_MIN_VALUE{
                32767}; // Note duplicated code: also set in mpi_utils. TODO fix
            ERRCHK_MPI(m_packets.size() < MPI_TAG_UB_MIN_VALUE);

            const auto start{std::chrono::system_clock::now()};
            packet->launch(comm, tag, inputs);
            packet->wait(outputs);
            const auto elapsed{std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - start)};
            if (ac::mpi::get_rank(comm) == 0 && ac::prod(packet->segment().dims) >= 64 * 64 * 32)
                std::cerr << "Rank " << ac::mpi::get_rank(comm) << ", " << packet->segment() << ", "
                          << elapsed.count() << std::endl;

            ac::mpi::increment_tag(&tag);
        }
        if (ac::mpi::get_rank(comm) == 0)
            std::cerr << "-----------------" << std::endl;
    }

    bool complete() const
    {
        const bool cc_allof_result{
            std::all_of(m_packets.begin(),
                        m_packets.end(),
                        std::mem_fn(&ac::comm::packet<T, Allocator>::complete))};

        // TODO remove and return the cc_allof_result after testing
        for (const auto& packet : m_packets) {
            if (!packet->complete()) {
                ERRCHK_MPI(cc_allof_result == false);
                return false;
            }
        }
        ERRCHK_MPI(cc_allof_result == true);
        return true;
    }
};

} // namespace ac::comm
