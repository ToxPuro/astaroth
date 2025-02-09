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

    void launch(const MPI_Comm&                                   parent_comm,
                const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (auto& packet : m_packets)
            packet->launch(parent_comm, inputs);
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
