#pragma once

#include <memory>
#include <vector>

#include "packet.h"
#include "partition.h"

template <typename T, typename MemoryResource> class HaloExchangeTask {
  private:
    std::vector<std::unique_ptr<Packet<T, MemoryResource>>> packets;

  public:
    HaloExchangeTask(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
                     const size_t n_aggregate_buffers)

    {
        // Must be larger than the boundary area to avoid boundary artifacts
        ERRCHK_MPI(local_nn >= local_rr);

        // Partition the mesh
        auto segments{partition(local_mm, local_nn, local_rr)};

        // Prune the segment containing the computational domain
        for (size_t i{0}; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, local_rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }

        // Create packed send/recv buffers
        for (const auto& segment : segments) {
            packets.push_back(std::make_unique<Packet<T, MemoryResource>>(local_mm, local_nn,
                                                                          local_rr, segment,
                                                                          n_aggregate_buffers));
        }
    }

    void launch(const MPI_Comm& parent_comm, const std::vector<ac::mr::device_ptr<T>>& inputs)
    {
        for (auto& packet : packets)
            packet->launch(parent_comm, inputs);
    }

    void wait(std::vector<ac::mr::device_ptr<T>>& outputs)
    {
        // Round-robin busy-wait to choose packet to unpack
        // while (!complete()) {
        //     for (auto& packet : packets)
        //         if (!packet->complete() && packet->ready())
        //             packet->wait(outputs);
        // }
        // Simple loop over the packets
        for (auto& packet : packets)
            packet->wait(outputs);
    }

    bool complete() const
    {
        for (const auto& packet : packets)
            if (!packet->complete())
                return false;
        return true;
    }
};
