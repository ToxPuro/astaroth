#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "pack.h"
#include "packet.h"
#include "partition.h"

namespace ac::comm {

template <typename T, typename MemoryResource = ac::mr::device_memory_resource>
class AsyncHaloExchangeTask {
  private:
    std::vector<std::unique_ptr<Packet<T, MemoryResource>>> packets{};

  public:
    AsyncHaloExchangeTask() = default;

    AsyncHaloExchangeTask(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
                          const size_t n_aggregate_buffers)

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
            packets.push_back(std::make_unique<Packet<T, MemoryResource>>(local_mm,
                                                                          local_nn,
                                                                          local_rr,
                                                                          segment,
                                                                          n_aggregate_buffers));
        }
    }

    // Experimental
    void pack_batched(const std::vector<ac::mr::base_ptr<T, MemoryResource>>& inputs)
    {
        // TODO
        // ac::pack_all(inputs, segments, packets);
    }

    void launch_batched(const MPI_Comm& parent_comm,
                        const std::vector<ac::mr::base_ptr<T, MemoryResource>>& inputs)
    {
        // TODO
        // pack_all(inputs, segments, packets);
        // for (auto& packet : packets)
        //     packet->launch_experimental(parent_comm, inputs);
    }

    // Experimental

    void launch_pipelined(const MPI_Comm& parent_comm,
                          const std::vector<ac::mr::base_ptr<T, MemoryResource>>& inputs)
    {
        for (auto& packet : packets)
            packet->launch_pipelined(parent_comm, inputs);
    }

    void wait(std::vector<ac::mr::base_ptr<T, MemoryResource>> outputs)
    {
        // Round-robin busy-wait to choose packet to unpack
        while (!complete()) {
            for (auto& packet : packets)
                if (!packet->complete() && packet->ready())
                    packet->wait(outputs);
        }
        // Simple loop over the packets
        // for (auto& packet : packets)
        //     packet->wait(outputs);
    }

    bool complete() const
    {
        const bool cc_allof_result{std::all_of(packets.begin(),
                                               packets.end(),
                                               std::mem_fn(&Packet<T, MemoryResource>::complete))};

        // TODO remove and return the cc_allof_result after testing
        for (const auto& packet : packets) {
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
