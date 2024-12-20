#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "pack_batched.h"
#include "packet.h"
#include "partition.h"

namespace ac::comm {

template <typename T, typename MemoryResource = ac::mr::device_memory_resource>
class AsyncHaloExchangeTask {
  private:
    std::vector<std::unique_ptr<Packet<T, MemoryResource>>> packets{};

    Shape _local_mm;                   // Experimental
    std::vector<ac::segment> segments; // Experimental

  public:
    AsyncHaloExchangeTask() = default;

    AsyncHaloExchangeTask(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
                          const size_t n_aggregate_buffers)
        : _local_mm{local_mm}

    {
        // Must be larger than the boundary area to avoid boundary artifacts
        ERRCHK_MPI(local_nn >= local_rr);

        // Partition the mesh
        segments = partition(local_mm, local_nn, local_rr);

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
    void launch_batched(const MPI_Comm& parent_comm,
                        const std::vector<ac::mr::base_ptr<T, MemoryResource>>& inputs)
    {
        std::vector<ac::mr::base_ptr<T, MemoryResource>> outputs;
        for (const auto& packet : packets)
            outputs.push_back(packet->get_send_buffer_ptr());

        pack_batched(_local_mm, inputs, segments, outputs);

        for (auto& packet : packets)
            packet->launch_batched(parent_comm);
    }

    // Experimental
    void wait_batched(std::vector<ac::mr::base_ptr<T, MemoryResource>> outputs)
    {
        for (auto& packet : packets)
            packet->wait_batched();

        std::vector<ac::mr::base_ptr<T, MemoryResource>> inputs;
        for (const auto& packet : packets)
            inputs.push_back(packet->get_recv_buffer_ptr());

        unpack_batched(segments, inputs, _local_mm, outputs);
    }

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
