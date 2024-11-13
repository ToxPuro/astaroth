#pragma once

#include <memory>
#include <vector>

#include "packet.h"

template <typename T> class HaloExchangeTask {
  private:
    std::vector<Packet<T>> packets;

  public:
    HaloExchangeTask(const MPI_Comm& parent_comm, const Shape& local_mm, const Shape& local_nn,
                     const Index& local_rr, const size_t n_aggregate_buffers)

    {
        // Must be larger than the boundary area to avoid boundary artifacts
        ERRCHK_MPI(local_nn >= local_rr);

        // Partition the mesh
        auto segments = partition(local_mm, local_nn, local_rr);

        // Prune the segment containing the computational domain
        for (size_t i = 0; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, local_rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }

        // Create packed send/recv buffers
        for (const auto& segment : segments) {
            packets.push_back(
                Packet<T>(parent_comm, local_mm, local_nn, local_rr, segment, n_aggregate_buffers));
        }
    }

    void launch(const PackPtrArray<T*>& inputs)
    {
        ERRCHK_MPI(complete());

        for (auto& packet : packets)
            packet.launch(inputs);
    }

    void wait(const PackPtrArray<T*>& outputs)
    {
        // Round-robin busy-wait to choose packet to unpack
        while (!complete()) {
            for (auto& packet : packets)
                if (!packet.complete() && packet.ready())
                    packet.wait(outputs);
        }
        // Simple loop over the packets
        // for (auto& packet : packets)
        //     packet.wait(outputs);
    }

    bool complete()
    {
        return std::all_of(packets.begin(), packets.end(),
                           [](const Packet<T>& packet) { return packet.complete(); });
    }
};
