#pragma once

#include <memory>
#include <vector>

#include "packet.h"

template <typename T, size_t N> class HaloExchangeTask {
  private:
    std::vector<std::unique_ptr<Packet<T, N>>> packets;

  public:
    HaloExchangeTask(const Shape<N>& local_mm, const Shape<N>& local_nn, const Index<N>& local_rr,
                     const size_t n_aggregate_buffers)

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
            packets.push_back(std::make_unique<Packet<T, N>>(local_mm, local_nn, local_rr, segment,
                                                             n_aggregate_buffers));
        }
    }

    template <size_t M> void launch(const MPI_Comm& parent_comm, const ac::array<T*, M>& inputs)
    {
        for (auto& packet : packets)
            packet->launch(parent_comm, inputs);
    }

    template <size_t M> void wait(ac::array<T*, M>& outputs)
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
