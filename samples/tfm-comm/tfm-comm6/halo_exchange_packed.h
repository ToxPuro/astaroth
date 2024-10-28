#pragma once

#include <memory>
#include <vector>

#include "packet.h"

template <typename T> struct HaloExchangeTask {
    std::vector<std::unique_ptr<Packet<T>>> packets;

    HaloExchangeTask(const Shape& local_mm, const Shape& local_nn, const Index& local_rr,
                     const size_t n_aggregate_buffers)

    {
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
            packets.push_back(std::make_unique<Packet<T>>(local_mm, local_nn, local_rr, segment,
                                                          n_aggregate_buffers));
        }
    }

    void launch(const MPI_Comm parent_comm, const PackInputs<T*> inputs)
    {
        for (auto& packet : packets)
            packet->launch(parent_comm, inputs);
    }

    void wait(const PackInputs<T*> outputs)
    {
        for (auto& packet : packets)
            packet->wait(outputs);
    }
};
