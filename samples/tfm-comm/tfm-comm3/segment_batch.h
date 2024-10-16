#pragma once
#include <stddef.h>
#include <stdint.h>

#include "packet.h"

struct HaloSegmentBatch_s {
    size_t ndims;
    uint64_t* local_mm;
    uint64_t* local_nn;
    uint64_t* local_nn_offset;

    size_t npackets;
    Packet* local_packets;
    Packet* remote_packets;
};

struct HaloSegmentBatch_s* halo_segment_batch_create(const size_t ndims, const uint64_t* local_mm,
                                                     const uint64_t* local_nn,
                                                     const uint64_t* local_nn_offset,
                                                     const uint64_t n_aggregate_buffers);

void halo_segment_batch_destroy(struct HaloSegmentBatch_s** batch);

void halo_segment_batch_launch(const MPI_Comm mpi_comm_, const uint64_t ninputs, double* inputs[],
                               struct HaloSegmentBatch_s* batch);

void halo_segment_batch_wait(struct HaloSegmentBatch_s* batch, const uint64_t noutputs,
                             double* outputs[]);

void test_halo_segment_batch(void);
