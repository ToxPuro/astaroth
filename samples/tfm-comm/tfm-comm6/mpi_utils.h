#pragma once
#include <stddef.h>
#include <stdint.h>

#include "errchk_mpi.h"

// void as_mpi_format(const size_t ndims, const uint64_t* dims, int* mpi_dims);

// void as_astaroth_format(const size_t ndims, const int* mpi_dims, uint64_t* dims);

/**
 * At each call, returns the next integer in range [0, INT_MAX].
 * If the counter overflows, the counter wraps around and starts again from 0.
 * NOTE: Not thread-safe.
 */
// int get_tag(void);

// void test_mpi_utils(void);

static inline Shape
decompose_mpi(const int nprocs, const size_t ndims)
{
    MPIShape mpi_decomp(ndims, 0);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as<int>(mpi_decomp.count), mpi_decomp.data));
    return Shape(mpi_decomp).reversed();
}

static inline bool
within_box(const Index& coords, const Shape& box_dims, const Index& box_offset)
{
    for (size_t i = 0; i < coords.count; ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

/**
 * Helper macros for printing
 */

#define MPI_SYNCHRONOUS_BLOCK_START(communicator)                                                  \
    {                                                                                              \
        fflush(stdout);                                                                            \
        MPI_Barrier(communicator);                                                                 \
        int rank__, nprocs_;                                                                       \
        ERRCHK_MPI_API(MPI_Comm_rank(communicator, &rank__));                                      \
        ERRCHK_MPI_API(MPI_Comm_size(communicator, &nprocs_));                                     \
        for (int i__ = 0; i__ < nprocs_; ++i__) {                                                  \
            if (i__ == rank__) {                                                                   \
                printf("---Rank %d---\n", rank__);

#define MPI_SYNCHRONOUS_BLOCK_END(communicator)                                                    \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(communicator);                                                                     \
    }                                                                                              \
    MPI_Barrier(communicator);                                                                     \
    }
