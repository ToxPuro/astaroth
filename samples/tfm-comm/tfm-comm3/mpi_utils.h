#pragma once
#include <stddef.h>
#include <stdint.h>

void to_mpi_format(const size_t ndims, const uint64_t* dims, int* mpi_dims);

void to_astaroth_format(const size_t ndims, const int* mpi_dims, uint64_t* dims);

/**
 * At each call, returns the next integer in range [0, INT_MAX].
 * If the counter overflows, the counter wraps around and starts again from 0.
 * NOTE: Not thread-safe.
 */
int get_tag(void);

void test_mpi_utils(void);

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

#define MPI_SYNCHRONOUS_BLOCK_END                                                                  \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(communicator);                                                                     \
    }                                                                                              \
    MPI_Barrier(communicator);                                                                     \
    }
