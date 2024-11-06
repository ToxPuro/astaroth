#pragma once
#include <stddef.h>
#include <stdint.h>

#include "datatypes.h"
#include "errchk_mpi.h"
#include "print_debug.h"

/**
 * Helper macros for printing
 */
#define MPI_SYNCHRONOUS_BLOCK_START(communicator)                                                  \
    {                                                                                              \
        MPI_Barrier(communicator);                                                                 \
        fflush(stdout);                                                                            \
        MPI_Barrier(communicator);                                                                 \
        int rank__, nprocs_;                                                                       \
        ERRCHK_MPI_API(MPI_Comm_rank(communicator, &rank__));                                      \
        ERRCHK_MPI_API(MPI_Comm_size(communicator, &nprocs_));                                     \
        for (int i__ = 0; i__ < nprocs_; ++i__) {                                                  \
            MPI_Barrier(communicator);                                                             \
            if (i__ == rank__) {                                                                   \
                printf("---Rank %d---\n", rank__);

#define MPI_SYNCHRONOUS_BLOCK_END(communicator)                                                    \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(communicator);                                                                     \
    }                                                                                              \
    MPI_Barrier(communicator);                                                                     \
    }

#define PRINT_DEBUG_MPI(expr, communicator)                                                        \
    do {                                                                                           \
        MPI_SYNCHRONOUS_BLOCK_START((communicator))                                                \
        PRINT_DEBUG((expr));                                                                       \
        MPI_SYNCHRONOUS_BLOCK_END((communicator))                                                  \
    } while (0)

/**
 * Wrappers for core functions
 */
// MPI_TAG_UB is required to be at least this large by the MPI 4.1 standard
// However, not all implementations seem to define it (note int*) and
// MPI_Comm_get_attr fails, so must be hardcoded here
constexpr int MPI_TAG_UB_MIN_VALUE = 32767;

int get_tag(void);

Direction get_direction(const Index& offset, const Shape& nn, const Index& rr);

void print_mpi_comm(const MPI_Comm& comm);

/** Creates a cartesian communicator with topology information attached
 * The resource must be freed after use with
 *  destroy_cart_comm(cart_comm)
 * or
 *  ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
 */
MPI_Comm create_cart_comm(const MPI_Comm& parent_comm, const Shape& global_nn);

void destroy_cart_comm(MPI_Comm& cart_comm);

/** Create and commit a subarray
 * The returned resource is ready to use.
 * The returned resource must be freed after use with either
 *  destroy_subarray(subarray)
 * or
 *  ERRCHK_MPI_API(MPI_Type_free(&subarray))
 * */
MPI_Datatype create_subarray(const Shape& dims, const Shape& subdims, const Index& offset,
                             const MPI_Datatype& dtype);

void destroy_subarray(MPI_Datatype& subarray);

Shape get_decomposition(const MPI_Comm& cart_comm);

Index get_coords(const MPI_Comm& cart_comm);

int get_rank(const MPI_Comm& cart_comm);

int get_neighbor(const MPI_Comm& cart_comm, const Direction& dir);

void wait_and_destroy_request(MPI_Request& req);

/** Map type to MPI enum representing the type
 * Usage: MPIType<double>::value // returns MPI_DOUBLE
 */
template <typename T>
constexpr MPI_Datatype
get_mpi_dtype()
{
    if (std::is_same<T, double>::value) {
        return MPI_DOUBLE;
    }
    else if (std::is_same<T, float>::value) {
        return MPI_DOUBLE;
    }
    else {
        ERROR_DESC("Unknown datatype");
        return MPI_DATATYPE_NULL;
    }
}
