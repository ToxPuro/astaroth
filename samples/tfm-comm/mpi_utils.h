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
 *  cart_comm_destroy(cart_comm)
 * or
 *  ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
 */
MPI_Comm cart_comm_create(const MPI_Comm& parent_comm, const Shape& global_nn);

void cart_comm_destroy(MPI_Comm& cart_comm);

/** Create and commit a subarray
 * The returned resource is ready to use.
 * The returned resource must be freed after use with either
 *  subarray_destroy(subarray)
 * or
 *  ERRCHK_MPI_API(MPI_Type_free(&subarray))
 * */
MPI_Datatype subarray_create(const Shape& dims, const Shape& subdims, const Index& offset,
                             const MPI_Datatype& dtype);

void subarray_destroy(MPI_Datatype& subarray);

/** Block until the request has completed and deallocate it.
 * The MPI_Request is set to MPI_REQUEST_NULL after deallocation.
 */
void request_wait_and_destroy(MPI_Request& req);

Shape get_decomposition(const MPI_Comm& cart_comm);

Index get_coords(const MPI_Comm& cart_comm);

int get_rank(const MPI_Comm& cart_comm);

int get_neighbor(const MPI_Comm& cart_comm, const Direction& dir);

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

/**
 * Helper wrappers for MPI types
 */
struct MPIRequest {
    MPI_Request handle;

    MPIRequest()
        : handle{MPI_REQUEST_NULL}
    {
    }
    // Move
    // MPIRequest(MPIRequest&& other) noexcept
    //     : handle{other.handle}
    // {
    //     other.handle = MPI_REQUEST_NULL;
    // }

    // Move assignment
    // MPIRequest& operator=(MPIRequest&& other) noexcept
    // {
    //     if (this != &other) {
    //         ERRCHK_MPI_API(handle == MPI_REQUEST_NULL);
    //         if (handle != MPI_REQUEST_NULL)
    //             synchronize();
    //         handle       = other.handle;
    //         other.handle = MPI_REQUEST_NULL;
    //     }
    //     return *this;
    // }

    ~MPIRequest()
    {
        ERRCHK_MPI(handle == MPI_REQUEST_NULL);
        request_wait_and_destroy(handle);
        ERRCHK_MPI(handle == MPI_REQUEST_NULL);
    }

    MPIRequest(MPIRequest&& other) noexcept  = delete; // Move
    MPIRequest& operator=(MPIRequest&&)      = delete; // Move assignment
    MPIRequest(const MPIRequest&)            = delete; // Copy
    MPIRequest& operator=(const MPIRequest&) = delete; // Copy assignment

    // Other functions
    void wait() { request_wait_and_destroy(handle); }
};

struct MPIComm {
    MPI_Comm handle;

    MPIComm(const MPI_Comm& parent_comm, const Shape& global_nn)
        : handle{cart_comm_create(parent_comm, global_nn)}
    {
    }

    ~MPIComm()
    {
        ERRCHK_MPI(handle != MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_free(&handle));
    }

    MPIComm(MPIComm&& other) noexcept  = delete; // Move
    MPIComm& operator=(MPIComm&&)      = delete; // Move assignment
    MPIComm(const MPIComm&)            = delete; // Copy
    MPIComm& operator=(const MPIComm&) = delete; // Copy assignment
};
