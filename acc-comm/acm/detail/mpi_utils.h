#pragma once
#include <mpi.h>

#include "datatypes.h"

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
 * Initializes MPI in funneled mode
 * Funneled mode that is required for correct CUDA/MPI programs that
 * utilize, e.g., asynchronous host-to-host memory copies
 */
void init_mpi_funneled();

/** Aborts the MPI program on MPI_COMM_WORLD with error code -1 */
void abort_mpi() noexcept;

/**
 * Finalizes the MPI context
 * Should be called last in the program.
 * With exception handling, this can be achieved by
 * ```c++
 * init_mpi_funneled();
 * try {
 *  MPI commands here
 *  ...
 * }
 * catch (const std::except& e) {
 *  Handle exception
 *  abort_mpi();
 * }
 * finalize_mpi();
 * ```
 */
void finalize_mpi();

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

/** Creates an MPI_Info structure with IO tuning parameters.
 * The resource must be freed after use to avoid memory leaks with
 * ERRCHK_MPI_API(MPI_Info_free(&info));
 */
MPI_Info info_create(void);

void info_destroy(MPI_Info& info);

/** Block until the request has completed and deallocate it.
 * The MPI_Request is set to MPI_REQUEST_NULL after deallocation.
 */
void request_wait_and_destroy(MPI_Request& req);

/** Getters */
int get_tag(void);

int get_rank(const MPI_Comm& cart_comm);

int get_size(const MPI_Comm& cart_comm);

int get_ndims(const MPI_Comm& comm);

Index get_coords(const MPI_Comm& cart_comm);

Shape get_decomposition(const MPI_Comm& cart_comm);

/** Returns the neighbor rank at the offset from current coordinates.  */
int get_neighbor(const MPI_Comm& cart_comm, const Direction& dir);

/** Returns the direction (integer coordinates) of the offset within the mesh.
For example, Coordinates Index offset{0,0,0} with Index rr{1,1,1} correspond
to Direction dir{-1, -1, -1} */
Direction get_direction(const Index& offset, const Shape& nn, const Index& rr);

/** Map type to MPI enum representing the type
 * Usage: MPIType<double>::value // returns MPI_DOUBLE
 */
template <typename T>
constexpr MPI_Datatype
get_mpi_dtype()
{
    if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return MPI_DOUBLE;
    }
    else {
        ERROR_DESC("Unknown datatype");
        return MPI_DATATYPE_NULL;
    }
}
