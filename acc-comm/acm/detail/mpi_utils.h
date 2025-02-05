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

namespace ac::mpi {

/**
 * Initializes MPI in funneled mode
 * Funneled mode that is required for correct CUDA/MPI programs that
 * utilize, e.g., asynchronous host-to-host memory copies
 */
void init_funneled();

/** Aborts the MPI program on MPI_COMM_WORLD with error code -1 */
void abort() noexcept;

/**
 * Finalizes the MPI context
 * Should be called last in the program.
 * With exception handling, this can be achieved by
 * ```c++
 * ac::mpi::init_funneled();
 * try {
 *  MPI commands here
 *  ...
 * }
 * catch (const std::except& e) {
 *  Handle exception
 *  ac::mpi::abort();
 * }
 * ac::mpi::finalize();
 * ```
 */
void finalize();

/** Creates a cartesian communicator with topology information attached
 * The resource must be freed after use with
 *  cart_comm_destroy(cart_comm)
 * or
 *  ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
 */
enum class RankReorderMethod { No, MPI_Default, Hierarchical };
MPI_Comm
cart_comm_create(const MPI_Comm& parent_comm, const ac::Shape& global_nn,
                 const RankReorderMethod& reorder_method = RankReorderMethod::Hierarchical);

void cart_comm_destroy(MPI_Comm* cart_comm);

/** Print information about the Cartesian communicator */
void print_mpi_comm(const MPI_Comm& comm);

/** Create and commit a subarray
 * The returned resource is ready to use.
 * The returned resource must be freed after use with either
 *  subarray_destroy(subarray)
 * or
 *  ERRCHK_MPI_API(MPI_Type_free(&subarray))
 * */
MPI_Datatype subarray_create(const ac::Shape& dims, const ac::Shape& subdims,
                             const ac::Index& offset, const MPI_Datatype& dtype);

void subarray_destroy(MPI_Datatype* subarray);

/** Creates an MPI_Info structure with IO tuning parameters.
 * The resource must be freed after use to avoid memory leaks with
 * ERRCHK_MPI_API(MPI_Info_free(&info));
 */
MPI_Info info_create(void);

void info_destroy(MPI_Info* info);

/** Block until the request has completed and deallocate it.
 * The MPI_Request is set to MPI_REQUEST_NULL after deallocation.
 */
void request_wait_and_destroy(MPI_Request* req);

/** Getters */

/** Returns the next tag within [0, MPI_TAG_UB_MIN_VALUE]. Note: all processes must take part in
 * calling get_next_tag, otherwise tags will become out of sync.
 * Commented out as error-prone. */
// int get_next_tag(void);

/** Increments the tag. The tag will be within the interval [0, 32767] afterwards. */
void increment_tag(int16_t& tag);

int get_rank(const MPI_Comm& cart_comm);

int get_size(const MPI_Comm& cart_comm);

int get_ndims(const MPI_Comm& comm);

/** Return coordinates of process rank */
ac::Index get_coords(const MPI_Comm& cart_comm, const int rank);

/** Return coordinates of the current process */
ac::Index get_coords(const MPI_Comm& cart_comm);

ac::Shape get_decomposition(const MPI_Comm& cart_comm);

/** Returns the neighbor rank at the offset from current coordinates.  */
int get_neighbor(const MPI_Comm& cart_comm, const ac::Direction& dir);

/** Returns the integer direction of the immediate neighbor (at Chebyshev distance 1) that has
 * ownership of the data at offset w.r.t. the local computational domain of the current process */
ac::Direction get_direction(const ac::Index& offset, const ac::Shape& nn, const ac::Index& rr);

ac::Shape get_local_nn(const MPI_Comm& cart_comm, const ac::Shape& global_nn);

ac::Index get_global_nn_offset(const MPI_Comm& cart_comm, const ac::Shape& global_nn);

ac::Shape get_local_mm(const MPI_Comm& cart_comm, const ac::Shape& global_nn, const ac::Index& rr);

ac::Shape get_global_mm(const ac::Shape& global_nn, const ac::Index& rr);

/** Map type to MPI enum representing the type
 * Usage: MPIType<double>::value // returns MPI_DOUBLE
 */
template <typename T>
constexpr MPI_Datatype
get_dtype()
{
    if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return MPI_DOUBLE;
    }
    else if constexpr (std::is_same_v<T, int>) {
        return MPI_INT;
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        return MPI_UINT64_T;
    }
    else {
        ERROR_DESC("Unknown datatype");
        return MPI_DATATYPE_NULL;
    }
}

/** Communication */
void scatter_advanced(const MPI_Comm& parent_comm, const MPI_Datatype& etype, //
                      const ac::Shape& global_mm, const ac::Index& subdomain_offset,
                      const void*      send_buffer, //
                      const ac::Shape& local_mm, const ac::Shape& local_nn,
                      const ac::Index& local_nn_offset, void* recv_buffer);

void gather_advanced(const MPI_Comm& parent_comm, const MPI_Datatype& etype, //
                     const ac::Shape& local_mm, const ac::Shape& local_nn,
                     const ac::Index& local_nn_offset,
                     const void*      send_buffer, //
                     const ac::Shape& global_mm, const ac::Index& subdomain_offset,
                     void* recv_buffer);

void scatter(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const ac::Shape& global_nn,
             const ac::Shape& local_rr, const void* send_buffer, void* recv_buffer);

void gather(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const ac::Shape& global_nn,
            const ac::Shape& local_rr, const void* send_buffer, void* recv_buffer);

/** IO */

/**
 * Synchronous collective read.
 * The elementary type of the underlying data is passed as the etype arameter.
 */
void read_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                     const ac::Shape& file_dims, const ac::Index& file_offset,
                     const ac::Shape& mesh_dims, const ac::Shape& mesh_subdims,
                     const ac::Index& mesh_offset, const std::string& path, void* data);

/**
 * Synchronous collective write.
 * The elementary type of the underlying data is passed as the etype parameter.
 */
void write_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                      const ac::Shape& file_dims, const ac::Index& file_offset,
                      const ac::Shape& mesh_dims, const ac::Shape& mesh_subdims,
                      const ac::Index& mesh_offset, const void* data, const std::string& path);

/** A simplified routine for reading a a domain of shape `global_nn` from disk to memory address
 * specified by `data` based on the arrangement defined by the communicator.
 * The input memory address must be able to hold at least prod(global_nn) elements of type etype.
 * TODO: consider renaming local_nn_offset to rr.
 */
void read_collective_simple(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                            const ac::Shape& global_nn, const ac::Index& local_nn_offset,
                            const std::string& path, void* data);

/** A simplified routine for writing a domain of shape `global_nn` starting at `data` on disk based
 * on the arrangement defined by the communicator.
 * The output memory address must be able to hold at least prod(global_nn) elements of type etype.
 * TODO: consider renaming local_nn_offset to rr.
 */
void write_collective_simple(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                             const ac::Shape& global_nn, const ac::Index& local_nn_offset,
                             const void* data, const std::string& path);

/** Writes a distributed snapshot. Each process should write to their own file. */
void write_distributed(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const ac::Shape& mm,
                       const void* data, const std::string& path);

/** Read a distributed snapshot. */
void read_distributed(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                      const ac::Shape& local_mm, const std::string& path, void* data);

/**
 * Collective synchronous reduction.
 * Operates in-place on the input data.
 * Operates on all axes.
 * Returns the number of processes that contribute to the result.
 */
int reduce(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const MPI_Op& op,
           const size_t count, void* data);

/**
 * Collective synchronous reduction.
 * Operates in-place on the input data.
 * The axis parameter is used to group the processes based on their spatial coordinates
 * Returns the number of processes that contributed to the result
 * (this can be used for, e.g., averaging)
 *
 * For example, reducing along axis=0 (the fastest varying) in a 2D decomposition with
 * processes
 *
 * axis 0  -->
 * axis 1 |  0 2
 *        v  1 3
 *
 * In this case, processes 0 and 1 do a mutual reduction (both will have the same result in their
 * buffers afterwards). Likewise for processes 2 and 3. This, because the spatial axis 0 component
 * of processes 0 and 1 is the same.
 *
 * A simple way to visualize this is to consider that
 * axis = 0 => reduced along the yz plane
 * axis = 1 => reduced along the xz plane
 * axis = 2 => reduced along the xy plane
 */
int reduce_axis(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const MPI_Op& op,
                const size_t& axis, const size_t count, void* data);

} // namespace ac::mpi

void test_mpi_utils();
