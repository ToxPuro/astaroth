#pragma once
#include <stddef.h>
#include <stdint.h>

#include <functional>

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
 * Functions for converting between Astaroth's uint64_t column major
 * and MPI's int row major formats
 */
template <size_t N>
auto
astaroth_to_mpi_format(const ac::array<uint64_t, N>& in)
{
    ac::array<int, N> out;
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

template <size_t N>
auto
astaroth_to_mpi_format(const ac::array<int64_t, N>& in)
{
    ac::array<int, N> out;
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

template <size_t N>
auto
mpi_to_astaroth_format(const ac::array<int, N>& in)
{
    ac::array<uint64_t, N> out;
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<uint64_t>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

/**
 * Wrappers for core functions
 */
// MPI_TAG_UB is required to be at least this large by the MPI 4.1 standard
// However, not all implementations seem to define it (note int*) and
// MPI_Comm_get_attr fails, so must be hardcoded here
constexpr int MPI_TAG_UB_MIN_VALUE{32767};

int get_tag(void);

template <size_t N>
Direction<N>
get_direction(const Index<N>& offset, const Shape<N>& nn, const Index<N>& rr)
{
    Direction<N> dir{};
    for (size_t i{0}; i < offset.size(); ++i)
        dir[i] = offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    return dir;
}

void print_mpi_comm(const MPI_Comm& comm);

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
template <size_t N>
MPI_Comm
cart_comm_create(const MPI_Comm& parent_comm, const Shape<N>& global_nn)
{
    // Get the number of processes
    int mpi_nprocs{-1};
    ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

    // Use MPI for finding the decomposition
    MPIShape<N> mpi_decomp{}; // Decompose all dimensions
    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(mpi_decomp.size()), mpi_decomp.data()));

    // Create the Cartesian communicator
    MPI_Comm cart_comm{MPI_COMM_NULL};
    MPIShape<N> mpi_periods{ones<int, N>()}; // Periodic in all dimensions
    int reorder{1}; // Enable reordering (but likely inop with most MPI implementations)
    ERRCHK_MPI_API(MPI_Cart_create(parent_comm, as<int>(mpi_decomp.size()), mpi_decomp.data(),
                                   mpi_periods.data(), reorder, &cart_comm));

    // Can also add custom decomposition and rank reordering here instead:
    // int reorder{0};
    // ...
    return cart_comm;
}

void cart_comm_destroy(MPI_Comm& cart_comm);

/** Create and commit a subarray
 * The returned resource is ready to use.
 * The returned resource must be freed after use with either
 *  subarray_destroy(subarray)
 * or
 *  ERRCHK_MPI_API(MPI_Type_free(&subarray))
 * */
template <size_t N>
MPI_Datatype
subarray_create(const Shape<N>& dims, const Shape<N>& subdims, const Index<N>& offset,
                const MPI_Datatype& dtype)
{
    MPIShape<N> mpi_dims{astaroth_to_mpi_format(dims)};
    MPIShape<N> mpi_subdims{astaroth_to_mpi_format(subdims)};
    MPIIndex<N> mpi_offset{astaroth_to_mpi_format(offset)};

    MPI_Datatype subarray = MPI_DATATYPE_NULL;
    ERRCHK_MPI_API(MPI_Type_create_subarray(as<int>(dims.size()), mpi_dims.data(),
                                            mpi_subdims.data(), mpi_offset.data(), MPI_ORDER_C,
                                            dtype, &subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&subarray));
    return subarray;
}

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

template <size_t N>
Shape<N>
get_decomposition(const MPI_Comm& cart_comm)
{
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    MPIShape<N> mpi_decomp{};
    MPIShape<N> mpi_periods{};
    MPIIndex<N> mpi_coords{};
    ERRCHK_MPI_API(MPI_Cart_get(cart_comm, mpi_ndims, mpi_decomp.data(), mpi_periods.data(),
                                mpi_coords.data()));
    return mpi_to_astaroth_format(mpi_decomp);
}

template <size_t N>
Index<N>
get_coords(const MPI_Comm& cart_comm)
{
    // Get the rank of the current process
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex<N> mpi_coords{fill<int, N>(-1)};
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));
    return mpi_to_astaroth_format(mpi_coords);
}

int get_rank(const MPI_Comm& cart_comm);

template <size_t N>
int
get_neighbor(const MPI_Comm& cart_comm, const Direction<N>& dir)
{
    // Get the rank of the current process
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex<N> mpi_coords{fill<int, N>(-1)};
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));

    // Get the direction of the neighboring process
    MPIIndex<N> mpi_dir{astaroth_to_mpi_format(dir)};

    // Get the coordinates of the neighbor
    MPIIndex<N> mpi_neighbor{mpi_coords + mpi_dir};

    // Get the rank of the neighboring process
    int neighbor_rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Cart_rank(cart_comm, mpi_neighbor.data(), &neighbor_rank));
    return neighbor_rank;
}

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

/**
 * Helper wrappers for MPI types
 */
#if false
class MPICommWrapper {
  private:
    MPI_Comm comm{MPI_COMM_NULL};

  public:
    explicit MPICommWrapper(const MPI_Comm& parent_comm)
    {
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));
    }

    MPICommWrapper(const MPICommWrapper&)            = delete; // Copy constructor
    MPICommWrapper& operator=(const MPICommWrapper&) = delete; // Copy assignment
    // MPICommWrapper(MPICommWrapper&& other) noexcept            = delete; // Move constructor
    // MPICommWrapper& operator=(MPICommWrapper&& other) noexcept = delete; // Move assignment

    // Move constructor
    MPICommWrapper(MPICommWrapper&& other) noexcept
        : comm{other.comm}
    {
        other.comm = MPI_COMM_NULL;
    }

    // Move assignment
    MPICommWrapper& operator=(MPICommWrapper&& other) noexcept
    {
        if (this != &other) {
            ERRCHK_MPI(comm == MPI_COMM_NULL);
            if (comm != MPI_COMM_NULL)
                ERRCHK_MPI_API(MPI_Comm_free(&comm));
            comm       = other.comm;
            other.comm = MPI_COMM_NULL;
        }
        return *this;
    }

    ~MPICommWrapper()
    {
        // ERRCHK_MPI(comm == MPI_COMM_NULL); // TODO consider enabling
        if (comm != MPI_COMM_NULL)
            ERRCHK_MPI_API(MPI_Comm_free(&comm));
    }

    MPI_Comm* get()
    {
        ERRCHK_MPI(comm == MPI_COMM_NULL); // Should not modify directly if non-null
        return &comm;
    }

    MPI_Comm value() const
    {
        ERRCHK_MPI(comm != MPI_COMM_NULL);
        return comm;
    }
};

class MPIRequestWrapper {
  private:
    MPI_Request req{MPI_REQUEST_NULL};

  public:
    MPIRequestWrapper() = default;

    MPIRequestWrapper(const MPIRequestWrapper&)            = delete; // Copy
    MPIRequestWrapper& operator=(const MPIRequestWrapper&) = delete; // Copy assignment

    // Move constructor
    MPIRequestWrapper(MPIRequestWrapper&& other) noexcept
        : req{other.req}
    {
        other.req = MPI_REQUEST_NULL;
    }

    // Move assignment
    MPIRequestWrapper& operator=(MPIRequestWrapper&& other) noexcept
    {
        if (this != &other) {
            ERRCHK_MPI_EXPR_DESC(req == MPI_REQUEST_NULL,
                                 "Attempted to overwrite an ongoing request. This should not "
                                 "happen. "
                                 "Call wait on the request to synchronize before move assignment.");
            req       = other.req;
            other.req = MPI_REQUEST_NULL;
        }
        return *this;
    }

    ~MPIRequestWrapper()
    {
        ERRCHK_MPI_EXPR_DESC(req == MPI_REQUEST_NULL,
                             "Attempted to destroy an ongoing request. This should not happen. "
                             "Call wait on the request to synchronize before deleting.");
        if (req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&req));
    }

    MPI_Request* get()
    {
        ERRCHK_MPI_EXPR_DESC(req == MPI_REQUEST_NULL,
                             "A request was still in flight. This should not happen: call "
                             "wait before attempting to modify req.");
        return &req;
    }

    bool ready() const
    {
        ERRCHK_MPI(req != MPI_REQUEST_NULL);

        int flag{0};
        MPI_Status status = {};
        status.MPI_ERROR  = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_Request_get_status(req, &flag, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        return flag;
    }

    void wait()
    {
        ERRCHK_MPI(req != MPI_REQUEST_NULL);

        MPI_Status status = {};
        status.MPI_ERROR  = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_Wait(&req, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        if (req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&req));
        ERRCHK_MPI(req == MPI_REQUEST_NULL);
    }

    bool complete() const { return req == MPI_REQUEST_NULL; }
};

class MPIFileWrapper {
  private:
    MPI_File file{MPI_FILE_NULL};

  public:
    MPIFileWrapper(const MPIFileWrapper&)            = delete; // Copy constructor
    MPIFileWrapper& operator=(const MPIFileWrapper&) = delete; // Copy assignment
    // MPIFileWrapper(MPIFileWrapper&& other) noexcept            = delete; // Move constructor
    // MPIFileWrapper& operator=(MPIFileWrapper&& other) noexcept = delete; // Move assignment

    // Move constructor
    MPIFileWrapper(MPIFileWrapper&& other) noexcept
        : file{other.file}
    {
        other.file = MPI_FILE_NULL;
    }

    // Move assignment
    MPIFileWrapper& operator=(MPIFileWrapper&& other) noexcept
    {
        if (this != &other) {
            ERRCHK_MPI_EXPR_DESC(file == MPI_FILE_NULL,
                                 "Tried to copy assign to a file that was still open");
            if (file != MPI_FILE_NULL)
                ERRCHK_MPI_API(MPI_File_close(&file));
            file       = other.file;
            other.file = MPI_FILE_NULL;
        }
        return *this;
    }

    ~MPIFileWrapper()
    {
        ERRCHK_MPI_EXPR_DESC(file == MPI_FILE_NULL, "Tried to delete a file that was still open");
        if (file != MPI_FILE_NULL)
            ERRCHK_MPI_API(MPI_File_close(&file));
    }

    MPI_File* get()
    {
        // ERRCHK_MPI(file == MPI_FILE_NULL); // Should not modify directly if non-null
        return &file;
    }

    MPI_File value() const
    {
        // ERRCHK_MPI(file != MPI_FILE_NULL);
        return file;
    }
};

using subarray_ptr_t = std::unique_ptr<MPI_Datatype, std::function<void(MPI_Datatype*)>>;
using info_ptr_t     = std::unique_ptr<MPI_Info, std::function<void(MPI_Info*)>>;

template <typename T, size_t N>
static inline subarray_ptr_t
datatype_make_unique(const Shape<N>& dims, const Shape<N>& subdims, const Index<N>& offset)
{
    auto* ptr           = new MPI_Datatype{MPI_DATATYPE_NULL};
    *ptr                = subarray_create(dims, subdims, offset, get_mpi_dtype<T>());
    static auto deleter = [](MPI_Datatype* in_ptr) {
        if (*in_ptr != MPI_DATATYPE_NULL)
            ERRCHK_MPI_API(MPI_Type_free(in_ptr));
        delete in_ptr;
    };
    return subarray_ptr_t{ptr, deleter};
}

static inline info_ptr_t
info_make_unique()
{
    auto* ptr = new MPI_Info{MPI_INFO_NULL};
    // ERRCHK_MPI_API(MPI_Info_create(&*info));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "blocksize", "4096"));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "striping_factor", "4"));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "striping_unit", "...")); // Size of stripe chunks
    // ERRCHK_MPI_API(MPI_Info_set(*info, "cb_buffer_size", "...")); // Collective buffer
    // size ERRCHK_MPI_API(MPI_Info_set(*info, "romio_ds_read", "...")); // Data sieving
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_ds_write", "...")); // Data sieving
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_cb_read", "...")); // Collective buffering
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_cb_write", "...")); // Collective buffering
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_no_indep_rw", "...")); // Enable/disable
    // independent rw
    static auto deleter = [](MPI_Info* in_ptr) {
        if (*in_ptr != MPI_INFO_NULL)
            ERRCHK_MPI_API(MPI_Info_free(in_ptr));
        delete in_ptr;
    };
    return info_ptr_t{ptr, deleter};
}
#endif
