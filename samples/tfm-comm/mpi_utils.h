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

Shape get_decomposition(const MPI_Comm& cart_comm);

Index get_coords(const MPI_Comm& cart_comm);

int get_rank(const MPI_Comm& cart_comm);

int get_neighbor(const MPI_Comm& cart_comm, const Direction& dir);

/** Map type to MPI enum representing the type
 * Usage: MPIType<double>::value // returns MPI_DOUBLE
 */
template <typename T>
MPI_Datatype
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

        int flag          = 0;
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

template <typename T>
static inline subarray_ptr_t
datatype_make_unique(const Shape& dims, const Shape& subdims, const Index& offset)
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
