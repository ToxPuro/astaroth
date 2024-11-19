#include "mpi_utils.h"

int
get_tag(void)
{
    static int tag{-1};
    ++tag;
    if (tag < 0 || tag >= MPI_TAG_UB_MIN_VALUE)
        tag = 0;
    return tag;
}

template <size_t N>
void
print_mpi_comm(const MPI_Comm& comm)
{
    int rank, nprocs, ndims;
    ERRCHK_MPI_API(MPI_Comm_rank(comm, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(comm, &nprocs));
    ERRCHK_MPI_API(MPI_Cartdim_get(comm, &ndims));

    MPIShape<N> mpi_decomp{};
    MPIShape<N> mpi_periods{};
    MPIIndex<N> mpi_coords{};
    ERRCHK_MPI_API(
        MPI_Cart_get(comm, ndims, mpi_decomp.data(), mpi_periods.data(), mpi_coords.data()));

    MPI_SYNCHRONOUS_BLOCK_START(comm);
    PRINT_DEBUG(mpi_decomp);
    PRINT_DEBUG(mpi_periods);
    PRINT_DEBUG(mpi_coords);
    MPI_SYNCHRONOUS_BLOCK_END(comm);
}

void
init_mpi_funneled()
{
    int provided;
    ERRCHK_MPI_API(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided));

    int claimed;
    ERRCHK_MPI_API(MPI_Query_thread(&claimed));
    ERRCHK_MPI(provided == claimed);

    int is_thread_main;
    ERRCHK_MPI_API(MPI_Is_thread_main(&is_thread_main));
    ERRCHK_MPI(is_thread_main);
}

void
abort_mpi() noexcept
{
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void
finalize_mpi()
{
    ERRCHK_MPI_API(MPI_Finalize());
}

void
cart_comm_destroy(MPI_Comm& cart_comm)
{
    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    cart_comm = MPI_COMM_NULL;
}

void
subarray_destroy(MPI_Datatype& subarray)
{
    ERRCHK_MPI_API(MPI_Type_free(&subarray));
    subarray = MPI_DATATYPE_NULL;
}

/** Creates an MPI_Info structure with IO tuning parameters.
 * The resource must be freed after use to avoid memory leaks with
 * info_destroy(info)
 * or
 * ERRCHK_MPI_API(MPI_Info_free(&info));
 */
MPI_Info
info_create(void)
{
    MPI_Info info{MPI_INFO_NULL};
    ERRCHK_MPI_API(MPI_Info_create(&info));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "blocksize", "4096"));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "striping_factor", "4"));
    // ERRCHK_MPI_API(MPI_Info_set(*info, "striping_unit", "...")); // Size of stripe chunks
    // ERRCHK_MPI_API(MPI_Info_set(*info, "cb_buffer_size", "...")); // Collective buffer
    // size ERRCHK_MPI_API(MPI_Info_set(*info, "romio_ds_read", "...")); // Data sieving
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_ds_write", "...")); // Data sieving
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_cb_read", "...")); // Collective buffering
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_cb_write", "...")); // Collective buffering
    // ERRCHK_MPI_API(MPI_Info_set(*info, "romio_no_indep_rw", "...")); // Enable/disable
    return info;
}

void
info_destroy(MPI_Info& info)
{
    ERRCHK_MPI_API(MPI_Info_free(&info));
    info = MPI_INFO_NULL;
}

void
request_wait_and_destroy(MPI_Request& req)
{
    MPI_Status status;
    status.MPI_ERROR = MPI_SUCCESS;
    ERRCHK_MPI_API(MPI_Wait(&req, &status));
    ERRCHK_MPI_API(status.MPI_ERROR);
    if (req != MPI_REQUEST_NULL)
        ERRCHK_MPI_API(MPI_Request_free(&req));
    ERRCHK_MPI(req == MPI_REQUEST_NULL);
}

int
get_rank(const MPI_Comm& cart_comm)
{
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));
    return rank;
}

/**
 * Managed MPI handles
 * However, the added layer of indirection and complexity may outweigh the benefits
 */
// #include <functional>
// #include <memory>
// using mpi_comm_ptr_t     = std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm*)>>;
// using mpi_datatype_ptr_t = std::unique_ptr<MPI_Datatype, std::function<void(MPI_Datatype*)>>;
// using mpi_request_ptr_t  = std::unique_ptr<MPI_Request, std::function<void(MPI_Request*)>>;

// class ManagedMPIComm {
//   private:
//     mpi_comm_ptr_t handle;

//     static MPI_Comm* alloc(const MPI_Comm& parent_comm, const Shape<N>& global_nn)
//     {
//         MPI_Comm* cart_comm = new MPI_Comm;
//         *cart_comm          = cart_comm_create(parent_comm, global_nn);
//         return cart_comm;
//     }

//     static void dealloc(MPI_Comm* cart_comm) noexcept
//     {
//         ERRCHK_MPI_API(MPI_Comm_free(cart_comm));
//         delete cart_comm;
//     }

//   public:
//     ManagedMPIComm(const MPI_Comm& parent_comm, const Shape<N>& global_nn)
//         : handle{alloc(parent_comm, global_nn), &dealloc}
//     {
//     }
// };
