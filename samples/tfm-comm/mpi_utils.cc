#include "mpi_utils.h"

int
get_tag(void)
{
    static int tag = -1;
    ++tag;
    if (tag < 0 || tag >= MPI_TAG_UB_MIN_VALUE)
        tag = 0;
    return tag;
}

Direction
get_direction(const Index& offset, const Shape& nn, const Index& rr)
{
    Direction dir(offset.count);
    for (size_t i = 0; i < offset.count; ++i)
        dir[i] = offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    return dir;
}

void
print_mpi_comm(const MPI_Comm& comm)
{
    int rank, nprocs, ndims;
    ERRCHK_MPI_API(MPI_Comm_rank(comm, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(comm, &nprocs));
    ERRCHK_MPI_API(MPI_Cartdim_get(comm, &ndims));

    MPIShape mpi_decomp(as<size_t>(ndims));
    MPIShape mpi_periods(as<size_t>(ndims));
    MPIIndex mpi_coords(as<size_t>(ndims));
    ERRCHK_MPI_API(MPI_Cart_get(comm, ndims, mpi_decomp.data, mpi_periods.data, mpi_coords.data));

    MPI_SYNCHRONOUS_BLOCK_START(comm);
    PRINT_DEBUG(mpi_decomp);
    PRINT_DEBUG(mpi_periods);
    PRINT_DEBUG(mpi_coords);
    MPI_SYNCHRONOUS_BLOCK_END(comm);
}

MPI_Comm
cart_comm_create(const MPI_Comm& parent_comm, const Shape& global_nn)
{
    // Get the number of processes
    int mpi_nprocs = -1;
    ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

    // Use MPI for finding the decomposition
    MPIShape mpi_decomp(global_nn.count, 0); // Decompose all dimensions
    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(mpi_decomp.count), mpi_decomp.data));

    // Create the Cartesian communicator
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPIShape mpi_periods(global_nn.count, 1); // Periodic in all dimensions
    int reorder = 1; // Enable reordering (but likely inop with most MPI implementations)
    ERRCHK_MPI_API(MPI_Cart_create(parent_comm, as<int>(mpi_decomp.count), mpi_decomp.data,
                                   mpi_periods.data, reorder, &cart_comm));

    // Can also add custom decomposition and rank reordering here instead:
    // int reorder = 0;
    // ...
    return cart_comm;
}

void
cart_comm_destroy(MPI_Comm& cart_comm)
{
    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    cart_comm = MPI_COMM_NULL;
}

MPI_Datatype
subarray_create(const Shape& dims, const Shape& subdims, const Index& offset,
                const MPI_Datatype& dtype)
{
    MPIShape mpi_dims(dims.reversed());
    MPIShape mpi_subdims(subdims.reversed());
    MPIIndex mpi_offset(offset.reversed());

    MPI_Datatype subarray = MPI_DATATYPE_NULL;
    ERRCHK_MPI_API(MPI_Type_create_subarray(as<int>(dims.count), mpi_dims.data, mpi_subdims.data,
                                            mpi_offset.data, MPI_ORDER_C, dtype, &subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&subarray));
    return subarray;
}

void
subarray_destroy(MPI_Datatype& subarray)
{
    ERRCHK_MPI_API(MPI_Type_free(&subarray));
    subarray = MPI_DATATYPE_NULL;
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

Shape
get_decomposition(const MPI_Comm& cart_comm)
{
    int mpi_ndims = -1;
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    MPIShape mpi_decomp(as<size_t>(mpi_ndims));
    MPIShape mpi_periods(as<size_t>(mpi_ndims));
    MPIIndex mpi_coords(as<size_t>(mpi_ndims));
    ERRCHK_MPI_API(
        MPI_Cart_get(cart_comm, mpi_ndims, mpi_decomp.data, mpi_periods.data, mpi_coords.data));
    return Shape(mpi_decomp.reversed());
}

Index
get_coords(const MPI_Comm& cart_comm)
{
    // Get the rank of the current process
    int rank = MPI_PROC_NULL;
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims = -1;
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data));
    return Index(mpi_coords.reversed());
}

int
get_rank(const MPI_Comm& cart_comm)
{
    int rank = MPI_PROC_NULL;
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));
    return rank;
}

int
get_neighbor(const MPI_Comm& cart_comm, const Direction& dir)
{
    // Get the rank of the current process
    int rank = MPI_PROC_NULL;
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims = -1;
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data));

    // Get the direction of the neighboring process
    MPIIndex mpi_dir(dir.reversed());

    // Get the coordinates of the neighbor
    MPIIndex mpi_neighbor = mpi_coords + mpi_dir;

    // Get the rank of the neighboring process
    int neighbor_rank = MPI_PROC_NULL;
    ERRCHK_MPI_API(MPI_Cart_rank(cart_comm, mpi_neighbor.data, &neighbor_rank));
    return neighbor_rank;
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

//     static MPI_Comm* alloc(const MPI_Comm& parent_comm, const Shape& global_nn)
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
//     ManagedMPIComm(const MPI_Comm& parent_comm, const Shape& global_nn)
//         : handle{alloc(parent_comm, global_nn), &dealloc}
//     {
//     }
// };
