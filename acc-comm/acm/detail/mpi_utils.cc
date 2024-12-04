#include "mpi_utils.h"

#include <algorithm>
#include <limits>

#include "errchk_mpi.h"
#include "type_conversion.h"

namespace ac::mpi {

/**
 * Datatypes
 */
using MPIIndex = ac::vector<int>;
using MPIShape = ac::vector<int>;

/**
 * Functions to convert between Astaroth's uint64_t column major
 * and MPI's int row major formats
 */
ac::vector<int>
astaroth_to_mpi_format(const ac::vector<uint64_t>& in)
{
    ac::vector<int> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

ac::vector<int>
astaroth_to_mpi_format(const Direction& in)
{
    ac::vector<int> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

ac::vector<uint64_t>
mpi_to_astaroth_format(const ac::vector<int>& in)
{
    ac::vector<uint64_t> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<uint64_t>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

void
print_mpi_comm(const MPI_Comm& comm)
{
    const int ndims = get_ndims(comm);

    MPIShape mpi_decomp(as<size_t>(ndims));
    MPIShape mpi_periods(as<size_t>(ndims));
    MPIIndex mpi_coords(as<size_t>(ndims));
    ERRCHK_MPI_API(
        MPI_Cart_get(comm, ndims, mpi_decomp.data(), mpi_periods.data(), mpi_coords.data()));

    MPI_SYNCHRONOUS_BLOCK_START(comm);
    PRINT_DEBUG(mpi_decomp);
    PRINT_DEBUG(mpi_periods);
    PRINT_DEBUG(mpi_coords);
    MPI_SYNCHRONOUS_BLOCK_END(comm);
}

void
init_funneled()
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
abort() noexcept
{
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void
finalize()
{
    ERRCHK_MPI_API(MPI_Finalize());
}

MPI_Comm
cart_comm_create(const MPI_Comm& parent_comm, const Shape& global_nn)
{
    // Get the number of processes
    int mpi_nprocs{-1};
    ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

    // Get ndims
    const size_t ndims{global_nn.size()};

    // Use MPI for finding the decomposition

    // Decompose all dimensions
    MPIShape mpi_decomp(ndims, 0);

    // Decompose only the slowest moving dimension (last dimension in Astaroth, first in MPI)
    // MPIShape mpi_decomp(ndims, 1);
    // mpi_decomp[0] = 0;

    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(ndims), mpi_decomp.data()));

    // Create the Cartesian communicator
    MPI_Comm cart_comm{MPI_COMM_NULL};
    MPIShape mpi_periods(ndims, 1); // Periodic in all dimensions
    int reorder{1}; // Enable reordering (but likely inop with most MPI implementations)
    ERRCHK_MPI_API(MPI_Cart_create(parent_comm, as<int>(ndims), mpi_decomp.data(),
                                   mpi_periods.data(), reorder, &cart_comm));

    // Can also add custom decomposition and rank reordering here instead:
    // int reorder{0};
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
    MPIShape mpi_dims{astaroth_to_mpi_format(dims)};
    MPIShape mpi_subdims{astaroth_to_mpi_format(subdims)};
    MPIIndex mpi_offset{astaroth_to_mpi_format(offset)};

    MPI_Datatype subarray{MPI_DATATYPE_NULL};
    ERRCHK_MPI_API(MPI_Type_create_subarray(as<int>(dims.size()), mpi_dims.data(),
                                            mpi_subdims.data(), mpi_offset.data(), MPI_ORDER_C,
                                            dtype, &subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&subarray));
    return subarray;
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

// int
// get_next_tag(void)
// {
//     // MPI_TAG_UB is required to be at least this large by the MPI 4.1 standard
//     // However, not all implementations seem to define it (note int*) and
//     // MPI_Comm_get_attr fails, so must be hardcoded here
//     constexpr int MPI_TAG_UB_MIN_VALUE{32767};

//     static int tag{-1};
//     ++tag;
//     if (tag < 0 || tag >= MPI_TAG_UB_MIN_VALUE)
//         tag = 0;
//     return tag;
// }

void
increment_tag(int16_t& tag)
{
    // MPI_TAG_UB is required to be at least this large by the MPI 4.1 standard
    // However, not all implementations seem to define it (note int*) and
    // MPI_Comm_get_attr fails, so must be hardcoded here
    constexpr int MPI_TAG_UB_MIN_VALUE{32767};
    static_assert(std::numeric_limits<int16_t>::max() == MPI_TAG_UB_MIN_VALUE);
    ERRCHK_MPI(tag >= 0);

    ++tag;
    if (tag < 0)
        tag = 0;
}

int
get_rank(const MPI_Comm& cart_comm)
{
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));
    return rank;
}

int
get_size(const MPI_Comm& cart_comm)
{
    int size{-1};
    ERRCHK_MPI_API(MPI_Comm_size(cart_comm, &size));
    return size;
}

int
get_ndims(const MPI_Comm& comm)
{
    int rank, nprocs, ndims;
    ERRCHK_MPI_API(MPI_Comm_rank(comm, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(comm, &nprocs));
    ERRCHK_MPI_API(MPI_Cartdim_get(comm, &ndims));
    (void)rank;   // Unused
    (void)nprocs; // Unused
    return ndims;
}

Index
get_coords(const MPI_Comm& cart_comm)
{
    // Get the rank of the current process
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));
    return mpi_to_astaroth_format(mpi_coords);
}

Shape
get_decomposition(const MPI_Comm& cart_comm)
{
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    MPIShape mpi_decomp(as<size_t>(mpi_ndims));
    MPIShape mpi_periods(as<size_t>(mpi_ndims));
    MPIIndex mpi_coords(as<size_t>(mpi_ndims));
    ERRCHK_MPI_API(MPI_Cart_get(cart_comm, mpi_ndims, mpi_decomp.data(), mpi_periods.data(),
                                mpi_coords.data()));
    return mpi_to_astaroth_format(mpi_decomp);
}

int
get_neighbor(const MPI_Comm& cart_comm, const Direction& dir)
{
    // Get the rank of the current process
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));

    // Get the direction of the neighboring process
    MPIIndex mpi_dir{astaroth_to_mpi_format(dir)};

    // Get the coordinates of the neighbor
    MPIIndex mpi_neighbor{mpi_coords + mpi_dir};

    // Get the rank of the neighboring process
    int neighbor_rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Cart_rank(cart_comm, mpi_neighbor.data(), &neighbor_rank));
    return neighbor_rank;
}

Direction
get_direction(const Index& offset, const Shape& nn, const Index& rr)
{
    Direction dir(offset.size());
    for (size_t i{0}; i < offset.size(); ++i)
        dir[i] = offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    return dir;
}

Shape
get_local_nn(const MPI_Comm& cart_comm, const Shape& global_nn)
{
    const Shape decomp{ac::mpi::get_decomposition(cart_comm)};
    return Shape{global_nn / decomp};
}

Index
get_global_nn_offset(const MPI_Comm& cart_comm, const Shape& global_nn)
{
    const Shape local_nn{get_local_nn(cart_comm, global_nn)};
    const Index coords{ac::mpi::get_coords(cart_comm)};
    return Index{coords * local_nn};
}

Shape
get_local_mm(const MPI_Comm& cart_comm, const Shape& global_nn, const Index& rr)
{
    const Shape local_nn{get_local_nn(cart_comm, global_nn)};
    return Shape{as<uint64_t>(2) * rr + local_nn};
}

void
read_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& in_file_dims,
                const Index& in_file_offset, const Shape& in_mesh_dims,
                const Shape& in_mesh_subdims, const Index& in_mesh_offset, const std::string& path,
                void* data)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = ac::mpi::info_create();

    // Subarrays
    MPI_Datatype global_subarray = ac::mpi::subarray_create(in_file_dims, in_mesh_subdims,
                                                            in_file_offset, etype);
    MPI_Datatype local_subarray  = ac::mpi::subarray_create(in_mesh_dims, in_mesh_subdims,
                                                            in_mesh_offset, etype);

    // File
    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, info, &file));
    ERRCHK_MPI_API(MPI_File_set_view(file, 0, etype, global_subarray, "native", info));

    // Check that the file is in the expected format
    int etype_bytes{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &etype_bytes));

    MPI_Offset bytes{0};
    ERRCHK_MPI_API(MPI_File_get_size(file, &bytes));
    ERRCHK_MPI_EXPR_DESC(as<uint64_t>(bytes) == prod(in_file_dims) * as<uint64_t>(etype_bytes),
                         "Tried to read a file that had unexpected file size. Ensure that "
                         "the file read/written using the same grid dimensions.");

    ERRCHK_MPI_API(MPI_File_read_all(file, data, 1, local_subarray, MPI_STATUS_IGNORE));

    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
    ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
    ERRCHK_MPI_API(MPI_Info_free(&info));
    ERRCHK_MPI_API(MPI_Comm_free(&comm));
}

void
write_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& in_file_dims,
                 const Index& in_file_offset, const Shape& in_mesh_dims,
                 const Shape& in_mesh_subdims, const Index& in_mesh_offset, const void* data,
                 const std::string& path)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = ac::mpi::info_create();

    // Subarrays
    MPI_Datatype global_subarray = ac::mpi::subarray_create(in_file_dims, in_mesh_subdims,
                                                            in_file_offset, etype);
    MPI_Datatype local_subarray  = ac::mpi::subarray_create(in_mesh_dims, in_mesh_subdims,
                                                            in_mesh_offset, etype);

    // File
    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(
        MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
    ERRCHK_MPI_API(MPI_File_set_view(file, 0, etype, global_subarray, "native", info));

    ERRCHK_MPI_API(MPI_File_write_all(file, data, 1, local_subarray, MPI_STATUS_IGNORE));

    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
    ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
    ERRCHK_MPI_API(MPI_Info_free(&info));
    ERRCHK_MPI_API(MPI_Comm_free(&comm));
}

void
read_collective_simple(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                       const Shape& global_nn, const Index& local_nn_offset,
                       const std::string& path, void* data)
{
    const Shape decomp{ac::mpi::get_decomposition(parent_comm)};
    const Shape local_nn{global_nn / decomp};

    const Index coords{ac::mpi::get_coords(parent_comm)};
    const Index global_nn_offset{coords * local_nn};

    const Shape local_mm{as<uint64_t>(2) * local_nn_offset + local_nn};
    read_collective(parent_comm, etype, global_nn, global_nn_offset, local_mm, local_nn,
                    local_nn_offset, path, data);
}

void
write_collective_simple(const MPI_Comm& parent_comm, const MPI_Datatype& etype,
                        const Shape& global_nn, const Index& local_nn_offset, const void* data,
                        const std::string& path)
{
    const Shape decomp{ac::mpi::get_decomposition(parent_comm)};
    const Shape local_nn{global_nn / decomp};

    const Index coords{ac::mpi::get_coords(parent_comm)};
    const Index global_nn_offset{coords * local_nn};

    const Shape local_mm{as<uint64_t>(2) * local_nn_offset + local_nn};
    write_collective(parent_comm, etype, global_nn, global_nn_offset, local_mm, local_nn,
                     local_nn_offset, data, path);
}

} // namespace ac::mpi

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
