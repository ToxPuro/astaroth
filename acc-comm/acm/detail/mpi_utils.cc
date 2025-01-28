#include "mpi_utils.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "errchk_mpi.h"
#include "type_conversion.h"

#include "decomp.h"

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
static auto
astaroth_to_mpi_format(const ac::vector<uint64_t>& in)
{
    ac::vector<int> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

static auto
astaroth_to_mpi_format(const Direction& in)
{
    ac::vector<int> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<int>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

static auto
mpi_to_astaroth_format(const ac::vector<int>& in)
{
    ac::vector<uint64_t> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = as<uint64_t>(in[i]);
    std::reverse(out.begin(), out.end());
    return out;
}

// TODO better way
template <typename T>
static auto
prodd(const std::vector<T>& vec)
{
    return std::reduce(vec.begin(), vec.end(), static_cast<T>(1), std::multiplies<T>());
}

// TODO better way
template <typename T>
static auto
mul(const ac::vector<T>& a, const ac::vector<T>& b)
{
    ac::vector<T> c(a.size());
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::multiplies<T>());
    return c;
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

static MPI_Comm
cart_comm_mpi_create(const MPI_Comm& parent_comm, const Shape& global_nn, const int reorder)
{
    // Get the number of processes
    int mpi_nprocs{-1};
    ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

    // Get ndims
    const size_t ndims{global_nn.size()};

    // Decompose all dimensions
    MPIShape mpi_decomp(ndims, 0);
    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(ndims), mpi_decomp.data()));

    // Decompose only the slowest moving dimension (last dimension in Astaroth, first in MPI)
    // MPIShape mpi_decomp(ndims, 1);
    // mpi_decomp[0] = 0;
    // ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(ndims), mpi_decomp.data()));

    // Create the Cartesian communicator
    MPI_Comm cart_comm{MPI_COMM_NULL};
    MPIShape mpi_periods(ndims, 1); // Periodic in all dimensions
    // int reorder{1}; // Enable reordering (but likely inop with most MPI implementations)
    ERRCHK_MPI_API(MPI_Cart_create(parent_comm,
                                   as<int>(ndims),
                                   mpi_decomp.data(),
                                   mpi_periods.data(),
                                   reorder,
                                   &cart_comm));

    // Can also add custom decomposition and rank reordering here instead:
    // int reorder{0};
    // ...
    return cart_comm;
}

static std::vector<uint64_t>
get_nprocs_per_layer(const uint64_t& nprocs, const std::vector<uint64_t>& max_per_layer)
{
    uint64_t curr_nprocs{nprocs};
    std::vector<uint64_t> nprocs_per_layer;
    for (const auto& elem : max_per_layer) {
        nprocs_per_layer.push_back(std::min(curr_nprocs, elem));
        curr_nprocs /= nprocs_per_layer.back();
    }
    nprocs_per_layer.push_back(curr_nprocs); // Push remainder
    ERRCHK_MPI(ac::mpi::prodd(nprocs_per_layer) == nprocs);
    return nprocs_per_layer;
}

static void
test_get_nprocs_per_layer()
{
    {
        constexpr uint64_t nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 4, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t nprocs{2};
        std::vector<uint64_t> max_per_layer{8, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t nprocs{8};
        std::vector<uint64_t> max_per_layer{4, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 2, 2, 4, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t nprocs{64};
        std::vector<uint64_t> max_nprocs_per_layer{2, 4};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_nprocs_per_layer)};
        const Shape global_nn{128, 128, 128};
        auto decomp{decompose_hierarchical(global_nn, nprocs_per_layer)};

        const auto global_decomp{hierarchical_decomposition_to_global(decomp)};
        PRINT_DEBUG(global_decomp);
        PRINT_DEBUG(decomp);
        ERRCHK((global_decomp == Shape{4, 4, 4}));
        ERRCHK(prod(global_decomp) == nprocs);
    }
}

static MPI_Comm
cart_comm_hierarchical_create(const MPI_Comm& parent_comm, const Shape& global_nn)
{
    // Get the number of processes
    int mpi_nprocs{-1};
    ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

    // Get ndims
    const size_t ndims{global_nn.size()};

    // Get node hierarchy
    const std::vector<uint64_t> max_nprocs_per_layer{2, 4};
    const auto nprocs_per_layer{
        get_nprocs_per_layer(as<uint64_t>(ac::mpi::get_size(parent_comm)), max_nprocs_per_layer)};

    // Decompose
    const auto hierarchical_decomposition{decompose_hierarchical(global_nn, nprocs_per_layer)};
    const auto global_decomposition{
        hierarchical_decomposition_to_global(hierarchical_decomposition)};
    const auto mpi_decomp{astaroth_to_mpi_format(global_decomposition)};

    // Create the Cartesian communicator
    MPI_Comm initial_cart_comm{MPI_COMM_NULL};
    MPIShape mpi_periods(ndims, 1); // Periodic in all dimensions
    const int reorder{0};           // Disable reordering
    ERRCHK_MPI_API(MPI_Cart_create(parent_comm,
                                   as<int>(ndims),
                                   mpi_decomp.data(),
                                   mpi_periods.data(),
                                   reorder,
                                   &initial_cart_comm));

    // Reorder
    // 1) Get coordinates of the current rank in the new reordered domain
    const auto coords{hierarchical_to_spatial(as<uint64_t>(ac::mpi::get_rank(initial_cart_comm)),
                                              hierarchical_decomposition)};
    // 2) Get the rank in the old reordered domain that corresponds to this position
    const auto new_rank{to_linear(coords, ac::mpi::get_decomposition(initial_cart_comm))};

    MPI_Comm reordered_cart_comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_split(parent_comm, 0, as<int>(new_rank), &reordered_cart_comm));

    // Create a new reordered cart comm
    MPI_Comm cart_comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Cart_create(reordered_cart_comm,
                                   as<int>(ndims),
                                   mpi_decomp.data(),
                                   mpi_periods.data(),
                                   reorder,
                                   &cart_comm));

    MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    PRINT_DEBUG(global_decomposition);
    PRINT_DEBUG(hierarchical_decomposition);
    PRINT_DEBUG(ac::mpi::get_rank(initial_cart_comm));
    PRINT_DEBUG(ac::mpi::get_rank(cart_comm));
    PRINT_DEBUG(ac::mpi::get_coords(initial_cart_comm));
    PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
    MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

    // Release resources
    ERRCHK_MPI_API(MPI_Comm_free(&initial_cart_comm));
    ERRCHK_MPI_API(MPI_Comm_free(&reordered_cart_comm));

    return cart_comm;
}

MPI_Comm
cart_comm_create(const MPI_Comm& parent_comm, const Shape& global_nn,
                 const RankReorderMethod& reorder_method)
{
    switch (reorder_method) {
    case RankReorderMethod::No:
        return cart_comm_mpi_create(parent_comm, global_nn, 0);
    case RankReorderMethod::MPI_Default:
        return cart_comm_mpi_create(parent_comm, global_nn, 1);
    case RankReorderMethod::Hierarchical:
        return cart_comm_hierarchical_create(parent_comm, global_nn);
    default:
        ERRCHK_EXPR_DESC(false, "Unhandled reorder_method");
        return MPI_COMM_NULL;
    }
}

void
cart_comm_destroy(MPI_Comm& cart_comm)
{
    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    cart_comm = MPI_COMM_NULL;
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

MPI_Datatype
subarray_create(const Shape& dims, const Shape& subdims, const Index& offset,
                const MPI_Datatype& dtype)
{
    MPIShape mpi_dims{astaroth_to_mpi_format(dims)};
    MPIShape mpi_subdims{astaroth_to_mpi_format(subdims)};
    MPIIndex mpi_offset{astaroth_to_mpi_format(offset)};

    MPI_Datatype subarray{MPI_DATATYPE_NULL};
    ERRCHK_MPI_API(MPI_Type_create_subarray(as<int>(dims.size()),
                                            mpi_dims.data(),
                                            mpi_subdims.data(),
                                            mpi_offset.data(),
                                            MPI_ORDER_C,
                                            dtype,
                                            &subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&subarray));
    return subarray;
}

MPI_Datatype
subarray_create_resized(const Shape& dims, const Shape& subdims, const Index& offset,
                        const MPI_Datatype& dtype, const MPI_Aint lower_bound,
                        const MPI_Aint extent)
{
    ERRCHK_MPI(dims.size() == subdims.size());
    ERRCHK_MPI(dims.size() == offset.size());
    MPI_Datatype subarray{subarray_create(dims, subdims, offset, dtype)};

    MPI_Datatype resized_subarray{MPI_DATATYPE_NULL};
    ERRCHK_MPI_API(MPI_Type_create_resized(subarray, lower_bound, extent, &resized_subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&resized_subarray));

    subarray_destroy(subarray);
    return resized_subarray;
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
get_coords(const MPI_Comm& cart_comm, const int rank)
{
    // Get dimensions of the communicator
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get coordinates of the process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));
    return mpi_to_astaroth_format(mpi_coords);
}

Index
get_coords(const MPI_Comm& cart_comm)
{
    // Get the rank of the current process
    int rank{MPI_PROC_NULL};
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // // Get dimensions of the communicator
    // int mpi_ndims{-1};
    // ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // // Get the coordinates of the current process
    // MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    // ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data()));
    // return mpi_to_astaroth_format(mpi_coords);
    return get_coords(cart_comm, rank);
}

Shape
get_decomposition(const MPI_Comm& cart_comm)
{
    int mpi_ndims{-1};
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    MPIShape mpi_decomp(as<size_t>(mpi_ndims));
    MPIShape mpi_periods(as<size_t>(mpi_ndims));
    MPIIndex mpi_coords(as<size_t>(mpi_ndims));
    ERRCHK_MPI_API(MPI_Cart_get(cart_comm,
                                mpi_ndims,
                                mpi_decomp.data(),
                                mpi_periods.data(),
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

Shape
get_global_mm(const Shape& global_nn, const Index& rr)
{
    return Shape{as<uint64_t>(2) * rr + global_nn};
}

void
scatter_advanced(const MPI_Comm& parent_comm, const MPI_Datatype& etype, //
                 const Shape& global_mm, const Index& subdomain_offset,
                 const void* send_buffer, //
                 const Shape& local_mm, const Shape& local_nn, const Index& local_nn_offset,
                 void* recv_buffer)
{
    constexpr int root{0};
    const size_t nprocs{as<size_t>(get_size(parent_comm))};

    constexpr int lb{0}; // Lower bound
    int extent{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &extent)); // Use element size as extent

    MPI_Datatype monolithic_subarray{
        subarray_create_resized(global_mm, local_nn, subdomain_offset, etype, lb, extent)};
    MPI_Datatype distributed_subarray{subarray_create(local_mm, local_nn, local_nn_offset, etype)};

    std::vector<int> counts(nprocs, 1);
    std::vector<int> displs(nprocs);
    for (size_t i{0}; i < nprocs; ++i) {
        const Index coords{get_coords(parent_comm, as<int>(i))};
        displs[i] = as<int>(to_linear(mul(coords, local_nn), global_mm));
    }

    ERRCHK_MPI_API(MPI_Scatterv(send_buffer,
                                counts.data(),
                                displs.data(),
                                monolithic_subarray,
                                recv_buffer,
                                1,
                                distributed_subarray,
                                root,
                                parent_comm));

    subarray_destroy(monolithic_subarray);
    subarray_destroy(distributed_subarray);
}

void
gather_advanced(const MPI_Comm& parent_comm, const MPI_Datatype& etype, //
                const Shape& local_mm, const Shape& local_nn, const Index& local_nn_offset,
                const void* send_buffer, //
                const Shape& global_mm, const Index& subdomain_offset, void* recv_buffer)
{
    constexpr int root{0};
    const size_t nprocs{as<size_t>(get_size(parent_comm))};

    constexpr int lb{0}; // Lower bound
    int extent{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &extent)); // Use element size as extent

    MPI_Datatype monolithic_subarray{
        subarray_create_resized(global_mm, local_nn, subdomain_offset, etype, lb, extent)};
    MPI_Datatype distributed_subarray{subarray_create(local_mm, local_nn, local_nn_offset, etype)};

    std::vector<int> counts(nprocs, 1);
    std::vector<int> displs(nprocs);
    for (size_t i{0}; i < nprocs; ++i) {
        const Index coords{get_coords(parent_comm, as<int>(i))};
        displs[i] = as<int>(to_linear(mul(coords, local_nn), global_mm));
    }

    ERRCHK_MPI_API(MPI_Gatherv(send_buffer,
                               1,
                               distributed_subarray,
                               recv_buffer,
                               counts.data(),
                               displs.data(),
                               monolithic_subarray,
                               root,
                               parent_comm));

    subarray_destroy(monolithic_subarray);
    subarray_destroy(distributed_subarray);
}

void
scatter(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& global_nn,
        const Shape& local_rr, const void* send_buffer, void* recv_buffer)
{
    const Shape local_mm{ac::mpi::get_local_mm(parent_comm, global_nn, local_rr)};
    const Shape local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
    const Index global_nn_offset{ac::mpi::get_global_nn_offset(parent_comm, global_nn)};
    const Index zero_offset(global_nn.size(), static_cast<uint64_t>(0));

    int etype_bytes{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &etype_bytes));

    MPI_Datatype monolithic_subarray_prototype{
        subarray_create(global_nn, local_nn, zero_offset, etype)};
    MPI_Datatype monolithic_subarray{MPI_DATATYPE_NULL};
    ERRCHK_MPI_API(MPI_Type_create_resized(monolithic_subarray_prototype,
                                           0,
                                           etype_bytes, // Extent
                                           &monolithic_subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&monolithic_subarray));
    MPI_Datatype distributed_subarray{subarray_create(local_mm, local_nn, local_rr, etype)};

    const int root_proc{0};

    // Basic scatter: works only on contiguous blocks of data
    // ERRCHK_MPI_API(MPI_Scatter(send_buffer,
    //                            1,
    //                            monolithic_subarray,
    //                            recv_buffer,
    //                            1,
    //                            distributed_subarray,
    //                            root_proc,
    //                            parent_comm));

    // Scatterv: can set offsets for noncontiguous blocks of data
    // with the resize extent
    const size_t nprocs{as<size_t>(get_size(parent_comm))};
    std::vector<int> counts(nprocs, 1);
    std::vector<int> displs(nprocs);
    for (size_t i{0}; i < nprocs; ++i) {
        const Index coords{get_coords(parent_comm, as<int>(i))};
        displs[i] = as<int>(to_linear(mul(coords, local_nn), global_nn));
    }

    ERRCHK_MPI_API(MPI_Scatterv(send_buffer,
                                counts.data(),
                                displs.data(),
                                monolithic_subarray,
                                recv_buffer,
                                1,
                                distributed_subarray,
                                root_proc,
                                parent_comm));

    subarray_destroy(monolithic_subarray);
    subarray_destroy(monolithic_subarray_prototype);
    subarray_destroy(distributed_subarray);
}

void
gather(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& global_nn,
       const Shape& local_rr, const void* send_buffer, void* recv_buffer)
{
    const Shape local_mm{ac::mpi::get_local_mm(parent_comm, global_nn, local_rr)};
    const Shape local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
    const Index global_nn_offset{ac::mpi::get_global_nn_offset(parent_comm, global_nn)};
    const Index zero_offset(global_nn.size(), static_cast<uint64_t>(0));

    int etype_bytes{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &etype_bytes));

    MPI_Datatype monolithic_subarray_prototype{
        subarray_create(global_nn, local_nn, zero_offset, etype)};
    MPI_Datatype monolithic_subarray{MPI_DATATYPE_NULL};
    ERRCHK_MPI_API(MPI_Type_create_resized(monolithic_subarray_prototype,
                                           0,
                                           etype_bytes, // Extent
                                           &monolithic_subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&monolithic_subarray));
    MPI_Datatype distributed_subarray{subarray_create(local_mm, local_nn, local_rr, etype)};

    const int root_proc{0};
    const size_t nprocs{as<size_t>(get_size(parent_comm))};
    std::vector<int> counts(nprocs, 1);
    std::vector<int> displs(nprocs);
    for (size_t i{0}; i < nprocs; ++i) {
        const Index coords{get_coords(parent_comm, as<int>(i))};
        displs[i] = as<int>(to_linear(mul(coords, local_nn), global_nn));
    }

    ERRCHK_MPI_API(MPI_Gatherv(send_buffer,
                               1,
                               distributed_subarray,
                               recv_buffer,
                               counts.data(),
                               displs.data(),
                               monolithic_subarray,
                               root_proc,
                               parent_comm));

    subarray_destroy(monolithic_subarray);
    subarray_destroy(monolithic_subarray_prototype);
    subarray_destroy(distributed_subarray);
}

void
read_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& file_dims,
                const Index& file_offset, const Shape& mesh_dims,
                const Shape& mesh_subdims, const Index& mesh_offset, const std::string& path,
                void* data)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = ac::mpi::info_create();

    // Subarrays
    MPI_Datatype global_subarray = ac::mpi::subarray_create(file_dims,
                                                            mesh_subdims,
                                                            file_offset,
                                                            etype);
    MPI_Datatype local_subarray  = ac::mpi::subarray_create(mesh_dims,
                                                           mesh_subdims,
                                                           mesh_offset,
                                                           etype);

    // File
    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, info, &file));
    ERRCHK_MPI_API(MPI_File_set_view(file, 0, etype, global_subarray, "native", info));

    // Check that the file is in the expected format
    int etype_bytes{-1};
    ERRCHK_MPI_API(MPI_Type_size(etype, &etype_bytes));

    MPI_Offset bytes{0};
    ERRCHK_MPI_API(MPI_File_get_size(file, &bytes));
    ERRCHK_MPI_EXPR_DESC(as<uint64_t>(bytes) == prod(file_dims) * as<uint64_t>(etype_bytes),
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
write_collective(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& file_dims,
                 const Index& file_offset, const Shape& mesh_dims,
                 const Shape& mesh_subdims, const Index& mesh_offset, const void* data,
                 const std::string& path)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = ac::mpi::info_create();

    // Subarrays
    MPI_Datatype global_subarray = ac::mpi::subarray_create(file_dims,
                                                            mesh_subdims,
                                                            file_offset,
                                                            etype);
    MPI_Datatype local_subarray  = ac::mpi::subarray_create(mesh_dims,
                                                           mesh_subdims,
                                                           mesh_offset,
                                                           etype);

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
    read_collective(parent_comm,
                    etype,
                    global_nn,
                    global_nn_offset,
                    local_mm,
                    local_nn,
                    local_nn_offset,
                    path,
                    data);
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
    write_collective(parent_comm,
                     etype,
                     global_nn,
                     global_nn_offset,
                     local_mm,
                     local_nn,
                     local_nn_offset,
                     data,
                     path);
}

void
write_distributed(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& local_mm,
                  const void* data, const std::string& path)
{
    // MPI IO
    MPI_File fp{MPI_FILE_NULL};
    ERRCHK_MPI_API(MPI_File_open(parent_comm,
                                 path.c_str(),
                                 MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                 MPI_INFO_NULL, // Note: MPI_INFO_NULL
                                 &fp));

    ERRCHK_MPI_API(MPI_File_write(fp, data, as<int>(prod(local_mm)), etype, MPI_STATUS_IGNORE));
    ERRCHK_MPI_API(MPI_File_close(&fp));

    // Posix
    // FILE* fp{fopen(path.c_str(), "wb")};
    // ERRCHK(fp != NULL);

    // int size{-1};
    // ERRCHK_MPI_API(MPI_Type_size(etype, &size));

    // const size_t count{prod(local_mm)};
    // const size_t elems_written{fwrite(data, as<size_t>(size), count, fp)};
    // ERRCHK(elems_written == count);

    // fclose(fp);
}

void
read_distributed(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const Shape& local_mm,
                 const std::string& path, void* data)
{
    // MPI IO
    MPI_File fp{MPI_FILE_NULL};
    ERRCHK_MPI_API(MPI_File_open(parent_comm,
                                 path.c_str(),
                                 MPI_MODE_RDONLY,
                                 MPI_INFO_NULL, // Note: MPI_INFO_NULL
                                 &fp));
    ERRCHK_MPI_API(MPI_File_write(fp, data, as<int>(prod(local_mm)), etype, MPI_STATUS_IGNORE));
    ERRCHK_MPI_API(MPI_File_close(&fp));

    // Posix
    // FILE* fp{fopen(path.c_str(), "rb")};
    // ERRCHK(fp != NULL);

    // int size{-1};
    // ERRCHK_MPI_API(MPI_Type_size(etype, &size));

    // const size_t count{prod(local_mm)};
    // const size_t elems_read{fread(data, as<size_t>(size), count, fp)};
    // ERRCHK(elems_read == count);

    // fclose(fp);
}

void
reduce(const MPI_Comm& parent_comm, const MPI_Datatype& etype, const MPI_Op& op, const size_t& axis,
       const size_t& count, void* data)
{
    const auto coords{get_coords(parent_comm)};
    const auto color{as<int>(coords[axis])};
    const auto key{get_rank(parent_comm)};

    MPI_Comm neighbors{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, key, &neighbors));
    ERRCHK_MPI_API(MPI_Allreduce(MPI_IN_PLACE, data, as<int>(count), etype, op, neighbors));
    ERRCHK_MPI_API(MPI_Comm_free(&neighbors));
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

// namespace test {

// void
// indent(const size_t depth)
// {
//     for (size_t i = 0; i < depth; ++i)
//         std::cout << "    ";
// }

// template <typename T>
// void
// print(const T& elem)
// {
//     std::cout << elem;
// }

// template <typename T>
// void
// print(const std::vector<T>& vec, const size_t depth = 0)
// {
//     test::indent(depth);
//     std::cout << "{ ";
//     for (const auto& elem : vec)
//         std::cout << elem << " ";
//     std::cout << "}" << std::endl;
// }

// template <typename T>
// void
// print(const std::vector<std::vector<T>>& vec, const size_t depth = 0)
// {
//     test::indent(depth);
//     std::cout << "{" << std::endl;
//     for (const auto& elem : vec) {
//         print(elem, depth + 1);
//     }
//     test::indent(depth);
//     std::cout << "}" << std::endl;
// }

// template <typename T>
// void
// print_debug(const std::string& label, const std::vector<T>& vec)
// {
//     std::cout << label << ":" << std::endl;
//     print(vec);
// }
// } // namespace test

void
test_mpi_utils()
{
    ac::mpi::test_get_nprocs_per_layer();

    // TODO proper test (but seems to work)
    // ac::mpi::init_funneled();
    // const Shape global_nn{256, 128, 64};
    // MPI_Comm cart_comm{ac::mpi::cart_comm_hierarchical_create(MPI_COMM_WORLD, global_nn)};
    // ac::mpi::cart_comm_destroy(cart_comm);
    // ac::mpi::finalize();

    // std::cout << "-----" << std::endl;
    // test::print(1);
    // std::cout << "-----" << std::endl;
    // test::print(std::vector{1, 2, 3});
    // std::cout << "-----" << std::endl;
    // test::print(std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}});
    // std::cout << "-----" << std::endl;
    // test::print_debug("lala",
    //                   std::vector{std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}},
    //                               std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}}});
    // std::cout << "-----" << std::endl;
    // test::print_debug("lala",
    //                   std::vector{std::vector{std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}},
    //                                           std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}}},
    //                               std::vector{std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}},
    //                                           std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}}}});
    // std::cout << "-----" << std::endl;

    /*
    Label: { // indent 0
        { // indent 1
            { 1 2 3 } // Indent 2
        } // indent 1
    }// indent 0

    indent after each endl
    */
}
