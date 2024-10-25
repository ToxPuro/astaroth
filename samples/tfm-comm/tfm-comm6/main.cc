#include <cstdlib>

#include "errchk.h"
#include "ndarray.h"
#include "shape.h"

#include <mpi.h>

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include "decomp.h"
#include "partition.h"

#include "print_debug.h"

#include "buffer.h"
#include "packet.h"

#include "pack.h"

static Direction
get_direction(const Index& offset, const Shape& nn, const Index& rr)
{
    Direction dir(offset.count);
    for (size_t i = 0; i < offset.count; ++i)
        dir[i] = offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    return dir;
}

static uint64_t
mod(const int64_t a, const int64_t b)
{
    const int64_t c = a % b;
    return c < 0 ? as<uint64_t>(c + b) : as<uint64_t>(c);
}

static void
mpi_comm_print_info(const MPI_Comm comm)
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

static MPI_Comm
create_cart_comm(const MPI_Comm parent_comm, const Shape& global_nn)
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

static Shape
get_decomposition(const MPI_Comm cart_comm)
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

static Index
get_coords(const MPI_Comm cart_comm)
{
    // Get the rank of the current process
    int rank = -1;
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));

    // Get dimensions of the communicator
    int mpi_ndims = -1;
    ERRCHK_MPI_API(MPI_Cartdim_get(cart_comm, &mpi_ndims));

    // Get the coordinates of the current process
    MPIIndex mpi_coords(as<size_t>(mpi_ndims), -1);
    ERRCHK_MPI_API(MPI_Cart_coords(cart_comm, rank, mpi_ndims, mpi_coords.data));
    return Index(mpi_coords.reversed());
}

static int
get_rank(const MPI_Comm cart_comm)
{
    int rank = -1;
    ERRCHK_MPI_API(MPI_Comm_rank(cart_comm, &rank));
    return rank;
}

static int
get_neighbor(const MPI_Comm cart_comm, const Direction dir)
{
    // Get the rank of the current process
    int rank = -1;
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
    int neighbor_rank = -1;
    ERRCHK_MPI_API(MPI_Cart_rank(cart_comm, mpi_neighbor.data, &neighbor_rank));
    return neighbor_rank;
}

int
main()
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {
        const Shape global_nn        = {4, 4};
        MPI_Comm cart_comm           = create_cart_comm(MPI_COMM_WORLD, global_nn);
        const Shape decomp           = get_decomposition(cart_comm);
        const Shape local_nn         = global_nn / decomp;
        const Index coords           = get_coords(cart_comm);
        const Index global_nn_offset = coords * local_nn;

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_DEBUG(global_nn);
        PRINT_DEBUG(local_nn);
        PRINT_DEBUG(decomp);
        PRINT_DEBUG(coords);
        PRINT_DEBUG(global_nn_offset);

        const Shape rr(global_nn.count, 1); // Symmetric halo
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

        NdArray<double> mesh(local_mm);
        mesh.fill_arange(get_rank(cart_comm) * prod(local_mm));
        mesh.display();

        auto segments = partition(local_mm, local_nn, rr);
        for (const auto& segment : segments)
            mesh.fill(1, segment.dims, segment.offset);

        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}

int
main_old()
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {

        const MPI_Comm parent_comm = MPI_COMM_WORLD;

        int nprocs;
        ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &nprocs));

        const Shape global_nn = {4, 4};
        Shape decomp          = decompose(global_nn, as<uint64_t>(nprocs));
        const Shape local_nn  = global_nn / decomp;

        const Shape rr       = {2, 2};
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

        /*
         * Create the library communicator
         */
        MPI_Comm cart_comm = create_cart_comm(parent_comm, global_nn);
        // MPI_Comm cart_comm;
        // MPIShape mpi_decomp(decomp.reversed());
        // MPIShape mpi_periods(global_nn.count, 1); // Set as periodic
        // int reorder = 0;
        // ERRCHK_MPI_API(MPI_Cart_create(parent_comm, as<int>(mpi_decomp.count), mpi_decomp.data,
        //                                mpi_periods.data, reorder, &cart_comm));
        // mpi_comm_print_info(cart_comm);

        NdArray<double> mesh(local_mm);

        // Debug: fill the segments with identifying information
        // MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD)
        auto segments = partition(local_mm, local_nn, rr);
        // for (size_t i = 0; i < segments.size(); ++i)
        //     mesh.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
        mesh.fill_arange();
        mesh.display();

        // Prune the segment containing the computational domain
        for (size_t i = 0; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }
        // PRINT_DEBUG(segments);
        // MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD)

        // Create packets
        // std::vector<Packet<double>> local_packets;
        // std::vector<Packet<double>> remote_packets;
        // for (const auto& segment : segments) {
        // }

        // Buffer<double> buf(prod(local_nn));
        // PackInputs<double*> inputs = {mesh.buffer.data};
        // pack(local_mm, Shape{2, 2}, Index{2, 2}, inputs, buf.data);
        // PRINT_DEBUG(buf);
        // mesh.fill(as<uint64_t>(0), local_mm, Index{0, 0});
        // unpack(buf.data, local_mm, Shape{2, 2}, Index{2, 2}, inputs);
        // mesh.display();

        // Shape pack_dims   = {8, 8};
        // Index pack_offset = {0, 0};

        // NdArray<double> a(local_mm);
        // NdArray<double> b(local_mm);
        // for (size_t i = 0; i < segments.size(); ++i) {
        //     a.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
        //     b.fill(as<uint64_t>(i + 1) + as<uint64_t>(segments.size()), segments[i].dims,
        //            segments[i].offset);
        // }
        // a.display();
        // b.display();

        // PackInputs<double*> inputs = {a.buffer.data, b.buffer.data};
        // Buffer<double> buf(inputs.count * prod(pack_dims));
        // pack(local_mm, pack_dims, pack_offset, inputs, buf.data);
        // PRINT_DEBUG(buf);

        // a.fill(as<uint64_t>(0), local_mm, Index{0, 0});
        // b.fill(as<uint64_t>(0), local_mm, Index{0, 0});
        // a.display();
        // b.display();
        // unpack(buf.data, local_mm, pack_dims, pack_offset, inputs);
        // a.display();
        // b.display();

        // Send the packets
        std::vector<Buffer<double>> local_buffers;
        std::vector<Buffer<double>> remote_buffers;

        std::vector<MPI_Request> recv_reqs;
        std::vector<MPI_Request> send_reqs;

        PackInputs<double*> inputs = {mesh.buffer.data};

        for (const auto& segment : segments) {
            Buffer<double> send_buffer(inputs.count * prod(segment.dims));
            Buffer<double> recv_buffer(inputs.count * prod(segment.dims));

            Index recv_offset = segment.offset;
            Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;
            // PRINT_DEBUG(recv_offset);
            // PRINT_DEBUG(send_offset);

            pack(local_mm, segment.dims, send_offset, inputs, send_buffer.data);
            // PRINT_DEBUG(send_buffer);

            // PRINT_DEBUG(get_direction(recv_offset, local_nn, rr));
            // PRINT_DEBUG(-get_direction(recv_offset, local_nn, rr));

            // Post recv
            // ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data, as_int(recv_buffer.count), mpi_dtype_,
            //                          recv_peer, tag, mpi_comm_, &remote_packet->req));

            // Post send
            // ERRCHK_MPI_API(MPI_Isend(local_packet->buffer.data,
            // as_int(local_packet->buffer.count),
            //                          mpi_dtype_, send_peer, tag, mpi_comm_, &local_packet->req));
        }

        // int64_t a = 1;
        // int64_t b = 5;
        // int64_t c = 10;
        // PRINT_DEBUG(mod(a - b, c));

        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}
