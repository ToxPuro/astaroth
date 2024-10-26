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
// #include "packet.h"

#include "pack.h"

#include "halo_exchange.h"
#include "halo_exchange_packed.h"

// static uint64_t
// mod(const int64_t a, const int64_t b)
// {
//     const int64_t c = a % b;
//     return c < 0 ? as<uint64_t>(c + b) : as<uint64_t>(c);
// }

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

        // Print grid information
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // PRINT_DEBUG(global_nn);
        // PRINT_DEBUG(local_nn);
        // PRINT_DEBUG(decomp);
        // PRINT_DEBUG(coords);
        // PRINT_DEBUG(global_nn_offset);
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        const Shape rr(global_nn.count, 2); // Symmetric halo
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

        NdArray<double> mesh(local_mm);
        // mesh.fill_arange(as<uint64_t>(get_rank(cart_comm)) * prod(local_mm));
        mesh.fill(as<uint64_t>(get_rank(cart_comm)), local_mm, Index(local_mm.count));

        // Print mesh
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // Halo communication using MPI_Datatypes
        // auto segments = partition(local_mm, local_nn, rr);

        /// DEBUG
        // for (size_t i = 0; i < segments.size(); ++i)
        //     mesh.fill(as<uint64_t>(i) + as<uint64_t>(get_rank(cart_comm)) * segments.size(),
        //               segments[i].dims, segments[i].offset);
        // // Print mesh
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
        /// DEBUG

        // Prune the segment containing the computational domain
        // for (size_t i = 0; i < segments.size(); ++i) {
        //     if (within_box(segments[i].offset, local_nn, rr)) {
        //         segments.erase(segments.begin() + as<long>(i));
        //         --i;
        //     }
        // }

        // std::vector<MPI_Request> reqs;
        // for (const Segment& segment : segments) {
        //     const auto recv_offset     = segment.offset;
        //     const auto send_offset     = ((local_nn + recv_offset - rr) % local_nn) + rr;
        //     MPI_Datatype recv_subarray = create_subarray(local_mm, segment.dims, recv_offset,
        //                                                  MPI_DOUBLE);
        //     MPI_Datatype send_subarray = create_subarray(local_mm, segment.dims, send_offset,
        //                                                  MPI_DOUBLE);

        //     const Direction send_direction = get_direction(segment.offset, local_nn, rr);
        //     const int send_neighbor        = get_neighbor(cart_comm, send_direction);
        //     const int recv_neighbor        = get_neighbor(cart_comm, -send_direction);

        //     const int tag = get_tag();

        //     MPI_Request req;
        //     ERRCHK_MPI_API(MPI_Isendrecv(mesh.buffer.data, 1, send_subarray, send_neighbor, tag,
        //     //
        //                                  mesh.buffer.data, 1, recv_subarray, recv_neighbor, tag,
        //                                  cart_comm, &req));
        //     reqs.push_back(req);

        //     ERRCHK_MPI_API(MPI_Type_free(&send_subarray));
        //     ERRCHK_MPI_API(MPI_Type_free(&recv_subarray));
        // }
        // ERRCHK_MPI_API(MPI_Waitall(reqs.size(), reqs.data(),
        //                            MPI_STATUSES_IGNORE)); // TODO proper error checking
        // ERRCHK_MPI_API(MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE));

        // Basic MPI halo exchange task
        auto recv_reqs = create_halo_exchange_task(cart_comm, local_mm, local_nn, rr,
                                                   mesh.buffer.data, mesh.buffer.data);
        while (!recv_reqs.empty()) {
            wait_request(recv_reqs.back());
            recv_reqs.pop_back();
        }

        // Packet MPI/CUDA halo exchange task
        PackInputs<double*> inputs = {mesh.buffer.data};
        HaloExchangeTask<double> task(local_mm, local_nn, rr, inputs.count);

        // // Prune the segment containing the computational domain
        // auto segments = partition(local_mm, local_nn, rr);
        // for (size_t i = 0; i < segments.size(); ++i) {
        //     if (within_box(segments[i].offset, local_nn, rr)) {
        //         segments.erase(segments.begin() + as<long>(i));
        //         --i;
        //     }
        // }

        // std::vector<MPI_Request> send_reqs(segments.size());
        // std::vector<MPI_Request> recv_reqs(segments.size());
        // std::vector<Buffer<double>> send_buffers(segments.size());
        // std::vector<Buffer<double>> recv_buffers(segments.size());
        // for (size_t i = 0; i < segments.size(); ++i) {

        //     const size_t buflen = inputs.count * prod(segments[i].dims);
        //     send_buffers[i]     = std::move(Buffer<double>(buflen));
        // }
        /*
        std::vector<MPI_Request> send_reqs;
        std::vector<MPI_Request> recv_reqs;
        for (const auto& segment : segments) {
            Buffer<double> send_buffer(inputs.count * prod(segment.dims));
            Buffer<double> recv_buffer(inputs.count * prod(segment.dims));

            Index recv_offset = segment.offset;
            Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;

            const Direction recv_direction = get_direction(segment.offset, local_nn, rr);
            const int recv_neighbor        = get_neighbor(cart_comm, recv_direction);
            const int send_neighbor        = get_neighbor(cart_comm, -recv_direction);

            const int tag = get_tag();

            // Post recv
            MPI_Request recv_req;
            ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data, as_int(recv_buffer.count), MPI_DOUBLE,
                                     recv_neighbor, tag, cart_comm, &recv_req));
            recv_reqs.push_back(recv_req);

            // Pack and post send
            pack(local_mm, segment.dims, send_offset, inputs, send_buffer.data);

            MPI_Request send_req;
            ERRCHK_MPI_API(MPI_Isend(send_buffer.data, as_int(send_buffer.count), MPI_DOUBLE,
                                     send_neighbor, tag, cart_comm, &send_req));
            send_reqs.push_back(send_req);
        }
        while (!recv_reqs.empty()) {
            wait_request(recv_reqs.back());
            recv_reqs.pop_back();

            unpack(recv_buffer.data, local_mm, segment.dims, send_offset, inputs);
        }
        */

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        mesh.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}

// int
// main_old()
// {
//     ERRCHK_MPI_API(MPI_Init(NULL, NULL));
//     try {

//         const MPI_Comm parent_comm = MPI_COMM_WORLD;

//         int nprocs;
//         ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &nprocs));

//         const Shape global_nn = {4, 4};
//         Shape decomp          = decompose(global_nn, as<uint64_t>(nprocs));
//         const Shape local_nn  = global_nn / decomp;

//         const Shape rr       = {2, 2};
//         const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

//         /*
//          * Create the library communicator
//          */
//         MPI_Comm cart_comm = create_cart_comm(parent_comm, global_nn);
//         // MPI_Comm cart_comm;
//         // MPIShape mpi_decomp(decomp.reversed());
//         // MPIShape mpi_periods(global_nn.count, 1); // Set as periodic
//         // int reorder = 0;
//         // ERRCHK_MPI_API(MPI_Cart_create(parent_comm, as<int>(mpi_decomp.count),
//         mpi_decomp.data,
//         //                                mpi_periods.data, reorder, &cart_comm));
//         // mpi_comm_print_info(cart_comm);

//         NdArray<double> mesh(local_mm);

//         // Debug: fill the segments with identifying information
//         // MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD)
//         auto segments = partition(local_mm, local_nn, rr);
//         // for (size_t i = 0; i < segments.size(); ++i)
//         //     mesh.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
//         mesh.fill_arange();
//         mesh.display();

//         // Prune the segment containing the computational domain
//         for (size_t i = 0; i < segments.size(); ++i) {
//             if (within_box(segments[i].offset, local_nn, rr)) {
//                 segments.erase(segments.begin() + as<long>(i));
//                 --i;
//             }
//         }
//         // PRINT_DEBUG(segments);
//         // MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD)

//         // Create packets
//         // std::vector<Packet<double>> local_packets;
//         // std::vector<Packet<double>> remote_packets;
//         // for (const auto& segment : segments) {
//         // }

//         // Buffer<double> buf(prod(local_nn));
//         // PackInputs<double*> inputs = {mesh.buffer.data};
//         // pack(local_mm, Shape{2, 2}, Index{2, 2}, inputs, buf.data);
//         // PRINT_DEBUG(buf);
//         // mesh.fill(as<uint64_t>(0), local_mm, Index{0, 0});
//         // unpack(buf.data, local_mm, Shape{2, 2}, Index{2, 2}, inputs);
//         // mesh.display();

//         // Shape pack_dims   = {8, 8};
//         // Index pack_offset = {0, 0};

//         // NdArray<double> a(local_mm);
//         // NdArray<double> b(local_mm);
//         // for (size_t i = 0; i < segments.size(); ++i) {
//         //     a.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
//         //     b.fill(as<uint64_t>(i + 1) + as<uint64_t>(segments.size()), segments[i].dims,
//         //            segments[i].offset);
//         // }
//         // a.display();
//         // b.display();

//         // PackInputs<double*> inputs = {a.buffer.data, b.buffer.data};
//         // Buffer<double> buf(inputs.count * prod(pack_dims));
//         // pack(local_mm, pack_dims, pack_offset, inputs, buf.data);
//         // PRINT_DEBUG(buf);

//         // a.fill(as<uint64_t>(0), local_mm, Index{0, 0});
//         // b.fill(as<uint64_t>(0), local_mm, Index{0, 0});
//         // a.display();
//         // b.display();
//         // unpack(buf.data, local_mm, pack_dims, pack_offset, inputs);
//         // a.display();
//         // b.display();

//         // Send the packets
//         std::vector<Buffer<double>> local_buffers;
//         std::vector<Buffer<double>> remote_buffers;

//         std::vector<MPI_Request> recv_reqs;
//         std::vector<MPI_Request> send_reqs;

//         PackInputs<double*> inputs = {mesh.buffer.data};

//         for (const auto& segment : segments) {
//             Buffer<double> send_buffer(inputs.count * prod(segment.dims));
//             Buffer<double> recv_buffer(inputs.count * prod(segment.dims));

//             Index recv_offset = segment.offset;
//             Index send_offset = ((local_nn + recv_offset - rr) % local_nn) + rr;
//             // PRINT_DEBUG(recv_offset);
//             // PRINT_DEBUG(send_offset);

//             pack(local_mm, segment.dims, send_offset, inputs, send_buffer.data);
//             // PRINT_DEBUG(send_buffer);

//             // PRINT_DEBUG(get_direction(recv_offset, local_nn, rr));
//             // PRINT_DEBUG(-get_direction(recv_offset, local_nn, rr));

//             // Post recv
//             // ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data, as_int(recv_buffer.count), mpi_dtype_,
//             //                          recv_peer, tag, mpi_comm_, &remote_packet->req));

//             // Post send
//             // ERRCHK_MPI_API(MPI_Isend(local_packet->buffer.data,
//             // as_int(local_packet->buffer.count),
//             //                          mpi_dtype_, send_peer, tag, mpi_comm_,
//             &local_packet->req));
//         }

//         // int64_t a = 1;
//         // int64_t b = 5;
//         // int64_t c = 10;
//         // PRINT_DEBUG(mod(a - b, c));

//         ERRCHK_MPI_API(MPI_Finalize());
//     }
//     catch (std::exception& e) {
//         ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
//     }
//     return EXIT_SUCCESS;
// }
