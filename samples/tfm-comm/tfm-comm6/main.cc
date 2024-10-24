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

static uint64_t
get_neighbor(const Direction& dir, const Direction& coords, const Direction& decomp)
{
    // Direction neighbor_coords(coords);
    // for (size_t i = 0; i < coords.count; ++i)
    //     neighbor_coords[i] = mod(neighbor_coords[i] + dir[i], decomp[i]);
    // return
    // TODO
    return 0;
}

int
main()
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {

        int nprocs;
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

        const Shape global_nn = {4, 4};
        Shape decomp          = decompose(global_nn, as<uint64_t>(nprocs));
        const Shape local_nn  = global_nn / decomp;

        const Shape rr       = {2, 2};
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

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
            PRINT_DEBUG(recv_offset);
            PRINT_DEBUG(send_offset);

            pack(local_mm, segment.dims, send_offset, inputs, send_buffer.data);
            PRINT_DEBUG(send_buffer);

            PRINT_DEBUG(get_direction(recv_offset, local_nn, rr));
            PRINT_DEBUG(-get_direction(recv_offset, local_nn, rr));

            // Post recv
            // ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data, as_int(recv_buffer.count), mpi_dtype_,
            //                          recv_peer, tag, mpi_comm_, &remote_packet->req));

            // Post send
            // ERRCHK_MPI_API(MPI_Isend(local_packet->buffer.data,
            // as_int(local_packet->buffer.count),
            //                          mpi_dtype_, send_peer, tag, mpi_comm_, &local_packet->req));
        }

        int64_t a = 1;
        int64_t b = 5;
        int64_t c = 10;
        PRINT_DEBUG(mod(a - b, c));

        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}
