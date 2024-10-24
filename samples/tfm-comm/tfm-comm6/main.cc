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
        for (size_t i = 0; i < segments.size(); ++i)
            mesh.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
        // mesh.display();

        // Prune the segment containing the computational domain
        for (size_t i = 0; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }
        PRINT_DEBUG(segments);
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

        Shape pack_dims   = {8, 8};
        Index pack_offset = {0, 0};

        NdArray<double> a(local_mm);
        NdArray<double> b(local_mm);
        for (size_t i = 0; i < segments.size(); ++i) {
            a.fill(as<uint64_t>(i + 1), segments[i].dims, segments[i].offset);
            b.fill(as<uint64_t>(i + 1) + as<uint64_t>(segments.size()), segments[i].dims,
                   segments[i].offset);
        }
        a.display();
        b.display();

        PackInputs<double*> inputs = {a.buffer.data, b.buffer.data};
        Buffer<double> buf(inputs.count * prod(pack_dims));
        pack(local_mm, pack_dims, pack_offset, inputs, buf.data);
        PRINT_DEBUG(buf);

        a.fill(as<uint64_t>(0), local_mm, Index{0, 0});
        b.fill(as<uint64_t>(0), local_mm, Index{0, 0});
        a.display();
        b.display();
        unpack(buf.data, local_mm, pack_dims, pack_offset, inputs);
        a.display();
        b.display();

        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}
