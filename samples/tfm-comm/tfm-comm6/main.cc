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
            segments.erase(segments.begin() + 1);
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
    Buffer<double> buf(prod(local_nn));
    // pack(local_mm, local_nn, rr, StaticArray<double*, PACK_MAX_INPUTS>);
    PackInputs<double*> test = {nullptr, nullptr};

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
