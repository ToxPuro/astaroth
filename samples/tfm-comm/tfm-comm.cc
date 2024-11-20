#include <cstdio>
#include <iostream>

#include "datatypes.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"
#include <mpi.h>

int
main()
{
    init_mpi_funneled();
    try {
        int rank, nprocs;
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
#if defined(DEVICE_ENABLED)
        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(rank % device_count));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif

        constexpr size_t ndims{2};
        const ac::shape<ndims> global_nn{4, 4};
        MPI_Comm cart_comm{cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const ac::shape<ndims> decomp{get_decomposition(cart_comm)};
        const ac::shape<ndims> local_nn{global_nn / decomp};
        const ac::index<ndims> coords{get_coords(cart_comm)};
        const ac::index<ndims> global_nn_offset{coords * local_nn};

        auto rr{ones<uint64_t, ndims>()}; // Symmetric halo
        auto local_mm{as<uint64_t>(2) * rr + local_nn};

        const size_t count{prod(local_mm)};
        Buffer<AcReal, HostMemoryResource> lnrho(count);
        Buffer<AcReal, HostMemoryResource> ux(count);
        Buffer<AcReal, HostMemoryResource> uy(count);
        Buffer<AcReal, HostMemoryResource> uz(count);

        // Raw pointer cast is required to convert from device_ptr wrapper returned with
        // device_vector.data() to a raw pointer
        ac::array<AcReal*, 10> inputs{
            lnrho.data(),
            ux.data(),
            uy.data(),
            uz.data(),
        };
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        abort_mpi();
    }
    finalize_mpi();
}
