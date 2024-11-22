#include <cstdlib>
#include <iostream>

#include "acc_runtime.h"

#include "tfm_utils.h"

#include "array.h"

#include "mpi_utils.h"
#include <mpi.h>

#include "errchk_cuda.h"
#include "errchk_mpi.h"

using Shape = ac::shape<3>;
using Index = ac::shape<3>;

int
main(const int argc, char* argv[])
{
    init_mpi_funneled();
    try {
        // Load config
        Arguments args;
        ERRCHK_MPI(acParseArguments(argc, argv, &args) == 0);

        printf("Arguments:\n");
        acPrintArguments(args);
        printf("\n");

        if (args.config_path == nullptr) {
            throw std::invalid_argument("Error: Must supply config path. Pass --h for usage.");
        }

        // Mesh configuration
        AcMeshInfo info;
        ERRCHK_MPI(acParseINI(args.config_path, &info) == 0);
        ERRCHK_MPI(acHostUpdateTFMSpecificGlobalParams(&info) == 0);
        ERRCHK_MPI(acHostUpdateMHDSpecificParams(&info) == 0);

        printf("MeshInfo:\n");
        acPrintMeshInfo(info);
        printf("\n");

        // // Setup communicator
        // const Shape global_nn{8, 8};
        // MPI_Comm cart_comm{cart_comm_create(MPI_COMM_WORLD, global_nn)};
        // const Shape decomp{get_decomposition<ndims>(cart_comm)};
        // const Shape local_nn{global_nn / decomp};
        // const Index coords{get_coords<ndims>(cart_comm)};
        // const Index global_nn_offset{coords * local_nn};

        // const Shape rr{as<uint64_t>(2) * ones<uint64_t, ndims>()}; // Symmetric halo
        // const Shape local_mm{as<uint64_t>(2) * rr + local_nn};
        // const int rank{get_rank(cart_comm)};

        // // Setup device
        // // Use the original rank to set the device s.t. the cpu-gpu binding set with slurm is
        // // correct
        // int original_rank{get_rank(MPI_COMM_WORLD)};
        // int device_count;
        // ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        // ERRCHK_CUDA_API(cudaSetDevice(original_rank % device_count));
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    finalize_mpi();
    return EXIT_SUCCESS;
}
