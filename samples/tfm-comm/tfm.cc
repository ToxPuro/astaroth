#include <cstdlib>
#include <exception>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "mpi_utils.h"
#include <mpi.h>

// #include "astaroth.h"

int
main()
{
    init_mpi_funneled();
    try {
        constexpr size_t count = 10;
        thrust::host_vector<double> hin(count);
        thrust::device_vector<double> din(count);
        thrust::device_vector<double> dout(count);
        thrust::host_vector<double> hout(count);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    finalize_mpi();

    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
