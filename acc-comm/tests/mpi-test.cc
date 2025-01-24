#include <cstdlib>
#include <iostream>

#include "acm/detail/mpi_utils.h"

template <typename MemoryResource>
void
test_reduce(const MPI_Comm& cart_comm, const Shape& global_nn)
{
    Shape decomp{ac::mpi::get_decomposition(cart_comm)};
    Index coords{ac::mpi::get_coords(cart_comm)};
    const size_t nprocs{prod(decomp)};

    // Checks that the reduce sum is the sum of all processes along a specific axis
    for (size_t axis{0}; axis < global_nn.size(); ++axis) {
        constexpr size_t count{10};
        const int value{as<int>((coords[axis] + 1) * nprocs)};
        ac::buffer<int, ac::mr::host_memory_resource> tmp{count, value};
        ac::buffer<int, MemoryResource> buf{count};
        migrate(tmp, buf);

        ac::mpi::reduce(cart_comm,
                        ac::mpi::get_dtype<int>(),
                        MPI_SUM,
                        axis,
                        buf.size(),
                        buf.data());

        migrate(buf, tmp);

        PRINT_DEBUG(decomp);
        PRINT_DEBUG(coords);
        PRINT_DEBUG(nprocs);
        PRINT_DEBUG_ARRAY(tmp.size(), tmp.data());

        // E.g. 4 procs on axis, the value in the buffer of each proc corresponds to its
        // coordinates on that axis
        for (size_t i{0}; i < count; ++i) {
            const size_t nprocs_on_axis{nprocs / decomp[axis]};
            ERRCHK(tmp[i] == value * as<int>(nprocs_on_axis));
        }
    }
}

void
test_scatter_gather(const MPI_Comm& cart_comm, const Shape& global_nn)
{
    std::cout << "Hello" << std::endl;
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        const Shape global_nn{128, 128, 128};
        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

        test_reduce<ac::mr::host_memory_resource>(cart_comm, global_nn);
        test_reduce<ac::mr::pinned_host_memory_resource>(cart_comm, global_nn);
        test_reduce<ac::mr::pinned_write_combined_host_memory_resource>(cart_comm, global_nn);
        test_reduce<ac::mr::device_memory_resource>(cart_comm, global_nn);

        test_scatter_gather(cart_comm, global_nn);

        ac::mpi::cart_comm_destroy(cart_comm);
    }
    catch (const std::exception& e) {
        PRINT_LOG("Exception caught");
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
