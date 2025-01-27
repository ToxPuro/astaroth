#include <cstdlib>
#include <iostream>

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/ndbuffer.h"
#include <numeric> // std::iota

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
    using T      = int;
    using Buffer = ac::ndbuffer<T, ac::mr::host_memory_resource>;

    const Index global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    const Index zero_offset(global_nn.size(), static_cast<int>(0));
    const Shape local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    const Index rr(global_nn.size(), static_cast<uint64_t>(2));
    const Shape local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};

    Buffer monolithic{global_nn};
    std::iota(monolithic.begin(), monolithic.end(), 1);
    Buffer distributed{local_mm};

    ac::mpi::scatter(cart_comm,
                     ac::mpi::get_dtype<T>(),
                     global_nn,
                     rr,
                     monolithic.data(),
                     distributed.data());

    MPI_SYNCHRONOUS_BLOCK_START(cart_comm);
    PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
    PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
    monolithic.display();
    distributed.display();
    MPI_SYNCHRONOUS_BLOCK_END(cart_comm);

    // Check
    Buffer monolithic_test{global_nn, 0};
    ac::mpi::gather(cart_comm,
                    ac::mpi::get_dtype<T>(),
                    global_nn,
                    rr,
                    distributed.data(),
                    monolithic_test.data());

    const auto rank{ac::mpi::get_rank(cart_comm)};
    if (rank == 0) {
        PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
        PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
        monolithic.display();
        monolithic_test.display();

        for (size_t i{0}; i < monolithic_test.size(); ++i)
            ERRCHK_MPI(monolithic.get()[i] == monolithic_test.get()[i]);
    }
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        {
            const Shape global_nn{128, 128, 128};
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_reduce<ac::mr::host_memory_resource>(cart_comm, global_nn);
            test_reduce<ac::mr::pinned_host_memory_resource>(cart_comm, global_nn);
            test_reduce<ac::mr::pinned_write_combined_host_memory_resource>(cart_comm, global_nn);
            test_reduce<ac::mr::device_memory_resource>(cart_comm, global_nn);
            ac::mpi::cart_comm_destroy(cart_comm);
        }
        {
            const Shape global_nn{16};
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(cart_comm);
        }
        {
            const Shape global_nn{8, 8};
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(cart_comm);
        }
        {
            const Shape global_nn{8, 4, 2};
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(cart_comm);
        }
    }
    catch (const std::exception& e) {
        PRINT_LOG("Exception caught");
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
