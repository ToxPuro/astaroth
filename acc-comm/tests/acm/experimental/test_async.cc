#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include "acm/detail/allocator.h"
#include "acm/detail/buffer.h"
#include "acm/detail/mpi_utils.h"

#include "acm/detail/experimental/mpi_utils_experimental.h"
#include "acm/detail/view.h"

template <typename T, typename Allocator>
void
print(const std::string& label, const ac::view<T, Allocator>& view)
{
    std::cout << label << ": " << std::endl;
    for (size_t i{0}; i < view.size(); ++i) {
        std::cout << "\t" << i << ": " << view[i] << std::endl;
    }
    std::cout << std::endl;
}

#define PRINT(x)                                                                                   \
    do {                                                                                           \
        print(#x, x);                                                                              \
    } while (0)

int
main()
{
    ac::mpi::init_funneled();

    using T = int;

    const size_t       count{4};
    ac::host_buffer<T> buf0{count, 0};
    ac::host_buffer<T> buf1{count, 0};

    auto in{ac::view<T, ac::mr::host_allocator>{buf0}};
    auto out{ac::view<T, ac::mr::host_allocator>{buf1}};

    // host_copy
    {
        std::iota(in.begin(), in.end(), 1);
        ac::copy(in, out);
        ERRCHK(equals(in, out));
    }

    // host_copy_async
    {
        std::fill(out.begin(), out.end(), 0);
        auto fut{ac::copy_async(in, out)};
        fut.wait();
        ERRCHK(equals(in, out));
    }

    // mpi_isend, mpi_irecv
    {
        std::fill(out.begin(), out.end(), 0);
        auto send_task{ac::mpi::isend(MPI_COMM_WORLD, 0, in, ac::mpi::get_rank(MPI_COMM_WORLD))};
        auto recv_task{ac::mpi::irecv(MPI_COMM_WORLD, 0, ac::mpi::get_rank(MPI_COMM_WORLD), out)};
        send_task.wait();
        recv_task.wait();
        ERRCHK(equals(in, out));
    }

    // mpi_isend, mpi_irecv: blocking send
    {
        std::fill(out.begin(), out.end(), 0);
        ac::mpi::isend(MPI_COMM_WORLD, 0, in, ac::mpi::get_rank(MPI_COMM_WORLD)).wait();
        auto recv_task{ac::mpi::irecv(MPI_COMM_WORLD, 0, ac::mpi::get_rank(MPI_COMM_WORLD), out)};
        recv_task.wait();
        ERRCHK(equals(in, out));
    }

    // mpi_allreduce
    {
        std::iota(in.begin(), in.end(), as<size_t>(ac::mpi::get_rank(MPI_COMM_WORLD)) * count);
        std::fill(out.begin(), out.end(), 0);
        auto reduce_task{ac::mpi::iallreduce(MPI_COMM_WORLD, in, MPI_SUM, out)};
        reduce_task.wait();
        const auto nprocs{as<size_t>(ac::mpi::get_size(MPI_COMM_WORLD))};
        for (size_t i{0}; i < in.size(); ++i)
            in[i] = as<int>(i * nprocs + in.size() * nprocs * (nprocs - 1) / 2);
        ERRCHK(equals(in, out));

        PRINT(in);
        PRINT(out);
    }

    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
