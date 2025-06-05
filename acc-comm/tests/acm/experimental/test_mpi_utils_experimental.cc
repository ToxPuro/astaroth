#include <cstdlib>
#include <exception>
#include <iostream>

#include "acm/detail/allocator.h"
#include "acm/detail/experimental/mpi_utils_experimental.h"
#include "acm/detail/experimental/verify_experimental.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/print_debug.h"
#include "acm/detail/view.h"

template <typename T, typename Allocator>
static void
test(const ac::mpi::comm& comm)
{
    const ac::shape local_nn{10};
    const ac::index local_rr{0};
    const ac::shape local_mm{local_nn + 2 * local_rr};
    const ac::shape global_nn{ac::mpi::size(comm) * local_nn};
    const ac::shape global_nn_offset{ac::mpi::rank(comm) * local_nn};

    ac::ndbuffer<T, Allocator> in{local_mm};
    ac::ndbuffer<T, Allocator> out{local_mm};
    ac::ndbuffer<T, Allocator> model{local_mm};

    // Setup the model solutoon
    const auto nprocs{ac::mpi::size(comm)};
    for (size_t i{0}; i < in.size(); ++i)
        model[i] = as<int>(i * nprocs + in.size() * nprocs * (nprocs - 1) / 2);

    // Unbuffered
    ac::to_global_iota(global_nn, global_nn_offset, local_mm, local_nn, local_rr, in.get());
    MPI_SYNCHRONOUS_BLOCK_START(comm.get());
    in.display();
    MPI_SYNCHRONOUS_BLOCK_END(comm.get());

    auto req{ac::mpi::iallreduce(comm.get(), in.get(), MPI_SUM, out.get())};
    // Note: in must not be modified before calling wait
    req.wait();
    ERRCHK_MPI(ac::equals(out.get(), model.get()));

    MPI_SYNCHRONOUS_BLOCK_START(comm.get());
    out.display();
    MPI_SYNCHRONOUS_BLOCK_END(comm.get());

    // Buffered
    ac::to_global_iota(global_nn, global_nn_offset, local_mm, local_nn, local_rr, in.get());
    ac::mpi::buffered_iallreduce<T, Allocator> task{};
    task.launch(comm.get(), in.get(), MPI_SUM, out.get());
    ac::fill(in.get(), -1); // Can modify the in buffer before waiting
    // But not the output buffer
    task.wait();
    ERRCHK_MPI(ac::equals(out.get(), model.get()));

    MPI_SYNCHRONOUS_BLOCK_START(comm.get());
    out.display();
    MPI_SYNCHRONOUS_BLOCK_END(comm.get());

    // Double buffered
    ac::to_global_iota(global_nn, global_nn_offset, local_mm, local_nn, local_rr, in.get());
    ac::mpi::twoway_buffered_iallreduce<T, Allocator> twb_task;
    twb_task.launch(comm.get(), in.get(), MPI_SUM);
    ac::fill(in.get(), -1);  // Can modify the in buffer before waiting
    ac::fill(out.get(), -1); // Can modify the out buffer before waiting
    twb_task.wait(out.get());
    ERRCHK_MPI(ac::equals(out.get(), model.get()));

    MPI_SYNCHRONOUS_BLOCK_START(comm.get());
    out.display();
    MPI_SYNCHRONOUS_BLOCK_END(comm.get());
}

int
main()
{
    ac::mpi::init_funneled();

    try {
        ac::mpi::comm comm{MPI_COMM_WORLD};
        test<int, ac::mr::host_allocator>(comm);
        test<int, ac::mr::device_allocator>(comm);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
