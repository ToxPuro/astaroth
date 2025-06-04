/**
 * Checks whether async IO functions actually operate asynchronously and raise an error (if sure
 * that not asynchronous) or a warning (if potentially synchronous but may also be due to the
 * buffering overhead).
 *
 * NOTE: does not check the correctness of the functions
 */
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include <mpi.h>

#include "acm/detail/errchk.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/errchk_print.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/print_debug.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/io.h"

using T = uint64_t;

template <class Clock, class Duration>
static auto
ms_elapsed_since(const std::chrono::time_point<Clock, Duration>& start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                 start)
        .count();
}

template <typename T>
static int
write_async_basic(const MPI_Comm& parent_comm, const size_t count, const T* data)
{
    MPI_Comm cart_comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &cart_comm));

    const auto rank{ac::mpi::get_rank(cart_comm)};
    char       outfile[4096];
    snprintf(outfile, 4096, "async-test-proc-%d.out", rank);

    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(
        MPI_File_open(cart_comm, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file));

    auto start{std::chrono::system_clock::now()};

    MPI_Request req{MPI_REQUEST_NULL};
    ERRCHK_MPI_API(MPI_File_iwrite_all(file, data, as<int>(count), ac::mpi::get_dtype<T>(), &req));

    const auto iwrite_ms_elapsed{ms_elapsed_since(start)};

    std::cout << "[" << iwrite_ms_elapsed << " ms] "
              << " MPI_File_iwrite_all" << std::endl;
    start = std::chrono::system_clock::now();

    ERRCHK_MPI_API(MPI_Wait(&req, MPI_STATUS_IGNORE));

    const auto wait_ms_elapsed{ms_elapsed_since(start)};
    std::cout << "[" << wait_ms_elapsed << " ms] "
              << " MPI_Wait" << std::endl;
    start = std::chrono::system_clock::now();

    ERRCHK_MPI_API(MPI_File_close(&file));

    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));

    // Check that writing happens asynchronously
    ERRCHK_MPI(iwrite_ms_elapsed <= wait_ms_elapsed);

    return 0;
}

static int
test_write_async_basic(const size_t problem_size_in_bytes)
{
    const ac::shape global_nn{problem_size_in_bytes / sizeof(T)};
    MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

    ac::host_ndbuffer<T> hbuf{global_nn};
    std::iota(hbuf.begin(), hbuf.end(), 1);
    const auto buf{hbuf.to_device()};

    ERRCHK(write_async_basic(cart_comm, buf.size(), buf.data()) == 0);

    ac::mpi::cart_comm_destroy(&cart_comm);
    return 0;
}

template <typename T>
static int
write_async_subdomain(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ERRCHK_MPI(global_nn.size() == rr.size());
    const auto                  local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                  local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::io::async_write_task<T> write{global_nn,
                                      ac::make_index(global_nn.size(), 0),
                                      local_mm,
                                      local_nn,
                                      rr};

    ac::host_ndbuffer<T> hbuf{local_mm};
    std::iota(hbuf.begin(), hbuf.end(), 1);
    const auto buf{hbuf.to_device()};

    const auto rank{ac::mpi::get_rank(cart_comm)};
    char       outfile[4096];
    snprintf(outfile, 4096, "async-subdomain-test-proc-%d.out", rank);

    auto start{std::chrono::system_clock::now()};
    write.launch_write_collective(cart_comm, buf.get(), std::string{outfile});

    const auto iwrite_ms_elapsed{ms_elapsed_since(start)};
    std::cout << "[" << iwrite_ms_elapsed << " ms] "
              << " MPI_File_iwrite_all" << std::endl;
    start = std::chrono::system_clock::now();

    write.wait_write_collective();

    const auto wait_ms_elapsed{ms_elapsed_since(start)};
    std::cout << "[" << wait_ms_elapsed << " ms] "
              << " MPI_Wait" << std::endl;
    start = std::chrono::system_clock::now();

    // Check that writing happens asynchronously
    // (though launch may take longer time due to buffering)
    WARNCHK(iwrite_ms_elapsed <= wait_ms_elapsed);
    return 0;
}

static int
test_write_async_subdomain(const size_t approx_problem_size_in_bytes)
{
    ac::shape global_nn{1, 1, 1}; // Note: multidimensional iwrite may hang on some systems
    for (size_t axis{0};; axis = (axis + 1) % global_nn.size()) {
        if (prod(global_nn) * sizeof(T) >= approx_problem_size_in_bytes)
            break;
        else
            global_nn[axis] *= 2;
    }
    const auto rr{ac::make_index(global_nn.size(), 3)};
    PRINT_DEBUG(global_nn);
    PRINT_DEBUG(prod(global_nn) * sizeof(T) / (1024 * 1024));
    PRINT_DEBUG(rr);

    MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

    write_async_subdomain<T>(cart_comm, global_nn, rr);

    ac::mpi::cart_comm_destroy(&cart_comm);
    return 0;
}

template <typename T>
static int
write_async_subdomain_batched(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                              const ac::index& rr, const size_t naggr_bufs)
{
    ERRCHK_MPI(naggr_bufs == 8);
    ERRCHK_MPI(global_nn.size() == rr.size());
    const auto                          local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                          local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::io::batched_async_write_task<T> write{global_nn,
                                              ac::make_index(global_nn.size(), 0),
                                              local_mm,
                                              local_nn,
                                              rr,
                                              naggr_bufs};

    ac::host_ndbuffer<T> hbuf{local_mm};
    std::iota(hbuf.begin(), hbuf.end(), 1);
    const auto buf{hbuf.to_device()};

    const auto rank{ac::mpi::get_rank(cart_comm)};

    std::vector<ac::device_view<T>> inputs;
    std::vector<std::string>        paths;
    for (size_t i{0}; i < naggr_bufs; ++i) {
        inputs.push_back(buf.get());

        char outfile[4096];
        snprintf(outfile, 4096, "async-subdomain-test-proc-%d-%zu.out", rank, i);
        paths.push_back(std::string(outfile));
    }

    auto start{std::chrono::system_clock::now()};
    write.launch(cart_comm, inputs, paths);

    const auto iwrite_ms_elapsed{ms_elapsed_since(start)};
    std::cout << "[" << iwrite_ms_elapsed << " ms] "
              << " MPI_File_iwrite_all" << std::endl;
    start = std::chrono::system_clock::now();

    write.wait();

    const auto wait_ms_elapsed{ms_elapsed_since(start)};
    std::cout << "[" << wait_ms_elapsed << " ms] "
              << " MPI_Wait" << std::endl;
    start = std::chrono::system_clock::now();

    // Check that writing happens asynchronously
    // (though launch may take longer time due to buffering)
    WARNCHK(iwrite_ms_elapsed <= wait_ms_elapsed);
    return 0;
}

static int
test_write_async_subdomain_batched(const size_t approx_problem_size_in_bytes)
{
    const size_t naggr_bufs{8};
    ac::shape    global_nn{1, 1, 1}; // Note: multidimensional iwrite may hang on some systems
    for (size_t axis{0};; axis = (axis + 1) % global_nn.size()) {
        if (prod(global_nn) * sizeof(T) * naggr_bufs >= approx_problem_size_in_bytes)
            break;
        else
            global_nn[axis] *= 2;
    }
    const auto rr{ac::make_index(global_nn.size(), 3)};
    PRINT_DEBUG(global_nn);
    PRINT_DEBUG(prod(global_nn) * sizeof(T) / (1024 * 1024));
    PRINT_DEBUG(rr);

    MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

    write_async_subdomain_batched<T>(cart_comm, global_nn, rr, naggr_bufs);

    ac::mpi::cart_comm_destroy(&cart_comm);
    return 0;
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        // const size_t approx_problem_size_in_bytes{128 * 1024 * 1024};
        constexpr size_t approx_problem_size_in_bytes{1024};
        ERRCHK_MPI(test_write_async_basic(approx_problem_size_in_bytes) == 0);
        ERRCHK_MPI(test_write_async_subdomain(approx_problem_size_in_bytes) == 0);
        ERRCHK_MPI(test_write_async_subdomain_batched(approx_problem_size_in_bytes) == 0);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
