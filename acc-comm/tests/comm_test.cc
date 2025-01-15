#include <cstdlib>

#include "acm/detail/datatypes.h"
#include "acm/detail/errchk.h"
#include "acm/detail/ndbuffer.h"

#include <algorithm>
#include <numeric>

#include <mpi.h>

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"

#include "acm/detail/buffer_exchange.h"
#include "acm/detail/halo_exchange.h"
#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/io.h"

#include <unistd.h>

#if defined(ACM_CUDA_ENABLED)
#include "acm/detail/errchk_cuda.h"
#elif defined(ACM_HIP_ENABLED)
#include "acm/detail/errchk_cuda.h"
#include "acm/detail/hip.h"
#else
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk.h"
static void
cudaStreamCreate(cudaStream_t* stream)
{
    *stream = nullptr;
}
static void
cudaStreamDestroy(cudaStream_t stream)
{
    (void)stream; // Unused
}
#endif

using UserType = double;

static void
benchmark(void)
{
    const size_t num_samples{5};

    // Stream creation
    for (size_t i{0}; i < num_samples; ++i) {
        cudaStream_t stream;
        BENCHMARK(cudaStreamCreate(&stream));
        BENCHMARK(cudaStreamDestroy(stream));
    }

    // ac::buffers
    const size_t count{(1000 * 1024 * 1024) / sizeof(double)};
    ac::buffer<double, ac::mr::host_memory_resource> hbuf(count);
    ac::buffer<double, ac::mr::device_memory_resource> dbuf(count);

    // C++ standard library
    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(std::copy(hbuf.data(), hbuf.data() + hbuf.size(), hbuf.data()));

    // Regular htoh and dtod
    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(hbuf, hbuf));

    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, dbuf));

    // Regular dtoh and htod
    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(hbuf, dbuf));

    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, hbuf));

    // Pinned
    ac::buffer<double, ac::mr::pinned_host_memory_resource> phbuf(count);
    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(phbuf, dbuf));

    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, phbuf));

    // Pinned write-combined
    ac::buffer<double, ac::mr::pinned_write_combined_host_memory_resource> pwchbuf(count);
    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(pwchbuf, dbuf));

    for (size_t i{0}; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, pwchbuf));
}

int
main()
{
    int provided, claimed, is_thread_main;
    ERRCHK_MPI_API(MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided));
    ERRCHK_MPI_API(MPI_Query_thread(&claimed));
    ERRCHK_MPI(provided == claimed);
    ERRCHK_MPI_API(MPI_Is_thread_main(&is_thread_main));
    ERRCHK_MPI(is_thread_main);

    try {
        int rank, nprocs;
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
#if defined(ACM_DEVICE_ENABLED)
        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(rank % device_count));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif

        // benchmark();

        const Shape global_nn{4, 4, 4};
        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const Shape decomp{ac::mpi::get_decomposition(cart_comm)};
        const Shape local_nn{global_nn / decomp};
        const Index coords{ac::mpi::get_coords(cart_comm)};
        const Index global_nn_offset{coords * local_nn};

        const Shape rr(global_nn.size(), 1); // Symmetric halo
        const Shape local_mm{as<uint64_t>(2) * rr + local_nn};

        ac::ndbuffer<UserType, ac::mr::host_memory_resource> hin(local_mm);
        ac::ndbuffer<UserType, ac::mr::host_memory_resource> hout(local_mm);

        ac::ndbuffer<UserType, ac::mr::device_memory_resource> din(local_mm);
        ac::ndbuffer<UserType, ac::mr::device_memory_resource> dout(local_mm);

        PRINT_LOG("Testing migration"); //-----------------------------------------
        std::iota(hin.begin(),
                  hin.end(),
                  static_cast<UserType>(ac::mpi::get_rank(cart_comm)) *
                      static_cast<UserType>(prod(local_mm)));
        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        migrate(hin.buffer, din.buffer);
        migrate(din.buffer, hout.buffer);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Should be arange");
        hout.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

#if true
        PRINT_LOG("Testing basic halo exchange"); //-------------------------------
        if (nprocs == 1) {
            std::iota(hin.begin(),
                      hin.end(),
                      static_cast<UserType>(ac::mpi::get_rank(cart_comm)) *
                          static_cast<UserType>(prod(local_mm)));
        }
        else {
            std::fill(hin.begin(), hin.end(), static_cast<UserType>(ac::mpi::get_rank(cart_comm)));
        }
        migrate(hin.buffer, din.buffer);

        // Basic MPI halo exchange task
        auto recv_reqs = launch_halo_exchange<UserType>(cart_comm,
                                                        local_mm,
                                                        local_nn,
                                                        rr,
                                                        din.buffer.data(),
                                                        din.buffer.data());
        while (!recv_reqs.empty()) {
            ac::mpi::request_wait_and_destroy(recv_reqs.back());
            recv_reqs.pop_back();
        }
        migrate(din.buffer, hin.buffer);
        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Should be properly exchanged");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
#endif

#if true
        PRINT_LOG("Testing packed halo exchange"); //-------------------------------
        if (nprocs == 1) {
            std::iota(hin.begin(),
                      hin.end(),
                      static_cast<UserType>(ac::mpi::get_rank(cart_comm)) *
                          static_cast<UserType>(prod(local_mm)));
        }
        else {
            std::fill(hin.begin(), hin.end(), static_cast<UserType>(ac::mpi::get_rank(cart_comm)));
        }
        migrate(hin.buffer, din.buffer);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Hin before exhange");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        ac::comm::AsyncHaloExchangeTask<UserType, ac::mr::device_memory_resource>
            halo_exchange{local_mm, local_nn, rr, 1};
        std::vector<ac::mr::device_ptr<UserType>> inputs{
            ac::mr::device_ptr<UserType>{din.size(), din.data()}};

        // Pipelined
        halo_exchange.launch(cart_comm, inputs);
        halo_exchange.wait(inputs);
        migrate(din.buffer, hin.buffer);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Should be properly exchanged");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
#endif

#if true
        PRINT_LOG("Testing IO"); //-------------------------------
        std::iota(hin.begin(),
                  hin.end(),
                  static_cast<UserType>(ac::mpi::get_rank(cart_comm)) *
                      static_cast<UserType>(prod(local_mm)));

        ac::io::AsyncWriteTask<UserType> iotask{global_nn,
                                                global_nn_offset,
                                                local_mm,
                                                local_nn,
                                                rr};
        // iotask.launch_write_collective(cart_comm, hin.buffer, "test.dat");
        // iotask.wait_write_collective();
        ac::mpi::write_collective(cart_comm,
                                  ac::mpi::get_dtype<UserType>(),
                                  global_nn,
                                  global_nn_offset,
                                  local_mm,
                                  local_nn,
                                  rr,
                                  hin.buffer.data(),
                                  "test.dat");
        std::fill(hin.begin(), hin.end(), 0);
        ac::mpi::read_collective(cart_comm,
                                 ac::mpi::get_dtype<UserType>(),
                                 global_nn,
                                 global_nn_offset,
                                 local_mm,
                                 local_nn,
                                 rr,
                                 "test.dat",
                                 hin.buffer.data());

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Should be arange");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // Test reductions
        // std::cout << "decomposition: " << ac::mpi::get_decomposition(cart_comm) << std::endl;
        std::vector<int> buf(10);
        // std::iota(buf.begin(), buf.end(), as<size_t>(ac::mpi::get_rank(cart_comm)) * buf.size());
        std::fill(buf.begin(), buf.end(), ac::mpi::get_rank(cart_comm));

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Reduce result before");
        std::cout << ac::mpi::get_coords(cart_comm) << "{ ";
        for (const auto& elem : buf)
            std::cout << elem << " ";
        std::cout << "}" << std::endl;
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        ac::mpi::reduce(cart_comm, ac::mpi::get_dtype<int>(), MPI_SUM, 0, buf.size(), buf.data());

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Reduce result after");
        std::cout << "{ ";
        for (const auto& elem : buf)
            std::cout << elem << " ";
        std::cout << "}" << std::endl;
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
#endif

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    }
    catch (std::exception& e) {
        PRINT_LOG("Exception caught");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
