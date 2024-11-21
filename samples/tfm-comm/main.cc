#include <cstdlib>

#include "datatypes.h"
#include "errchk.h"
// #include "ndarray.h"
#include "ndbuffer.h"

#include <algorithm>
#include <numeric>

#include <mpi.h>

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include "buffer_exchange.h"
#include "halo_exchange.h"
#include "halo_exchange_packed.h"
#include "io.h"

#include <unistd.h>

#if defined(CUDA_ENABLED)
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
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
#if defined(DEVICE_ENABLED)
        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(rank % device_count));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif

#if defined(DEVICE_ENABLED)
        benchmark();
#endif
        constexpr size_t N{2};

        const ac::shape<N> global_nn{4, 4};
        MPI_Comm cart_comm{cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const ac::shape<N> decomp{get_decomposition<N>(cart_comm)};
        const ac::shape<N> local_nn{global_nn / decomp};
        const ac::index<N> coords{get_coords<N>(cart_comm)};
        const ac::index<N> global_nn_offset{coords * local_nn};

        const ac::shape<N> rr{ones<uint64_t, N>()}; // Symmetric halo
        const ac::shape<N> local_mm{as<uint64_t>(2) * rr + local_nn};

        ac::ndbuffer<AcReal, N, ac::mr::host_memory_resource> hin(local_mm);
        ac::ndbuffer<AcReal, N, ac::mr::host_memory_resource> hout(local_mm);

        ac::ndbuffer<AcReal, N, ac::mr::device_memory_resource> din(local_mm);
        ac::ndbuffer<AcReal, N, ac::mr::device_memory_resource> dout(local_mm);

        PRINT_LOG("Testing migration"); //-----------------------------------------
        std::iota(hin.begin(), hin.end(),
                  static_cast<AcReal>(get_rank(cart_comm)) * static_cast<AcReal>(prod(local_mm)));
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
            std::iota(hin.begin(), hin.end(),
                      static_cast<AcReal>(get_rank(cart_comm)) *
                          static_cast<AcReal>(prod(local_mm)));
        }
        else {
            std::fill(hin.begin(), hin.end(), static_cast<AcReal>(get_rank(cart_comm)));
        }
        migrate(hin.buffer, din.buffer);

        // Basic MPI halo exchange task
        auto recv_reqs = launch_halo_exchange<AcReal>(cart_comm, local_mm, local_nn, rr,
                                                      din.buffer.data(), din.buffer.data());
        while (!recv_reqs.empty()) {
            request_wait_and_destroy(recv_reqs.back());
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
            std::iota(hin.begin(), hin.end(),
                      static_cast<AcReal>(get_rank(cart_comm)) *
                          static_cast<AcReal>(prod(local_mm)));
        }
        else {
            std::fill(hin.begin(), hin.end(), static_cast<AcReal>(get_rank(cart_comm)));
        }
        migrate(hin.buffer, din.buffer);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Hin before exhange");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        HaloExchangeTask<AcReal, N, ac::mr::device_memory_resource> halo_exchange{local_mm,
                                                                                  local_nn, rr, 1};
        std::vector<ac::buffer<AcReal, ac::mr::device_memory_resource>*> inputs{&din.buffer};
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
        std::iota(hin.begin(), hin.end(),
                  static_cast<AcReal>(get_rank(cart_comm)) * static_cast<AcReal>(prod(local_mm)));

        IOTaskAsync<AcReal, N> iotask{global_nn, global_nn_offset, local_mm, local_nn, rr};
        // iotask.launch_write_collective(cart_comm, hin.buffer, "test.dat");
        // iotask.wait_write_collective();
        mpi_write_collective(cart_comm, global_nn, global_nn_offset, local_mm, local_nn, rr,
                             hin.buffer.data(), "test.dat");
        std::fill(hin.begin(), hin.end(), 0);
        mpi_read_collective(cart_comm, global_nn, global_nn_offset, local_mm, local_nn, rr,
                            "test.dat", hin.buffer.data());

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Should be arange");
        hin.display();
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
