#include <cstdlib>

#include "datatypes.h"
#include "errchk.h"
#include "ndarray.h"

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
void
cudaStreamCreate(cudaStream_t* stream)
{
    *stream = nullptr;
}
void
cudaStreamDestroy(cudaStream_t stream)
{
    (void)stream; // Unused
}
#endif

static void
benchmark(void)
{
    const size_t num_samples = 5;

    // Stream creation
    for (size_t i = 0; i < num_samples; ++i) {
        cudaStream_t stream;
        BENCHMARK(cudaStreamCreate(&stream));
        BENCHMARK(cudaStreamDestroy(stream));
    }

    // Buffers
    const size_t count = (1000 * 1024 * 1024) / sizeof(double);
    Buffer<double, HostMemoryResource> hbuf(count);
    Buffer<double, DeviceMemoryResource> dbuf(count);

    // C++ standard library
    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(std::copy(hbuf.data(), hbuf.data() + hbuf.size(), hbuf.data()));

    // Regular htoh and dtod
    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(hbuf, hbuf));

    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, dbuf));

    // Regular dtoh and htod
    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(hbuf, dbuf));

    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, hbuf));

    // Pinned
    Buffer<double, PinnedHostMemoryResource> phbuf(count);
    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(phbuf, dbuf));

    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(dbuf, phbuf));

    // Pinned write-combined
    Buffer<double, PinnedWriteCombinedHostMemoryResource> pwchbuf(count);
    for (size_t i = 0; i < num_samples; ++i)
        BENCHMARK(migrate(pwchbuf, dbuf));

    for (size_t i = 0; i < num_samples; ++i)
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

        const Shape global_nn{4, 4};
        MPI_Comm cart_comm           = cart_comm_create(MPI_COMM_WORLD, global_nn);
        const Shape decomp           = get_decomposition(cart_comm);
        const Shape local_nn         = global_nn / decomp;
        const Index coords           = get_coords(cart_comm);
        const Index global_nn_offset = coords * local_nn;

        const Shape rr(global_nn.count, 1); // Symmetric halo
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

        NdArray<AcReal, HostMemoryResource> hin(local_mm);
        NdArray<AcReal, HostMemoryResource> hout(local_mm);

        NdArray<AcReal, DeviceMemoryResource> din(local_mm);
        NdArray<AcReal, DeviceMemoryResource> dout(local_mm);

        PRINT_LOG("Testing migration"); //-----------------------------------------
        hin.arange(static_cast<AcReal>(get_rank(cart_comm)) * static_cast<AcReal>(prod(local_mm)));
        // hin.fill(static_cast<AcReal>(get_rank(cart_comm)), local_mm, Index(local_mm.count));
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
            hin.arange(static_cast<AcReal>(get_rank(cart_comm)) *
                       static_cast<AcReal>(prod(local_mm)));
        }
        else {
            hin.fill(static_cast<AcReal>(get_rank(cart_comm)), local_mm, Index(local_mm.count));
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
            hin.arange(static_cast<AcReal>(get_rank(cart_comm)) *
                       static_cast<AcReal>(prod(local_mm)));
        }
        else {
            hin.fill(static_cast<AcReal>(get_rank(cart_comm)), local_mm, Index(local_mm.count));
        }
        migrate(hin.buffer, din.buffer);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        PRINT_LOG("Hin before exhange");
        hin.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        HaloExchangeTask<AcReal> halo_exchange{local_mm, local_nn, rr, 1};
        PackPtrArray<AcReal*> inputs{din.buffer.data()};
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
        hin.arange(static_cast<AcReal>(get_rank(cart_comm)) * static_cast<AcReal>(prod(local_mm)));

        IOTaskAsync<AcReal> iotask{global_nn, global_nn_offset, local_mm, local_nn, rr};
        // iotask.launch_write(cart_comm, hin.buffer, "test.dat");
        // iotask.wait_write();
        mpi_write(cart_comm, global_nn, global_nn_offset, local_mm, local_nn, rr, hin.buffer.data(),
                  "test.dat");
        hin.buffer.fill(0);
        mpi_read(cart_comm, global_nn, global_nn_offset, local_mm, local_nn, rr, "test.dat",
                 hin.buffer.data());

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
