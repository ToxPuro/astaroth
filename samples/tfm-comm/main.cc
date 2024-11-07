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
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {
        int nprocs;
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(nprocs % device_count));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());
        benchmark();

        const Shape global_nn        = {4, 4};
        MPI_Comm cart_comm           = cart_comm_create(MPI_COMM_WORLD, global_nn);
        const Shape decomp           = get_decomposition(cart_comm);
        const Shape local_nn         = global_nn / decomp;
        const Index coords           = get_coords(cart_comm);
        const Index global_nn_offset = coords * local_nn;

        // Print grid information
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // PRINT_DEBUG(global_nn);
        // PRINT_DEBUG(local_nn);
        // PRINT_DEBUG(decomp);
        // PRINT_DEBUG(coords);
        // PRINT_DEBUG(global_nn_offset);
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        const Shape rr(global_nn.count, 1); // Symmetric halo
        const Shape local_mm = as<uint64_t>(2) * rr + local_nn;

        NdArray<AcReal> mesh(local_mm);
        // mesh.fill_arange(as<uint64_t>(get_rank(cart_comm)) * prod(local_mm));
        mesh.fill(static_cast<AcReal>(get_rank(cart_comm)), local_mm, Index(local_mm.count));

        // Print mesh
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // Basic MPI halo exchange task
        // auto recv_reqs = create_halo_exchange_task<AcReal>(cart_comm, local_mm, local_nn, rr,
        //                                            mesh.buffer.data(),
        //                                            mesh.buffer.data());
        // while (!recv_reqs.empty()) {
        //     request_wait_and_destroy(recv_reqs.back());
        //     recv_reqs.pop_back();
        // }

        // Migrate
        const size_t count = 10;
        Buffer<AcReal, HostMemoryResource> hbuf(count);
        Buffer<AcReal, DeviceMemoryResource> dbuf(count);

        HostToDeviceBufferExchangeTask<AcReal> htod(count);
        hbuf.arange(static_cast<AcReal>(count * get_rank(cart_comm)));
        htod.launch(hbuf);
        htod.wait(dbuf);
        hbuf.fill(0);
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        hbuf.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        DeviceToHostBufferExchangeTask<AcReal> dtoh(count);
        dtoh.launch(dbuf);
        dtoh.wait(hbuf);
        hbuf.display();

        // Packed MPI/CUDA halo exchange task
        PackPtrArray<AcReal*> inputs = {mesh.buffer.data()};
        HaloExchangeTask<AcReal> task(local_mm, local_nn, rr, inputs.count);
        task.launch(cart_comm, inputs);
        task.wait(inputs);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        mesh.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // IO
        IOTask<AcReal> iotask(global_nn, global_nn_offset, local_mm, local_nn, rr);
        iotask.write(cart_comm, mesh.buffer.data(), "test.dat");
        // iotask.launch_write(cart_comm, mesh.buffer, "test.dat");
        // iotask.wait_write();
        iotask.read(cart_comm, "test.dat", mesh.buffer.data());

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        mesh.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
