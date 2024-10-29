#include <cstdlib>

#include "datatypes.h"
#include "errchk.h"
#include "ndarray.h"

#include <mpi.h>

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include "halo_exchange.h"
#include "halo_exchange_packed.h"
#include "io.h"

int
main()
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {
        const Shape global_nn        = {8, 8, 8};
        MPI_Comm cart_comm           = create_cart_comm(MPI_COMM_WORLD, global_nn);
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

        const MPI_Datatype mpi_dtype = get_dtype<AcReal>();
        NdArray<AcReal> mesh(local_mm);
        // mesh.fill_arange(as<uint64_t>(get_rank(cart_comm)) * prod(local_mm));
        mesh.fill(as<uint64_t>(get_rank(cart_comm)), local_mm, Index(local_mm.count));

        // Print mesh
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // Basic MPI halo exchange task
        // auto recv_reqs = create_halo_exchange_task<AcReal>(cart_comm, local_mm, local_nn, rr,
        //                                            mesh.buffer.data, mesh.buffer.data);
        // while (!recv_reqs.empty()) {
        //     wait_request(recv_reqs.back());
        //     recv_reqs.pop_back();
        // }

        // Packet MPI/CUDA halo exchange task
        PackInputs<AcReal*> inputs = {mesh.buffer.data};
        HaloExchangeTask<AcReal> task(local_mm, local_nn, rr, inputs.count);
        task.launch(cart_comm, inputs);
        task.wait(inputs);

        // Print mesh
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        mesh.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // IO
        if (/* DISABLES CODE */ (false)) {
            MPI_Datatype global_subarray = create_subarray(global_nn, local_nn, global_nn_offset,
                                                           mpi_dtype);
            MPI_Datatype local_subarray  = create_subarray(local_mm, local_nn, rr, mpi_dtype);

            MPI_Info info;
            ERRCHK_MPI_API(MPI_Info_create(&info));
            ERRCHK_MPI_API(MPI_Info_set(info, "blocksize", "4096"));
            ERRCHK_MPI_API(MPI_Info_set(info, "striping_factor", "4"));
            // ERRCHK_MPI_API(MPI_Info_set(info, "striping_unit", "...")); // Size of stripe chunks
            // ERRCHK_MPI_API(MPI_Info_set(info, "cb_buffer_size", "...")); // Collective buffer
            // size ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_read", "...")); // Data sieving
            // ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_write", "...")); // Data sieving
            // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_read", "...")); // Collective buffering
            // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_write", "...")); // Collective buffering
            // ERRCHK_MPI_API(MPI_Info_set(info, "romio_no_indep_rw", "...")); // Enable/disable
            // independent rw

            // Write
            MPI_File file = MPI_FILE_NULL;
            ERRCHK_MPI_API(MPI_File_open(cart_comm, "mesh.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                         info, &file));
            ERRCHK_MPI_API(MPI_File_set_view(file, 0, mpi_dtype, global_subarray, "native", info));
            // ERRCHK_MPI_API(
            //     MPI_File_write_all(file, mesh.buffer.data, 1, local_subarray,
            //     MPI_STATUS_IGNORE));
            MPI_Request req;
            ERRCHK_MPI_API(MPI_File_iwrite_all(file, mesh.buffer.data, 1, local_subarray, &req));
            wait_request(req);
            ERRCHK_MPI_API(MPI_File_close(&file));

            // Read
            ERRCHK_MPI_API(MPI_File_open(cart_comm, "mesh.dat", MPI_MODE_RDONLY, info, &file));
            ERRCHK_MPI_API(MPI_File_set_view(file, 0, mpi_dtype, global_subarray, "native", info));
            MPI_Status status = {.MPI_ERROR = MPI_SUCCESS};
            ERRCHK_MPI_API(MPI_File_read_all(file, mesh.buffer.data, 1, local_subarray, &status));
            ERRCHK_MPI_API(status.MPI_ERROR);
            ERRCHK_MPI_API(MPI_File_close(&file));

            ERRCHK_MPI_API(MPI_Info_free(&info));
            ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
            ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        }
        IOTask<AcReal> iotask(global_nn, global_nn_offset, local_mm, local_nn, rr);
        // iotask.write(cart_comm, "test.dat", mesh.buffer.data);
        // iotask.launch_write(cart_comm, "test.dat", mesh.buffer.data);
        // iotask.wait_write();
        iotask.write(cart_comm, "test.dat", mesh.buffer.data);
        iotask.read(cart_comm, "test.dat", mesh.buffer.data);

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        mesh.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        ERRCHK_MPI_EXPR_DESC(false, "Exception caught");
    }
    return EXIT_SUCCESS;
}
