#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
// #include <cuda_runtime.h>

#include "errchk_mpi.h"
// #include "errchk_cuda.h"

// #include <thrust/host_vector.h>

int
main(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    // int aa;
    // ERRCHK_MPI_API(MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &aa));

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    {
        // thrust::host_vector<double> hvec(10);
        // MPI_File file;
        // MPI_Request req;
        // MPI_Status status = {};
        // ERRCHK_MPI_API(MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE |
        // MPI_MODE_WRONLY,
        //                              MPI_INFO_NULL, &file));
        // // ERRCHK_MPI_API(MPI_File_iwrite_at(file, 0, hvec.data(), hvec.size(), MPI_DOUBLE,
        // &req));
        // // ERRCHK_MPI_API(MPI_File_iwrite_all(file, hvec.data(), hvec.size(), MPI_DOUBLE, &req));
        // // ERRCHK_MPI_API(MPI_Wait(&req, &status));
        // ERRCHK_MPI_API(MPI_File_write_all(file, hvec.data(), hvec.size(), MPI_DOUBLE, &status));
        // ERRCHK_MPI_API(status.MPI_ERROR);
        // ERRCHK_MPI_API(MPI_File_close(&file));

        int buf = 1;
        MPI_File file;
        MPI_Request req;
        MPI_Status status = {};
        ERRCHK_MPI_API(MPI_File_open(MPI_COMM_WORLD, "out.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     MPI_INFO_NULL, &file));
        ERRCHK_MPI_API(
            MPI_File_iwrite_all(file, &buf, 1, MPI_INT, &req)); // This line causes a segmentation
                                                                // on one machine but not on another
        ERRCHK_MPI_API(MPI_Wait(&req, &status));
        // ERRCHK_MPI_API(MPI_File_write_all(file, &buf, 1, MPI_INT, &status)); // This completes
        // without errors
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&file));
    }

    printf("Completed succesfully\n");
    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
