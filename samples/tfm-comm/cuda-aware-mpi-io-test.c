#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "errchk_mpi.h"

#include "buffer.h"

#define MPI_SYNCHRONOUS_BLOCK_START                                                                \
    {                                                                                              \
        fflush(stdout);                                                                            \
        MPI_Barrier(MPI_COMM_WORLD);                                                               \
        int rank__, nprocs_;                                                                       \
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank__));                                    \
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs_));                                   \
        for (int i__ = 0; i__ < nprocs_; ++i__) {                                                  \
            if (i__ == rank__) {                                                                   \
                printf("---Rank %d---\n", rank__);

#define MPI_SYNCHRONOUS_BLOCK_END                                                                  \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(MPI_COMM_WORLD);                                                                   \
    }                                                                                              \
    MPI_Barrier(MPI_COMM_WORLD);                                                                   \
    }

int
main(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const size_t count = 10;
    AcBuffer buffer    = acBufferCreate(count, false);

    MPI_File file;
    MPI_Request req;
    MPI_Status status = {0};
    ERRCHK_MPI_API(MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                 MPI_INFO_NULL, &file));
    ERRCHK_MPI_API(MPI_File_iwrite_at(file, 0, buffer.data, buffer.count, MPI_DOUBLE, &req));
    ERRCHK_MPI_API(MPI_Wait(&req, &status));
    ERRCHK_MPI_API(status.MPI_ERROR);
    ERRCHK_MPI_API(MPI_File_close(&file));

    acBufferDestroy(&buffer);

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
