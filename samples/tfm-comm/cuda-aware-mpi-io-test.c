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

    const size_t count = 64;
    AcBuffer buffer    = acBufferCreate(count, false);
    AcBuffer dbuffer   = acBufferCreate(count, true);

    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = (double)i;
    acBufferMigrate(buffer, &dbuffer);

    MPI_File file;
    MPI_Request req;
    MPI_Status status = {0};
    ERRCHK_MPI_API(MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                 MPI_INFO_NULL, &file));
    ERRCHK_MPI_API(MPI_File_iwrite_at(file, 0, dbuffer.data, dbuffer.count, MPI_DOUBLE, &req));
    ERRCHK_MPI_API(MPI_Wait(&req, &status));
    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(status.MPI_ERROR);

    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = (double)0;
    acBufferMigrate(buffer, &dbuffer);

    ERRCHK_MPI_API(
        MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    ERRCHK_MPI_API(MPI_File_read_at(file, 0, dbuffer.data, count, MPI_DOUBLE, &status));
    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(status.MPI_ERROR);

    acBufferMigrate(dbuffer, &buffer);
    for (size_t i = 0; i < count; ++i)
        printf("%zu: %g\n", i, buffer.data[i]);

    acBufferDestroy(&buffer);

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
