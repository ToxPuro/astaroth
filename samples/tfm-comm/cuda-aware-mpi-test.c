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
main_basic(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const size_t count = 10;
    int* a             = malloc(sizeof(a[0]) * count);
    int* b             = malloc(sizeof(b[0]) * count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = rank;
        b[i] = -1;
    }

    MPI_SYNCHRONOUS_BLOCK_START
    printf("a: ");
    for (size_t i = 0; i < count; ++i)
        printf("%d ", a[i]);
    printf("\n");
    printf("b: ");
    for (size_t i = 0; i < count; ++i)
        printf("%d ", b[i]);
    printf("\n");
    MPI_SYNCHRONOUS_BLOCK_END

    MPI_Request send_req, recv_req;
    ERRCHK_MPI_API(MPI_Irecv(b, count, MPI_INT, (nprocs + rank + 1) % nprocs,
                             (nprocs + rank + 1) % nprocs, MPI_COMM_WORLD, &recv_req));
    ERRCHK_MPI_API(MPI_Isend(a, count, MPI_INT, (nprocs + rank - 1) % nprocs, rank, MPI_COMM_WORLD,
                             &send_req));

    MPI_Status send_status = {0}, recv_status = {0};
    ERRCHK_MPI_API(MPI_Wait(&recv_req, &recv_status));
    ERRCHK_MPI_API(MPI_Wait(&send_req, &send_status));
    ERRCHK_MPI_API(recv_status.MPI_ERROR);
    ERRCHK_MPI_API(send_status.MPI_ERROR);

    MPI_SYNCHRONOUS_BLOCK_START
    printf("a: ");
    for (size_t i = 0; i < count; ++i)
        printf("%d ", a[i]);
    printf("\n");
    printf("b: ");
    for (size_t i = 0; i < count; ++i)
        printf("%d ", b[i]);
    printf("\n");
    MPI_SYNCHRONOUS_BLOCK_END

    free(a);
    free(b);
    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}

int main(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const size_t count = 5;
    AcBuffer a = acBufferCreate(count, false);
    AcBuffer b = acBufferCreate(count, false);

    for (size_t i = 0; i < count; ++i) {
        a.data[i] = rank;
        b.data[i] = -1;
    }


    MPI_SYNCHRONOUS_BLOCK_START
    acBufferPrint("a", a);
    acBufferPrint("b", b);
    MPI_SYNCHRONOUS_BLOCK_END

    // Send to GPU
    AcBuffer da = acBufferCreate(count, true);
    AcBuffer db = acBufferCreate(count, true);
    acBufferMigrate(a, &da);

    MPI_Request send_req, recv_req;
    ERRCHK_MPI_API(MPI_Irecv(db.data, count, MPI_DOUBLE, (nprocs + rank + 1) % nprocs,
                             (nprocs + rank + 1) % nprocs, MPI_COMM_WORLD, &recv_req));
    ERRCHK_MPI_API(MPI_Isend(da.data, count, MPI_DOUBLE, (nprocs + rank - 1) % nprocs, rank, MPI_COMM_WORLD,
                             &send_req));

    MPI_Status send_status = {0}, recv_status = {0};
    ERRCHK_MPI_API(MPI_Wait(&recv_req, &recv_status));
    ERRCHK_MPI_API(MPI_Wait(&send_req, &send_status));
    ERRCHK_MPI_API(recv_status.MPI_ERROR);
    ERRCHK_MPI_API(send_status.MPI_ERROR);

    // Receive from GPU
    acBufferMigrate(db, &b);

    MPI_SYNCHRONOUS_BLOCK_START
    acBufferPrint("a", a);
    acBufferPrint("b", b);
    MPI_SYNCHRONOUS_BLOCK_END

    acBufferDestroy(&db);
    acBufferDestroy(&da);
    acBufferDestroy(&b);
    acBufferDestroy(&a);

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
