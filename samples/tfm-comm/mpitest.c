#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "errchk_mpi.h"

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
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const int n = 10;
    int a[n];
    int b[n];

    for (int i = 0; i < n; ++i) {
        a[i] = rank;
        b[i] = -1;
    }

    MPI_SYNCHRONOUS_BLOCK_START
    printf("a:\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", a[i]);
    printf("\n");

    printf("b:\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", b[i]);
    printf("\n");
    MPI_SYNCHRONOUS_BLOCK_END

    MPI_Request send_req = {0}, recv_req = {0};
    ERRCHK_MPI_API(MPI_Irecv(b, n, MPI_INT, (nprocs + rank + 1) % nprocs,
                             (nprocs + rank + 1) % nprocs, MPI_COMM_WORLD, &recv_req));
    ERRCHK_MPI_API(
        MPI_Isend(a, n, MPI_INT, (nprocs + rank - 1) % nprocs, rank, MPI_COMM_WORLD, &send_req));

    MPI_Status send_status = {0}, recv_status = {0};
    ERRCHK_MPI_API(MPI_Wait(&recv_req, &recv_status));
    ERRCHK_MPI_API(MPI_Wait(&send_req, &send_status));

    ERRCHK_MPI_API(recv_status.MPI_ERROR);
    ERRCHK_MPI_API(send_status.MPI_ERROR);

    if (recv_req != MPI_REQUEST_NULL)
        ERRCHK_MPI_API(MPI_Request_free(&recv_req));
    if (send_req != MPI_REQUEST_NULL)
        ERRCHK_MPI_API(MPI_Request_free(&send_req));

    MPI_SYNCHRONOUS_BLOCK_START
    printf("a:\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", a[i]);
    printf("\n");

    printf("b:\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", b[i]);
    printf("\n");
    MPI_SYNCHRONOUS_BLOCK_END

    ERRCHK_MPI_API(MPI_Finalize());
}

// #include "nalloc.h"

// int
// main(void)
// {
//     MPI_Init(NULL, NULL);

//     const size_t n = 8;
//     double *a, *b;
//     nalloc(n, a);
//     nalloc(n, b);

//     int rank, nprocs;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

//     for (size_t i = 0; i < n; ++i)
//         a[i] = rank;

//     for (int i = 0; i < nprocs; ++i) {
//         MPI_Barrier(MPI_COMM_WORLD);
//         if (i == rank) {
//             printf("Rank %d a\n", rank);
//             for (size_t j = 0; j < n; ++j)
//                 printf("%zu: %g\n", j, a[j]);
//             printf("\n");
//         }
//         MPI_Barrier(MPI_COMM_WORLD);
//     }

//     // MPI_Status status;
//     // MPI_Sendrecv(a, n, MPI_DOUBLE, (rank + 1) % nprocs, (rank + 1) % nprocs, b, n, MPI_DOUBLE,
//     //              (rank + 1) % nprocs, rank, MPI_COMM_WORLD, &status);

//     // MPI_Request req;
//     // MPI_Isendrecv(a, n, MPI_DOUBLE, (rank + 1) % nprocs, (rank + 1) % nprocs, b, n,
//     MPI_DOUBLE,
//     //               (rank + 1) % nprocs, rank, MPI_COMM_WORLD, &req);
//     // MPI_Wait(&req, &status);
//     // if (status.MPI_ERROR != MPI_SUCCESS) // Firewall blocks sender getting information of the
//     send
//     //                                      // count (do with 2, otherwise dining philosophers)
//     //     fprintf(stderr, "Failure\n");

//     MPI_Request send_req, recv_req;
//     MPI_Irecv(b, n, MPI_DOUBLE, (rank + 1) % nprocs, rank, MPI_COMM_WORLD, &recv_req);
//     MPI_Isend(a, n, MPI_DOUBLE, (rank + 1) % nprocs, (rank + 1) % nprocs, MPI_COMM_WORLD,
//     &send_req);

//     MPI_Status send_status, recv_status;
//     MPI_Wait(&send_req, &send_status);
//     MPI_Wait(&recv_req, &recv_status);

//     if (send_status.MPI_ERROR != MPI_SUCCESS){
//         if (send_status.MPI_ERROR == MPI_ERR_COUNT)
//             fprintf(stderr, "Invalid count\n");
//         if (send_status.MPI_ERROR == MPI_ERR_TAG)
//             fprintf(stderr, "Invalid tag\n");
//         fprintf(stderr, "send failure\n");

//         char description[MPI_MAX_ERROR_STRING];
//             int resultlen;
//             MPI_Error_string(send_status.MPI_ERROR, description, &resultlen);
//             printf(description);
// }

//     for (int i = 0; i < nprocs; ++i) {
//         MPI_Barrier(MPI_COMM_WORLD);
//         fflush(stdout);
//         if (i == rank) {
//             printf("Rank %d b\n", rank);
//             for (size_t j = 0; j < n; ++j)
//                 printf("%zu: %g\n", j, b[j]);
//             printf("\n");
//         }
//         fflush(stdout);
//         MPI_Barrier(MPI_COMM_WORLD);
//     }

//     ndealloc(b);
//     ndealloc(a);

//     MPI_Finalize();
//     return EXIT_SUCCESS;
// }
