#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "nalloc.h"

int
main(void)
{
    MPI_Init(NULL, NULL);

    const size_t n = 8;
    double *a, *b;
    nalloc(n, a);
    nalloc(n, b);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (size_t i = 0; i < n; ++i)
        a[i] = rank;

    for (int i = 0; i < nprocs; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            printf("Rank %d a\n", rank);
            for (size_t j = 0; j < n; ++j)
                printf("%zu: %g\n", j, a[j]);
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Status status;
    // MPI_Sendrecv(a, n, MPI_DOUBLE, (rank + 1) % nprocs, (rank + 1) % nprocs, b, n, MPI_DOUBLE,
    //              (rank + 1) % nprocs, rank, MPI_COMM_WORLD, &status);

    MPI_Request req;
    MPI_Isendrecv(a, n, MPI_DOUBLE, (rank + 1) % nprocs, (rank + 1) % nprocs, b, n, MPI_DOUBLE,
                  (rank + 1) % nprocs, rank, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &status);

    for (int i = 0; i < nprocs; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);
        if (i == rank) {
            printf("Rank %d b\n", rank);
            for (size_t j = 0; j < n; ++j)
                printf("%zu: %g\n", j, b[j]);
            printf("\n");
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ndealloc(b);
    ndealloc(a);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
