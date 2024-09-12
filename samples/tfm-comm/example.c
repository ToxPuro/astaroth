#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

//static const int ndims = 3;
#define NDIMS (2)

static void
print(const char* label, const int count, const int* arr)
{
    printf("%s: (", label);
    for (int i = 0; i < count; ++i)
        printf("%d%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}

static int
prod(const int count, const int* arr)
{
    int res = 1;
    for (int i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

static void
decompose(const int nprocs, const int ndims, int* dims)
{
    for (int i = 0; i < ndims; ++i)
        dims[i] = 1;

    int i = 0;
    while (prod(ndims, dims) != nprocs) {
        dims[i] *= 2;
        i = (i + 1) % ndims;
    }
}

int
main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Decompose
    int dims[NDIMS];
    decompose(nprocs, NDIMS, dims);
    const int periods[] = {[0 ... NDIMS-1] = 1};
    //print("Periods", sizeof(periods)/sizeof(periods[0]), periods);

    MPI_Dims_create(nprocs, NDIMS, dims);
    if (pid == 0)
        print("MPI dims", NDIMS, dims);
    MPI_Barrier(MPI_COMM_WORLD);

    // Reorder
    int keys[nprocs];
    for (int counter = 0; counter < nprocs; ++counter){
        // const int bx = 2;
        // const int by = 2;
        // const int i = counter % bx;
        // const int j = (counter / dims[0]) % by;
        // const int bi = (counter % dims[0]) / bx;
        // const int bj = counter / (dims[0]*by);
        // keys[counter] = i + bi*bx*by + j*bx + bj * dims[0]*by;
        // // keys[counter] = i + bi * bx*by + j * dims[0] + bj * dims[0]*by;
        const int i = counter % dims[0];
        const int j = counter / dims[1];

        const int nx = dims[1];
        const int bn = 2;
        const int i0 = i % bn;
        const int j0 = j % bn;
        const int i1 = i / bn;
        const int j1 = j / bn;
        keys[counter] = i0 + j0 * bn + i1 * bn * bn + j1 * nx * bn;

        if (pid == 0){
            printf("Map %d -> %d\n", counter, keys[counter]);
            // printf("\t(%d, %d)\n", j, i);
            // printf("\t(%d, %d)\n", bj, bi);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Comm reordered_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, keys[pid], &reordered_comm);

    MPI_Comm comm_cart;
    MPI_Cart_create(reordered_comm, NDIMS, dims, periods, 0, &comm_cart);

    for (int i = 0; i < nprocs; ++i) {
        if (i == pid) {
            int new_pid;
            MPI_Comm_rank(reordered_comm, &new_pid);
            int coords[NDIMS];
            MPI_Cart_coords(comm_cart, new_pid, NDIMS, coords);
            printf("Hello from %d. New pid %d. ", pid, new_pid);
            print("Mapping", NDIMS, coords);

            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}