#include "comm.h"

#include "decomp.h"
#include "errchk.h"
#include "math_utils.h"
#include "print.h"
#include "type_conversion.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h> // memset

#define SUCCESS (0)
#define FAILURE (-1)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#define USE_RANK_REORDERING (0)

static void
print(const char* label, const int count, const int* arr)
{
    printf("%s: (", label);
    for (int i = 0; i < count; ++i)
        printf("%d%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}

int
comm_run(void)
{
    MPI_Init(NULL, NULL);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#if USE_RANK_REORDERING
    // Hierarchical decomposition
    const size_t gcds_per_gpu  = 2;
    const size_t gpus_per_node = 4;
    const size_t nnodes        = nprocs / (gcds_per_gpu * gpus_per_node);
    ERRCHK(gcds_per_gpu * gpus_per_node * nnodes == as_size_t(nprocs));
    const size_t global_dims[]          = {128, 128, 128};
    const size_t partitions_per_layer[] = {gcds_per_gpu, gpus_per_node, nnodes};
    const size_t ndims                  = ARRAY_SIZE(global_dims);
    const size_t nlayers                = ARRAY_SIZE(partitions_per_layer);
    AcDecompositionInfo info            = acDecompositionInfoCreate(ndims, global_dims, nlayers,
                                                                    partitions_per_layer);
    if (rank == 0)
        acDecompositionInfoPrint(info);

    int dims[] = {
        (int)info.global_decomposition[0],
        (int)info.global_decomposition[1],
        (int)info.global_decomposition[2],
    };
    const int periods[] = {1, 1, 1};

    MPI_Dims_create(nprocs, ndims, dims);

    int keys[nprocs];
    for (size_t i = 0; i < prod(info.ndims, info.global_decomposition); ++i) {

        int64_t pid[info.ndims];
        acGetPid3D(i, info, info.ndims, pid);

        size_t pid_unsigned[info.ndims];
        as_size_t_array(info.ndims, pid, pid_unsigned);
        reverse(info.ndims, pid_unsigned);

        size_t decomposition[info.ndims];
        copy(info.ndims, info.global_decomposition, decomposition);
        reverse(info.ndims, decomposition);

        size_t row_wise_i = to_linear(info.ndims, pid_unsigned, decomposition);
        keys[i]           = row_wise_i;

        if (rank == 0) {
            printf("%zu -> %zu", i, row_wise_i);

            reverse(info.ndims, pid_unsigned);
            acPrintArray_size_t("", info.ndims, pid_unsigned);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    acDecompositionInfoDestroy(&info);

    MPI_Comm reordered_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, keys[rank], &reordered_comm);

    MPI_Comm comm_cart;
    MPI_Cart_create(reordered_comm, ndims, dims, periods, 0, &comm_cart);
    MPI_Comm_free(&reordered_comm);
#else
    MPI_Comm reordered_comm = MPI_COMM_WORLD;
    const int ndims         = 3;
    int dims[ndims];
    memset(dims, 0, ndims * sizeof(dims[0]));

    MPI_Dims_create(nprocs, ndims, dims);
    if (rank == 0)
        print("Mapping", ndims, dims);
    int periods[] = {1, 1, 1};

    MPI_Comm comm_cart;
    MPI_Cart_create(reordered_comm, ndims, dims, periods, 0, &comm_cart);
#endif

    for (int i = 0; i < nprocs; ++i) {
        if (i == rank) {
            int new_rank;
            MPI_Comm_rank(comm_cart, &new_rank);
            int coords[ndims];
            MPI_Cart_coords(comm_cart, new_rank, ndims, coords);
            printf("Hello from %d. New rank %d. ", rank, new_rank);
            print("Mapping", ndims, coords);

            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Comm_free(&comm_cart);

    MPI_Finalize();
    return SUCCESS;
}