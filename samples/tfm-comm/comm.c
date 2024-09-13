#include "comm.h"

#include "decomp.h"
#include "errchk.h"
#include "math_utils.h"
#include "print.h"
#include "type_conversion.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h> // memset

#define USE_RANK_REORDERING (1)

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define ERRCHK_MPI(retval)                                                                         \
    ERRCHK(retval);                                                                                \
    if ((retval) == 0) {                                                                           \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

static MPI_Comm
create_rank_reordered_cart_comm(const MPI_Comm parent, const size_t ndims,
                                const size_t global_dims[])
{
    int rank, nprocs;
    MPI_Comm_rank(parent, &rank);
    MPI_Comm_size(parent, &nprocs);

#if USE_RANK_REORDERING
    // Hierarchical decomposition
    const size_t gcds_per_gpu  = as_size_t(min(nprocs, 2));
    const size_t gpus_per_node = min(as_size_t(nprocs) / gcds_per_gpu, 4);
    const size_t nnodes        = as_size_t(nprocs) / (gcds_per_gpu * gpus_per_node);
    ERRCHK_MPI(gcds_per_gpu * gpus_per_node * nnodes == as_size_t(nprocs));
    ERRCHK_MPI(nnodes >= 1);

    const size_t partitions_per_layer[] = {gcds_per_gpu, gpus_per_node, nnodes};
    const size_t nlayers                = ARRAY_SIZE(partitions_per_layer);
    AcDecompositionInfo info            = acDecompositionInfoCreate(ndims, global_dims, nlayers,
                                                                    partitions_per_layer);
    if (rank == 0)
        acDecompositionInfoPrint(info);

    int keys[nprocs];
    for (size_t i = 0; i < as_size_t(nprocs); ++i) {

        int64_t pid[info.ndims];
        acGetPid3D(i, info, info.ndims, pid);

        size_t pid_unsigned[info.ndims];
        as_size_t_array(info.ndims, pid, pid_unsigned);
        reverse(info.ndims, pid_unsigned);

        size_t decomposition[info.ndims];
        copy(info.ndims, info.global_decomposition, decomposition);
        reverse(info.ndims, decomposition);

        size_t row_wise_i = to_linear(info.ndims, pid_unsigned, decomposition);
        keys[i]           = as_int(row_wise_i);

        if (rank == 0) {
            printf("%zu -> %zu\n", i, row_wise_i);

            print_array("\tproper", info.ndims, pid);

            reverse(info.ndims, pid_unsigned);
            print_array("\tmapped", info.ndims, pid_unsigned);
        }
    }
    fflush(stdout);
    MPI_Barrier(parent);

    MPI_Comm reordered_comm;
    MPI_Comm_split(parent, 0, keys[rank], &reordered_comm);

    int dims[info.ndims], periods[info.ndims];
    as_int_array(info.ndims, info.global_decomposition, dims);
    iset(1, info.ndims, periods);

    MPI_Comm comm_cart;
    MPI_Cart_create(reordered_comm, as_int(ndims), dims, periods, 0, &comm_cart);
    MPI_Comm_free(&reordered_comm);

    // // Check that the mapping is correct (TODO)
    // for (size_t i = 0; i < as_size_t(nprocs); ++i) {

    //     int64_t a[info.ndims];
    //     acGetPid3D(i, info, info.ndims, a);

    //     int coords[info.ndims];
    //     as_int_array(info.ndims, a, coords);

    //     int new_rank;
    //     MPI_Cart_rank(comm_cart, coords, &new_rank);

    //     set(0, info.ndims, coords);
    //     MPI_Cart_coords(comm_cart, new_rank, info.ndims, coords);

    //     ERRCHK_MPI(i == new_rank);
    //     //     int new_rank;
    //     // MPI_Comm_rank(comm_cart, &new_rank);

    //     // int b[info.ndims];
    //     // MPI_Cart_coords(comm_cart, new_rank, info.ndims, b);

    //     // if (rank == 0)
    //     //     for (size_t j = 0; j < info.ndims; ++j) {
    //     //         print_i64_t_array("a", info.ndims, a);
    //     //         print_array("b", info.ndims, b);
    //     //         ERRCHK_MPI(a[j] == b[j]);
    //     //     }
    // }

    acDecompositionInfoDestroy(&info);
#else
    (void)global_dims; // Unused
    int dims[ndims], periods[ndims];
    iset(0, ndims, dims);
    iset(1, ndims, periods);
    MPI_Dims_create(nprocs, ndims, dims);

    if (rank == 0)
        print_array("Mapping", as_size_t(ndims), dims);

    MPI_Comm comm_cart;
    MPI_Cart_create(parent, ndims, dims, periods, 0, &comm_cart);
#endif
    return comm_cart;
}

static void
test_indexing(const MPI_Comm comm_cart)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (int i = 0; i < nprocs; ++i) {
        if (i == rank) {
            int new_rank, ndims;
            MPI_Comm_rank(comm_cart, &new_rank);
            MPI_Cartdim_get(comm_cart, &ndims);
            int coords[ndims];
            MPI_Cart_coords(comm_cart, new_rank, ndims, coords);
            printf("Hello from %d. New rank %d. ", rank, new_rank);
            print_array("Mapping", as_size_t(ndims), coords);

            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

static void
get_local_dims(const size_t ndims, const size_t nn[], const MPI_Comm comm_cart, size_t local_nn[])
{
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get(comm_cart, as_int(ndims), dims, periods, coords);

    size_t decomp[ndims];
    as_size_t_array(ndims, dims, decomp);

    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = nn[i] / decomp[i];

    ERRCHK_MPI(prod(ndims, local_nn) * prod(ndims, decomp) == prod(ndims, nn));
}

int
comm_run(void)
{
    MPI_Init(NULL, NULL);

    const size_t global_nn[] = {8, 8};
    const size_t ndims       = ARRAY_SIZE(global_nn);
    MPI_Comm comm_cart       = create_rank_reordered_cart_comm(MPI_COMM_WORLD, ndims, global_nn);
    test_indexing(comm_cart);

    // Global original rank
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Allocate a buffer
    size_t local_nn[ndims];
    get_local_dims(ndims, global_nn, comm_cart, local_nn);

    const size_t r = 2;
    size_t local_mm[ndims];
    copy(ndims, local_nn, local_mm);
    add_to_array(2 * r, ndims, local_mm);

    // print_array("Local nn", ndims, local_nn);
    // print_array("Local mm", ndims, local_mm);
    size_t* buffer = (size_t*)malloc(sizeof(buffer[0]) * prod(ndims, local_mm));
    set(as_size_t(rank + 1), prod(ndims, local_mm), buffer);
    // for (size_t i = 0; i < prod(ndims, local_mm); ++i)
    //     buffer[i] = i;
    ERRCHK_MPI(buffer);

    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get(comm_cart, as_int(ndims), dims, periods, coords);

    // Do halo comm
    size_t domain[ndims], subdomain[ndims], offsets[ndims];
    copy(ndims, local_mm, domain);
    set(r, ndims, subdomain);
    set(0, ndims, offsets);
    subdomain[0] = local_nn[0]; // tmp debug hack

    reverse(ndims, domain);
    reverse(ndims, subdomain);
    reverse(ndims, offsets);

    int sizes[ndims], subsizes[ndims], starts[ndims];
    as_int_array(ndims, domain, sizes);
    as_int_array(ndims, subdomain, subsizes);
    as_int_array(ndims, offsets, starts);

    MPI_Datatype subarray_type;
    MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, starts, //
                             MPI_ORDER_C, MPI_UNSIGNED_LONG_LONG, &subarray_type);
    MPI_Type_commit(&subarray_type);

    int up, down, left, right;
    print("up", up);
    print("down", down);
    MPI_Cart_shift(comm_cart, 0, 1, &up, &down);
    MPI_Cart_shift(comm_cart, 1, 1, &left, &right);

    MPI_Sendrecv(&buffer[r + r * local_mm[0]], 1, subarray_type, down, 0,
                 &buffer[r + (r + local_nn[1]) * local_mm[0]], 1, subarray_type, up, 0, //
                 comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&buffer[r + local_nn[1] * local_mm[0]], 1, subarray_type, up, 1, //
                 &buffer[r], 1, subarray_type, down, 1,                           //
                 comm_cart, MPI_STATUS_IGNORE);

    MPI_Type_free(&subarray_type);

    // print_array("Subdomain", ndims, subdomain);

    MPI_Barrier(comm_cart);
    for (int coordy = 0; coordy < dims[ndims - 2]; ++coordy) {
        for (int coordx = 0; coordx < dims[ndims - 1]; ++coordx) {
            if (coordx == coords[ndims - 1] && coordy == coords[ndims - 2]) {
                print_array("Hello from", ndims, coords);
                for (size_t j = local_mm[1] - 1; j < local_mm[0]; --j) {
                    for (size_t i = 0; i < local_mm[0]; ++i) {
                        printf("%zu ", buffer[i + j * local_mm[0]]);
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(comm_cart);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    free(buffer);

    MPI_Comm_free(&comm_cart);
    MPI_Finalize();
    return SUCCESS;
}