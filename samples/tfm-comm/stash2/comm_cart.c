#include "comm_cart.h"

#include "math_utils.h"
#include "print.h"
#include "type_conversion.h"

MPI_Comm
create_rank_reordered_cart_comm(const MPI_Comm parent, const size_t ndims,
                                const size_t* global_dims)
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

    acDecompositionInfoDestroy(&info);
#else
    (void)global_dims; // Unused
    int dims[ndims], periods[ndims];
    iset(0, ndims, dims);
    iset(1, ndims, periods);
    MPI_Dims_create(nprocs, as_int(ndims), dims);

    MPI_Comm comm_cart;
    MPI_Cart_create(parent, as_int(ndims), dims, periods, 0, &comm_cart);
#endif
    return comm_cart;
}
