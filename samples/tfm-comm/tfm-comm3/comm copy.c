#include "comm.h"

#include "dynarr.h"
#include "errchk.h"
#include "math_utils.h"
#include "ntuple.h"
#include "partition.h"
#include "segment.h"

#include <stdio.h>

#include <mpi.h>

#include "errchk_mpi.h"
#include "error.h"
#include "misc.h"
#include "mpi_utils.h"
#include "print.h"

/*
 * Comm
 */
typedef struct {
    int mpi_nprocs;
    int mpi_ndims;
    int* mpi_decomposition;

    MPI_Comm mpi_comm;
    MPI_Datatype mpi_dtype;
    int mpi_rank;

    int* mpi_global_nn;
    int* mpi_global_nn_offset;
    int* mpi_local_nn;
    int* mpi_local_nn_offset;

    size_t ndims;
    uint64_t* global_nn;
    uint64_t* global_nn_offset;
    uint64_t* local_nn;
    uint64_t* local_nn_offset;
    uint64_t* decomposition;
} CommCtx;

static CommCtx
commctx_create(const size_t ndims)
{
    CommCtx ctx = {
        .mpi_nprocs = 0,
        .mpi_comm   = MPI_COMM_NULL,
        .mpi_dtype  = MPI_DOUBLE,
        .mpi_rank   = MPI_PROC_NULL,

        .mpi_ndims            = as_int(ndims),
        .mpi_decomposition    = ac_calloc(ndims, sizeof(ctx.mpi_decomposition[0])),
        .mpi_global_nn        = ac_calloc(ndims, sizeof(ctx.mpi_global_nn[0])),
        .mpi_global_nn_offset = ac_calloc(ndims, sizeof(ctx.mpi_global_nn_offset[0])),
        .mpi_local_nn         = ac_calloc(ndims, sizeof(ctx.mpi_local_nn[0])),
        .mpi_local_nn_offset  = ac_calloc(ndims, sizeof(ctx.mpi_local_nn_offset[0])),

        .ndims            = ndims,
        .decomposition    = ac_calloc(ndims, sizeof(ctx.decomposition[0])),
        .global_nn        = ac_calloc(ndims, sizeof(ctx.global_nn[0])),
        .global_nn_offset = ac_calloc(ndims, sizeof(ctx.global_nn_offset[0])),
        .local_nn         = ac_calloc(ndims, sizeof(ctx.local_nn[0])),
        .local_nn_offset  = ac_calloc(ndims, sizeof(ctx.local_nn_offset[0])),
    };
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &ctx.nprocs));
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), ctx.mpi_dims));
    return ctx;
}

static void
commctx_destroy(CommCtx* ctx)
{
    ac_free(ctx->local_nn_offset);
    ac_free(ctx->local_nn);
    ac_free(ctx->global_nn_offset);
    ac_free(ctx->global_nn);
    ac_free(ctx->decomposition);
    ctx->ndims = 0;

    ac_free(ctx->mpi_local_nn_offset);
    ac_free(ctx->mpi_local_nn);
    ac_free(ctx->mpi_global_nn_offset);
    ac_free(ctx->mpi_global_nn);
    ac_free(ctx->mpi_decomposition);
    ctx->mpi_ndims = 0;

    ctx->mpi_nprocs = 0;
    ctx->mpi_rank   = MPI_PROC_NULL;
    ctx->mpi_dtype  = MPI_DATATYPE_NULL;
    ctx->mpi_comm   = MPI_COMM_NULL;
}

static CommCtx ctx = {0};

void
acCommInit(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
}

void
acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
            uint64_t* global_nn_offset)
{
    ErrorCode retval = ERRORCODE_SUCCESS;

    // Create the communication context
    ctx = commctx_create(ndims);
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &ctx.nprocs));

    // Decompose
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), ctx.mpi_dims));

    // Create the primary communicator
    set_array_int(1, ndims, mpi_periods);
    ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &ctx.mpi_comm));

    // Set errors as non-fatal
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(ctx.mpi_comm, MPI_ERRORS_RETURN));

    // Setup the rest of the global variables
    mpi_ndims_ = as_int(ndims);

    // Compute local_nn
    to_astaroth_format(ndims, mpi_dims, dims);
    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];

    // Compute global_nn_offset
    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(ctx.mpi_comm, &rank));

    ERRCHK_MPI_API(MPI_Cart_coords(ctx.mpi_comm, rank, mpi_ndims_, mpi_coords));

    to_astaroth_format(ndims, mpi_coords, coords);
    for (size_t i = 0; i < ndims; ++i)
        global_nn_offset[i] = local_nn[i] * coords[i];

    return retval;
}

void
acCommQuit(void)
{
    if (ctx.mpi_comm != MPI_COMM_NULL)
        ERRCHK_MPI_API(MPI_Comm_free(&ctx.mpi_comm));
    commctx_destroy(&ctx);
    ERRCHK_MPI_API(MPI_Finalize());
}

void
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(ctx.mpi_comm, rank));
    ERRCHK_MPI_API(MPI_Comm_size(ctx.mpi_comm, nprocs));
}

void
print_comm(void)
{
    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(ctx.mpi_comm, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(ctx.mpi_comm, &nprocs));

    int* mpi_dims    = ac_calloc(as_size_t(mpi_ndims_), sizeof(mpi_dims[0]));
    int* mpi_periods = ac_calloc(as_size_t(mpi_ndims_), sizeof(mpi_periods[0]));
    int* mpi_coords  = ac_calloc(as_size_t(mpi_ndims_), sizeof(mpi_coords[0]));
    ERRCHK_MPI_API(MPI_Cart_get(ctx.mpi_comm, mpi_ndims_, mpi_dims, mpi_periods, mpi_coords));

    for (int i = 0; i < nprocs; ++i) {
        acCommBarrier();
        if (rank == i) {
            printf("Rank %d of %d:\n", rank, nprocs);
            print_array("\t mpi_dims", as_uint64_t(mpi_ndims_), mpi_dims);
            print_array("\t mpi_periods", as_uint64_t(mpi_ndims_), mpi_periods);
            print_array("\t mpi_coords", as_uint64_t(mpi_ndims_), mpi_coords);
        }
        acCommBarrier();
    }
    ac_free(mpi_dims);
    ac_free(mpi_periods);
    ac_free(mpi_coords);
}

void
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(ctx.mpi_comm));
}

int
acCommTest(void)
{
    int retval = 0;
    retval |= test_get_errorcode_description();
    retval |= test_ntuple();
    retval |= test_alloc();

    // TODO: return errcounts from test_math_utils
    test_math_utils();
    test_segment();
    test_dynarr();
    test_partition();
    // test_mpi_utils();
    // test_pack();

    return retval;
}
