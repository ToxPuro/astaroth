#include "comm.h"
#include <stddef.h>
#include <stdlib.h>

#include <mpi.h>

#include "comm_cart.h"
#include "comm_data.h"
#include "errchk.h"
#include "math_utils.h"
#include "print.h"
#include "type_conversion.h"

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define ERRCHK_MPI(retval)                                                                         \
    ERRCHK(retval);                                                                                \
    if ((retval) == 0) {                                                                           \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

#define ERRCHK_MPI_API(errorcode)                                                                  \
    if ((errorcode) != MPI_SUCCESS) {                                                              \
        char description[MPI_MAX_ERROR_STRING];                                                    \
        int resultlen;                                                                             \
        MPI_Error_string(errorcode, description, &resultlen);                                      \
        ERRCHKK((errorcode) == MPI_SUCCESS, description);                                          \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

// Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)

typedef struct {
    size_t ndims;
    size_t* local_nn;
    size_t* global_nn;
    size_t* rr;
    // MPI_Comm comm_cart;
} State;

State
state_create(const size_t ndims, const size_t* local_nn, const size_t* global_nn, const size_t* rr)
{
    State state = (State){
        .ndims     = ndims,
        .local_nn  = malloc(sizeof(state.local_nn[0]) * ndims),
        .global_nn = malloc(sizeof(state.global_nn[0]) * ndims),
        .rr        = malloc(sizeof(state.rr[0]) * ndims),
        // .comm_cart = create_rank_reordered_cart_comm(MPI_COMM_WORLD, ndims, global_nn),
    };
    copy(ndims, local_nn, state.local_nn);
    copy(ndims, global_nn, state.global_nn);
    copy(ndims, rr, state.rr);
    return state;
}

void
state_destroy(State* state)
{
    // ERRCHK_MPI_API(MPI_Comm_free(&state->comm_cart));
    free(state->rr);
    free(state->global_nn);
    free(state->local_nn);
    state->ndims = 0;
}

static State global_state;

void
to_mpi_format(const size_t ndims, const size_t* dims, int* mpi_dims)
{
    as_int_array(ndims, dims, mpi_dims);
    reversei(ndims, mpi_dims);
}

void
to_astaroth_format(const size_t ndims, const int* mpi_dims, size_t* dims)
{
    as_size_t_array(ndims, mpi_dims, dims);
    reverse(ndims, dims);
}

static void
get_local_dims(const size_t ndims, const size_t* global_nn, const MPI_Comm comm_cart,
               size_t* local_nn)
{
    int mpi_dims[ndims], mpi_periods[ndims], mpi_coords[ndims];
    MPI_Cart_get(comm_cart, as_int(ndims), mpi_dims, mpi_periods, mpi_coords);

    size_t dims[ndims];
    to_astaroth_format(ndims, mpi_dims, dims);

    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];
}

static void
dims_create(const int nprocs, const size_t ndims, size_t* dims, size_t* periods)
{
    int mpi_dims[ndims], mpi_periods[ndims];
    iset(0, ndims, mpi_dims);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    to_astaroth_format(ndims, mpi_dims, dims);
    to_astaroth_format(ndims, mpi_periods, periods);
}

static void
get_local_nn(const size_t ndims, const size_t* global_nn, const size_t* dims, size_t* local_nn)
{
    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];
}

int
acCommInit(const size_t ndims, const size_t* global_nn, const size_t* rr)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    // int nprocs;
    // ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // size_t dims[ndims], periods[ndims];
    // dims_create(nprocs, ndims, dims, periods);
    // // print_array("dims", ndims, dims);

    // size_t local_nn[ndims];
    // get_local_nn(ndims, global_nn, dims, local_nn);
    // // print_array("local_nn", ndims, local_nn);

    // int mpi_dims[ndims], mpi_periods[ndims];
    // to_mpi_format(ndims, dims, mpi_dims);
    // to_mpi_format(ndims, periods, mpi_periods);

    // MPI_Comm comm_cart;
    // ERRCHK_MPI_API(
    //     MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &comm_cart));

    // global_state = state_create(ndims, local_nn, global_nn, rr);

    return SUCCESS;
}

int
acCommHaloExchange(const size_t ndims, const size_t* nn, const size_t* rr, const size_t nfields)
{
    // Basic info about the processes
    int nprocs;
    MPI_Comm parent = MPI_COMM_WORLD;
    ERRCHK_MPI_API(MPI_Comm_size(parent, &nprocs));

    // Decompose
    int mpi_dims[ndims], mpi_periods[ndims];
    iset(0, ndims, mpi_dims);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    // Create communicator
    MPI_Comm comm_cart;
    ERRCHK_MPI_API(MPI_Cart_create(parent, as_int(ndims), mpi_dims, mpi_periods, 0, &comm_cart));

    // Check position
    // int rank;
    // ERRCHK_MPI_API(MPI_Comm_rank(comm_cart, &rank));

    // int mpi_coords[ndims];
    // ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));

    // size_t coords[ndims];
    // to_astaroth_format(ndims, mpi_coords, coords);
    // if (rank == 0) {
    //     print_array("Coords", ndims, coords);
    //     printf("Left neighbor coords\n"),
    //     --mpi_coords[ndims-1];
    //     ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords, rank));
    //     print
    // }

    CommData comm_data = acCommDataCreate(ndims, nn, rr, nfields);
    // acCommDataPrint("comm_data", comm_data);
    acCommDataDestroy(&comm_data);

    // Release resources
    ERRCHK_MPI_API(MPI_Comm_free(&comm_cart));
    return SUCCESS;
}

int
acCommQuit(void)
{
    // state_destroy(&global_state);
    ERRCHK_MPI_API(MPI_Finalize());
    return SUCCESS;
}
