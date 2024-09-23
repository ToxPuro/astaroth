#include "comm.h"
#include <stddef.h>
#include <stdlib.h>

#include <mpi.h>

#include "comm_cart.h"
#include "comm_data.h"
#include "errchk.h"
#include "math_utils.h"
#include "ndarray.h"
#include "print.h"
#include "type_conversion.h"

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define ERRCHK_MPI(retval)                                                                         \
    {                                                                                              \
        ERRCHK(retval);                                                                            \
        if ((retval) == 0) {                                                                       \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
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

// static State global_state;

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

// static void
// get_local_dims(const size_t ndims, const size_t* global_nn, const MPI_Comm comm_cart,
//                size_t* local_nn)
// {
//     int mpi_dims[ndims], mpi_periods[ndims], mpi_coords[ndims];
//     MPI_Cart_get(comm_cart, as_int(ndims), mpi_dims, mpi_periods, mpi_coords);

//     size_t dims[ndims];
//     to_astaroth_format(ndims, mpi_dims, dims);

//     for (size_t i = 0; i < ndims; ++i)
//         local_nn[i] = global_nn[i] / dims[i];
// }

int
acCommInit(const size_t ndims, const size_t* global_nn, const size_t* rr)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    (void)ndims;
    (void)global_nn;
    (void)rr;
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

/////////////////////

static void
dims_create(const int nprocs, const size_t ndims, size_t* dims, size_t* periods)
{
    int mpi_dims[ndims], mpi_periods[ndims];
    iset(0, ndims, mpi_dims);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    to_astaroth_format(ndims, mpi_dims, dims);
    to_astaroth_format(ndims, mpi_periods, periods);
    ERRCHK_MPI(prod(ndims, dims) == as_size_t(nprocs));
}

static void
get_local_nn(const size_t ndims, const size_t* global_nn, const size_t* dims, size_t* local_nn)
{
    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];
}

static void
get_mm(const size_t ndims, const size_t* nn, const size_t* rr, size_t* mm)
{
    for (size_t i = 0; i < ndims; ++i)
        mm[i] = 2 * rr[i] + nn[i];
}

#include <limits.h>

static int
get_tag(const size_t packet, const size_t npackets, const size_t launch)
{
    ERRCHK_MPI(packet < npackets);
    const size_t tag = (packet + launch * npackets) % (as_size_t(INT_MAX) + as_size_t(1));
    return as_int(tag);
}

static void
test_get_tag(void)
{
    // Criteria:
    // Given npackets and nlaunches
    //  1. At any given starting index, the (npackets*nlaunches) subsequent tags should be unique
    // const size_t npackets = 7;
    // for (size_t i = 0; i < 128; ++i) {
    //     for (size_t j = 0; j < npackets; ++j) {
    //         // const size_t launch = INT_MAX - 50 + i;
    //         const size_t launch = SIZE_MAX - 50 + i;
    //         const size_t packet = j;
    //         printf("%zu, %zu -> %d\n", launch, packet, get_tag(packet, npackets, launch));
    //     }
    // }
    // ERRCHK_MPI(get_tag(0, 1, get_tag(0, 1, INT_MAX + 1)) == 0);
    // ERRCHK_MPI(get_tag(0, 1, get_tag(0, 1, INT_MAX)) == INT_MAX);
    // ERRCHK_MPI(get_tag(0, 1, get_tag(0, 1, INT_MAX - 1)) == INT_MAX - 1);
    // ERRCHK_MPI(get_tag(0, 1, as_size_t(INT_MAX) + as_size_t(1)) == 0);
    ERRCHK_MPI(get_tag(0, 1, SIZE_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 1, SIZE_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(6, 7, SIZE_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 7, SIZE_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(20, 21, SIZE_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 21, SIZE_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(20, 21, SIZE_MAX + (size_t)1) == 20);
    ERRCHK_MPI(get_tag(0, 1, INT_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 1, INT_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(6, 7, INT_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 7, INT_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(20, 21, INT_MAX) == INT_MAX);
    ERRCHK_MPI(get_tag(0, 21, INT_MAX + (size_t)1) == 0);
    ERRCHK_MPI(get_tag(20, 21, INT_MAX + (size_t)1) == 20);
}

static void
get_mpi_coords_neighbor(const size_t ndims, const size_t* nn, const size_t* rr,
                        const size_t* offset, const int* mpi_coords, int* mpi_coords_neighbor)
{
    for (size_t i = 0; i < ndims; ++i)
        mpi_coords_neighbor[i] = offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    reversei(ndims, mpi_coords_neighbor);
    for (size_t i = 0; i < ndims; ++i)
        mpi_coords_neighbor[i] += mpi_coords[i];
}

static void
get_mpi_coords_neighbor_inv(const size_t ndims, const size_t* nn, const size_t* rr,
                            const size_t* offset, const int* mpi_coords, int* mpi_coords_neighbor)
{
    for (size_t i = 0; i < ndims; ++i)
        mpi_coords_neighbor[i] = offset[i] < rr[i] ? 1 : offset[i] >= rr[i] + nn[i] ? -1 : 0;
    reversei(ndims, mpi_coords_neighbor);
    for (size_t i = 0; i < ndims; ++i)
        mpi_coords_neighbor[i] += mpi_coords[i];
}

// static void haloExchange(const MPI_Comm comm_cart, CommData comm_data)

//     size_t launch_counter = 0;

// const size_t ndims = comm_data.local_packets[0].ndims;
// for (size_t i = 0; i < comm_data.npackets; ++i) {
//     int sizes[ndims], subsizes[ndims], starts[ndims];
//     to_mpi_format(ndims, local_mm, sizes);
//     to_mpi_format(ndims, comm_data.local_packets[i].dims, subsizes);
//     to_mpi_format(ndims, comm_data.local_packets[i].offset, starts);
// }
// ++launch_counter;
// }

static void
haloExchange(const MPI_Comm comm_cart, const size_t ndims, const size_t* local_nn,
             const size_t* local_mm, const size_t* rr, CommData comm_data, size_t* buffer)
{
    static size_t launch_counter = 0;

    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(comm_cart, &rank));

    MPI_Request reqs[comm_data.npackets];
    for (size_t i = 0; i < comm_data.npackets; ++i) {
        int sizes[ndims], subsizes[ndims], send_starts[ndims], recv_starts[ndims];
        to_mpi_format(ndims, local_mm, sizes);
        to_mpi_format(ndims, comm_data.local_packets[i].dims, subsizes);
        to_mpi_format(ndims, comm_data.local_packets[i].offset, recv_starts);

        for (size_t j = 0; j < ndims; ++j)
            send_starts[j] = as_int(mod(as_int64_t(recv_starts[j]) - as_int64_t(rr[ndims - 1 - j]),
                                        as_int64_t(local_nn[ndims - 1 - j])) +
                                    as_int64_t(rr[ndims - 1 - j]));

        MPI_Datatype send_subarray, recv_subarray;
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, send_starts,
                                                MPI_ORDER_C, MPI_UNSIGNED_LONG_LONG,
                                                &send_subarray));
        ERRCHK_MPI_API(MPI_Type_commit(&send_subarray));
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, recv_starts,
                                                MPI_ORDER_C, MPI_UNSIGNED_LONG_LONG,
                                                &recv_subarray));
        ERRCHK_MPI_API(MPI_Type_commit(&recv_subarray));

        // Get tag
        const int tag = get_tag(i, comm_data.npackets, launch_counter);

        // Get neighbors
        int mpi_coords[ndims];
        ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));

        const size_t* offset = comm_data.local_packets[i].offset;
        int mpi_coords_send_neighbor[ndims];
        get_mpi_coords_neighbor(ndims, local_nn, rr, offset, mpi_coords, mpi_coords_send_neighbor);

        int send_neighbor;
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_send_neighbor, &send_neighbor));

        int mpi_coords_recv_neighbor[ndims];
        get_mpi_coords_neighbor_inv(ndims, local_nn, rr, offset, mpi_coords,
                                    mpi_coords_recv_neighbor);

        int recv_neighbor;
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_recv_neighbor, &recv_neighbor));

        ERRCHK_MPI_API(MPI_Isendrecv(buffer, 1, send_subarray, send_neighbor, tag, buffer, 1,
                                     recv_subarray, recv_neighbor, tag, comm_cart, &reqs[i]));

        ERRCHK_MPI_API(MPI_Type_free(&send_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&recv_subarray));
    }
    ERRCHK_MPI_API(MPI_Waitall(as_int(comm_data.npackets), reqs, MPI_STATUSES_IGNORE));

    ++launch_counter;
}

int
acCommTest(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    // Get nprocs
    int nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // Global grid
    const size_t global_nn[] = {4, 4, 4};
    const size_t ndims       = ARRAY_SIZE(global_nn);
    const size_t rr[]        = {1, 1, 1, 1};
    const size_t fields[]    = {1};
    const size_t nfields     = ARRAY_SIZE(fields);

    // Decompose
    size_t dims[ndims], periods[ndims];
    dims_create(nprocs, ndims, dims, periods);

    // Create communicator
    int mpi_dims[ndims], mpi_periods[ndims];
    to_mpi_format(ndims, dims, mpi_dims);
    to_mpi_format(ndims, periods, mpi_periods);
    MPI_Comm comm_cart;
    ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &comm_cart));

    // Get rank
    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(comm_cart, &rank));

    // Get local dims
    size_t local_nn[ndims];
    get_local_nn(ndims, global_nn, dims, local_nn);
    for (size_t i = 0; i < ndims; ++i)
        ERRCHK_MPI(local_nn[i] >= rr[i]);

    size_t local_mm[ndims];
    get_mm(ndims, local_nn, rr, local_mm);

    // Reserve resources
    size_t* buffer = malloc(sizeof(buffer[0]) * prod(ndims, local_mm));
    set(as_size_t(rank + 1), prod(ndims, local_mm), buffer);
    CommData comm_data = acCommDataCreate(ndims, local_nn, rr, nfields);

    // Print data
    if (rank == 0) {
        print_array("global_nn", ndims, global_nn);
        print_array("rr", ndims, rr);
        print_array("dims", ndims, dims);
        print_array("local_nn", ndims, local_nn);
        print_array("local_mm", ndims, local_mm);
        print_ndarray("Mesh", ndims, local_mm, buffer);
        // acCommDataPrint("comm_data", comm_data);
    }

// Send packets
#if 0
    for (int i = 0; i < nprocs; ++i) {
        MPI_Barrier(comm_cart);
        if (rank == i) {
            print("Rank", i);
            int mpi_coords[ndims];
            ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));
            print_array("\tCoords", ndims, mpi_coords);

            for (size_t j = 0; j < comm_data.npackets; ++j) {
                const size_t* offset = comm_data.local_packets[j].offset;
                const size_t* dims   = comm_data.local_packets[j].dims;
                print("\t\tPacket", j);
                print_array("\t\tDims", ndims, dims);
                print_array("\t\t\tOffset", ndims, offset);

                int mpi_coords_neighbor[ndims];
                get_mpi_coords_neighbor(ndims, local_nn, rr, offset, mpi_coords,
                                        mpi_coords_neighbor);
                print_array("\t\t\tMPI coordinate offset", ndims, mpi_coords_neighbor);
                // if (any)
                int neighbor;
                ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_neighbor, &neighbor));
                print("\t\t\tNeighbor rank", neighbor);
            }

            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(comm_cart);
    }
#endif

    haloExchange(comm_cart, ndims, local_nn, local_mm, rr, comm_data, buffer);
    haloExchange(comm_cart, ndims, local_nn, local_mm, rr, comm_data, buffer);
    haloExchange(comm_cart, ndims, local_nn, local_mm, rr, comm_data, buffer);
    haloExchange(comm_cart, ndims, local_nn, local_mm, rr, comm_data, buffer);
#if 0
    MPI_Request send_reqs[comm_data.npackets];
    MPI_Request recv_reqs[comm_data.npackets];
    size_t launch_counter = 0;
    for (size_t i = 0; i < comm_data.npackets; ++i) {
        int sizes[ndims], subsizes[ndims], starts[ndims];
        to_mpi_format(ndims, local_mm, sizes);
        to_mpi_format(ndims, comm_data.local_packets[i].dims, subsizes);
        to_mpi_format(ndims, comm_data.local_packets[i].offset, starts);

        // Subarrays
        MPI_Datatype recv_subarray;
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, starts, MPI_ORDER_C,
                                                MPI_UNSIGNED_LONG_LONG, &recv_subarray));
        ERRCHK_MPI_API(MPI_Type_commit(&recv_subarray));

        // Get tag
        const int tag = get_tag(i, comm_data.npackets, launch_counter);

        // Get source
        int mpi_coords[ndims];
        ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));

        const size_t* offset = comm_data.local_packets[i].offset;
        int mpi_coords_neighbor[ndims];
        get_mpi_coords_neighbor(ndims, local_nn, rr, offset, mpi_coords, mpi_coords_neighbor);

        int neighbor;
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_neighbor, &neighbor));

        ERRCHK_MPI_API(
            MPI_Irecv(buffer, 1, recv_subarray, neighbor, tag, comm_cart, &recv_reqs[i]));
        // MPI_Sendrecv(buffer, 1, recv_subarray, neighbor, tag, &buffer[0], 1, recv_subarray,
        //              neighbor, tag, comm_cart, MPI_STATUS_IGNORE);

        // Get source
        // int mpi_coords[ndims];
        // ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));
        // // for (size_t i = 0; i < ndims; ++i)
        // //     mpi_coords[ndims] += starts[i] < rr[i] ? -1 : starts[i] > rr[i] ? 1 : 0;
        // int source;
        // // ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords, &source));
        // // if (rank == 0) {
        // print_array("coords", ndims, mpi_coords);
        // // }

        // ERRCHK_MPI_API(MPI_Irecv(buffer, 1, subarray, source, tag, comm_cart,
        // MPI_STATUS_IGNORE)); MPI_Sendrecv(&buffer[0], 1, subarray, neighbor, tag, comm_cart,
        // MPI_STATUS_IGNORE);
        ERRCHK_MPI_API(MPI_Type_free(&recv_subarray));
    }
    for (size_t i = 0; i < comm_data.npackets; ++i) {
        int sizes[ndims], subsizes[ndims], starts[ndims];
        to_mpi_format(ndims, local_mm, sizes);
        to_mpi_format(ndims, comm_data.local_packets[i].dims, subsizes);
        to_mpi_format(ndims, comm_data.local_packets[i].offset, starts);

        for (size_t j = 0; j < ndims; ++j)
            starts[j] = as_int(mod(as_int64_t(starts[j]) - as_int64_t(rr[ndims - 1 - j]),
                                   as_int64_t(local_nn[ndims - 1 - j])) +
                               as_int64_t(rr[ndims - 1 - j]));

        // Subarrays
        MPI_Datatype send_subarray;
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, starts, MPI_ORDER_C,
                                                MPI_UNSIGNED_LONG_LONG, &send_subarray));
        ERRCHK_MPI_API(MPI_Type_commit(&send_subarray));

        // Get tag
        const int tag = get_tag(i, comm_data.npackets, launch_counter);

        // Get source
        int mpi_coords[ndims];
        ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));

        const size_t* offset = comm_data.local_packets[i].offset;
        int mpi_coords_neighbor[ndims];
        get_mpi_coords_neighbor_inv(ndims, local_nn, rr, offset, mpi_coords, mpi_coords_neighbor);

        int neighbor;
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_neighbor, &neighbor));

        ERRCHK_MPI_API(
            MPI_Isend(buffer, 1, send_subarray, neighbor, tag, comm_cart, &send_reqs[i]));

        ERRCHK_MPI_API(MPI_Wait(&send_reqs[i], MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Wait(&recv_reqs[i], MPI_STATUS_IGNORE));
        // MPI_Waitall(comm_data.npackets, send_reqs, MPI_STATUSES_IGNORE);
        // MPI_Waitall(comm_data.npackets, recv_reqs, MPI_STATUSES_IGNORE);
        ERRCHK_MPI_API(MPI_Type_free(&send_subarray));
    }
    ++launch_counter;
#endif

    // Print data
    if (rank == 0) {
        print_array("global_nn", ndims, global_nn);
        print_array("rr", ndims, rr);
        print_array("dims", ndims, dims);
        print_array("local_nn", ndims, local_nn);
        print_array("local_mm", ndims, local_mm);
        print_ndarray("Mesh", ndims, local_mm, buffer);
        // acCommDataPrint("comm_data", comm_data);
    }

    test_get_tag();

    // Release resources
    acCommDataDestroy(&comm_data);
    free(buffer);
    ERRCHK_MPI_API(MPI_Comm_free(&comm_cart));

    ERRCHK_MPI_API(MPI_Finalize());
    return 0;
}
