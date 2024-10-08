#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"
#include "misc.h"
#include "nalloc.h"
#include "print.h"
#include "type_conversion.h"

// Debug
#include "partition.h"

#define nalloc_ones(count, arr)                                                                    \
    do {                                                                                           \
        nalloc(count, arr);                                                                        \
        for (size_t __ones_counter__ = 0; __ones_counter__ < count; ++__ones_counter__)            \
            arr[__ones_counter__] = 1;                                                             \
    } while (0)

#define nalloc_arange(min, count, arr)                                                             \
    do {                                                                                           \
        nalloc(count, arr);                                                                        \
        for (size_t __arange_counter__ = 0; __arange_counter__ < count; ++__arange_counter__)      \
            arr[__arange_counter__] = __arange_counter__ + min;                                    \
    } while (0)

// static void
// get_mm(const size_t ndims, const size_t* nn, const size_t* rr, size_t* mm)
// {
//     for (size_t i = 0; i < ndims; ++i)
//         mm[i] = nn[i] + 2 * rr[i];
// }

int
main(void)
{
    // Initialize MPI
    acCommInit();

    // Setup the communications module
    const size_t global_nn[] = {4, 6};
    const size_t rr[]        = {1, 2, 1, 1, 1};
    const size_t ndims       = ARRAY_SIZE(global_nn);

    size_t *local_nn, *global_nn_offset;
    nalloc(ndims, local_nn);
    nalloc(ndims, global_nn_offset);
    acCommSetup(ndims, global_nn, local_nn, global_nn_offset);

    // Get communicator info
    int rank, nprocs;
    acCommGetProcInfo(&rank, &nprocs);

    // Setup the local buffers
    size_t* local_mm;
    nalloc(ndims, local_mm);
    for (size_t i = 0; i < ndims; ++i)
        local_mm[i] = local_nn[i] + 2 * rr[i];
    const size_t buflen = prod(ndims, local_mm);

    double *buf0, *buf1, *buf2;
    nalloc(buflen, buf0);
    nalloc(buflen, buf1);
    nalloc(buflen, buf2);

    // Setup and launch the halo exchange
    // double* buffers[] = {buf0, buf1, buf2};
    double* buffers[]     = {buf0};
    const size_t nbuffers = ARRAY_SIZE(buffers);

    // Debug: partition the buffer
    SegmentArray segments;
    dynarr_create_with_destructor(segment_destroy, &segments);
    partition(ndims, local_mm, local_nn, rr, &segments);
    for (size_t j = 0; j < nbuffers; ++j) {
        // for (size_t i = 0; i < buflen; ++i)
        //     buffers[j][i] = as_double(i + j * buflen + rank * buflen * nbuffers);
        for (size_t i = 0; i < segments.length; ++i)
            set_ndarray_double(i + 1 + rank * segments.length, ndims, local_mm,
                               segments.data[i].dims, segments.data[i].offset, buffers[j]);
    }
    dynarr_destroy(&segments);

    // for (int i = 0; i < nprocs; ++i) {
    //     acCommBarrier();
    //     fflush(stdout);
    //     if (rank == i) {
    //         printf("Proc %d-------------\n", i);
    //         for (size_t j = 0; j < nbuffers; ++j)
    //             printd_ndarray(ndims, local_mm, buffers[j]);
    //     }
    //     fflush(stdout);
    //     acCommBarrier();
    // }

    HaloSegmentBatch batch = halo_segment_batch_create(ndims, local_mm, local_nn, rr, nbuffers);
    halo_segment_batch_launch(nbuffers, buffers, batch);
    halo_segment_batch_wait(batch, nbuffers, buffers);
    halo_segment_batch_destroy(&batch);

    // Print the result
    for (int i = 0; i < nprocs; ++i) {
        acCommBarrier();
        fflush(stdout);
        if (rank == i) {
            printf("Proc %d-------------\n", i);
            for (size_t j = 0; j < nbuffers; ++j)
                printd_ndarray(ndims, local_mm, buffers[j]);
        }
        fflush(stdout);
        acCommBarrier();
    }

    // Cleanup
    ndealloc(local_mm);
    ndealloc(global_nn_offset);
    ndealloc(local_nn);
    acCommQuit();
    return EXIT_SUCCESS;
}

// static void
// get_mm(const size_t ndims, const size_t* nn, const size_t* rr, size_t* mm)
// {
//     for (size_t i = 0; i < ndims; ++i)
//         mm[i] = 2 * rr[i] + nn[i];
// }

// int
// main(void)
// {
//     acCommInit();

//     const size_t nn[]  = {3, 3};
//     const size_t rr[]  = {1, 1, 1, 1};
//     const size_t ndims = ARRAY_SIZE(nn);

//     size_t mm[ndims];
//     get_mm(ndims, nn, rr, mm);
//     const size_t count = prod(ndims, mm);

//     size_t buf0[count];
//     size_t buf1[count];

//     int rank, nprocs;
//     acCommGetProcInfo(&rank, &nprocs);

//     for (size_t i = 0; i < count; ++i) {
//         buf0[i] = i; // as_size_t(rank);
//         buf1[i] = 2 * i;
//     }
//     size_t* buffers[]     = {buf0, buf1};
//     const size_t nbuffers = ARRAY_SIZE(buffers);
//     print("nbuffers", nbuffers);
//     print_ndarray("Mesh", ndims, mm, buf0);

//     // HaloExchangeTask* task = acHaloExchangeTaskCreate(ndims, mm, nn, rr, nbuffers);
//     // acHaloExchangeTaskLaunch(task, nbuffers, buffers);
//     // acHaloExchangeTaskSynchronize(task);

//     for (int i = 0; i < nprocs; ++i) {
//         acCommBarrier();
//         if (i == rank)
//             print_ndarray("Mesh", ndims, mm, buf0);
//         acCommBarrier();
//     }
//     // acHaloExchangeTaskDestroy(&task);

//     acCommQuit();
//     return EXIT_SUCCESS;
// }
