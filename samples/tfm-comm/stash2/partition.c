#include "partition.h"

#include "dynamic_array.h"
#include "errchk.h"
#include "math_utils.h"
#include "ndarray.h"
#include "print.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// #include "tgarray.h"

size_t
volume(const size_t ndims, const size_t* mmin, const size_t* mmax)
{
    size_t dims[ndims];
    subtract_arrays(ndims, mmax, mmin, dims);
    return prod(ndims, dims);
}

size_t
partition_recursive(const size_t ndims, const size_t* mmin, const size_t* nmin, const size_t* nmax,
                    const size_t* mmax, const size_t axis, DynamicArray* offsets,
                    DynamicArray* dimensions)
{
    if (axis >= ndims) {
        static size_t counter = 0;
        print("Counter", counter++);
        print_array("mmin", ndims, mmin);
        print_array("mmax", ndims, mmax);

        size_t dims[ndims];
        subtract_arrays(ndims, mmax, mmin, dims);
        print_array("dims", ndims, dims);
        printf("\n");

        if (offsets != NULL)
            array_append_multiple(ndims, mmin, offsets);
        if (dimensions != NULL)
            array_append_multiple(ndims, dims, dimensions);
        return 1;
    }
    else {
        size_t npartitions = 0;
        {
            size_t new_mmin[ndims];
            copy(ndims, mmin, new_mmin);

            size_t new_mmax[ndims];
            copy(ndims, mmax, new_mmax);

            new_mmin[axis] = mmin[axis];
            new_mmax[axis] = nmin[axis];
            if (volume(ndims, new_mmin, new_mmax) > 0)
                npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                                   offsets, dimensions);
        }
        {
            size_t new_mmin[ndims];
            copy(ndims, mmin, new_mmin);

            size_t new_mmax[ndims];
            copy(ndims, mmax, new_mmax);

            new_mmin[axis] = nmin[axis];
            new_mmax[axis] = nmax[axis];
            if (volume(ndims, new_mmin, new_mmax) > 0)
                npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                                   offsets, dimensions);
        }
        {
            size_t new_mmin[ndims];
            copy(ndims, mmin, new_mmin);

            size_t new_mmax[ndims];
            copy(ndims, mmax, new_mmax);

            new_mmin[axis] = nmax[axis];
            new_mmax[axis] = mmax[axis];
            if (volume(ndims, new_mmin, new_mmax) > 0)
                npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                                   offsets, dimensions);
        }
        return npartitions;
    }
}

/** Partitions the domain mm  */
void
partition(const size_t ndims, const size_t* mm, const size_t* nn_offset, const size_t* nn)
{
    size_t mmin[ndims], nmin[ndims], nmax[ndims], mmax[ndims];
    for (size_t i = 0; i < ndims; ++i) {
        mmin[i] = 0;
        nmin[i] = nn_offset[i];
        nmax[i] = nn_offset[i] + nn[i];
        mmax[i] = mm[i];
    }

    for (size_t i = 0; i < ndims; ++i) {
        ERRCHK(mmin[i] <= nmin[i]);
        ERRCHK(nmin[i] < nmax[i]);
        ERRCHK(nmax[i] <= mmax[i]);
    }

    const size_t npartitions = partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, NULL, NULL);
    DynamicArray dyn_offsets = array_create(0 * npartitions);
    DynamicArray dyn_dims    = array_create(ndims * npartitions);
    partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, &dyn_offsets, &dyn_dims);

    print_ndarray("offsets", 2, (size_t[]){ndims, npartitions}, dyn_offsets.data);
    print_ndarray("dims", 2, (size_t[]){ndims, npartitions}, dyn_dims.data);

    array_destroy(&dyn_offsets);
    array_destroy(&dyn_dims);
    print("Npartitions", npartitions);
}

static void
iterate_twod_array(const size_t nrows, const size_t ncols, const size_t arr[nrows][ncols])
{
    for (size_t j = 0; j < nrows; ++j) {
        for (size_t i = 0; i < ncols; ++i) {
            printf("%zu, %zu: %zu\n", arr[j][i]);
        }
    }
}

int
partition_test(void)
{
    const size_t mm[]        = {10, 10};
    const size_t nn[]        = {8, 8};
    const size_t nn_offset[] = {1, 1};
    const size_t ndims       = ARRAY_SIZE(mm);
    partition(ndims, mm, nn_offset, nn);

    return 0;
}

// void
// partition_test(const size_t ndims, const size_t* nn, const size_t* rr,
//                        const HaloSegmentBatch batch)
// {
//     // Ensure that
//     // 1. none of the segments overlap
//     // 2. the number of points covered by the segments encompass the whole
//     //    ghost zone
//     // 3. all of the segments are within mm

//     { // Local packets
//         size_t mm[ndims];
//         get_mm(ndims, nn, rr, mm);
//         const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

//         size_t count = 0;
//         for (size_t i = 0; i < batch.npackets; ++i) {
//             const HaloSegment a = batch.local_packets[i];
//             ERRCHK(ndims == a.ndims);
//             for (size_t j = i + 1; j < batch.npackets; ++j) {
//                 const HaloSegment b = batch.local_packets[j];

//                 ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
//             }
//             count += prod(ndims, a.dims);

//             for (size_t j = 0; j < ndims; ++j)
//                 ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
//         }
//         ERRCHK(count == model_count);
//     }
//     { // Remote packets
//         size_t mm[ndims];
//         get_mm(ndims, nn, rr, mm);
//         const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

//         size_t count = 0;
//         for (size_t i = 0; i < batch.npackets; ++i) {
//             const HaloSegment a = batch.remote_packets[i];
//             ERRCHK(ndims == a.ndims);
//             for (size_t j = i + 1; j < batch.npackets; ++j) {
//                 const HaloSegment b = batch.remote_packets[j];

//                 ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
//             }
//             count += prod(ndims, a.dims);

//             for (size_t j = 0; j < ndims; ++j)
//                 ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
//         }
//         ERRCHK(count == model_count);
//     }
// }
