#include "partition.h"

#include "dynarr.h"
#include "math_utils.h"
#include "misc.h"
#include "print.h"

typedef dynarr_s(size_t) DynamicArray;

static size_t
get_volume(const size_t ndims, const size_t* mmin, const size_t* mmax)
{
    size_t* dims;
    nalloc(ndims, dims);
    subtract_arrays(ndims, mmax, mmin, dims);
    const size_t result = prod(ndims, dims);
    ndealloc(dims);
    return result;
}

static size_t
partition_recursive(const size_t ndims, const size_t* mmin, const size_t* nmin, const size_t* nmax,
                    const size_t* mmax, const size_t axis, DynamicArray* segment_dims,
                    DynamicArray* segment_offsets)
{
    if (get_volume(ndims, mmin, mmax) == 0) {
        return 0;
    }
    else if (axis >= ndims) {
        if (segment_dims != NULL) {
            size_t* dims;
            nalloc(ndims, dims);
            subtract_arrays(ndims, mmax, mmin, dims);
            dynarr_append_multiple(ndims, dims, segment_dims);
            ndealloc(dims);
        }
        if (segment_offsets != NULL)
            dynarr_append_multiple(ndims, mmin, segment_offsets);

        return 1;
    }
    else {
        size_t npartitions = 0;
        size_t *new_mmin, *new_mmax;
        nalloc(ndims, new_mmin);
        nalloc(ndims, new_mmax);

        // Left
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = mmin[axis];
        new_mmax[axis] = nmin[axis];
        npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                           segment_dims, segment_offsets);

        // Center
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmin[axis];
        new_mmax[axis] = nmax[axis];
        npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                           segment_dims, segment_offsets);

        // Right
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmax[axis];
        new_mmax[axis] = mmax[axis];
        npartitions += partition_recursive(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1,
                                           segment_dims, segment_offsets);

        ndealloc(new_mmin);
        ndealloc(new_mmax);

        return npartitions;
    }
}

size_t
partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
          size_t* nelems, size_t* dims, size_t* offsets)
{
    size_t *mmin, *nmin, *nmax, *mmax;
    nalloc(ndims, mmin);
    nalloc(ndims, nmin);
    nalloc(ndims, nmax);
    nalloc(ndims, mmax);

    for (size_t i = 0; i < ndims; ++i) {
        mmin[i] = 0;
        nmin[i] = nn_offset[i];
        nmax[i] = nn_offset[i] + nn[i];
        mmax[i] = mm[i];

        ERRCHK(mmin[i] <= nmin[i]);
        ERRCHK(nmin[i] < nmax[i]);
        ERRCHK(nmax[i] <= mmax[i]);
    }

    DynamicArray segment_dims, segment_offsets;
    dynarr_create(&segment_dims);
    dynarr_create(&segment_offsets);

    const size_t npartitions = partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, &segment_dims,
                                                   &segment_offsets);
    // print("npartitions", npartitions);
    // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), segment_dims.data);
    // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), segment_offsets.data);

    // Copy the count required to hold the data in segment_offsets and segment_dims
    ERRCHK(segment_dims.length == segment_offsets.length);
    ERRCHK(segment_offsets.length == ndims * npartitions);
    ERRCHK(nelems != NULL);
    *nelems = segment_dims.length;

    // Copy the segment information to the output arrays
    if (dims != NULL)
        ncopy(segment_dims.length, segment_dims.data, dims);
    if (offsets != NULL)
        ncopy(segment_offsets.length, segment_offsets.data, offsets);

    dynarr_destroy(&segment_dims);
    dynarr_destroy(&segment_offsets);

    ndealloc(mmin);
    ndealloc(nmin);
    ndealloc(nmax);
    ndealloc(mmax);

    return npartitions;
}

void
test_partition(void)
{
    {
        const size_t mm[]        = {8};
        const size_t nn[]        = {6, 6};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        // print("npartitions", npartitions);
        // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 3);

        ndealloc(dims);
        ndealloc(offsets);
    }
    {
        const size_t mm[]        = {8, 8};
        const size_t nn[]        = {6, 6};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        // print("npartitions", npartitions);
        // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 9);

        ndealloc(dims);
        ndealloc(offsets);
    }
    {
        const size_t mm[]        = {8, 8, 8};
        const size_t nn[]        = {6, 6, 6};
        const size_t nn_offset[] = {1, 1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        // print("npartitions", npartitions);
        // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 27);

        ndealloc(dims);
        ndealloc(offsets);
    }
    {
        const size_t mm[]        = {5, 6, 7, 8};
        const size_t nn[]        = {3, 4, 5, 6};
        const size_t nn_offset[] = {1, 1, 1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        // print("npartitions", npartitions);
        // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 3 * 3 * 3 * 3);

        ndealloc(dims);
        ndealloc(offsets);
    }
    {
        const size_t mm[]        = {4, 4, 4};
        const size_t nn[]        = {4, 4, 4};
        const size_t nn_offset[] = {0, 0, 0};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        // print("npartitions", npartitions);
        // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 1);

        ndealloc(dims);
        ndealloc(offsets);
    }
    {
        const size_t mm[]        = {4, 4};
        const size_t nn[]        = {3, 3};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        size_t nelems;
        partition(ndims, mm, nn, nn_offset, &nelems, NULL, NULL);

        size_t *dims, *offsets;
        nalloc(nelems, dims);
        nalloc(nelems, offsets);
        const size_t npartitions = partition(ndims, mm, nn, nn_offset, &nelems, dims, offsets);
        print("npartitions", npartitions);
        print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), dims);
        print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), offsets);

        ERRCHK(npartitions == 4);

        ndealloc(dims);
        ndealloc(offsets);
    }
}
