#include "partition.h"

#include "math_utils.h"
#include "misc.h"
#include "print.h"

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
          DynamicArray* segment_dims, DynamicArray* segment_offsets)
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

    const size_t npartitions = partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, segment_dims,
                                                   segment_offsets);
    // print("npartitions", npartitions);
    // print_ndarray("segment_dims", 2, ((size_t[]){ndims, npartitions}), segment_dims->data);
    // print_ndarray("segment_offsets", 2, ((size_t[]){ndims, npartitions}), segment_offsets->data);

    ndealloc(mmin);
    ndealloc(nmin);
    ndealloc(nmax);
    ndealloc(mmax);

    return npartitions;
}

static void
partition_recursive_new(const size_t ndims, const size_t* mmin, const size_t* nmin,
                        const size_t* nmax, const size_t* mmax, const size_t axis,
                        SegmentArray* segments)
{
    ERRCHK(segments != NULL);
    if (get_volume(ndims, mmin, mmax) == 0) {
        return;
    }
    else if (axis >= ndims) {

        size_t* dims;
        nalloc(ndims, dims);
        subtract_arrays(ndims, mmax, mmin, dims);
        const size_t* offset = mmin;

        dynarr_append(segment_create(ndims, dims, offset), segments);

        ndealloc(dims);
    }
    else {

        size_t *new_mmin, *new_mmax;
        nalloc(ndims, new_mmin);
        nalloc(ndims, new_mmax);

        // Left
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = mmin[axis];
        new_mmax[axis] = nmin[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        // Center
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmin[axis];
        new_mmax[axis] = nmax[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        // Right
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmax[axis];
        new_mmax[axis] = mmax[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        ndealloc(new_mmin);
        ndealloc(new_mmax);
    }
}

void
partition_new(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
              SegmentArray* segments)
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

    partition_recursive_new(ndims, mmin, nmin, nmax, mmax, 0, segments);

    ndealloc(mmin);
    ndealloc(nmin);
    ndealloc(nmax);
    ndealloc(mmax);
}

/*
typedef struct {
    size_t ndims;
    size_t* mmin;
    size_t* nmin;
    size_t* nmax;
    size_t* mmax;
} PartitionDescriptor;

static PartitionDescriptor
partition_descriptor_create(const size_t ndims, const size_t* mmin, const size_t* nmin,
                            const size_t* nmax, const size_t* mmax)
{
    PartitionDescriptor pd;

    pd.ndims = ndims;
    ndup(ndims, mmin, pd.mmin);
    ndup(ndims, nmin, pd.nmin);
    ndup(ndims, nmax, pd.nmax);
    ndup(ndims, mmax, pd.mmax);

    return pd;
}

static void
partition_descriptor_destroy(PartitionDescriptor* pd)
{
    ndealloc(pd->mmin);
    ndealloc(pd->nmin);
    ndealloc(pd->nmax);
    ndealloc(pd->mmax);
    pd->ndims = 0;
}

static PartitionDescriptor
partition_descriptor_create_from_offset(const size_t ndims, const size_t* mm, const size_t* nn,
                                        const size_t* nn_offset)
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

    PartitionDescriptor pd = partition_descriptor_create(ndims, mmin, nmin, nmax, mmax);

    ndealloc(mmin);
    ndealloc(nmin);
    ndealloc(nmax);
    ndealloc(mmax);

    return pd;
}

static PartitionDescriptor
partition_descriptor_duplicate(const PartitionDescriptor pd)
{
    return partition_descriptor_create(pd.ndims, pd.mmin, pd.nmin, pd.nmax, pd.mmax);
}

static void
partition_recursive_newer(const size_t axis, const PartitionDescriptor pd, SegmentArray* segments)
{
    ERRCHK(segments != NULL);
    if (get_volume(pd.ndims, pd.mmin, pd.mmax) == 0) {
        return;
    }
    else if (axis >= pd.ndims) {

        size_t* dims;
        nalloc(pd.ndims, dims);
        subtract_arrays(pd.ndims, pd.mmax, pd.mmin, dims);
        const size_t* offset = pd.mmin;

        dynarr_append(segment_create(pd.ndims, dims, pd.mmin), segments);

        ndealloc(dims);
    }
    else {

        PartitionDescriptor pd_new = partition_descriptor_duplicate(pd);
        pd_new.mmin[axis] = pd.mmin[axis];
        pd_new.mmax[axis] = pd.mmin[axis];
        partition_descriptor_destroy(&pd_new);

        size_t *new_mmin, *new_mmax;
        nalloc(ndims, new_mmin);
        nalloc(ndims, new_mmax);

        // Left
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = mmin[axis];
        new_mmax[axis] = nmin[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        // Center
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmin[axis];
        new_mmax[axis] = nmax[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        // Right
        ncopy(ndims, mmin, new_mmin);
        ncopy(ndims, mmax, new_mmax);
        new_mmin[axis] = nmax[axis];
        new_mmax[axis] = mmax[axis];
        partition_recursive_new(ndims, new_mmin, nmin, nmax, new_mmax, axis + 1, segments);

        ndealloc(new_mmin);
        ndealloc(new_mmax);
    }
}
*/

void
test_partition(void)
{
    {
        const size_t mm[]        = {8, 8};
        const size_t nn[]        = {6, 6};
        const size_t nn_offset[] = {1, 1};
        const size_t ndims       = ARRAY_SIZE(mm);

        SegmentArray segments;
        dynarr_create_with_destructor(segment_destroy, &segments);
        partition_new(ndims, mm, nn, nn_offset, &segments);
        ERRCHK(segments.length == 9);

        dynarr_destroy(&segments);
    }

    /*
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
    }*/
}
