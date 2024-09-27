#include "partition.h"

#include "array.h"
#include "dynamic_array.h"
#include "errchk.h"
#include "math_utils.h"
#include "ndarray.h"
#include "print.h"

size_t
volume(const size_t ndims, const size_t* mmin, const size_t* mmax)
{
    size_t dims[ndims];
    subtract_arrays(ndims, mmax, mmin, dims);
    return prod(ndims, dims);
}

size_t
partition_recursive(const size_t ndims, const size_t* mmin, const size_t* nmin, const size_t* nmax,
                    const size_t* mmax, const size_t axis, DynamicArray* dimensions,
                    DynamicArray* offsets)
{
    if (axis >= ndims) {
        // static size_t counter = 0;
        // print("Counter", counter++);
        // print_array("mmin", ndims, mmin);
        // print_array("mmax", ndims, mmax);

        size_t dims[ndims];
        subtract_arrays(ndims, mmax, mmin, dims);
        // print_array("dims", ndims, dims);
        // printf("\n");

        if (dimensions != NULL)
            array_append_multiple(ndims, dims, dimensions);
        if (offsets != NULL)
            array_append_multiple(ndims, mmin, offsets);
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
                                                   dimensions, offsets);
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
                                                   dimensions, offsets);
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
                                                   dimensions, offsets);
        }
        return npartitions;
    }
}

/** Partitions the domain mm based on subdomain nn, offset by nn_offset  */
size_t
partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
          const size_t npartitions, size_t dims[npartitions][ndims],
          size_t offsets[npartitions][ndims])
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

    const size_t count = partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, NULL, NULL);
    if (npartitions == 0 || dims == NULL || offsets == NULL)
        return count;

    ERRCHK(npartitions == count);

    DynamicArray dims_dynamic    = array_create(1);
    DynamicArray offsets_dynamic = array_create(1);

    partition_recursive(ndims, mmin, nmin, nmax, mmax, 0, &dims_dynamic, &offsets_dynamic);
    to_static_array(dims_dynamic, npartitions, ndims, dims);
    to_static_array(offsets_dynamic, npartitions, ndims, offsets);

    array_destroy(&dims_dynamic);
    array_destroy(&offsets_dynamic);

    return count;
}
