#include "partition.h"

#include "nalloc.h"

// static Partition
// partition_create(const size_t ndims)
// {
//     Partition partition = (Partition){
//         .ndims  = ndims,
//         .dims   = NULL,
//         .offset = NULL,
//     };
//     nalloc(ndims, partition.dims);
//     nalloc(ndims, partition.offset);
//     return partition;
// }

// static void
// partition_destroy(Partition* partition)
// {
//     ndealloc(partition->offset);
//     ndealloc(partition->dims);
//     partition->dims   = NULL;
//     partition->offset = NULL;
//     partition->ndims  = 0;
// }

// size_t
// partition_recursive()

// size_t
// partitions_create(const size_t ndims, const size_t* mm, const size_t* nn, const size_t*
// nn_offset,
//                   const size_t npartitions, Partition* partitions)
// {
//     size_t *mmin, *nmin, *nmax, *mmax;
//     nalloc(ndims, mmin);
//     nalloc(ndims, nmin);
//     nalloc(ndims, nmax);
//     nalloc(ndims, mmax);

//     for (size_t i = 0; i < ndims; ++i) {
//         mmin[i] = 0;
//         nmin[i] = nn_offset[i];
//         nmax[i] = nn_offset[i] + nn[i];
//         mmax[i] = mm[i];

//         ERRCHK(mmin[i] <= nmin[i]);
//         ERRCHK(nmin[i] < nmax[i]);
//         ERRCHK(nmax[i] <= mmax[i]);
//     }

//     ndealloc(mmin);
//     ndealloc(nmin);
//     ndealloc(nmax);
//     ndealloc(mmax);
// }

// void
// partitions_destroy(const size_t npartitions, Partition** partitions)
// {
//     WARNING("not implemented");
// }

#include "print.h"

// #define as_matrix(ncols, arr) (*arr)[ncols] = (size_t(*)[ncols])arr

void
test_partition(void)
{
    WARNING("not implemented");
    const size_t nrows = 3;
    const size_t ncols = 2;
    size_t* arr;
    nalloc(nrows * ncols, arr);

    print_ndarray("arr", 2, ((size_t[]){ncols, nrows}), arr);

    size_t(*mat)[ncols] = (size_t(*)[ncols])arr; // 2D array from dynamically allocated memory
    print_array("mat row", ncols, mat[0]);

    ndealloc(arr);
}
