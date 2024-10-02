#include "partition.h"

#include "dynarr.h"
#include "nalloc.h"
#include "type_conversion.h"

// #define dynarr(T)                                                                                  \
//     struct {                                                                                       \
//         size_t length;                                                                             \
//         size_t capacity;                                                                           \
//         T* data;                                                                                   \
//     }

// #define dynarr_create(arr)                                                                         \
//     do {                                                                                           \
//         (arr)->length   = 0;                                                                       \
//         (arr)->capacity = 1;                                                                       \
//         nalloc(1, (arr)->data);                                                                    \
//     } while (0)

// #define dynarr_destroy(arr)                                                                        \
//     do {                                                                                           \
//         ndealloc((arr)->data);                                                                     \
//         (arr)->capacity = 0;                                                                       \
//         (arr)->length   = 0;                                                                       \
//     } while (0)

// #define dynarr_append(elem, arr)                                                                   \
//     do {                                                                                           \
//         if ((arr)->length == (arr)->capacity)                                                      \
//             nrealloc(++(arr)->capacity, (arr)->data);                                              \
//         (arr)->data[(arr)->length++] = (elem);                                                     \
//     } while (0)

// #define dynarr_append_multiple(count, elems, arr)                                                  \
//     do {                                                                                           \
//         for (size_t __dynarr_i__ = 0; __dynarr_i__ < (count); ++__dynarr_i__)                      \
//             dynarr_append((elems)[__dynarr_i__], (arr));                                           \
//     } while (0)

// #define dynarr_remove(index, arr)                                                                  \
//     do {                                                                                           \
//         ERRCHK((index) < (arr)->length);                                                           \
//         const size_t __dynarr_count__ = (arr)->length - (index) - 1;                               \
//         if (__dynarr_count__ > 0)                                                                  \
//             ncopy(__dynarr_count__, &(arr)->data[(index) + 1], &(arr)->data[(index)]);             \
//         --(arr)->length;                                                                           \
//     } while (0)

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

    // dynarr_test();

    ndealloc(arr);
}
