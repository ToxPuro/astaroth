#include "dynarr.h"

typedef dynarr(int) dynarr_int;
typedef dynarr(size_t) dynarr_size_t;

void
test_dynarr(void)
{
    {
        dynarr_int arr;
        dynarr_create(&arr);
        dynarr_append(1, &arr);
        dynarr_append(2, &arr);
        dynarr_append(3, &arr);
        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 2);
        ERRCHK(arr.data[2] == 3);
        dynarr_remove(1, &arr);
        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);

        const size_t count = 2;
        int* elems;
        nalloc(count, elems);
        for (size_t i = 0; i < count; ++i)
            elems[i] = as_int(10 + i);
        dynarr_append_multiple(count, elems, &arr);

        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);
        ERRCHK(arr.data[2] == 10);
        ERRCHK(arr.data[3] == 11);
        ndealloc(elems);

        dynarr_destroy(&arr);
    }
    {
        dynarr_size_t arr;
        dynarr_create(&arr);
        dynarr_append(1, &arr);
        dynarr_append(2, &arr);
        dynarr_append(3, &arr);
        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 2);
        ERRCHK(arr.data[2] == 3);
        dynarr_remove(1, &arr);
        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);

        const size_t count = 2;
        size_t* elems;
        nalloc(count, elems);
        for (size_t i = 0; i < count; ++i)
            elems[i] = as_size_t(10 + i);
        dynarr_append_multiple(count, elems, &arr);

        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);
        ERRCHK(arr.data[2] == 10);
        ERRCHK(arr.data[3] == 11);
        ndealloc(elems);

        dynarr_destroy(&arr);
    }
}
