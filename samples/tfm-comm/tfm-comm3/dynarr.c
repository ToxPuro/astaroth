#include "dynarr.h"

typedef dynarr_s(int) dynarr_int;
typedef dynarr_s(size_t) dynarr_size_t;

typedef struct {
    size_t count;
    size_t* elems;
} TestStruct;

static TestStruct
test_struct_create(const size_t count)
{
    TestStruct s = (TestStruct){
        .count = count,
        .elems = ac_calloc(count, sizeof(s.elems[0])),
    };
    return s;
}

static void
test_struct_destroy(TestStruct* s)
{
    ac_free(s->elems);
    s->count = 0;
}

typedef dynarr_s(TestStruct) dynarr_test_struct;

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
        int* elems         = ac_calloc(count, sizeof(elems[0]));
        for (size_t i = 0; i < count; ++i)
            elems[i] = as_int(10 + i);
        dynarr_append_multiple(count, elems, &arr);

        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);
        ERRCHK(arr.data[2] == 10);
        ERRCHK(arr.data[3] == 11);
        ac_free(elems);

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
        size_t* elems      = ac_calloc(count, sizeof(elems[0]));
        for (size_t i = 0; i < count; ++i)
            elems[i] = as_size_t(10 + i);
        dynarr_append_multiple(count, elems, &arr);

        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 3);
        ERRCHK(arr.data[2] == 10);
        ERRCHK(arr.data[3] == 11);

        dynarr_remove_multiple(1, 2, &arr);
        ERRCHK(arr.data[0] == 1);
        ERRCHK(arr.data[1] == 11);

        ac_free(elems);

        dynarr_destroy(&arr);
    }
    {
        dynarr_test_struct arr;
        dynarr_create_with_destructor(test_struct_destroy, &arr);

        dynarr_append(test_struct_create(10), &arr);
        dynarr_append(test_struct_create(11), &arr);
        dynarr_append(test_struct_create(12), &arr);
        dynarr_append(test_struct_create(13), &arr);

        dynarr_destroy(&arr);
    }
}
