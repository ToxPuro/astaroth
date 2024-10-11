#include "dynarr.h"

#include <stdint.h>

#include "alloc.h"
#include "errchk.h"

DynamicArray
dynarr_create(const size_t capacity, const size_t elem_size)
{
    DynamicArray arr = (DynamicArray){
        .len       = 0,
        .capacity  = capacity,
        .elem_size = elem_size,
        .data      = ac_calloc(capacity, elem_size),
    };
    return arr;
}

void
dynarr_destroy(DynamicArray* arr)
{
    ac_free(arr->data);
    arr->data      = NULL;
    arr->elem_size = 0;
    arr->capacity  = 0;
    arr->len       = 0;
}

void
dynarr_append(const size_t elem_size, const void* elem, DynamicArray* arr)
{
    ERRCHK(elem_size == arr->elem_size);
    if (arr->len == arr->capacity)
        arr->data = ac_realloc(++arr->capacity, arr->elem_size, arr->data);

    ac_copy(1, arr->elem_size, elem, (uint8_t*)arr->data + arr->len * arr->elem_size);
    ++arr->len;
}

void
dynarr_append_multiple(const size_t count, const size_t elem_size, const void* elems,
                       DynamicArray* arr)
{
    for (size_t i = 0; i < count; ++i)
        dynarr_append(elem_size, (const uint8_t*)elems + i * arr->elem_size, arr);
}

void
dynarr_remove(const size_t index, DynamicArray* arr)
{
    ERRCHK(index < arr->len);
    if (index >= arr->len)
        return;

    const size_t count = arr->len - index - 1;
    if (count > 0) {
        const void* src = (uint8_t*)arr->data + (index + 1) * arr->elem_size;
        void* dst       = (uint8_t*)arr->data + index * arr->elem_size;
        ac_copy(count, arr->elem_size, src, dst);
    }
    --arr->len;
}

void
dynarr_remove_multiple(const size_t index, const size_t count, DynamicArray* arr)
{
    for (size_t i = 0; i < count; ++i)
        dynarr_remove(index, arr);
}

void*
dynarr_get(const size_t index, const DynamicArray arr)
{
    ERRCHK(index < arr.len);
    if (index >= arr.len)
        return NULL;

    return (uint8_t*)arr.data + index * arr.elem_size;
}

#include "print.h"
#include <stdio.h>

void
print_dynarr(const char* label, const DynamicArray arr)
{
    printf("%s:\n", label);
    print("\t.len", arr.len);
    print("\t.capacity", arr.capacity);
    print("\t.elem_size", arr.elem_size);
    printf("\t.data: %p\n", arr.data);
}

unsigned
test_dynarr(void)
{
    // Drawbacks: cumbersome and error-prone to use, no type safety
    DynamicArray arr = dynarr_create(10, sizeof(size_t));
    ERRCHK(arr.len == 0);
    dynarr_append(sizeof(size_t), (size_t[]){1}, &arr);
    ERRCHK(arr.len == 1);
    dynarr_append_multiple(4, sizeof(size_t), (size_t[]){2, 3, 4, 5}, &arr);
    ERRCHK(arr.len == 5);
    dynarr_remove_multiple(1, 3, &arr);
    ERRCHK(arr.len == 2);
    ERRCHK(*((size_t*)dynarr_get(0, arr)) == 1);
    ERRCHK(*((size_t*)dynarr_get(1, arr)) == 5);
    print_dynarr("test", arr);
    dynarr_destroy(&arr);

    return 1;
}
