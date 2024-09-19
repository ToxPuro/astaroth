#include "dynamic_array.h"

#include <stdlib.h>

#include "errchk.h"

DynamicArray
array_create(const size_t capacity)
{
    DynamicArray arr = (DynamicArray){
        .len      = 0,
        .capacity = capacity,
        .data     = (DATYPE*)malloc(sizeof(arr.data[0]) * capacity),
    };
    ERRCHK(arr.data);
    return arr;
}

void
array_append(const DATYPE element, DynamicArray* array)
{
    if (array->len == array->capacity) {
        array->capacity += 128;
        array->data = (DATYPE*)realloc(array->data, sizeof(array->data[0]) * array->capacity);
        WARNING("Array too small, reallocated");
        ERRCHK(array->data);
    }
    array->data[array->len] = element;
    ++array->len;
}

void
array_append_multiple(const size_t count, const DATYPE* elements, DynamicArray* array)
{
    for (size_t i = 0; i < count; ++i)
        array_append(elements[i], array);
}

void
array_destroy(DynamicArray* array)
{
    free(array->data);
    array->len      = 0;
    array->capacity = 0;
}
