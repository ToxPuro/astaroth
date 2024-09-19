#include "dynamic_array.h"

#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "ndarray.h"

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

// DynamicNdArray
// ndarray_create(const size_t ndims, const size_t* dims)
// {
//     DynamicNdArray ndarray = (DynamicNdArray){
//         .ndims = ndims,
//         .dims  = malloc(sizeof(ndarray.dims[0]) * ndims),
//         .array = array_create(prod(ndims, dims)),
//     };
//     ERRCHK(ndarray.dims);
//     copy(ndims, dims, ndarray.dims);

//     return ndarray;
// }

// void
// ndarray_destroy(DynamicNdArray* ndarray)
// {
//     array_destroy(&ndarray->array);
//     free(ndarray->dims);
//     ndarray->dims  = NULL;
//     ndarray->ndims = 0;
// }

// void
// ndarray_print(const char* label, const DynamicNdArray ndarray)
// {
//     printf("%s:\n", label);
//     print_ndarray(ndarray.ndims, ndarray.dims, ndarray.array.data);
// }

// void
// ndarray_append(const size_t ndims, const size_t* row, DynamicNdArray* ndarray)
// {
//     array_append_multiple(ndims, row, &ndarray->array);
// }
