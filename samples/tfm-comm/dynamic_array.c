#include "dynamic_array.h"

#include <stdlib.h>
#include <string.h>

#include "errchk.h"
#include "math_utils.h"
#include "ndarray.h"

DynamicArray
array_create(const size_t capacity)
{
    DynamicArray arr = (DynamicArray){
        .length   = 0,
        .capacity = capacity,
        .data     = (size_t*)malloc(sizeof(arr.data[0]) * capacity),
    };
    ERRCHK(arr.data);
    return arr;
}

void
array_append(const size_t element, DynamicArray* array)
{
    if (array->length == array->capacity) {
        array->capacity += 128;
        array->data = (size_t*)realloc(array->data, sizeof(array->data[0]) * array->capacity);
        // WARNING("Array too small, reallocated");
        ERRCHK(array->data);
    }
    array->data[array->length] = element;
    ++array->length;
}

void
array_append_multiple(const size_t count, const size_t* elements, DynamicArray* array)
{
    for (size_t i = 0; i < count; ++i)
        array_append(elements[i], array);
}

void
array_remove(const size_t index, DynamicArray* array)
{
    ERRCHK(index < array->length);

    const size_t count = array->length - index - 1;
    if (count > 0)
        memmove(&array->data[index], &array->data[index + 1], count * sizeof(array->data[0]));
    --array->length;
}

size_t
array_get(const DynamicArray array, const size_t index)
{
    ERRCHK(index < array.length);
    return array.data[index];
}

void
to_static_array(const DynamicArray in, const size_t nrows, const size_t ncols,
                size_t out[nrows][ncols])
{
    memmove(out, in.data, sizeof(out[0][0]) * nrows * ncols);
}

void
array_destroy(DynamicArray* array)
{
    free(array->data);
    array->length   = 0;
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
