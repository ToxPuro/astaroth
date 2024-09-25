// Clean
#define TG TG_TYPE_GENERIC_FUNCTION
#define T TG_DTYPE

// #include "print.h"
#include <string.h>

typedef struct {
    size_t length;
    size_t capacity;
    T* data;
} TG(DynamicArray);

TG(DynamicArray) TG(array_create)(const size_t capacity)
{
    TG(DynamicArray)
    arr = (TG(DynamicArray)){
        .length   = 0,
        .capacity = capacity,
        .data     = malloc(sizeof(arr.data[0]) * capacity),
    };
    ERRCHK(arr.data);
    return arr;
}

void
TG(array_append)(const T element, TG(DynamicArray) * array)
{
    if (array->length == array->capacity) {
        array->capacity += 128;
        array->data = realloc(array->data, sizeof(array->data[0]) * array->capacity);
        WARNING("Array too small, reallocated");
        ERRCHK(array->data);
    }
    array->data[array->length] = element;
    ++array->length;
}

void
TG(array_remove)(const size_t index, TG(DynamicArray) * array)
{
    ERRCHK(index < array->length);

    const size_t count = array->length - index - 1;
    if (count > 0)
        memmove(&array->data[index], &array->data[index + 1], count * sizeof(array->data[0]));
    --array->length;
}

T
TG(array_get)(const TG(DynamicArray) array, const size_t index)
{
    ERRCHK(index < array.length);
    return array.data[index];
}

// void
// TG(array_print)(const char* label, const TG(DynamicArray) array)
// {
//     print_array(label, array.length, array.data);
// }

// DynamicArray_size_t test = array_create_size_t(1);
//     array_append(0, &test);
//     array_append(1, &test);
//     array_append(2, &test);
//     array_append(3, &test);
//     array_remove(2, &test);
//     print_array("Test", test.length, test.data);
//     array_destroy(&test);

void
TG(array_destroy)(TG(DynamicArray) * array)
{
    free(array->data);
    array->length   = 0;
    array->capacity = 0;
}

#undef TG
#undef T
