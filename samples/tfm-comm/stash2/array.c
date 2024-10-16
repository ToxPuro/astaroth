#include "array.h"

#include <string.h>

#include "errchk.h"

Array
arrayCreate(const size_t element_size, const size_t initial_capacity)
{
    Array array = (Array){
        .length       = 0,
        .capacity     = initial_capacity,
        .element_size = element_size,
        .data         = malloc(element_size * initial_capacity),
    };
    ERRCHK(array.data != NULL);

    return array;
}

void
arrayAppend(const void* element, Array* array)
{
    if (array->length == array->capacity) {
        array->capacity += 128 * array->element_size;
        array->data = realloc(array->data, array->element_size * array->capacity);
        WARNING("Array too small, reallocated");
        ERRCHK(array->data);
    }
    memmove(&array->data[array->length * array->element_size], element, array->element_size);
    ++array->length;
}

char*
arrayGet(const Array array, const size_t index)
{
    ERRCHK(index < array.length);
    return &array.data[index * array.element_size];
}

void
arrayRemove(const size_t index, Array* array)
{
    ERRCHK(index < array->length);

    if (index + 1 < array->length) {
        const size_t dst   = index * array->element_size;
        const size_t src   = (index + 1) * array->element_size;
        const size_t bytes = (array->length - index - 1) * array->element_size;
        memmove(&array->data[dst], &array->data[src], bytes);
    }
    --array->length;
}

void
arrayDestroy(Array* array)
{
    free(array->data);
    array->data         = NULL;
    array->capacity     = 0;
    array->element_size = 0;
    array->length       = 0;
}
