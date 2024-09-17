#include "array.h"

#include <stdlib.h>

double*
array_create(const size_t count, const bool on_device)
{
    return (double*)malloc(sizeof(double) * count);
}

void
array_destroy(double** array, const bool on_device)
{
    free(*array);
    *array = NULL;
}
