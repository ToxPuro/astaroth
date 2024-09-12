#include "print.h"

#include <stdio.h>

void
acPrint_size_t(const char* label, const size_t value)
{
    printf("%s: %zu\n", label, value);
}

void
acPrintArray_size_t(const char* label, const size_t count, const size_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i)
        printf("%zu%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}