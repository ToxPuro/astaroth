#include "print.h"

#include <stdio.h>

void
print_size_t(const char* label, const size_t value)
{
    printf("%s: %zu\n", label, value);
}

void
print_int64_t(const char* label, const int64_t value)
{
    printf("%s: %ld\n", label, value);
}

void
print_int(const char* label, const int value)
{
    printf("%s: %d\n", label, value);
}

void
print_size_t_array(const char* label, const size_t count, const size_t arr[])
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i)
        printf("%zu%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}

void
print_int64_t_array(const char* label, const size_t count, const int64_t arr[])
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i)
        printf("%ld%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}

void
print_int_array(const char* label, const size_t count, const int arr[])
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i)
        printf("%d%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}
