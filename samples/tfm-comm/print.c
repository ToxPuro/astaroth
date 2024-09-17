#include "print.h"

#include <stdio.h>

void
print_type_size_t(const size_t value)
{
    printf("%zu", value);
}

void
print_type_int64_t(const int64_t value)
{
    printf("%lld", value);
}

void
print_type_int(const int value)
{
    printf("%d", value);
}

void
print_type_double(const double value)
{
    printf("%g", value);
}
void
print_size_t(const char* label, const size_t value)
{
    printf("%s: ", label);
    print_type(value);
}

void
print_int64_t(const char* label, const int64_t value)
{
    printf("%s: ", label);
    print_type(value);
}

void
print_int(const char* label, const int value)
{
    printf("%s: ", label);
    print_type(value);
}

void
print_double(const char* label, const double value)
{
    printf("%s: ", label);
    print_type(value);
}
void
print_array_size_t(const char* label, const size_t count, const size_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
        printf(")");
    }
}

void
print_array_int64_t(const char* label, const size_t count, const int64_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
        printf(")");
    }
}

void
print_array_int(const char* label, const size_t count, const int* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
        printf(")");
    }
}

void
print_array_double(const char* label, const size_t count, const double* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
        printf(")");
    }
}
