#include "print.h"

#include <stdio.h>

const char*
format_specifier_size_t(const size_t value)
{
    (void)value;
    static const char specifier[] = "%zu";
    return specifier;
}

const char*
format_specifier_int64_t(const int64_t value)
{
    (void)value;
    static const char specifier[] = "%lld";
    return specifier;
}

const char*
format_specifier_int(const int value)
{
    (void)value;
    static const char specifier[] = "%d";
    return specifier;
}

const char*
format_specifier_double(const double value)
{
    (void)value;
    static const char specifier[] = "%g";
    return specifier;
}
void
print_type_size_t(const size_t value)
{
    printf(format_specifier(value), value);
}

void
print_type_int64_t(const int64_t value)
{
    printf(format_specifier(value), value);
}

void
print_type_int(const int value)
{
    printf(format_specifier(value), value);
}

void
print_type_double(const double value)
{
    printf(format_specifier(value), value);
}
void
print_size_t(const char* label, const size_t value)
{
    printf("%s: ", label);
    print_type(value);
    printf("\n");
}

void
print_int64_t(const char* label, const int64_t value)
{
    printf("%s: ", label);
    print_type(value);
    printf("\n");
}

void
print_int(const char* label, const int value)
{
    printf("%s: ", label);
    print_type(value);
    printf("\n");
}

void
print_double(const char* label, const double value)
{
    printf("%s: ", label);
    print_type(value);
    printf("\n");
}
void
print_array_size_t(const char* label, const size_t count, const size_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
    }
    printf(")\n");
}

void
print_array_int64_t(const char* label, const size_t count, const int64_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
    }
    printf(")\n");
}

void
print_array_int(const char* label, const size_t count, const int* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
    }
    printf(")\n");
}

void
print_array_double(const char* label, const size_t count, const double* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i) {
        print_type(arr[i]);
        printf("%s", i < count - 1 ? ", " : "");
    }
    printf(")\n");
}
