#include "type_conversion.h"

#include <limits.h> // INT_MAX

#include "errchk.h"

size_t
int64_t_as_size_t(const int64_t i)
{
    ERRCHK(i >= 0);
    return (size_t)i;
}

size_t
int_as_size_t(const int i)
{
    ERRCHK(i >= 0);
    return (size_t)i;
}

int64_t
size_t_as_int64_t(const size_t i)
{
    ERRCHK(i <= INT64_MAX);
    return (int64_t)i;
}

int64_t
int_as_int64_t(const int i)
{
    return (int64_t)i;
}

int
size_t_as_int(const size_t i)
{
    ERRCHK(i <= INT_MAX);
    return (int)i;
}

int
int64_t_as_int(const int64_t i)
{
    ERRCHK(i <= INT_MAX);
    return (int)i;
}

void
int64_t_as_size_t_array(const size_t count, const int64_t* a, size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_size_t(a[i]);
}
void
int_as_size_t_array(const size_t count, const int* a, size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_size_t(a[i]);
}
void
size_t_as_int64_t_array(const size_t count, const size_t* a, int64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int64_t(a[i]);
}
void
int_as_int64_t_array(const size_t count, const int* a, int64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int64_t(a[i]);
}
void
size_t_as_int_array(const size_t count, const size_t* a, int* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int(a[i]);
}
void
int64_t_as_int_array(const size_t count, const int64_t* a, int* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int(a[i]);
}
