#include "type_conversion.h"

#include <limits.h> // INT_MAX

#include "errchk.h"

// uint64_t
// double_as_uint64_t(const double x)
// {
//     ERRCHK(x >= 0);
//     ERRCHK((uint64_t)x <= SIZE_MAX);
//     ERRCHK(x <= (double)SIZE_MAX);
//     const uint64_t tmp = (uint64_t)x;
//     ERRCHK((double)tmp == x);
//     return (uint64_t)x;
// }

double
uint64_t_as_double(const uint64_t x)
{
    const double y   = (double)x;
    const uint64_t z = (uint64_t)y;
    ERRCHK(z == x);
    return y;
}

uint64_t
uint64_t_as_uint64_t(const uint64_t i)
{
    return i;
}

int
int_as_int(const int i)
{
    return i;
}

uint64_t
int64_t_as_uint64_t(const int64_t i)
{
    ERRCHK(i >= 0);
    return (uint64_t)i;
}

uint64_t
int_as_uint64_t(const int i)
{
    ERRCHK(i >= 0);
    return (uint64_t)i;
}

int64_t
uint64_t_as_int64_t(const uint64_t i)
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
uint64_t_as_int(const uint64_t i)
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
int64_t_as_uint64_t_array(const size_t count, const int64_t* a, uint64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_uint64_t(a[i]);
}
void
int_as_uint64_t_array(const size_t count, const int* a, uint64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_uint64_t(a[i]);
}
void
uint64_t_as_int64_t_array(const size_t count, const uint64_t* a, int64_t* b)
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
uint64_t_as_int_array(const size_t count, const uint64_t* a, int* b)
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
