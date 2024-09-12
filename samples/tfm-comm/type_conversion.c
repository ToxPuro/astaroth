#include "type_conversion.h"

#include "errchk.h"

size_t
as_size_t(const int64_t i)
{
    ERRCHK(i >= 0);
    return (size_t)i;
}

int64_t
as_int64_t(const size_t i)
{
    ERRCHK(i <= INT64_MAX);
    return (int64_t)i;
}

void
as_size_t_array(const size_t count, const int64_t* a, size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_size_t(a[i]);
}

void
as_int64_t_array(const size_t count, const size_t* a, int64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int64_t(a[i]);
}