#include "alloc.h"

#include <stdlib.h>
#include <string.h>

#include "errchk.h"

void*
ac_malloc(const size_t bytes)
{
    void* ptr = malloc(bytes);
    ERRCHK(ptr != NULL);
    return ptr;
}

void*
ac_calloc(const size_t count, const size_t size)
{
    void* ptr = calloc(count, size);
    ERRCHK(ptr);
    return ptr;
}

void
ac_free(void* ptr)
{
    ERRCHK(ptr);
    free(ptr);
}

void*
ac_realloc(const size_t count, const size_t size, void* ptr)
{
    ptr = realloc(ptr, count * size);
    ERRCHK(ptr);
    return ptr;
}

void
ac_copy(const size_t count, const size_t size, const void* in, void* out)
{
    ERRCHK(in != NULL);
    ERRCHK(out != NULL);
    memmove(out, in, count * size);
}

void*
ac_dup(const size_t count, const size_t size, const void* in)
{
    ERRCHK(in != NULL);
    void* out = ac_calloc(count, size);
    ac_copy(count, size, in, out);
    return out;
}

bool
ac_cmp(const size_t count, const size_t size, const void* a, const void* b)
{
    return memcmp(a, b, count * size) == 0 ? true : false;
}
