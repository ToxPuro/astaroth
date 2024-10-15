#include "alloc.h"

#include <stdlib.h>
#include <string.h>

#include "errchk.h"

void*
ac_malloc(const size_t bytes)
{
    void* ptr = malloc(bytes);
    ERRCHK(ptr);
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
    ERRCHK(in);
    ERRCHK(out);
    memmove(out, in, count * size);
}

void*
ac_dup(const size_t count, const size_t size, const void* in)
{
    ERRCHK(in);
    void* out = ac_calloc(count, size);
    ac_copy(count, size, in, out);
    return out;
}

bool
ac_cmp(const size_t count, const size_t size, const void* a, const void* b)
{
    return memcmp(a, b, count * size) == 0 ? true : false;
}

void
ac_reverse(const size_t count, const size_t size, void* a)
{
    void* tmp = ac_dup(count, size, a);

    for (size_t i = 0; i < count; ++i)
        ac_copy(1, size, (uint8_t*)tmp + (count - 1 - i) * size, (uint8_t*)a + i * size);

    ac_free(tmp);
}

int
test_alloc(void)
{
    const size_t count = 10;
    size_t in[count], out[count];
    for (size_t i = 0; i < count; ++i)
        in[i] = i;
    ac_copy(count, sizeof(in[0]), in, out);
    ac_reverse(count, sizeof(in[0]), out);

    for (size_t i = 1; i < count; ++i) {
        if (out[i] >= out[i - 1]) {
            ERROR("ac_reverse test failed: expected out[%d] >= out[%d] but the values were %zu and "
                  "%zu",
                  i, i - 1, out[i], out[i - 1]);
            return -1;
        }
    }

    return 0;
}
