#include "array.h"

void
copy(const size_t count, const size_t* in, size_t* out)
{
    for (size_t i = 0; i < count; ++i)
        out[i] = in[i];
}

void
test_array(void)
{
}
