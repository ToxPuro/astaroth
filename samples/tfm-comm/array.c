#include "array.h"

#include "errchk.h"

void
copy(const size_t count, const size_t* in, size_t* out)
{
    // Disallow aliasing
    ERRCHK(!(in >= out && in < out + count));
    ERRCHK(!(in + count >= out && in + count < out + count));

    for (size_t i = 0; i < count; ++i)
        out[i] = in[i];
}

void
copyi(const size_t count, const int* in, int* out)
{
    // Disallow aliasing
    ERRCHK(!(in >= out && in < out + count));
    ERRCHK(!(in + count >= out && in + count < out + count));

    for (size_t i = 0; i < count; ++i)
        out[i] = in[i];
}

void
test_array(void)
{
}
