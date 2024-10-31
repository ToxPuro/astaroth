#include "buffer.h"

#include "math_utils.h"

void
test_buffer()
{
    const size_t count = 10;
    Buffer<double> a(count);
    a.fill_arange(0, count);
    Buffer<double> b(count);
    a.migrate(b);
    ERRCHK(equals(a.count, a.data, b.data));
}
