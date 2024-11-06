#include "buffer.h"

#include "errchk.h"

void
test_buffer()
{
    const size_t count = 10;
    Buffer<double> a(count);
    a.arange();
    Buffer<double> b(count);
    migrate(a, b);
    ERRCHK(std::equal(a.data(), a.data() + count, b.data()));
}
