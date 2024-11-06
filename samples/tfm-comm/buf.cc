#include "buf.h"

#include "errchk.h"

void
test_buf()
{
    const size_t count = 10;
    GenericBuffer<double> a(count);
    a.arange();
    GenericBuffer<double> b(count);
    migrate(a, b);
    ERRCHK(std::equal(a.data(), a.data() + count, b.data()));
}
