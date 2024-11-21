#include "ndvector.h"

void
test_ndvector(void)
{
    using Shape = ac::shape<2>;
    ac::ndvector<uint64_t, 2, HostMemoryResource> ndvec(Shape{4, 4}, 0);
    fill<uint64_t, 2>(1, Shape{2, 2}, Shape{1, 1}, ndvec);
    ERRCHK(ndvec.data()[0] == 0);
    ERRCHK(ndvec.data()[5] == 1);
    ERRCHK(ndvec.data()[10] == 1);
    ERRCHK(ndvec.data()[15] == 0);
}
