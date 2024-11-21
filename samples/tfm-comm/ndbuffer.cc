#include "ndbuffer.h"

void
test_ndbuffer(void)
{
    using Shape = ac::shape<2>;
    ac::ndbuffer<uint64_t, 2, HostMemoryResource> mesh(Shape{4, 4}, 0);
    fill<uint64_t, 2>(1, Shape{2, 2}, Shape{1, 1}, mesh);
    ERRCHK(mesh.buffer[0] == 0);
    ERRCHK(mesh.buffer[5] == 1);
    ERRCHK(mesh.buffer[10] == 1);
    ERRCHK(mesh.buffer[15] == 0);
}
