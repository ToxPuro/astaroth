#include "ndbuffer.h"

void
test_ndbuffer(void)
{
    ac::ndbuffer<uint64_t, ac::mr::host_memory_resource> mesh(Shape{4, 4}, 0);
    ac::fill<uint64_t>(1, Shape{2, 2}, Shape{1, 1}, mesh);
    ERRCHK(mesh.buffer()[0] == 0);
    ERRCHK(mesh.buffer()[5] == 1);
    ERRCHK(mesh.buffer()[10] == 1);
    ERRCHK(mesh.buffer()[15] == 0);
    PRINT_LOG_INFO("OK");
}
