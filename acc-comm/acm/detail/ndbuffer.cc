#include "ndbuffer.h"

void
test_ndbuffer(void)
{
    ac::ndbuffer<uint64_t, ac::mr::host_allocator> mesh(ac::Shape{4, 4}, 0);
    ac::fill<uint64_t>(1, ac::Shape{2, 2}, ac::Shape{1, 1}, mesh);
    ERRCHK(mesh[0] == 0);
    ERRCHK(mesh[5] == 1);
    ERRCHK(mesh[10] == 1);
    ERRCHK(mesh[15] == 0);
    PRINT_LOG_INFO("OK");
}
