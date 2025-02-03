#include <cstdlib>

#include "pointer.h"

void
test_pointer()
{
    ac::mr::host_pointer<int> a{0, nullptr};
    ac::mr::device_pointer<int> b{0, nullptr};
}
