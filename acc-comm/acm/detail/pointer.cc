#include <cstdlib>

#include "pointer.h"

void
test_pointer()
{
    ac::mr::host_ptr<int> a{0, nullptr};
    ac::mr::device_ptr<int> b{0, nullptr};
}
