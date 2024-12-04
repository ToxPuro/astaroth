#include <cstdlib>

#include "memory_resource.h"

void
test_memory_resource()
{
    ac::mr::device_ptr<int> ptr{0, nullptr};
}

int
main()
{
    test_memory_resource();
    return EXIT_SUCCESS;
}
