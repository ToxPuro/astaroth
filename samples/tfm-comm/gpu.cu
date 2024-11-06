#include <cstdlib>
#include <iostream>
#include <memory>

#include "errchk.h"
#include "errchk_cuda.h"

#include "buffer.h"
#include "buffer_exchange.h"

int
main(void)
{
    std::cout << "hello" << std::endl;

    const size_t count = 10;

    Buffer<double, HostMemoryResource> a(count);
    Buffer<double, PinnedHostMemoryResource> b(count);
    Buffer<double, PinnedWriteCombinedHostMemoryResource> c(count);
    Buffer<double, DeviceMemoryResource> d(count);
    migrate(a, a);
    migrate(a, b);
    migrate(a, c);
    migrate(a, d);

    migrate(b, a);
    migrate(b, b);
    migrate(b, c);
    migrate(b, d);

    migrate(c, a);
    migrate(c, b);
    migrate(c, c);
    migrate(c, d);

    migrate(d, a);
    migrate(d, b);
    migrate(d, c);
    migrate(d, d);

    a.arange();
    b.fill(0);
    c.fill(0);
    migrate(a, b);
    migrate(b, c);
    migrate(c, d);
    a.fill(0);
    migrate(d, a);
    PRINT_LOG("Initial");
    a.display();

    // a.fill(0);

    BufferExchangeTask<double, PinnedHostMemoryResource, DeviceMemoryResource> htod(count);
    htod.launch(a);
    htod.wait(d);

    PRINT_LOG("On device, launching back to host");
    BufferExchangeTask<double, DeviceMemoryResource, PinnedHostMemoryResource> dtoh(count);
    dtoh.launch(d);
    dtoh.wait(a);
    PRINT_LOG("After");
    a.display();

    return EXIT_SUCCESS;
}
