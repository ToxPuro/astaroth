#include "buffer_exchange.h"

#include <numeric>

#include "errchk.h"

void
test_buffer_exchange(void)
{
    const size_t count{10};
    Buffer<double, HostMemoryResource> a(count);
    Buffer<double, DeviceMemoryResource> b(count);
    Buffer<double, PinnedHostMemoryResource> c(count);

    BufferExchangeTask<double, PinnedHostMemoryResource, DeviceMemoryResource> htod(count);
    std::iota(a.begin(), a.end(), 0);
    htod.launch(a);
    htod.wait(b);

    BufferExchangeTask<double, DeviceMemoryResource, PinnedHostMemoryResource> dtoh(count);
    dtoh.launch(b);
    dtoh.wait(c);
    ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
}
