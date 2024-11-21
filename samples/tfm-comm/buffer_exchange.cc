#include "buffer_exchange.h"

#include <numeric>

#include "errchk.h"

void
test_buffer_exchange(void)
{
    const size_t count{10};
    ac::vector<double, HostMemoryResource> a(count);
    ac::vector<double, DeviceMemoryResource> b(count);
    ac::vector<double, PinnedHostMemoryResource> c(count);

    BufferExchangeTask<double, PinnedHostMemoryResource, DeviceMemoryResource> htod(count);
    std::iota(a.begin(), a.end(), 0);
    htod.launch(a);
    htod.wait(b);

    BufferExchangeTask<double, DeviceMemoryResource, PinnedHostMemoryResource> dtoh(count);
    dtoh.launch(b);
    dtoh.wait(c);
    ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
}
