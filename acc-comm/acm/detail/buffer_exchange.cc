#include "buffer_exchange.h"

#include <numeric>

#include "errchk.h"

void
test_buffer_exchange(void)
{
    const size_t count{10};
    ac::buffer<double, ac::mr::host_memory_resource> a(count);
    ac::buffer<double, ac::mr::device_memory_resource> b(count);
    ac::buffer<double, ac::mr::pinned_host_memory_resource> c(count);

    BufferExchangeTask<double, ac::mr::pinned_host_memory_resource, ac::mr::device_memory_resource>
        htod(count);
    std::iota(a.begin(), a.end(), 0);
    htod.launch(a);
    htod.wait(b);

    BufferExchangeTask<double, ac::mr::device_memory_resource, ac::mr::pinned_host_memory_resource>
        dtoh(count);
    dtoh.launch(b);
    dtoh.wait(c);
    ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
    PRINT_LOG_INFO("OK");
}
