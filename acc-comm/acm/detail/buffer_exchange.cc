#include "buffer_exchange.h"

#include <numeric>

#include "errchk.h"

void
test_buffer_exchange(void)
{
    const size_t                                      count{10};
    ac::buffer<double, ac::mr::host_allocator>        a(count);
    ac::buffer<double, ac::mr::device_allocator>      b(count);
    ac::buffer<double, ac::mr::pinned_host_allocator> c(count);

    BufferExchangeTask<double, ac::mr::pinned_host_allocator, ac::mr::device_allocator> htod(count);
    std::iota(a.begin(), a.end(), 0);
    htod.launch(a);
    htod.wait(b);

    BufferExchangeTask<double, ac::mr::device_allocator, ac::mr::pinned_host_allocator> dtoh(count);
    dtoh.launch(b);
    dtoh.wait(c);
    ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
    PRINT_LOG_INFO("OK");
}
