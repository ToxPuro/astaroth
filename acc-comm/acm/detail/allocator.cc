#include <cstdlib>
#include <memory>

#include "allocator.h"

void
test_allocator()
{
    using namespace ac::mr;
    using host_unique_ptr = std::unique_ptr<double, decltype(&host_allocator::dealloc)>;
    auto a{host_unique_ptr{static_cast<double*>(host_allocator::alloc(10 * sizeof(double))),
                           host_allocator::dealloc}};

    using device_unique_ptr = std::unique_ptr<double, decltype(&device_allocator::dealloc)>;
    auto b{device_unique_ptr{static_cast<double*>(device_allocator::alloc(10 * sizeof(double))),
                             device_allocator::dealloc}};

    using pinned_host_unique_ptr = std::unique_ptr<double,
                                                   decltype(&pinned_host_allocator::dealloc)>;
    auto c{pinned_host_unique_ptr{static_cast<double*>(
                                      pinned_host_allocator::alloc(10 * sizeof(double))),
                                  pinned_host_allocator::dealloc}};

    using pinned_write_combined_host_unique_ptr = std::
        unique_ptr<double, decltype(&pinned_write_combined_host_allocator::dealloc)>;
    auto d{pinned_write_combined_host_unique_ptr{static_cast<double*>(
                                                     pinned_write_combined_host_allocator::alloc(
                                                         10 * sizeof(double))),
                                                 pinned_write_combined_host_allocator::dealloc}};
    PRINT_LOG_INFO("OK");
}
