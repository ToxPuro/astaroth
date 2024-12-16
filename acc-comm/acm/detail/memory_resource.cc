#include <cstdlib>
#include <memory>

#include "memory_resource.h"

void
test_memory_resource()
{
    using namespace ac::mr;
    using host_unique_ptr = std::unique_ptr<double, decltype(&host_memory_resource::dealloc)>;
    auto a{host_unique_ptr{static_cast<double*>(host_memory_resource::alloc(10 * sizeof(double))),
                           host_memory_resource::dealloc}};

    using device_unique_ptr = std::unique_ptr<double, decltype(&device_memory_resource::dealloc)>;
    auto b{
        device_unique_ptr{static_cast<double*>(device_memory_resource::alloc(10 * sizeof(double))),
                          device_memory_resource::dealloc}};

    using pinned_host_unique_ptr = std::unique_ptr<double,
                                                   decltype(&pinned_host_memory_resource::dealloc)>;
    auto c{pinned_host_unique_ptr{static_cast<double*>(
                                      pinned_host_memory_resource::alloc(10 * sizeof(double))),
                                  pinned_host_memory_resource::dealloc}};

    using pinned_write_combined_host_unique_ptr = std::unique_ptr<
        double, decltype(&pinned_write_combined_host_memory_resource::dealloc)>;
    auto d{
        pinned_write_combined_host_unique_ptr{static_cast<double*>(
                                                  pinned_write_combined_host_memory_resource::alloc(
                                                      10 * sizeof(double))),
                                              pinned_write_combined_host_memory_resource::dealloc}};
}
