#include "buffer.h"

void
test_buffer()
{
    Buffer<double> a(10);
    Buffer<double, DeviceAllocator> b(10);
    fill(static_cast<double>(1), a);
    const auto c = arange<double>(0, 10);
}
