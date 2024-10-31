#include "buffer.h"

void
test_buffer()
{
    Buffer<double> a(10);
    // Buffer<double> b(10, BUFFER_DEVICE);
    a.fill(static_cast<double>(1));
    const auto c = arange<double>(0, 10);
}
