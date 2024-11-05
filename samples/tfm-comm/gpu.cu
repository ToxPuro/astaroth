#include <cstdlib>
#include <iostream>
#include <memory>

// #if defined(__CUDACC__)
// #define DEVICE_ENABLED
// #include "errchk_cuda.h"
// #include <cuda_runtime.h>
// #elif defined(__HIP_PLATFORM_AMD__)
// #define DEVICE_ENABLED
// #include "errchk_cuda.h"
// #include "hip.h"
// #include <hip/hip_runtime.h>
// #else
// #include "errchk.h"
// #define cudaStream_t void*
// #endif

#include "errchk.h"
#include "errchk_cuda.h"

#include "buf.h"

int
main(void)
{
    std::cout << "hello" << std::endl;

    const size_t count = 10;

    GenericBuffer<double, HostMemoryResource> a(count);
    GenericBuffer<double, PinnedHostMemoryResource> b(count);
    GenericBuffer<double, PinnedWriteCombinedHostMemoryResource> c(count);
    GenericBuffer<double, DeviceMemoryResource> d(count);
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
    a.display();

    return EXIT_SUCCESS;
}
