#include "buffer.h"

#include "cuda_utils.h"
#include "errchk.h"

void
test_buffer()
{
    {
        const size_t count = 10;
        Buffer<double> a(count);
        a.arange();
        Buffer<double> b(count);
        migrate(a, b);
        ERRCHK(std::equal(a.data(), a.data() + count, b.data()));
    }
    {
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
        migrate(a, b);
        c.fill(0);
        migrate(b, c);
        migrate(c, d);
        c.fill(0);
        migrate(d, c);
        b.fill(0);
        migrate(c, b);
        ERRCHK(std::equal(a.data(), a.data() + a.size(), b.data()));
    }
    // {
    //     const size_t count = 10;
    //     Buffer<double, HostMemoryResource> a(count);
    //     Buffer<double, DeviceMemoryResource> b(count);
    //     Buffer<double, PinnedHostMemoryResource> c(count);

    //     std::unique_ptr<cudaStream_t, decltype(&cuda_stream_destroy)>
    //         stream{cuda_stream_create(cudaStreamNonBlocking), &cuda_stream_destroy};

    //     a.arange();
    //     migrate_async(*stream, a, b);
    //     migrate_async(*stream, b, c);
    //     ERRCHK_CUDA_API(cudaStreamSynchronize(*stream));
    //     ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
    // }
}
