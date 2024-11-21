#include "buffer.h"

#include <numeric>

#include "errchk.h"

void
test_buffer()
{
    {
        const size_t count{10};
        Buffer<double> a(count);
        std::iota(a.begin(), a.end(), 0);
        Buffer<double> b(count);
        migrate(a, b);
        ERRCHK(std::equal(a.begin(), a.end(), b.begin()));
    }
    {
        const size_t count{10};

        Buffer<double, ac::mr::host_memory_resource> a(count);
        Buffer<double, ac::mr::pinned_host_memory_resource> b(count);
        Buffer<double, ac::mr::pinned_write_combined_host_memory_resource> c(count);
        Buffer<double, ac::mr::device_memory_resource> d(count);
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

        std::iota(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        migrate(a, b);
        std::fill(c.begin(), c.end(), 0);
        migrate(b, c);
        migrate(c, d);
        std::fill(c.begin(), c.end(), 0);
        migrate(d, c);
        std::fill(b.begin(), b.end(), 0);
        migrate(c, b);
        ERRCHK(std::equal(a.begin(), a.end(), b.begin()));
    }
    // {
    //     const size_t count{10};
    //     Buffer<double, ac::mr::host_memory_resource> a(count);
    //     Buffer<double, ac::mr::device_memory_resource> b(count);
    //     Buffer<double, ac::mr::pinned_host_memory_resource> c(count);

    //     std::unique_ptr<cudaStream_t, decltype(&cuda_stream_destroy)>
    //         stream{cuda_stream_create(cudaStreamNonBlocking), &cuda_stream_destroy};

    //     a.arange();
    //     migrate_async(*stream, a, b);
    //     migrate_async(*stream, b, c);
    //     ERRCHK_CUDA_API(cudaStreamSynchronize(*stream));
    //     ERRCHK(std::equal(a.data(), a.data() + a.size(), c.data()));
    // }
}
