#include <cstdlib>
#include <iostream>
#include <memory>

#if defined(__CUDACC__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
#define cudaStream_t void*
#endif

#include "errchk.h"
#include "errchk_cuda.h"

#include "buf.h"

#if true
template <typename T> struct host_memory_resource {
    static T* alloc(const size_t count)
    {
        std::cout << "malloc host" << std::endl;
        T* ptr = (T*)malloc(count * sizeof(T));
        ERRCHK(ptr);
        return ptr;
    }

    static void dealloc(T* ptr) noexcept
    {
        std::cout << "dealloc host" << std::endl;
        WARNCHK(ptr);
        free(ptr);
    }
};
template <typename T> struct pinned_host_memory_resource : public host_memory_resource<T> {
    static T* alloc(const size_t count)
    {
        std::cout << "malloc host pinned" << std::endl;
        T* ptr = (T*)malloc(count * sizeof(T));
        ERRCHK(ptr);
        return ptr;
    }

    static void dealloc(T* ptr) noexcept
    {
        std::cout << "dealloc host pinned" << std::endl;
        WARNCHK(ptr);
        free(ptr);
    }
};
template <typename T>
struct pinned_write_combined_host_memory_resource : public host_memory_resource<T> {
    static T* alloc(const size_t count)
    {
        std::cout << "malloc host pinned wc" << std::endl;
        T* ptr = (T*)malloc(count * sizeof(T));
        ERRCHK(ptr);
        return ptr;
    }

    static void dealloc(T* ptr) noexcept
    {
        std::cout << "dealloc host pinned wc" << std::endl;
        WARNCHK(ptr);
        free(ptr);
    }
};
template <typename T> struct device_memory_resource {
    static T* alloc(const size_t count)
    {
        std::cout << "malloc device" << std::endl;
        T* ptr = nullptr;
        ERRCHK_CUDA_API(cudaMalloc(&ptr, count * sizeof(T)));
        return ptr;
    }

    static void dealloc(T* ptr) noexcept
    {
        std::cout << "dealloc device" << std::endl;
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFree(ptr));
    }
};

template <typename T, template <typename> typename Allocator = host_memory_resource> struct Buffer {
    const size_t count;
    std::unique_ptr<T[], decltype(&Allocator<T>::dealloc)> data;

    Buffer(const size_t in_count)
        : count{in_count}, data{Allocator<T>::alloc(in_count), Allocator<T>::dealloc}
    {
    }

    void arange()
    {
        static_assert(std::is_base_of<host_memory_resource<T>, Allocator<T>>::value);
        for (size_t i = 0; i < count; ++i)
            data[i] = static_cast<T>(i);
    }

    void display() const
    {
        static_assert(std::is_base_of<host_memory_resource<T>, Allocator<T>>::value);
        for (size_t i = 0; i < count; ++i)
            std::cout << i << ": " << data[i] << std::endl;
    }
};

template <typename T, template <typename> typename AllocatorA,
          template <typename> typename AllocatorB>
typename std::enable_if<std::is_base_of<host_memory_resource<T>, AllocatorA<T>>::value &&
                        std::is_base_of<host_memory_resource<T>, AllocatorB<T>>::value>::type
migrate(const Buffer<T, AllocatorA>& in, const Buffer<T, AllocatorB>& out)
{
    std::cout << "htoh" << std::endl;
}

template <typename T, template <typename> typename AllocatorA,
          template <typename> typename AllocatorB>
typename std::enable_if<std::is_base_of<device_memory_resource<T>, AllocatorA<T>>::value &&
                        std::is_base_of<host_memory_resource<T>, AllocatorB<T>>::value>::type
migrate(const Buffer<T, AllocatorA>& in, const Buffer<T, AllocatorB>& out)
{
    std::cout << "dtoh" << std::endl;
}

template <typename T, template <typename> typename AllocatorA,
          template <typename> typename AllocatorB>
typename std::enable_if<std::is_base_of<host_memory_resource<T>, AllocatorA<T>>::value &&
                        std::is_base_of<device_memory_resource<T>, AllocatorB<T>>::value>::type
migrate(const Buffer<T, AllocatorA>& in, const Buffer<T, AllocatorB>& out)
{
    std::cout << "htod" << std::endl;
}

template <typename T, template <typename> typename AllocatorA,
          template <typename> typename AllocatorB>
typename std::enable_if<std::is_base_of<device_memory_resource<T>, AllocatorA<T>>::value &&
                        std::is_base_of<device_memory_resource<T>, AllocatorB<T>>::value>::type
migrate(const Buffer<T, AllocatorA>& in, const Buffer<T, AllocatorB>& out)
{
    std::cout << "dtod" << std::endl;
}
#endif

int
main(void)
{
    std::cout << "hello" << std::endl;

#if true
    Buffer<double> hbuf(10);
    hbuf.arange(); //
    hbuf.data[0] = 100;
    hbuf.display();

    Buffer<double, pinned_host_memory_resource> hbuf_pinned(10);
    hbuf_pinned.arange();

    Buffer<double, host_memory_resource> a(10);
    Buffer<double, pinned_host_memory_resource> b(10);
    Buffer<double, pinned_write_combined_host_memory_resource> c(10);
    Buffer<double, device_memory_resource> d(10);
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
#else

    HostBufferDefault<double> hbuf(10);
    hbuf.arange(); //
    hbuf.data[0] = 100;
    hbuf.display();

    HostBufferPinned<double> hbuf_pinned(10);
    hbuf_pinned.arange(); //
    hbuf_pinned.data[0] = 100;
    hbuf_pinned.display();
#endif

    return EXIT_SUCCESS;
}
