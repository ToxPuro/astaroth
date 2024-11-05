#pragma once
#include <cstddef>
#include <memory>

#include "mem.h"

template <typename T, template <typename> typename MemoryResource = HostMemoryResource>
struct GenericBuffer {
    const size_t count;
    std::unique_ptr<T[], decltype(&MemoryResource<T>::ndealloc)> data;

    GenericBuffer(const size_t in_count)
        : count{in_count}, data{MemoryResource<T>::nalloc(in_count), MemoryResource<T>::ndealloc}
    {
    }

    void fill(const T& value)
    {
        static_assert(std::is_base_of<HostMemoryResource<T>, MemoryResource<T>>::value);
        for (size_t i = 0; i < count; ++i)
            data[i] = value;
    }

    void arange(const size_t min = 0)
    {
        static_assert(std::is_base_of<HostMemoryResource<T>, MemoryResource<T>>::value);
        for (size_t i = 0; i < count; ++i)
            data[i] = static_cast<T>(min + i);
    }

    void display() const
    {
        static_assert(std::is_base_of<HostMemoryResource<T>, MemoryResource<T>>::value);
        for (size_t i = 0; i < count; ++i)
            std::cout << i << ": " << data[i] << std::endl;
    }
};

#if defined(DEVICE_ENABLED)
template <typename T, template <typename> typename MemoryResourceA,
          template <typename> typename MemoryResourceB>
typename std::enable_if<std::is_base_of<HostMemoryResource<T>, MemoryResourceA<T>>::value &&
                        std::is_base_of<HostMemoryResource<T>, MemoryResourceB<T>>::value>::type
migrate(const GenericBuffer<T, MemoryResourceA>& in, GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("htoh");
    ERRCHK(in.count == out.count);
    ERRCHK_CUDA_API(cudaMemcpy(out.data.get(), in.data.get(), sizeof(in.data.get()[0]) * in.count,
                               cudaMemcpyHostToHost));
}

template <typename T, template <typename> typename MemoryResourceA,
          template <typename> typename MemoryResourceB>
typename std::enable_if<std::is_base_of<DeviceMemoryResource<T>, MemoryResourceA<T>>::value &&
                        std::is_base_of<HostMemoryResource<T>, MemoryResourceB<T>>::value>::type
migrate(const GenericBuffer<T, MemoryResourceA>& in, GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("dtoh");
    ERRCHK(in.count == out.count);
    ERRCHK_CUDA_API(cudaMemcpy(out.data.get(), in.data.get(), sizeof(in.data.get()[0]) * in.count,
                               cudaMemcpyDeviceToHost));
}

template <typename T, template <typename> typename MemoryResourceA,
          template <typename> typename MemoryResourceB>
typename std::enable_if<std::is_base_of<HostMemoryResource<T>, MemoryResourceA<T>>::value &&
                        std::is_base_of<DeviceMemoryResource<T>, MemoryResourceB<T>>::value>::type
migrate(const GenericBuffer<T, MemoryResourceA>& in, GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("htod");
    ERRCHK(in.count == out.count);
    ERRCHK_CUDA_API(cudaMemcpy(out.data.get(), in.data.get(), sizeof(in.data.get()[0]) * in.count,
                               cudaMemcpyHostToDevice));
}

template <typename T, template <typename> typename MemoryResourceA,
          template <typename> typename MemoryResourceB>
typename std::enable_if<std::is_base_of<DeviceMemoryResource<T>, MemoryResourceA<T>>::value &&
                        std::is_base_of<DeviceMemoryResource<T>, MemoryResourceB<T>>::value>::type
migrate(const GenericBuffer<T, MemoryResourceA>& in, GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("dtod");
    ERRCHK(in.count == out.count);
    ERRCHK_CUDA_API(cudaMemcpy(out.data.get(), in.data.get(), sizeof(in.data.get()[0]) * in.count,
                               cudaMemcpyDeviceToDevice));
}
#else
template <typename T, template <typename> typename MemoryResourceA,
          template <typename> typename MemoryResourceB>
void
migrate(const GenericBuffer<T, MemoryResourceA>& in, const GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.count == out.count);
    std::copy(in.data.get(), in.data.get() + in.count, out.data.get());
}
#endif
