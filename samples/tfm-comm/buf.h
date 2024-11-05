#pragma once
#include <cstddef>
#include <memory>

#include "mem.h"

template <typename T, typename MemoryResource = HostMemoryResource> struct GenericBuffer {
    const size_t count;
    std::unique_ptr<T[], decltype(&MemoryResource::dealloc)> data;

    GenericBuffer(const size_t in_count)
        : count{in_count},
          data{static_cast<T*>(MemoryResource::alloc(in_count * sizeof(T))),
               MemoryResource::dealloc}
    {
    }

    void fill(const T& value)
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            data[i] = value;
    }

    void arange(const size_t min = 0)
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            data[i] = static_cast<T>(min + i);
    }

    void display() const
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            std::cout << i << ": " << data[i] << std::endl;
    }
};

#if defined(DEVICE_ENABLED)
template <typename MemoryResourceA, typename MemoryResourceB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of<DeviceMemoryResource, MemoryResourceA>::value) {
        if constexpr (std::is_base_of<DeviceMemoryResource, MemoryResourceB>::value) {
            PRINT_LOG("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of<DeviceMemoryResource, MemoryResourceB>::value) {
            PRINT_LOG("htod");
            return cudaMemcpyHostToDevice;
        }
        else {
            PRINT_LOG("htoh");
            return cudaMemcpyHostToHost;
        }
    }
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const GenericBuffer<T, MemoryResourceA>& in, GenericBuffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.count == out.count);
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(
        cudaMemcpy(out.data.get(), in.data.get(), in.count * sizeof(in.data.get()[0]), kind));
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const cudaStream_t stream, const GenericBuffer<T, MemoryResourceA>& in,
              GenericBuffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.count == out.count);
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(cudaMemcpyAsync(out.data.get(), in.data.get(),
                                    in.count * sizeof(in.data.get()[0]), kind, stream));
}
#else
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const GenericBuffer<T, MemoryResourceA>& in, const GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.count == out.count);
    std::copy(in.data.get(), in.data.get() + in.count, out.data.get());
}
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const void* stream, const GenericBuffer<T, MemoryResourceA>& in,
              GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh async (not async in reality, blocks)");
    (void)stream; // Unused
    migrate(in, out);
}
#endif
