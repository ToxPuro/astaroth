#pragma once
#include <cstddef>
#include <memory>

#include "mem.h"

template <typename T, typename MemoryResource = HostMemoryResource> class GenericBuffer {
  private:
    const size_t count;
    std::unique_ptr<T[], decltype(&MemoryResource::dealloc)> resource;

  public:
    GenericBuffer(const size_t in_count)
        : count{in_count},
          resource{static_cast<T*>(MemoryResource::alloc(in_count * sizeof(T))),
                   MemoryResource::dealloc}
    {
    }

    T* data() const { return resource.get(); }
    size_t size() const { return count; }

    void fill(const T& value)
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            resource[i] = value;
    }

    void arange(const size_t min = 0)
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            resource[i] = static_cast<T>(min + i);
    }

    void display() const
    {
        static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        for (size_t i = 0; i < count; ++i)
            std::cout << i << ": " << resource[i] << std::endl;
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
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(in.resource[0]), kind));
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const cudaStream_t stream, const GenericBuffer<T, MemoryResourceA>& in,
              GenericBuffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(
        cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in.resource[0]), kind, stream));
}
#else
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const GenericBuffer<T, MemoryResourceA>& in, const GenericBuffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.size() == out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
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

void test_buf();
