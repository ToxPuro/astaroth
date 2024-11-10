#pragma once
#include <cstddef>
#include <memory>

#include "mem.h"

template <typename T, typename MemoryResource = HostMemoryResource> class Buffer {
  private:
    const size_t count;
    std::unique_ptr<T[], decltype(&MemoryResource::dealloc)> resource;

  public:
    explicit Buffer(const size_t in_count)
        : count{in_count},
          resource{static_cast<T*>(MemoryResource::alloc(in_count * sizeof(T))),
                   MemoryResource::dealloc}
    {
    }

    // Enable subscript notation
    T& operator[](const size_t i) { ERRCHK(i < count); return resource[i]; }
    const T& operator[](const size_t i) const { ERRCHK(i < count); return resource[i]; }

    T* data() const { return resource.get(); }
    size_t size() const { return count; }

    void fill(const T& value)
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>);
        for (size_t i = 0; i < count; ++i)
            resource[i] = value;
    }

    void arange(const size_t min = 0)
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>);
        for (size_t i = 0; i < count; ++i)
            resource[i] = static_cast<T>(min + i);
    }

    void display() const
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>);
        for (size_t i = 0; i < count; ++i)
            std::cout << i << ": " << resource[i] << std::endl;
    }
};

#if defined(DEVICE_ENABLED)
template <typename MemoryResourceA, typename MemoryResourceB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<DeviceMemoryResource, MemoryResourceA>) {
        if constexpr (std::is_base_of_v<DeviceMemoryResource, MemoryResourceB>) {
            PRINT_LOG("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<DeviceMemoryResource, MemoryResourceB>) {
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
migrate(const Buffer<T, MemoryResourceA>& in, Buffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(in.data()[0]), kind));
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const cudaStream_t stream, const Buffer<T, MemoryResourceA>& in,
              Buffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind = get_kind<MemoryResourceA, MemoryResourceB>();
    ERRCHK_CUDA_API(
        cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in.data()[0]), kind, stream));
}
#else
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const Buffer<T, MemoryResourceA>& in, const Buffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.size() == out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
}
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const void* stream, const Buffer<T, MemoryResourceA>& in,
              Buffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh async (note: blocking, stream ignored)");
    (void)stream; // Unused
    migrate(in, out);
}
#endif

void test_buffer();
