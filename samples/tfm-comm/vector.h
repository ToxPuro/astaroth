#pragma once
#include <cstddef>
#include <memory>

#include "mem.h"

namespace ac {
template <typename T, typename MemoryResource> class vector {
  private:
    const size_t count;
    std::unique_ptr<T, decltype(&MemoryResource::dealloc)> resource;

  public:
    explicit vector(const size_t in_count)
        : count{in_count},
          resource{static_cast<T*>(MemoryResource::alloc(in_count * sizeof(T))),
                   MemoryResource::dealloc}
    {
    }

    explicit vector(const size_t in_count, const T& fill_value)
        : vector(in_count)
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>,
                      "Only supported for host memory types");
        std::fill(begin(), end(), fill_value);
    }

    // Enable subscript notation
    T& operator[](const size_t i)
    {
        ERRCHK(i < count);
        return data()[i];
    }
    const T& operator[](const size_t i) const
    {
        ERRCHK(i < count);
        return data()[i];
    }

    T* data() { return resource.get(); }
    const T* data() const { return resource.get(); }
    size_t size() const { return count; }

    T* begin() { return data(); }
    const T* begin() const { return data(); }
    T* end() { return data() + size(); }
    const T* end() const { return data() + size(); }

    // // Initializer list constructor
    // // ac::vector<int, 3> a{1,2,3}
    // vector(const std::initializer_list<T>& init_list)
    //     : vector(init_list.size())
    // {
    //     static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>,
    //                   "Only enabled for host vector");
    //     std::copy(init_list.begin(), init_list.end(), begin());
    // }

    void display() const
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>,
                      "Only enabled for host vector");
        for (size_t i{0}; i < count; ++i)
            std::cout << i << ": " << resource[i] << std::endl;
    }

    // friend std::ostream& operator<<(std::ostream& os, const ac::vector<T>& obj)
    // {
    //     static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>,
    //                   "Only enabled for host vector");
    //     os << "{ ";
    //     for (const auto& elem : obj)
    //         os << elem << " ";
    //     os << "}";
    //     return os;
    // }
};
} // namespace ac

#if defined(DEVICE_ENABLED)
#if defined(CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "hip.h"
#include <hip/hip_runtime.h>
#endif

#include "errchk_cuda.h"

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
migrate(const ac::vector<T, MemoryResourceA>& in, ac::vector<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind{get_kind<MemoryResourceA, MemoryResourceB>()};
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(in[0]), kind));
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const cudaStream_t stream, const ac::vector<T, MemoryResourceA>& in,
              ac::vector<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind{get_kind<MemoryResourceA, MemoryResourceB>()};
    ERRCHK_CUDA_API(
        cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in[0]), kind, stream));
}
#else
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const ac::vector<T, MemoryResourceA>& in, ac::vector<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.size() == out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
}
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const void* stream, const ac::vector<T, MemoryResourceA>& in,
              ac::vector<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh async (note: blocking, stream ignored)");
    (void)stream; // Unused
    migrate(in, out);
}
#endif

void test_vector();
