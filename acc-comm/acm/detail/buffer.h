#pragma once
#include <cstddef>
#include <memory>

#include "memory_resource.h"
#include "pointer.h"

namespace ac {
template <typename T, typename MemoryResource> class buffer {
  private:
    const size_t m_count;
    std::unique_ptr<T, decltype(&MemoryResource::dealloc)> m_resource;

  public:
    explicit buffer(const size_t count)
        : m_count{count},
          m_resource{static_cast<T*>(MemoryResource::alloc(count * sizeof(T))),
                   MemoryResource::dealloc}
    {
    }

    explicit buffer(const size_t count, const T& fill_value)
        : buffer(count)
    {
        static_assert(std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
                      "Only supported for host memory types");
        std::fill(begin(), end(), fill_value);
    }

    // Enable subscript notation
    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return data()[i];
    }
    const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return data()[i];
    }

    T* data() { return m_resource.get(); }
    const T* data() const { return m_resource.get(); }
    size_t size() const { return m_count; }

    T* begin() { return data(); }
    const T* begin() const { return data(); }
    T* end() { return data() + size(); }
    const T* end() const { return data() + size(); }

    auto get() { return ac::mr::base_ptr<T, MemoryResource>{size(), data()}; }
    auto get() const { return ac::mr::base_ptr<T, MemoryResource>{size(), data()}; }

    // // Initializer list constructor
    // // ac::buffer<int, 3> a{1,2,3}
    // buffer(const std::initializer_list<T>& init_list)
    //     : buffer(init_list.size())
    // {
    //     static_assert(std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
    //                   "Only enabled for host buffer");
    //     std::copy(init_list.begin(), init_list.end(), begin());
    // }

    void display() const
    {
        static_assert(std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
                      "Only enabled for host buffer");
        for (size_t i{0}; i < size(); ++i)
            std::cout << i << ": " << m_resource.get()[i] << std::endl;
    }

    // friend std::ostream& operator<<(std::ostream& os, const ac::buffer<T, MemoryResource>& obj)
    // {
    //     static_assert(std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
    //                   "Only enabled for host buffer");
    //     os << "{ ";
    //     for (const auto& elem : obj)
    //         os << elem << " ";
    //     os << "}";
    //     return os;
    // }
};
} // namespace ac

#if defined(ACM_DEVICE_ENABLED)

#include "cuda_utils.h"
#include "errchk_cuda.h"

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const ac::buffer<T, MemoryResourceA>& in, ac::buffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind{ac::mr::get_kind<MemoryResourceA, MemoryResourceB>()};
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(in[0]), kind));
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const cudaStream_t stream, const ac::buffer<T, MemoryResourceA>& in,
              ac::buffer<T, MemoryResourceB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind{ac::mr::get_kind<MemoryResourceA, MemoryResourceB>()};
    ERRCHK_CUDA_API(
        cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in[0]), kind, stream));
}
#else
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate(const ac::buffer<T, MemoryResourceA>& in, ac::buffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh");
    ERRCHK(in.size() == out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
}
template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
migrate_async(const void* stream, const ac::buffer<T, MemoryResourceA>& in,
              ac::buffer<T, MemoryResourceB>& out)
{
    PRINT_LOG("non-cuda htoh async (note: blocking, stream ignored)");
    (void)stream; // Unused
    migrate(in, out);
}
#endif

void test_buffer();
