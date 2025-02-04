#pragma once
#include <cstddef>
#include <iostream>
#include <memory>

#include "allocator.h"
#include "pointer.h"

namespace ac {
template <typename T, typename Allocator> class buffer {
  private:
    const size_t                                      m_count;
    std::unique_ptr<T, decltype(&Allocator::dealloc)> m_resource;

  public:
    explicit buffer(const size_t count)
        : m_count{count},
          m_resource{static_cast<T*>(Allocator::alloc(count * sizeof(T))), Allocator::dealloc}
    {
    }

    explicit buffer(const size_t count, const T& fill_value)
        : buffer{count}
    {
        static_assert(std::is_base_of_v<ac::mr::host_allocator, Allocator>,
                      "Only supported for host memory types");
        std::fill(begin(), end(), fill_value);
    }

    auto size() const { return m_count; }

    auto data() const { return m_resource.get(); }
    auto data() { return m_resource.get(); }

    auto begin() const { return data(); }
    auto begin() { return data(); }

    auto end() const { return data() + size(); }
    auto end() { return data() + size(); }

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

    const ac::mr::pointer<T, Allocator> get() const
    {
        return ac::mr::pointer<T, Allocator>{size(), data()};
    }
    ac::mr::pointer<T, Allocator> get() { return ac::mr::pointer<T, Allocator>{size(), data()}; }

    void display() const
    {
        static_assert(std::is_base_of_v<ac::mr::host_allocator, Allocator>,
                      "Only enabled for host buffer");
        for (size_t i{0}; i < size(); ++i)
            std::cout << i << ": " << m_resource.get()[i] << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os, const ac::buffer<T, Allocator>& obj)
    {
        static_assert(std::is_base_of_v<ac::mr::host_allocator, Allocator>,
                      "Only enabled for host buffer");
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }
};

template <typename T> using host_buffer        = buffer<T, ac::mr::host_allocator>;
template <typename T> using pinned_host_buffer = buffer<T, ac::mr::pinned_host_allocator>;
template <typename T>
using pinned_write_combined_host_buffer   = buffer<T, ac::mr::pinned_write_combined_host_allocator>;
template <typename T> using device_buffer = buffer<T, ac::mr::device_allocator>;

} // namespace ac

template <typename T, typename AllocatorA, typename AllocatorB>
void
migrate(const ac::buffer<T, AllocatorA>& a, ac::buffer<T, AllocatorB>& b)
{
    ac::mr::copy(a.get(), b.get());
}

// TODO: Draft
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// migrate_async(const cudaStream_t stream, const ac::buffer<T, AllocatorA>& a,
//               ac::buffer<T, AllocatorB>& b)
// {
//     ac::mr::copy_async(a.get(), b.get());
// }

#if defined(ACM_DEVICE_ENABLED)
#include "cuda_utils.h"
#include "errchk_cuda.h"

template <typename T, typename AllocatorA, typename AllocatorB>
[[deprecated("Use ac::mr::copy_async instead")]] void
migrate_async(const cudaStream_t stream, const ac::buffer<T, AllocatorA>& in,
              ac::buffer<T, AllocatorB>& out)
{
    ERRCHK(in.size() == out.size());
    const cudaMemcpyKind kind{ac::mr::get_kind<AllocatorA, AllocatorB>()};
    ERRCHK_CUDA_API(
        cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in[0]), kind, stream));
}
#else
template <typename T, typename AllocatorA, typename AllocatorB>
void
migrate_async(const void* stream, const ac::buffer<T, AllocatorA>& in,
              ac::buffer<T, AllocatorB>& out)
{
    PRINT_LOG_DEBUG("non-cuda htoh async (note: blocking, stream ignored)");
    (void)stream; // Unused
    migrate(in, out);
}
#endif

// #if defined(ACM_DEVICE_ENABLED)

// #include "cuda_utils.h"
// #include "errchk_cuda.h"

// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// migrate(const ac::buffer<T, AllocatorA>& in, ac::buffer<T, AllocatorB>& out)
// {
//     ERRCHK(in.size() == out.size());
//     const cudaMemcpyKind kind{ac::mr::get_kind<AllocatorA, AllocatorB>()};
//     ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(in[0]), kind));
// }

// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// migrate_async(const cudaStream_t stream, const ac::buffer<T, AllocatorA>& in,
//               ac::buffer<T, AllocatorB>& out)
// {
//     ERRCHK(in.size() == out.size());
//     const cudaMemcpyKind kind{ac::mr::get_kind<AllocatorA, AllocatorB>()};
//     ERRCHK_CUDA_API(
//         cudaMemcpyAsync(out.data(), in.data(), in.size() * sizeof(in[0]), kind, stream));
// }
// #else
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// migrate(const ac::buffer<T, AllocatorA>& in, ac::buffer<T, AllocatorB>& out)
// {
//     PRINT_LOG_DEBUG("non-cuda htoh");
//     ERRCHK(in.size() == out.size());
//     std::copy(in.data(), in.data() + in.size(), out.data());
// }
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// migrate_async(const void* stream, const ac::buffer<T, AllocatorA>& in,
//               ac::buffer<T, AllocatorB>& out)
// {
//     PRINT_LOG_DEBUG("non-cuda htoh async (note: blocking, stream ignored)");
//     (void)stream; // Unused
//     migrate(in, out);
// }
// #endif

void test_buffer();
