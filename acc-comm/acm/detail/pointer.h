#pragma once

#include "view.h"

namespace ac::mr {

template <typename T, typename Allocator> using pointer = view<T, Allocator>;
template <typename T> using host_pointer                = host_view<T>;
template <typename T> using device_pointer              = device_view<T>;

} // namespace ac::mr

#if 0
#include <algorithm>
#include <type_traits>

#include "allocator.h"
#include "cuda_utils.h"
// #include "type_conversion.h"

namespace ac::mr {

template <typename T, typename Allocator> class pointer {
  private:
    size_t m_count{0};
    T*     m_data{nullptr};

  public:
    explicit pointer(const size_t count, T* data)
        : m_count{count}, m_data{data}
    {
    }

    __host__ __device__ auto size() const { return m_count; }

    auto data() const { return m_data; }
    auto data() { return m_data; }

    auto begin() const { return data(); }
    auto begin() { return data(); }

    auto end() const { return data() + size(); }
    auto end() { return data() + size(); }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_data[i];
    }

    __host__ __device__ const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_data[i];
    }
};

#if defined(ACM_DEVICE_ENABLED)

template <typename AllocatorA, typename AllocatorB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<ac::mr::device_allocator, AllocatorA>) {
        if constexpr (std::is_base_of_v<ac::mr::device_allocator, AllocatorB>) {
            PRINT_LOG_TRACE("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG_TRACE("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<ac::mr::device_allocator, AllocatorB>) {
            PRINT_LOG_TRACE("htod");
            return cudaMemcpyHostToDevice;
        }
        else {
            PRINT_LOG_TRACE("htoh");
            return cudaMemcpyHostToHost;
        }
    }
}

template <typename T, typename U, typename AllocatorA, typename AllocatorB>
void
copy(const size_t count, const size_t in_offset, const pointer<T, AllocatorA>& in,
     const size_t out_offset, pointer<U, AllocatorB>& out)
{
    // Check that T is the same type as U apart from constness
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);

    ERRCHK(in_offset + count <= in.size());
    ERRCHK(out_offset + count <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(&out[out_offset],
                               &in[in_offset],
                               count * sizeof(T),
                               get_kind<AllocatorA, AllocatorB>()));
}

// TODO: Draft
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// copy_async(const cudaStream_t stream, const size_t count, const size_t in_offset,
//            const pointer<T, AllocatorA>& in, const size_t out_offset, pointer<T, AllocatorB>&
//            out)
// {
//     ERRCHK(in_offset + count <= in.size());
//     ERRCHK(out_offset + count <= out.size());
//     ERRCHK_CUDA_API(cudaMemcpyAsync(&out[out_offset],
//                                     &in[in_offset],
//                                     count * sizeof(T),
//                                     get_kind<AllocatorA, AllocatorB>(),
//                                     stream));
// }

template <typename T> using host_pointer   = pointer<T, ac::mr::host_allocator>;
template <typename T> using device_pointer = pointer<T, ac::mr::device_allocator>;

// Does not work with MPI: why?
// template <typename T>
// void
// fill(const uint8_t fill_value, device_pointer<T> ptr)
// {
//     ERRCHK_CUDA_API(cudaMemset(ptr.data(), as<int>(fill_value), sizeof(ptr[0]) * ptr.size()));
// }
#else

template <typename T> using host_pointer   = pointer<T, ac::mr::host_allocator>;
template <typename T> using device_pointer = host_pointer<T>;

template <typename T, typename U>
void
copy(const size_t count, const size_t in_offset, const host_pointer<T>& in, const size_t out_offset,
     host_pointer<U>& out)
{
    // Check that T is the same type as U apart from constness
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);

    ERRCHK(in_offset + count <= in.size());
    ERRCHK(out_offset + count <= out.size());
    std::copy_n(&in[in_offset], count, &out[out_offset]);
}

// TODO: Draft
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// copy_async(const cudaStream_t stream, const size_t count, const size_t in_offset,
//            const pointer<T, AllocatorA>& in, const size_t out_offset, pointer<T, AllocatorB>&
//            out)
// {
//     PRINT_INFO("Device code not enabled, using synchronous host-to-host copy instead");
//     copy(count, in_offset, in, out_offset, out);
// }

#endif

template <typename T, typename U, typename AllocatorA, typename AllocatorB>
void
copy(const pointer<T, AllocatorA>& in, pointer<U, AllocatorB> out)
{
    // Check that T is the same type as U apart from constness
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    copy(in.size(), 0, in, 0, out);
}

// Does not work with MPI: why?
// template <typename T, typename Allocator>
// void
// fill(const uint8_t fill_value, pointer<T, Allocator> ptr)
// {
//     static_assert(std::is_base_of_v<ac::mr::host_allocator, Allocator>,
//                   "Only supported for host memory types");
//     memset(ptr.data(), as<int>(fill_value), sizeof(ptr[0]) * ptr.size());
// }

// TODO: Draft
// template <typename T, typename AllocatorA, typename AllocatorB>
// void
// copy_async(const pointer<T, AllocatorA>& in, pointer<T, AllocatorB>&& out)
// {
//     copy_async(in.size(), 0, in, 0, out);
// }

} // namespace ac::mr

template <typename T>
bool
equals(const ac::mr::host_pointer<T>& a, const ac::mr::host_pointer<T>& b)
{
    if (a.size() != b.size())
        return false;

    for (size_t i{0}; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

#endif
