#pragma once

#include "memory_resource.h"

namespace ac::mr {

template <typename T, typename MemoryResource> class pointer {
  private:
    size_t m_count{0};
    T*     m_data{nullptr};

  public:
    pointer(const size_t count, T* data)
        : m_count{count}, m_data{data}
    {
    }

    auto size() const { return m_count; }

    auto data() const { return m_data; }
    auto data() { return m_data; }

    auto begin() const { return data(); }
    auto begin() { return data(); }

    auto end() const { return data() + size(); }
    auto end() { return data() + size(); }

    // Enable the subscript[] operator
    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_data[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_data[i];
    }
};

#if defined(ACM_DEVICE_ENABLED)

template <typename MemoryResourceA, typename MemoryResourceB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceA>) {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
            PRINT_LOG_TRACE("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG_TRACE("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
            PRINT_LOG_TRACE("htod");
            return cudaMemcpyHostToDevice;
        }
        else {
            PRINT_LOG_TRACE("htoh");
            return cudaMemcpyHostToHost;
        }
    }
}

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
copy(const size_t count, const size_t in_offset, const pointer<T, MemoryResourceA>& in,
     const size_t out_offset, pointer<T, MemoryResourceB>& out)
{
    ERRCHK(in_offset + count <= in.size());
    ERRCHK(out_offset + count <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(&out[out_offset],
                               &in[in_offset],
                               count * sizeof(T),
                               get_kind<MemoryResourceA, MemoryResourceB>()));
}

template <typename T> using host_pointer   = pointer<T, ac::mr::host_memory_resource>;
template <typename T> using device_pointer = pointer<T, ac::mr::device_memory_resource>;

#else

template <typename T> using host_pointer   = pointer<T, ac::mr::host_memory_resource>;
template <typename T> using device_pointer = host_pointer<T>;

template <typename T>
void
copy(const size_t count, const size_t in_offset, const host_pointer<T>& in, const size_t out_offset,
     host_pointer<T>& out)
{
    ERRCHK(in_offset + count <= in.size());
    ERRCHK(out_offset + count <= out.size());
    std::copy_n(&in[in_offset], count, &out[out_offset]);
}

#endif

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
copy(const pointer<T, MemoryResourceA>& in, pointer<T, MemoryResourceB>& out)
{
    copy(in.size(), 0, in, 0, out);
}

} // namespace ac::mr

void test_pointer(void);
