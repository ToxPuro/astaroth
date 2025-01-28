#pragma once

#include "memory_resource.h"

namespace ac::mr {

template <typename T, typename MemoryResource> class base_ptr {
  private:
    const size_t m_count;
    T* m_data;

  public:
    explicit base_ptr(const size_t count, T* data)
        : m_count{count}, m_data{data}
    {
    }

    size_t size() const { return m_count; }
    T* data() { return m_data; }
    T* data() const { return m_data; }
    T* get() { return data(); }
    T* get() const { return data(); }

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
template <typename T> using host_ptr = base_ptr<T, ac::mr::host_memory_resource>;

#if defined(ACM_DEVICE_ENABLED)

template <typename MemoryResourceA, typename MemoryResourceB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceA>) {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
            PRINT_LOG("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
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
copy(const base_ptr<T, MemoryResourceA> in, base_ptr<T, MemoryResourceB> out)
{
    ERRCHK(in.size() <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(out.data(),
                               in.data(),
                               in.size() * sizeof(T),
                               get_kind<MemoryResourceA, MemoryResourceB>()));
}

template <typename T> using device_ptr = base_ptr<T, ac::mr::device_memory_resource>;

#else

template <typename T> using device_ptr = host_ptr<T>;

template <typename T>
void
copy(const host_ptr<T>& in, host_ptr<T>& out)
{
    ERRCHK(in.size() <= out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
}

#endif

} // namespace ac::mr

void test_pointer(void);
