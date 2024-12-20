#pragma once

#include "memory_resource.h"

namespace ac::mr {

template <typename T, typename MemoryResource> class base_ptr {
  protected:
    const size_t _size;
    T* _data;

  public:
    explicit base_ptr(const size_t size, T* data)
        : _size{size}, _data{data}
    {
    }

    size_t size() const { return _size; }
    T* data() { return _data; }
    T* data() const { return _data; }

    // Enable the subscript[] operator
    T& operator[](const size_t i)
    {
        ERRCHK(i < _size);
        return _data[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < _size);
        return _data[i];
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
