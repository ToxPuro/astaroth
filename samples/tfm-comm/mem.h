#pragma once
#include <cstddef>
#include <memory>

#include <iostream>

#include "errchk.h"

template <typename T> using mem_t = std::unique_ptr<T[], void (*)(T*)>;

namespace host {
template <typename T>
void
dealloc(T* ptr) noexcept
{
    std::cout << "dealloc host" << std::endl;
    WARNCHK(ptr);
    free(ptr);
}

template <typename T>
mem_t<T>
make_unique(const size_t count)
{
    std::cout << "malloc host" << std::endl;
    T* ptr = (T*)malloc(count * sizeof(T));
    ERRCHK(ptr);
    return mem_t<T>(ptr, host::dealloc);
}
} // namespace host

namespace host::pinned {
template <typename T>
void
dealloc(T* ptr) noexcept
{
    std::cout << "dealloc host pinned" << std::endl;
    WARNCHK(ptr);
    WARNCHK_CUDA_API(cudaFreeHost(ptr));
}

template <typename T>
mem_t<T>
make_unique(const size_t count)
{
    std::cout << "malloc host pinned" << std::endl;
    T* ptr;
    ERRCHK_CUDA_API(cudaHostAlloc(&ptr, count * sizeof(ptr[0]), cudaHostAllocDefault));
    ERRCHK(ptr);
    return mem_t<T>(ptr, dealloc);
}
} // namespace host::pinned

namespace host::pinned::wc {
template <typename T>
mem_t<T>
make_unique(const size_t count)
{
    std::cout << "malloc host pinned wc" << std::endl;
    T* ptr = (T*)malloc(count * sizeof(T));
    ERRCHK(ptr);
    return mem_t<T>(ptr, dealloc);
}
} // namespace host::pinned::wc

namespace device {
template <typename T>
void
dealloc(T* ptr) noexcept
{
    std::cout << "dealloc device" << std::endl;
    WARNCHK(ptr);
    WARNCHK_CUDA_API(cudaFree(ptr));
}

template <typename T>
mem_t<T>
make_unique(const size_t count)
{
    std::cout << "malloc device" << std::endl;
    T* ptr;
    ERRCHK_CUDA_API(cudaMalloc(&ptr, count * sizeof(ptr[0])));
    ERRCHK(ptr);
    return mem_t<T>(ptr, dealloc);
}
} // namespace device
