#pragma once

struct BaseBuffer {};

namespace host {
template <typename T> struct Buffer : public BaseBuffer {
    size_t count;
    std::unique_ptr<T> data;

    Buffer(const size_t count)
        : count(count), data(std::make_unique<T>(count))
    {
    }
};
} // namespace host

namespace cuda {
namespace device {
template <typename T>
void
dealloc(T* ptr)
{
    std::cout << "device dealloc" << std::endl;
    free(ptr);
}

template <typename T>
auto
make_unique(const size_t count)
{
    std::cout << "device alloc" << std::endl;
    T* ptr = (T*)malloc(count * sizeof(T));
    return std::unique_ptr<T, decltype(&device::dealloc<T>)>{ptr, &device::dealloc<T>};
}

// template <typename T> struct Buffer : public BaseBuffer {
//     size_t count;
//     std::unique_ptr<T, decltype(&cuda::device::dealloc<T>)> data;

//     Buffer(const size_t count)
//         : count(count), data(cuda::device::make_unique<T>(count))
//     {
//     }
// };
} // namespace device
}; // namespace cuda

namespace cuda {
namespace device {
template <typename T> using Buffer = host::Buffer<T>;
} // namespace device
} // namespace cuda

template <typename T>
void
migrate(const cuda::device::Buffer<T>& in, cuda::device::Buffer<T>& out)
{
    std::cout << "dtod" << std::endl;
}
template <typename T>
void
migrate(const cuda::device::Buffer<T>& in, host::Buffer<T>& out)
{
    std::cout << "dtoh" << std::endl;
}
template <typename T>
void
migrate(const host::Buffer<T>& in, cuda::device::Buffer<T>& out)
{
    std::cout << "htod" << std::endl;
}
template <typename T>
void
migrate(const host::Buffer<T>& in, host::Buffer<T>& out)
{
    std::cout << "htoh" << std::endl;
}
// template <typename T> using DeviceBuffer = HostBuffer<T>;
