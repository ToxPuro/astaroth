#pragma once
#include <cstddef>

template <typename T> struct BaseBuffer {};

namespace host {
template <typename T> struct Buffer : public BaseBuffer<T> {
    static constexpr bool on_device = false;
    size_t count;
    std::unique_ptr<T> data;

    Buffer(const size_t count)
        : count(count), data(std::make_unique<T>(count))
    {
        std::cout << "host alloc and implicit dealloc" << std::endl;
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
    return std::unique_ptr<T, decltype(&dealloc<T>)>{ptr, &dealloc<T>};
}
template <typename T> struct Buffer : public BaseBuffer<T> {
    static constexpr bool on_device = true;
    size_t count;
    std::unique_ptr<T, decltype(&dealloc<T>)> data;

    Buffer(const size_t count)
        : count(count), data(make_unique<T>(count))
    {
    }
};
} // namespace device

namespace pinned {
template <typename T>
void
dealloc(T* ptr)
{
    std::cout << "cuda host dealloc" << std::endl;
    free(ptr);
}

template <typename T>
auto
make_unique(const size_t count)
{
    std::cout << "cuda host pinned alloc" << std::endl;
    T* ptr = (T*)malloc(count * sizeof(T));
    return std::unique_ptr<T, decltype(&dealloc<T>)>{ptr, &dealloc<T>};
}
template <typename T> struct Buffer : public BaseBuffer<T> {
    static constexpr bool on_device = false;
    size_t count;
    std::unique_ptr<T, decltype(&dealloc<T>)> data;

    Buffer(const size_t count)
        : count(count), data(make_unique<T>(count))
    {
    }
};

namespace wc {
template <typename T>
auto
make_unique(const size_t count)
{
    std::cout << "cuda host pinned write combined alloc" << std::endl;
    T* ptr = (T*)malloc(count * sizeof(T));
    return std::unique_ptr<T, decltype(&dealloc<T>)>{ptr, &dealloc<T>};
}
template <typename T> struct Buffer : public BaseBuffer<T> {
    static constexpr bool on_device = false;
    size_t count;
    std::unique_ptr<T, decltype(&dealloc<T>)> data;

    Buffer(const size_t count)
        : count(count), data(make_unique<T>(count))
    {
    }
};
} // namespace wc
} // namespace pinned

} // namespace cuda

template <typename T, typename U>
void
migrate(const T& in, U& out)
{
    if constexpr (T::on_device) {
        if constexpr (U::on_device)
            std::cout << "dtod" << std::endl;
        else
            std::cout << "dtoh" << std::endl;
    }
    else {
        if constexpr (U::on_device)
            std::cout << "htod" << std::endl;
        else
            std::cout << "htoh" << std::endl;
    }
}
