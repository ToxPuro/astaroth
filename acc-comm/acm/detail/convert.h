#pragma once

#include "static_array.h"
#include "vector.h"

namespace ac {

template <typename Container, typename T = typename Container::value_type>
auto
unwrap(const Container& in)
{
    std::vector<decltype(std::declval<T>().data())> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [](const T& obj) { return obj.data(); });
    return out;
}

template <typename Container, typename T = typename Container::value_type>
auto
unwrap(Container& in)
{
    std::vector<decltype(std::declval<T>().data())> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [](T& obj) { return obj.data(); });
    return out;
}

#include "ndbuffer.h"
template <typename T, typename MemoryResource>
auto
ptr_cast(const ac::ndbuffer<T, MemoryResource>& in)
{
    return ac::mr::base_ptr<T, MemoryResource>{in.size(), in.data()};
}

template <typename T, typename MemoryResource>
auto
ptr_cast(const std::vector<ac::ndbuffer<T, MemoryResource>>& in)
{
    std::vector<T*> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

#include "buffer.h"
template <typename T, typename MemoryResource>
auto
ptr_cast(const ac::buffer<T, MemoryResource>& in)
{
    return ac::mr::base_ptr<T, MemoryResource>{in.size(), in.data()};
}

template <typename T, typename MemoryResource>
auto
ptr_cast(const std::vector<ac::buffer<T, MemoryResource>>& in)
{
    std::vector<T*> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

} // namespace ac
