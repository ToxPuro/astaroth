#pragma once
#include <vector>

namespace ac {

template <typename Container, typename T = typename Container::value_type>
auto
unwrap_data(const Container& in)
{
    std::vector<decltype(std::declval<T>().data())> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [](const T& obj) { return obj.data(); });
    return out;
}

template <typename Container, typename T = typename Container::value_type>
auto
unwrap_data(Container& in)
{
    std::vector<decltype(std::declval<T>().data())> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [](T& obj) { return obj.data(); });
    return out;
}

template <typename T>
auto
unwrap_get(const std::vector<T>& in)
{
    std::vector<decltype(in.front().get())> out;

    for (const auto& elem : in)
        out.push_back(elem.get());

    return out;
}

template <typename T>
auto
unwrap_get(std::vector<T>& in)
{
    std::vector<decltype(in.front().get())> out;

    for (const auto& elem : in)
        out.push_back(elem.get());

    return out;
}

template <typename T>
auto
unwrap_ptr_get(const std::vector<T>& in)
{
    std::vector<decltype(in.front()->get())> out;

    for (const auto& elem : in)
        out.push_back(elem->get());

    return out;
}

template <typename T>
auto
unwrap_ptr_get(std::vector<T>& in)
{
    std::vector<decltype(in.front()->get())> out;

    for (const auto& elem : in)
        out.push_back(elem->get());

    return out;
}

#if 0
template <typename T, typename Allocator>
auto
ptr_cast(const ac::ndbuffer<T, Allocator>& in)
{
    return ac::mr::pointer<T, Allocator>{in.size(), in.data()};
}

template <typename T, typename Allocator>
auto
ptr_cast(const std::vector<ac::ndbuffer<T, Allocator>>& in)
{
    std::vector<ac::mr::pointer<T, Allocator>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

template <typename T, typename Allocator>
auto
ptr_cast(ac::ndbuffer<T, Allocator>& in)
{
    return ac::mr::pointer<T, Allocator>{in.size(), in.data()};
}

template <typename T, typename Allocator>
auto
ptr_cast(std::vector<ac::ndbuffer<T, Allocator>>& in)
{
    std::vector<ac::mr::pointer<T, Allocator>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

template <typename T, typename Allocator>
auto
ptr_cast(const ac::buffer<T, Allocator>& in)
{
    return ac::mr::pointer<T, Allocator>{in.size(), in.data()};
}

template <typename T, typename Allocator>
auto
ptr_cast(const std::vector<ac::buffer<T, Allocator>>& in)
{
    std::vector<ac::mr::pointer<T, Allocator>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

template <typename T, typename Allocator>
auto
ptr_cast(ac::buffer<T, Allocator>& in)
{
    return ac::mr::pointer<T, Allocator>{in.size(), in.data()};
}

template <typename T, typename Allocator>
auto
ptr_cast(std::vector<ac::buffer<T, Allocator>>& in)
{
    std::vector<ac::mr::pointer<T, Allocator>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}
#endif

} // namespace ac
