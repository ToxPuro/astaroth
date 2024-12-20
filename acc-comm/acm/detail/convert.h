#pragma once

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

#if 0
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
    std::vector<ac::mr::base_ptr<T, MemoryResource>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

template <typename T, typename MemoryResource>
auto
ptr_cast(ac::ndbuffer<T, MemoryResource>& in)
{
    return ac::mr::base_ptr<T, MemoryResource>{in.size(), in.data()};
}

template <typename T, typename MemoryResource>
auto
ptr_cast(std::vector<ac::ndbuffer<T, MemoryResource>>& in)
{
    std::vector<ac::mr::base_ptr<T, MemoryResource>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

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
    std::vector<ac::mr::base_ptr<T, MemoryResource>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}

template <typename T, typename MemoryResource>
auto
ptr_cast(ac::buffer<T, MemoryResource>& in)
{
    return ac::mr::base_ptr<T, MemoryResource>{in.size(), in.data()};
}

template <typename T, typename MemoryResource>
auto
ptr_cast(std::vector<ac::buffer<T, MemoryResource>>& in)
{
    std::vector<ac::mr::base_ptr<T, MemoryResource>> out;
    for (const auto& elem : in)
        out.push_back(ptr_cast(elem));
    return out;
}
#endif

} // namespace ac
