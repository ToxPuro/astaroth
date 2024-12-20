#pragma once

#include "static_array.h"
#include "vector.h"

namespace ac {

template <typename Container, typename T = typename Container::value_type>
auto
unwrap(const Container& in)
{
    std::vector<decltype(std::declval<T>().get())> out(in.size());
    std::transform(in.begin(), in.end(), out.begin(), [](const T& obj) { return obj.get(); });
    return out;
}

} // namespace ac
