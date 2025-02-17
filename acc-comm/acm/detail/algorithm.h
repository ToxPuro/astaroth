#pragma once
#include <algorithm>
#include <functional>
#include <numeric>

#include "errchk.h"
#include "ntuple.h"
#include "pointer.h"

#include "pack.h"

namespace ac {

template <typename... Inputs>
bool
same_size(const Inputs&... inputs)
{
    const size_t count{std::get<0>(std::tuple(inputs...)).size()};
    return ((inputs.size() == count) && ...);
}

template <typename T, typename Function>
void
transform(const ac::mr::host_pointer<T>& input, const Function& fn, ac::mr::host_pointer<T> output)
{
    ERRCHK(same_size(input, output));
    std::transform(input.begin(), input.end(), output.begin(), fn);
}

template <typename T, typename Function>
void
transform(const ac::mr::host_pointer<T>& a, const ac::mr::host_pointer<T>& b, const Function& fn,
          ac::mr::host_pointer<T> output)
{
    ERRCHK(same_size(a, b, output));
    for (size_t i{0}; i < output.size(); ++i)
        output[i] = fn(a[i], b[i]);
}

template <typename T, typename Function>
void
transform(const ac::mr::host_pointer<T>& a, const ac::mr::host_pointer<T>& b,
          const ac::mr::host_pointer<T>& c, const Function& fn, ac::mr::host_pointer<T> output)
{
    ERRCHK(same_size(a, b, c, output));
    for (size_t i{0}; i < output.size(); ++i)
        output[i] = fn(a[i], b[i], c[i]);
}

template <typename T, typename Function>
void
transform(const ac::mr::host_pointer<T>& a, const ac::mr::host_pointer<T>& b,
          const ac::mr::host_pointer<T>& c, const ac::mr::host_pointer<T>& d, const Function& fn,
          ac::mr::host_pointer<T> output)
{
    ERRCHK(same_size(a, b, c, d, output));
    for (size_t i{0}; i < output.size(); ++i)
        output[i] = fn(a[i], b[i], c[i], d[i]);
}

// More generic but confusing order of parameters needed to resolve ambiguity
// template <typename T, typename Function, typename... Inputs>
// void
// transform(ac::mr::host_pointer<T> output, const Function& fn, const Inputs&... inputs)
// {
//     static_assert(sizeof...(inputs) > 0);
//     ERRCHK(same_size(output, inputs...));

//     for (size_t i{0}; i < output.size(); ++i)
//         output[i] = fn(inputs[i]...);
// }

template <typename T, typename Function>
void
segmented_reduce(const size_t num_segments, const size_t stride,
                 const ac::mr::host_pointer<T>& input, const Function& fn, const T& initial_value,
                 ac::mr::host_pointer<T> output)
{
    for (size_t segment{0}; segment < num_segments; ++segment) {
        output[segment] = std::reduce(input.begin() + segment * stride,
                                      input.begin() + (segment + 1) * stride,
                                      initial_value,
                                      fn);
    }
}

} // namespace ac

void test_algorithm();
