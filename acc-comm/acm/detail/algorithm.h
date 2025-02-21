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

template <typename T>
void
xcorr(const ac::shape& mm, const ac::shape& nn, const ac::shape& nn_offset,
      const ac::mr::host_pointer<T>& input, const ac::shape& nk,
      const ac::mr::host_pointer<T>& kernel, ac::mr::host_pointer<T> output)
{
    ERRCHK(input.data() != output.data());
    ERRCHK(same_size(mm, nn, nn_offset, nk));
    ERRCHK(input.size() == output.size());
    for (uint64_t block_idx{0}; block_idx < prod(nn); ++block_idx) {
        const auto block_coords{ac::to_spatial(block_idx, nn)};
        const auto out_coords{nn_offset + block_coords};
        const auto out_idx{ac::to_linear(out_coords, mm)};

        output[out_idx] = 0;
        for (uint64_t kernel_idx{0}; kernel_idx < prod(nk); ++kernel_idx) {
            const auto kernel_coords{ac::to_spatial(kernel_idx, nk)};
            const auto diff{(nk - static_cast<uint64_t>(1)) / static_cast<uint64_t>(2)};
            const auto in_coords{out_coords - diff + kernel_coords};
            const auto in_idx{ac::to_linear(in_coords, mm)};
            output[out_idx] += input[in_idx] * kernel[kernel_idx];
        }
    }
}

} // namespace ac

void test_algorithm();
