#pragma once
#include <future>

#include "acm/detail/device_utils.h"
#include "acm/detail/view.h"

namespace ac {

#if defined(ACM_DEVICE_ENABLED)
template <typename T, typename InAllocator, typename U, typename OutAllocator>
void
copy(const ac::view<T, InAllocator>& in, ac::view<U, OutAllocator> out)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    ERRCHK(in.size() <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(out.data(),
                               in.data(),
                               in.size() * sizeof(in[0]),
                               ac::device::get_kind<InAllocator, OutAllocator>()));
}

template <typename T, typename InAllocator, typename U, typename OutAllocator>
[[nodiscard]] auto
copy_async(const ac::view<T, InAllocator>& in, ac::view<U, OutAllocator> out)
{
    // Check that T is the same type as U apart from constness
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);

    // Check that buffers are the correct length
    ERRCHK(in.size() <= out.size());

    ac::device::stream stream;
    ERRCHK_CUDA_API(cudaMemcpyAsync(out.data(),
                                    in.data(),
                                    in.size() * sizeof(in[0]),
                                    ac::device::get_kind<InAllocator, OutAllocator>(),
                                    stream.get()));
    return stream;
}
#else
template <typename T, typename U>
void
copy(const ac::host_view<T>& in, ac::host_view<U> out)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    ERRCHK(in.size() <= out.size());
    std::copy(in.begin(), in.end(), out.begin());
}

template <typename T, typename U>
[[nodiscard]] auto
copy_async(const ac::host_view<T>& in, ac::host_view<U> out)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    ERRCHK(in.size() <= out.size());
    return std::async(std::launch::async, copy<T, U>, in, out);
}
#endif

template <typename T, typename U>
bool
equals(const ac::host_view<T>& a, const ac::host_view<U>& b)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    if (a.size() != b.size())
        return false;

    for (size_t i{0}; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;

    return true;
}

template <typename T>
void
fill(ac::host_view<T> view, const T& value)
{
    std::fill(view.begin(), view.end(), value);
}

} // namespace ac
