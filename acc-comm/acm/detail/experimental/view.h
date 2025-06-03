#pragma once
#include <cstdlib>
#include <future>

#include "acm/detail/allocator.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/experimental/device_utils_experimental.h"

namespace ac {

template <typename T> class base_view {
  private:
    const size_t m_count{0};
    T*           m_data{nullptr};

  public:
    using value_type = T;

    base_view(const size_t count, T* data)
        : m_count{count}, m_data{data}
    {
    }

    base_view(T* begin, T* end)
        : base_view{as<size_t>(end - begin), begin}
    {
    }

    auto size() const { return m_count; }

    auto data() const { return m_data; }
    auto data() { return m_data; }

    auto begin() const { return data(); }
    auto begin() { return data(); }

    auto end() const { return data() + size(); }
    auto end() { return data() + size(); }

    // Enable the subscript[] operator
    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_data[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_data[i];
    }
};

// Example function
// template <typename T, typename U>
// void
// fn(const ac::view<T>& input, ac::view<U> output)
// {
//     static_assert(std::is_same_v<std::remove_const_t<T>, U>);
//     ERRCHK(input.size() <= output.size());
//     for (size_t i{0}; i < input.size(); ++i)
//         output[i] += input[i];
// }

// View specialization
template <typename T, typename Allocator> class view : public base_view<T> {
  public:
    using base_view<T>::base_view;

    template <template <typename, typename> typename Container>
    view(const Container<T, Allocator>& buffer)
        : view{buffer.size(), buffer.data()}
    {
    }
};

template <typename T, typename Allocator>
auto
make_view(const size_t count, const size_t offset, const view<T, Allocator>& view)
{
    ERRCHK(offset + count <= view.size());
    return ac::view<T, Allocator>{count, view.begin() + offset};
}

// Type aliases
template <typename T> using host_view   = view<T, ac::mr::host_allocator>;
template <typename T> using device_view = view<T, ac::mr::device_allocator>;

// Algorithms
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
copy_async(const host_view<T>& in, host_view<U> out)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    ERRCHK(in.size() <= out.size());
    return std::async(std::launch::async, copy<T, U>, in, out);
}
#endif

} // namespace ac
