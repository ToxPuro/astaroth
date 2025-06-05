#pragma once
#include <cstdlib>

#include "acm/detail/allocator.h"
#include "acm/detail/type_conversion.h"

namespace ac {

template <typename T> class base_view {
  private:
    const size_t m_count{0};
    T*           m_data{nullptr};

  public:
    using value_type = T;

    base_view() = default;

    explicit base_view(const size_t count, T* data)
        : m_count{count}, m_data{data}
    {
    }

    explicit base_view(T* begin, T* end)
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

} // namespace ac
