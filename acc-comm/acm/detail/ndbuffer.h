#pragma once
#include <iomanip>

#include "buffer.h"
#include "vector"

#include "math_utils.h"

template <typename T>
static void
ndbuffer_print_recursive(const size_t ndims, const uint64_t* dims, const T* array)
{
    if (ndims == 1) {
        for (size_t i{0}; i < dims[0]; ++i)
            std::cout << std::setw(4) << array[i];
        std::cout << std::endl;
    }
    else {
        const uint64_t offset{prod(ndims - 1, dims)};
        for (size_t i{0}; i < dims[ndims - 1]; ++i) {
            if (ndims > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);
            if (ndims == 4)
                printf("Cube %zu:\n", i);
            if (ndims == 3)
                printf("Layer %zu:\n", i);
            if (ndims == 2)
                printf("Row %zu: ", i);
            ndbuffer_print_recursive<T>(ndims - 1, dims, &array[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
[[deprecated("Use transform::print instead.")]] void
ndbuffer_print(const char* label, const size_t ndims, const uint64_t* dims, const T* array)
{
    ERRCHK(array != NULL);
    printf("%s:\n", label);
    ndbuffer_print_recursive<T>(ndims, dims, array);
}

namespace ac {
template <typename T, typename Allocator> struct ndbuffer {
  private:
    Shape                    m_shape;
    ac::buffer<T, Allocator> m_buffer;

  public:
    explicit ndbuffer(const Shape& shape)
        : m_shape{shape}, m_buffer{prod(shape)}
    {
    }

    explicit ndbuffer(const Shape& shape, const T& fill_value)
        : m_shape{shape}, m_buffer{prod(shape), fill_value}
    {
    }

    auto size() const { return m_buffer.size(); }

    auto data() const { return m_buffer.data(); }
    auto data() { return m_buffer.data(); }

    auto begin() const { return m_buffer.begin(); }
    auto begin() { return m_buffer.begin(); }

    auto end() const { return m_buffer.end(); }
    auto end() { return m_buffer.end(); }

    const ac::mr::pointer<T, Allocator> get() const { return m_buffer.get(); }
    ac::mr::pointer<T, Allocator>       get() { return m_buffer.get(); }

    Shape shape() const { return m_shape; }

    void reshape(const Shape& shape)
    {
        ERRCHK(prod(shape) == m_buffer.size());
        ERRCHK(prod(shape) == prod(m_shape));
        m_shape = shape;
    }

    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return data()[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return data()[i];
    }

    void display() { ndbuffer_print_recursive(m_shape.size(), m_shape.data(), m_buffer.data()); }
};

template <typename T> using host_ndbuffer        = ndbuffer<T, ac::mr::host_allocator>;
template <typename T> using pinned_host_ndbuffer = ndbuffer<T, ac::mr::pinned_host_allocator>;
template <typename T>
using pinned_write_combined_host_ndbuffer   = ndbuffer<T,
                                                       ac::mr::pinned_write_combined_host_allocator>;
template <typename T> using device_ndbuffer = ndbuffer<T, ac::mr::device_allocator>;

} // namespace ac

template <typename T, typename AllocatorA, typename AllocatorB>
void
migrate(const ac::ndbuffer<T, AllocatorA>& a, ac::ndbuffer<T, AllocatorB>& b)
{
    ac::mr::copy(a.get(), b.get());
}

template <typename T>
void
ndbuffer_fill(const T& value, const size_t ndims, const uint64_t* dims, const uint64_t* subdims,
              const uint64_t* start, T* arr)
{
    if (ndims == 0) {
        *arr = value;
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const uint64_t offset{prod(ndims - 1, dims)};
        for (size_t i{start[ndims - 1]}; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            ndbuffer_fill<T>(value, ndims - 1, dims, subdims, start, &arr[i * offset]);
    }
}

namespace ac {
template <typename T>
void
fill(const T& fill_value, const Shape& subdims, const Shape& offset,
     ac::ndbuffer<T, ac::mr::host_allocator>& ndbuf)
{
    ERRCHK(offset + subdims <= ndbuf.shape());
    ndbuffer_fill<T>(fill_value,
                     ndbuf.shape().size(),
                     ndbuf.shape().data(),
                     subdims.data(),
                     offset.data(),
                     ndbuf.data());
}
} // namespace ac

void test_ndbuffer(void);
