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
void
ndbuffer_print(const char* label, const size_t ndims, const uint64_t* dims, const T* array)
{
    ERRCHK(array != NULL);
    printf("%s:\n", label);
    ndbuffer_print_recursive<T>(ndims, dims, array);
}

namespace ac {
template <typename T, typename MemoryResource> struct ndbuffer {
  private:
    ac::vector<uint64_t>          m_shape;
    ac::buffer<T, MemoryResource> m_buffer;

  public:
    explicit ndbuffer(const ac::vector<uint64_t>& shape)
        : m_shape{shape}, m_buffer(prod(shape))
    {
    }

    explicit ndbuffer(const ac::vector<uint64_t>& shape, const T& fill_value)
        : m_shape{shape}, m_buffer(prod(shape), fill_value)
    {
    }

    auto size() const { return m_buffer.size(); }

    auto data() const { return m_buffer.data(); }
    auto data() { return m_buffer.data(); }

    auto begin() const { return m_buffer.data(); }
    auto begin() { return m_buffer.data(); }

    auto end() const { return m_buffer.data() + m_buffer.size(); }
    auto end() { return m_buffer.data() + m_buffer.size(); }

    auto get() const { return ac::mr::pointer<T, MemoryResource>{size(), data()}; }
    auto get() { return ac::mr::pointer<T, MemoryResource>{size(), data()}; }

    auto& shape() const { return m_shape; }
    auto& shape() { return m_shape; }

    auto& buffer() const { return m_buffer; }
    auto& buffer() { return m_buffer; }

    template <typename OtherMemoryResource>
    void migrate(ac::ndbuffer<T, OtherMemoryResource>& other)
    {
        migrate(m_buffer, other.m_buffer);
    }

    void display() { ndbuffer_print_recursive(m_shape.size(), m_shape.data(), m_buffer.data()); }
};
} // namespace ac

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
fill(const T& fill_value, const ac::vector<uint64_t>& subdims, const ac::vector<uint64_t>& offset,
     ac::ndbuffer<T, ac::mr::host_memory_resource>& ndbuf)
{
    ERRCHK(offset + subdims <= ndbuf.shape());
    ndbuffer_fill<T>(fill_value,
                     ndbuf.shape().size(),
                     ndbuf.shape().data(),
                     subdims.data(),
                     offset.data(),
                     ndbuf.buffer().data());
}
} // namespace ac

void test_ndbuffer(void);
