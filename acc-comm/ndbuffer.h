#pragma once
#include <iomanip>

#include "buffer.h"

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
    Shape shape;
    ac::buffer<T, MemoryResource> buffer;

    explicit ndbuffer(const Shape& in_shape)
        : shape{in_shape}, buffer(prod(in_shape))
    {
    }

    explicit ndbuffer(const Shape& in_shape, const T& fill_value)
        : shape{in_shape}, buffer(prod(in_shape), fill_value)
    {
    }

    T* begin() { return buffer.data(); }
    const T* begin() const { return buffer.data(); }
    T* end() { return buffer.data() + buffer.size(); }
    const T* end() const { return buffer.data() + buffer.size(); }

    template <typename OtherMemoryResource>
    void migrate(ac::ndbuffer<T, OtherMemoryResource>& other)
    {
        migrate(buffer, other.buffer);
    }

    void display() { ndbuffer_print_recursive(shape.size(), shape.data(), buffer.data()); }
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

template <typename T>
void
fill(const T& fill_value, const Shape& subdims, const Shape& offset,
     ac::ndbuffer<T, ac::mr::host_memory_resource>& ndbuf)
{
    ERRCHK(offset + subdims <= ndbuf.shape);
    ndbuffer_fill<T>(fill_value, ndbuf.shape.size(), ndbuf.shape.data(), subdims.data(),
                     offset.data(), ndbuf.buffer.data());
}

void test_ndbuffer(void);
