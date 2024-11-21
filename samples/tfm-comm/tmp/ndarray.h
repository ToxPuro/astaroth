#pragma once

// #include "buffer.h"
#include "datatypes.h"
#include "math_utils.h"

#include <iomanip>
#include <iostream>

template <typename T>
void
ndarray_fill(const T& value, const size_t ndims, const uint64_t* dims, const uint64_t* subdims,
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
            ndarray_fill<T>(value, ndims - 1, dims, subdims, start, &arr[i * offset]);
    }
}

template <typename T>
static void
ndarray_print_recursive(const size_t ndims, const uint64_t* dims, const T* array)
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
            ndarray_print_recursive<T>(ndims - 1, dims, &array[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
void
ndarray_print(const char* label, const size_t ndims, const uint64_t* dims, const T* array)
{
    ERRCHK(array != NULL);
    printf("%s:\n", label);
    ndarray_print_recursive<T>(ndims, dims, array);
}

template <typename T, size_t N, typename MemoryResource = HostMemoryResource> struct NdArray {
    ac::shape<N> shape{};
    Buffer<T, MemoryResource> buffer;

    // Constructor
    explicit NdArray(const ac::shape<N>& in_shape)
        : shape{in_shape}, buffer{prod(in_shape)}
    {
    }

    T* data() { return buffer.data(); }
    const T* data() const { return buffer.data(); }
    size_t size() const { return buffer.size(); }

    T* begin() { return data(); }
    const T* begin() const { return data(); }
    T* end() { return data() + size(); }
    const T* end() const { return data() + size(); }

    void fill(const T& fill_value, const ac::shape<N>& subdims, const ac::index<N>& offset)
    {
        ERRCHK(subdims.size() == offset.size());
        ndarray_fill<T>(fill_value, shape.size(), shape.data(), subdims.data(), offset.data(),
                        buffer.data());
    }

    void display() { ndarray_print_recursive(shape.size(), shape.data(), buffer.data()); }

    friend std::ostream& operator<<(std::ostream& os, const NdArray<T, N, MemoryResource>& obj)
    {
        static_assert(std::is_base_of_v<HostMemoryResource, MemoryResource>,
                      "Can currently print only host memory");
        os << "{";
        os << "shape: " << obj.shape << ", ";
        os << "buffer: " << obj.buffer;
        os << "}";
        return os;
    }
};

void test_ndarray(void);
