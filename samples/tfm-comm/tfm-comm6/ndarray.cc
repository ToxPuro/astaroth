#include "ndarray.h"

#include <iostream>

#include "buffer.h"
#include "shape.h"

#include "print_debug.h"

uint64_t
to_linear(const Index& coords, const Shape& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

Index
to_spatial(const uint64_t index, const Shape& shape)
{
    Index coords(shape.count);
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

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

        const uint64_t offset = prod(ndims - 1, dims);
        for (size_t i = start[ndims - 1]; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            ndarray_fill<T>(value, ndims - 1, dims, subdims, start, &arr[i * offset]);
    }
}

#include <iomanip>
template <typename T>
static void
ndarray_print_recursive(const size_t ndims, const uint64_t* dims, const T* array)
{
    if (ndims == 1) {
        for (size_t i = 0; i < dims[0]; ++i)
            std::cout << std::setw(3) << array[i];
        std::cout << std::endl;
    }
    else {
        const uint64_t offset = prod(ndims - 1, dims);
        for (size_t i = 0; i < dims[ndims - 1]; ++i) {
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
ndarray_print(const char* label, const size_t ndims, const size_t* dims, const T* array)
{
    ERRCHK(array != NULL);
    printf("%s:\n", label);
    ndarray_print_recursive<T>(ndims, dims, array);
}

template <typename T> struct NdArray {
    Shape shape;
    Buffer<T> buffer;

    // Constructor
    NdArray(const Shape& shape) : shape(shape), buffer(prod(shape)) {}

    NdArray(const Shape& shape, const T& fill_value) : shape(shape), buffer(prod(shape), fill_value)
    {
    }

    NdArray(const Shape& shape, const Buffer<T>& buffer) : shape(shape), buffer(buffer) {}

    void fill(const T& fill_value, const Shape& subdims, const Index& offset)
    {
        ERRCHK(subdims.count == offset.count);
        ndarray_fill<T>(fill_value, shape.count, shape.data, subdims.data, offset.data,
                        buffer.data);
    }

    void print() { ndarray_print_recursive(shape.count, shape.data, buffer.data); }
};

template <typename T>
std::ostream&
operator<<(std::ostream& os, const NdArray<T>& obj)
{
    os << "{";
    os << "\tshape: " << obj.shape << std::endl;
    os << "\tbuffer: " << obj.buffer << std::endl;
    os << "}";
    return os;
}

void
test_ndarray(void)
{
    std::cout << "hello" << std::endl;
    // NdArray<double> a(Shape{2, 2, 2});
    // NdArray<double> b(Shape{3, 3, 3}, 50);
    // // b.buffer = ones(20);
    // a = b;
    // Buffer<double> a(10);
    // auto a = ones<double>(10);
    // auto a = zeros<double>(10);
    // auto a = arange<double>(5, 20);
    // Buffer<double> a(10);
    // a.fill(1, dims, subdims, offset);
    // PRINT_DEBUG(a);
    // std::cout << a.buffer << std::endl;

    // NdArray<double> a(Shape{4, 4}, Buffer<double>(10, 50));

    // NdArray<double> mesh(Shape{8});
    // Shape dims(2);
    // mesh.fill(1, 7, 1);
    // std::cout << mesh << std::endl;

    Shape dims    = {4, 4, 4};
    Shape subdims = {2, 2, 2};
    Index offset  = {1, 1, 1};
    NdArray<double> mesh(dims);
    mesh.fill(1, subdims, offset);
    mesh.print();
    std::cout << mesh << std::endl;
}
